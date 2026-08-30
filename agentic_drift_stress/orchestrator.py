"""
LangGraph orchestrator -- agentic_drift_stress.

Wires the 5 agents (drift_detector, feature_intelligence, target_drift_agent,
performance_drift_agent, alert_agent + remediation_agent) into a SINGLE graph,
routed by state["trigger"]:
- "batch": every inference batch -> data_drift + feature_intelligence (if
  drift is significant) + best-effort target_drift (PROXY/PENDING) +
  staleness_check + alerting/remediation.
- "labeled_window": fired when a labeled window is full -> performance_drift
  (LABELED) + confirmed target_drift (CONFIRMED) + alerting/remediation
  (may trigger process_root_cause, see alert_agent.py).

Stateful agents (DriftDetector, TargetDriftAgent, PerformanceDriftAgent) live
in a process-local registry keyed by model_id, NOT in the checkpointed
LangGraph State -- only serializable results go in State.
"""

from __future__ import annotations

import os
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Set, TypedDict

import pandas as pd

from drift_detector import DriftDetector, DriftReport, DriftSeverity
from target_drift_agent import TargetDriftAgent, TargetDriftResult
from performance_drift_agent import (
    PerformanceDriftAgent,
    PerformanceDriftResult,
    TaskType,
    LabelAvailability,
)
from feature_intelligence import (
    FeatureIntelligenceContext,
    FeatureRiskAssessor,
    CombinedFeatureVerdict,
)
from alert_agent import AlertAgent
from remediation_agent import RemediationAgent, remediation_node as _remediation_node_impl
from diagnostic_report import DiagnosticReportGenerator, report_node as _report_node_impl

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# Whitelist our dataclasses/Enums explicitly: JsonPlusSerializer's implicit
# fallback for unregistered types is deprecated and will be blocked in a
# future langgraph version.
_ALLOWED_MSGPACK_MODULES = [
    ("drift_detector", "DriftSeverity"), ("drift_detector", "FeatureType"),
    ("drift_detector", "FeatureDriftResult"), ("drift_detector", "DriftReport"),
    ("target_drift_agent", "LabelStatus"), ("target_drift_agent", "TargetDriftResult"),
    ("performance_drift_agent", "TaskType"), ("performance_drift_agent", "LabelAvailability"),
    ("performance_drift_agent", "PerformanceSeverity"), ("performance_drift_agent", "PerformanceMetricResult"),
    ("performance_drift_agent", "PerformanceDriftResult"),
    ("feature_intelligence", "RiskLevel"),
    ("alert_agent", "AlertPriority"), ("alert_agent", "AlertChannel"),
    ("alert_agent", "RemediationType"), ("alert_agent", "Alert"),
    ("remediation_agent", "RemediationAction"),
]


# Guard: block deploys where AGENTIC_ENV=production would silently fall back
# to the in-memory checkpointer (lost on restart, no horizontal scaling).
# Override with AGENTIC_ALLOW_INMEMORY_STATE=1 when that's intentional (e.g.
# a prod smoke test).
def _assert_not_unsafe_in_production(component: str) -> None:
    env = os.environ.get("AGENTIC_ENV", "").strip().lower()
    allow_override = os.environ.get("AGENTIC_ALLOW_INMEMORY_STATE", "").strip() == "1"
    if env == "production" and not allow_override:
        raise RuntimeError(
            f"AGENTIC_ENV=production but {component} defaults to an IN-MEMORY "
            f"backend (no restart tolerance, no horizontal scaling). Provide a "
            f"persistent/shared backend (e.g. PostgresSaver for the "
            f"checkpointer, a Redis/DB registry for stateful agents) before "
            f"deploying, or set AGENTIC_ALLOW_INMEMORY_STATE=1 if intentional "
            f"(e.g. smoke test)."
        )


def _make_checkpointer() -> MemorySaver:
    """MemorySaver for local dev / smoke tests. In production, swap for
    PostgresSaver(serde=JsonPlusSerializer(...)) with the same
    allowed_msgpack_modules."""
    _assert_not_unsafe_in_production("the LangGraph checkpointer (get_default_checkpointer)")
    return MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_MSGPACK_MODULES))


# Singleton kept as a practical default now that there's a single compiled
# graph (so a single checkpointer).
_DEFAULT_SHARED_CHECKPOINTER: Optional[MemorySaver] = None


def get_default_checkpointer() -> MemorySaver:
    global _DEFAULT_SHARED_CHECKPOINTER
    if _DEFAULT_SHARED_CHECKPOINTER is None:
        _DEFAULT_SHARED_CHECKPOINTER = _make_checkpointer()
    return _DEFAULT_SHARED_CHECKPOINTER


# ------------------------------------------------------------------------ #
# Stateful agent registry, outside the LangGraph State (see module docstring)
# ------------------------------------------------------------------------ #

class _BoundedIdSet:
    """Bounded (FIFO-eviction) set for processed_batch_ids -- avoids an
    unbounded memory leak for models monitored continuously for months.
    Only recent batch_id values are needed for idempotence."""

    def __init__(self, maxlen: int = 5000):
        self._order: Deque[str] = deque(maxlen=maxlen)
        self._members: Set[str] = set()

    def __contains__(self, item: str) -> bool:
        return item in self._members

    def add(self, item: str) -> None:
        if item in self._members:
            return
        if len(self._order) == self._order.maxlen:
            oldest = self._order.popleft()
            self._members.discard(oldest)
        self._order.append(item)
        self._members.add(item)


@dataclass
class ModelAgents:
    drift_detector: DriftDetector
    target_drift_agent: TargetDriftAgent
    performance_drift_agent: PerformanceDriftAgent
    feature_intelligence_context: Optional[FeatureIntelligenceContext] = None
    # Idempotence guard: never re-append the same batch_id twice to
    # PerformanceDriftAgent's internal window (e.g. on a LangGraph retry).
    processed_batch_ids: _BoundedIdSet = field(default_factory=_BoundedIdSet)
    # Per-model lock: protects mutations on the stateful agents themselves
    # (detect_drift/detect/update all touch non-thread-safe internal state),
    # not just registry access.
    lock: threading.Lock = field(default_factory=threading.Lock)


_AGENT_REGISTRY: Dict[str, ModelAgents] = {}
# Single lock shared by both registries below (they're mutated together in
# the same flows: stage_batch + bootstrap_agents). Protects intra-process
# concurrency only -- no multi-process scale-out.
_STATE_LOCK = threading.Lock()

# Cap for lists that accumulate cycle after cycle in the checkpointed State
# (alerts, remediation_actions, notes), so a long-running model doesn't grow
# the checkpoint unboundedly. Full history belongs in an external
# alerting/tracking system, not in LangGraph State.
_MAX_STATE_HISTORY = 200


def _append_capped(existing: List[Any], new_items: List[Any], cap: int = _MAX_STATE_HISTORY) -> List[Any]:
    return (existing + new_items)[-cap:]


# ------------------------------------------------------------------------ #
# Raw per-batch data (features_df, y_true, y_pred, y_proba), also kept out of
# the checkpointed State -- here because a pd.DataFrame breaks LangGraph's
# msgpack serialization outright. Only model_id/batch_id go through State;
# nodes look the raw data back up from this registry.
# ------------------------------------------------------------------------ #

_BATCH_DATA: Dict[str, Dict[str, Any]] = {}


def stage_batch(
    batch_id: str,
    features_df: Optional[pd.DataFrame] = None,
    y_true: Optional[Any] = None,
    y_pred: Optional[Any] = None,
    y_proba: Optional[Any] = None,
) -> None:
    """Call before graph.invoke(...) (trigger='batch' or 'labeled_window')
    to stage the cycle's raw data. Deliberately kept out of MonitoringState."""
    with _STATE_LOCK:
        _BATCH_DATA[batch_id] = {
            "features_df": features_df, "y_true": y_true, "y_pred": y_pred, "y_proba": y_proba,
        }


def get_batch(batch_id: str) -> Dict[str, Any]:
    with _STATE_LOCK:
        return _BATCH_DATA.get(batch_id, {})


def clear_batch(batch_id: str) -> None:
    """Call once a cycle is done so this registry doesn't grow unbounded."""
    with _STATE_LOCK:
        _BATCH_DATA.pop(batch_id, None)


def bootstrap_agents(
    model_id: str,
    task_type: TaskType,
    reference_features: pd.DataFrame,
    y_reference: Any,
    y_true_baseline: Any,
    y_pred_baseline: Any,
    y_proba_baseline: Optional[Any] = None,
    feature_intelligence_context: Optional[FeatureIntelligenceContext] = None,
    **performance_kwargs: Any,
) -> ModelAgents:
    """Single entry point to (re)initialize a monitored model's stateful
    agents. Call once at startup, then on every real retrain (a new
    baseline resets the reference for all 3 detectors)."""
    _assert_not_unsafe_in_production("the stateful agent registry (_AGENT_REGISTRY)")

    detector = DriftDetector()
    detector.set_reference(reference_features)

    target_agent = TargetDriftAgent()
    target_agent.set_reference(y_reference)

    perf_agent = PerformanceDriftAgent(task_type=task_type, **performance_kwargs)
    perf_agent.set_baseline(y_true_baseline, y_pred_baseline, y_proba_baseline)

    agents = ModelAgents(
        drift_detector=detector,
        target_drift_agent=target_agent,
        performance_drift_agent=perf_agent,
        feature_intelligence_context=feature_intelligence_context,
    )
    with _STATE_LOCK:
        _AGENT_REGISTRY[model_id] = agents
    return agents


def get_agents(model_id: str) -> ModelAgents:
    with _STATE_LOCK:
        if model_id not in _AGENT_REGISTRY:
            raise KeyError(
                f"Agents not initialized for model_id='{model_id}'. "
                "Call bootstrap_agents() before invoking the graph."
            )
        return _AGENT_REGISTRY[model_id]


# ------------------------------------------------------------------------ #
# Shared (checkpointed) state. Never holds ModelAgents objects.
# ------------------------------------------------------------------------ #

class MonitoringState(TypedDict, total=False):
    model_id: str
    batch_id: str
    trigger: str  # "batch" | "labeled_window" -- set by the caller before invoke()
    # No features_df/y_true/y_pred/y_proba here -- see _BATCH_DATA above.

    # Detection outputs (dataclasses/enums, not plain dicts)
    data_drift_report: Optional[DriftReport]
    target_drift_result: Optional[TargetDriftResult]
    performance_drift_result: Optional[PerformanceDriftResult]
    # No raw feature_intelligence here (FeatureIntelligenceContext can carry a
    # correlation_matrix DataFrame -- same serialization issue as features_df).
    # Resolved via agents.feature_intelligence_context instead.
    combined_feature_verdicts: List[Dict[str, Any]]

    # Alerting / remediation outputs
    alerts: List[dict]
    # Last-alerted signature (timestamp) per source, to avoid re-dispatching
    # the same alert every cycle while the underlying report stays in state.
    _alert_signatures: Dict[str, str]
    # Raw Alert objects (not to_dict()), consumed by remediation_node. Every
    # State key is checkpointed regardless of a leading underscore -- this
    # only works because Alert is in _ALLOWED_MSGPACK_MODULES.
    _alert_objects: List[Any]
    remediation_actions: List[dict]
    # Raw RemediationAction objects for THIS CYCLE ONLY (mirrors
    # _alert_objects above), consumed by diagnostic_report.py::report_node.
    # RemediationAction is in _ALLOWED_MSGPACK_MODULES, but that alone isn't
    # enough: a key returned by a node must ALSO be declared here for the
    # compiled graph to track/checkpoint it as a channel between nodes --
    # this key was missing, so remediation_node's output silently never
    # reached report_node (see audit note below build_graph()).
    _remediation_action_objects: List[Any]
    escalate: bool
    pending_approval: bool

    # Diagnostic report (6th agent, optional -- see diagnostic_report.py).
    # Only the on-disk path goes in State, not the report content itself.
    report_path: Optional[str]
    # None if PDF rendering failed this cycle (best-effort, see
    # diagnostic_report.py::DiagnosticReportGenerator.save()) -- callers
    # fall back to attaching report_path (Markdown) in that case.
    report_pdf_path: Optional[str]
    report_status: Optional[str]
    # Must be declared here (not just returned by report_node) to be
    # tracked as a channel by the compiled graph -- see the audit note
    # above about _remediation_action_objects for what happens otherwise.
    report_executive_summary: Optional[str]

    # Labeling-pipeline staleness
    label_pending_streak: int
    staleness_alert: Optional[str]

    notes: List[str]


# ------------------------------------------------------------------------ #
# Nodes -- fast loop
# ------------------------------------------------------------------------ #

def data_drift_node(state: MonitoringState) -> MonitoringState:
    agents = get_agents(state["model_id"])
    batch = get_batch(state["batch_id"])

    # Runs on both trigger paths now. A labeled_window batch may not have
    # features staged; degrade gracefully by keeping the previous report
    # rather than failing.
    if batch.get("features_df") is None:
        notes = state.get("notes", [])
        return {**state, "notes": _append_capped(
            notes, ["data_drift: no features_df staged for this batch -- keeping previous data_drift_report."]
        )}

    with agents.lock:
        report = agents.drift_detector.detect_drift(batch["features_df"])
    return {**state, "data_drift_report": report}


def _next_after_data_drift(state: MonitoringState) -> str:
    """After data_drift (and feature_intelligence if triggered): performance_drift
    for 'labeled_window', target_drift (best-effort) for 'batch'."""
    return "performance_drift" if state.get("trigger") == "labeled_window" else "target_drift"


def feature_intelligence_router(state: MonitoringState) -> str:
    """Only recompute feature intelligence (SHAP/VIF) when data_drift raised
    at least a WARNING -- avoids paying that cost every cycle when stable."""
    report: Optional[DriftReport] = state.get("data_drift_report")
    if report is None:
        return _next_after_data_drift(state)
    if report.overall_severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL):
        return "feature_intelligence"
    return _next_after_data_drift(state)


def feature_intelligence_node(state: MonitoringState) -> MonitoringState:
    agents = get_agents(state["model_id"])
    context = agents.feature_intelligence_context
    report: Optional[DriftReport] = state.get("data_drift_report")

    if context is None or report is None:
        return {**state, "combined_feature_verdicts": []}

    assessor = FeatureRiskAssessor(context)
    # Reuse the detector's own WARNING threshold (not assess_with_drift's
    # unrelated default) so this verdict agrees with AlertAgent's severity.
    psi_high_threshold = agents.drift_detector.psi_warning
    verdicts: List[CombinedFeatureVerdict] = [
        assessor.assess_with_drift(r.feature_name, psi=r.psi_score, psi_high_threshold=psi_high_threshold)
        for r in report.feature_results
        if r.severity in (DriftSeverity.WARNING, DriftSeverity.CRITICAL)
    ]
    return {
        **state,
        "combined_feature_verdicts": [
            {"feature": v.feature, "psi": v.psi, "priority": v.priority, "rationale": v.rationale}
            for v in verdicts
        ],
    }


def target_drift_node(state: MonitoringState) -> MonitoringState:
    """Same function for both loops: without y_true it falls back to
    PROXY/PENDING; with y_true it produces CONFIRMED (logic lives in
    TargetDriftAgent.detect(), see target_drift_agent.py)."""
    agents = get_agents(state["model_id"])
    batch = get_batch(state["batch_id"])
    with agents.lock:
        result = agents.target_drift_agent.detect(
            y_true=batch.get("y_true"),
            y_pred=batch.get("y_pred"),
        )
    return {**state, "target_drift_result": result}


def staleness_check_node(state: MonitoringState) -> MonitoringState:
    """Fast-loop only. A consecutive PENDING streak means normal label
    latency; total absence of signal over several cycles likely means the
    labeling pipeline is broken -- a distinct alert from statistical drift."""
    result: Optional[TargetDriftResult] = state.get("target_drift_result")
    streak = state.get("label_pending_streak", 0)

    if result is not None and result.status.value == "pending":
        streak += 1
    else:
        streak = 0  # PROXY counts as a live signal, not just CONFIRMED

    WARNING_CYCLES, CRITICAL_CYCLES = 5, 15
    alert = None
    if streak >= CRITICAL_CYCLES:
        alert = "critical"
    elif streak >= WARNING_CYCLES:
        alert = "warning"

    return {**state, "label_pending_streak": streak, "staleness_alert": alert}


# ------------------------------------------------------------------------ #
# Nodes -- slow loop
# ------------------------------------------------------------------------ #

def performance_drift_node(state: MonitoringState) -> MonitoringState:
    agents = get_agents(state["model_id"])
    batch_id = state["batch_id"]
    batch = get_batch(batch_id)

    # Idempotence check + window update must be atomic together, or two
    # concurrent invocations of the same batch_id could both append it.
    with agents.lock:
        if batch_id in agents.processed_batch_ids:
            notes = state.get("notes", [])
            return {**state, "performance_drift_result": None,
                    "notes": _append_capped(notes, [f"batch {batch_id} already processed by performance_drift, skipping (idempotence)."])}

        if batch.get("y_true") is None or batch.get("y_pred") is None:
            return {**state, "performance_drift_result": None}

        result = agents.performance_drift_agent.update(
            y_true=batch["y_true"], y_pred=batch["y_pred"], y_proba=batch.get("y_proba"),
        )
        if result.status == LabelAvailability.LABELED:
            agents.processed_batch_ids.add(batch_id)

    return {**state, "performance_drift_result": result}


# ------------------------------------------------------------------------ #
# Alerting / remediation nodes reuse alert_node / remediation_node as-is.
# ------------------------------------------------------------------------ #

_alert_agent = AlertAgent()
_remediation_agent = RemediationAgent()
_report_generator = DiagnosticReportGenerator()


def alerting_node(state: MonitoringState) -> MonitoringState:
    """Doesn't call process_all() directly: data_drift_report/performance_
    drift_result persist in shared state across cycles (needed for
    root_cause), so process_all() would re-dispatch the same alert every
    cycle. Instead, only calls process_X when X is genuinely new since the
    last pass (compared by result timestamp)."""
    agents = get_agents(state["model_id"])
    fi_context = agents.feature_intelligence_context

    data_drift_report = state.get("data_drift_report")
    target_drift_result = state.get("target_drift_result")
    performance_result = state.get("performance_drift_result")

    seen: Dict[str, str] = dict(state.get("_alert_signatures", {}))
    new_alerts: List[Any] = []

    def _is_fresh(key: str, signature: Optional[str]) -> bool:
        if signature is None or seen.get(key) == signature:
            return False
        seen[key] = signature
        return True

    if data_drift_report is not None and _is_fresh("data_drift", data_drift_report.timestamp):
        new_alerts += _alert_agent.process_data_drift(data_drift_report, feature_intelligence=fi_context)

    if target_drift_result is not None and _is_fresh("target_drift", target_drift_result.timestamp):
        new_alerts += _alert_agent.process_target_drift(target_drift_result)

    if performance_result is not None and _is_fresh("performance_drift", performance_result.timestamp):
        new_alerts += _alert_agent.process_performance_drift(performance_result)

    # root_cause freshness is keyed on performance_result.timestamp (event) +
    # suspected features (content), not on data_drift_report.timestamp, since
    # data_drift recomputes every fast-loop cycle while performance_result
    # only changes at the slow loop's rate -- keying on data_drift's
    # timestamp would re-page on every fast-loop batch.
    if data_drift_report is not None and performance_result is not None:
        candidate_root_cause = _alert_agent.process_root_cause(data_drift_report, performance_result, fi_context)
        if candidate_root_cause:
            suspected = tuple(sorted(candidate_root_cause[0].context.get("suspected_features", [])))
            root_cause_signature = f"{performance_result.timestamp}|{suspected}"
            if _is_fresh("root_cause", root_cause_signature):
                new_alerts += candidate_root_cause

    if new_alerts:
        # dispatch_async, not dispatch: don't block this node (and the fast
        # loop) on a slow/down channel's retries/backoff.
        _alert_agent.dispatcher.dispatch_async(new_alerts)

    escalate = any(a.priority.value in ("critical", "page") for a in new_alerts)

    return {
        **state,
        "alerts": _append_capped(state.get("alerts", []), [a.to_dict() for a in new_alerts]),
        # Overwritten (not accumulated): remediation_node should only see
        # this cycle's fresh alerts, not the whole history.
        "_alert_objects": new_alerts,
        "escalate": escalate,
        "_alert_signatures": seen,
    }


def remediation_node(state: MonitoringState) -> MonitoringState:
    return _remediation_node_impl(state, agent=_remediation_agent)


def report_node(state: MonitoringState) -> MonitoringState:
    """Assembles + persists the full diagnostic report for the cycle. Runs
    after remediation so it reflects the final remediation status."""
    return _report_node_impl(state, generator=_report_generator)


def _route_after_remediation(state: MonitoringState) -> str:
    return "human_review" if state.get("pending_approval") else END


def human_review_node(state: MonitoringState) -> MonitoringState:
    """Explicit stop: one or more RemediationAction items await human
    approval (RETRAIN/RECALIBRATE by default, see remediation_agent.py).
    Doesn't block the graph -- resumption happens via a separate run after
    RemediationAction.approve() is called and dispatcher.dispatch() is
    re-run on the approved actions."""
    notes = state.get("notes", [])
    return {**state, "notes": _append_capped(notes, ["Actions pending human approval -- see remediation_actions."])}


# ------------------------------------------------------------------------ #
# Single graph (instead of separate fast_graph/slow_graph), with
# data_drift/feature_intelligence on BOTH trigger paths -- see
# _next_after_data_drift / feature_intelligence_router.
# ------------------------------------------------------------------------ #

def _route_after_target_drift(state: MonitoringState) -> str:
    """staleness_check only applies to the 'batch' path -- 'labeled_window'
    has fresh labels by definition, nothing to measure."""
    return "staleness_check" if state.get("trigger") == "batch" else "alerting"


def build_graph(checkpointer=None):
    """Single entry point. checkpointer=None uses the shared default
    singleton (get_default_checkpointer()); pass an explicit one (e.g.
    PostgresSaver in prod) if needed."""
    graph = StateGraph(MonitoringState)

    graph.add_node("data_drift", data_drift_node)
    graph.add_node("feature_intelligence", feature_intelligence_node)
    graph.add_node("performance_drift", performance_drift_node)
    graph.add_node("target_drift", target_drift_node)
    graph.add_node("staleness_check", staleness_check_node)
    graph.add_node("alerting", alerting_node)
    graph.add_node("remediation", remediation_node)
    graph.add_node("diagnostic_report", report_node)
    graph.add_node("human_review", human_review_node)

    # Single entry for both triggers.
    graph.set_entry_point("data_drift")

    # data_drift -> [feature_intelligence] -> diverges by trigger.
    graph.add_conditional_edges(
        "data_drift", feature_intelligence_router,
        {"feature_intelligence": "feature_intelligence", "performance_drift": "performance_drift", "target_drift": "target_drift"},
    )
    graph.add_conditional_edges(
        "feature_intelligence", _next_after_data_drift,
        {"performance_drift": "performance_drift", "target_drift": "target_drift"},
    )

    # "labeled_window" path: performance_drift -> target_drift (CONFIRMED)
    graph.add_edge("performance_drift", "target_drift")

    # Both paths converge here. staleness_check only for "batch".
    graph.add_conditional_edges(
        "target_drift", _route_after_target_drift,
        {"staleness_check": "staleness_check", "alerting": "alerting"},
    )
    graph.add_edge("staleness_check", "alerting")

    # Common trunk: alerting -> remediation -> diagnostic_report -> [human_review] -> END
    graph.add_edge("alerting", "remediation")
    graph.add_edge("remediation", "diagnostic_report")
    graph.add_conditional_edges("diagnostic_report", _route_after_remediation, {"human_review": "human_review", END: END})
    graph.add_edge("human_review", END)

    return graph.compile(checkpointer=checkpointer or get_default_checkpointer())


# ------------------------------------------------------------------------ #
# Invocation (outside LangGraph):
#
# graph.invoke({..., "trigger": "batch"}, config=...)          : called on
#   every inference batch (e.g. from the serving service, or a Kafka
#   consumer on the inference-log topic). stage_batch() only needs features_df.
# graph.invoke({..., "trigger": "labeled_window"}, config=...) : NOT
#   triggered by LangGraph. An external scheduler (Airflow/cron) or a
#   labeling-pipeline webhook invokes the SAME compiled graph once
#   window_size fresh labels are available for model_id. Same thread_id=
#   model_id in both cases (same checkpointer by construction). Also stage
#   features_df for this window if possible, so data_drift/feature_
#   intelligence run on a fresh report for this cycle rather than reusing
#   the last 'batch' cycle's report.
# ------------------------------------------------------------------------ #
