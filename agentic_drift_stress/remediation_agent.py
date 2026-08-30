"""
Remediation Agent (5th agent, optional).

Consumes Alerts already classified by AlertAgent (not raw detection
results) and translates them into structured RemediationAction items,
routed to pluggable handlers -- same Dispatcher design as AlertAgent, for
consistency with the rest of the architecture.

Why a separate layer instead of folding this into AlertAgent:
1. AlertAgent stays a pure classifier with no infra side effects (easily
   testable, replayable, no risk of unwanted actions). Deciding to ACT is a
   different concern, with its own guardrails (human approval, cooldown,
   retrain budget, audit).
2. Free text ("Retrain recommended.") isn't actionable by a downstream
   LangGraph node without re-parsing the string. `Alert.remediation_type`
   (an enum) and `RemediationAction` (a structure) are, without losing the
   detailed message (kept verbatim in `RemediationAction.description`).

Default safety policy (see _DEFAULT_POLICY): any action that mutates the
production model (RETRAIN, RECALIBRATE) requires human approval before
execution, even if a handler capable of triggering it is wired in. Only
non-mutating actions (open a ticket, escalate, monitor) are fully automatic
by default. This is deliberate and overridable via `policy=`.

AUDIT FIX (inadequate action selection on the critical path): AlertAgent
already does the hard diagnostic work -- e.g. process_data_drift's
_diagnose_data_quality gives a FINAL VERDICT (pipeline issue vs genuine
population shift) and process_root_cause corroborates a performance
degradation with a specific drifted, SHAP-direct-impact feature. Two gaps
meant that diagnosis wasn't landing as an ADEQUATE decision downstream:

1. Every RemediationAction got the exact same automatable/requires_approval
   pair for a given RemediationType, regardless of how urgent the source
   Alert actually was (a WARNING-level RETRAIN and a PAGE-level one looked
   identical to an approver -- no signal on HOW FAST to act). Fixed by
   deriving an explicit `urgency`/`sla` pair from `Alert.priority`
   (_URGENCY_BY_PRIORITY below), attached to every RemediationAction.
2. When a generic "model_weights" RETRAIN alert and a `root_cause` RETRAIN
   alert fire in the SAME cycle (the common case: root_cause only exists
   because model_weights was already CRITICAL, see
   alert_agent.py::process_root_cause), RemediationAgent produced TWO
   independent, equally-weighted "retrain" actions with no indication
   either was more specific -- exactly the situation where "which one do I
   actually act on" matters most and was left for the human to figure out.
   Fixed by `_mark_superseded_actions`: the generic action is kept (for
   count/audit parity -- nothing is silently dropped, see
   tests/test_remediation_survives_the_graph.py) but flagged
   `superseded_by`/`is_primary=False`, so exactly ONE action per root-cause
   episode is presented as the one to act on.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from typing import Any, Callable, Dict, List, Optional

from alert_agent import Alert, AlertPriority, RemediationType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------- #
# Default policy: RemediationType -> (automatable, requires_approval)
# ---------------------------------------------------------------------- #

_DEFAULT_POLICY: Dict[RemediationType, Dict[str, bool]] = {
    RemediationType.RETRAIN:                    {"automatable": True,  "requires_approval": True},
    RemediationType.RECALIBRATE:                {"automatable": True,  "requires_approval": True},
    RemediationType.DATA_PIPELINE_INVESTIGATION: {"automatable": True,  "requires_approval": False},  # opening a ticket is risk-free
    RemediationType.FEATURE_INVESTIGATION:      {"automatable": True,  "requires_approval": False},
    RemediationType.ESCALATE:                   {"automatable": True,  "requires_approval": False},   # notifying != mutating the model
    RemediationType.MONITOR:                    {"automatable": True,  "requires_approval": False},
    RemediationType.NONE:                       {"automatable": False, "requires_approval": False},
}


# ---------------------------------------------------------------------- #
# Urgency / SLA: derived from Alert.priority, NOT from RemediationType --
# two RETRAIN actions can carry very different urgency depending on what
# triggered them (a routine WARNING-level population shift vs. a PAGE-level
# critical-class collapse). This is what turns "requires_approval=True"
# into an ADEQUATE, actionable instruction for the approver: not just
# "someone must click approve eventually", but "by when".
# ---------------------------------------------------------------------- #

_URGENCY_BY_PRIORITY: Dict[AlertPriority, Dict[str, str]] = {
    AlertPriority.PAGE:     {"urgency": "immediate", "sla": "act now -- business-critical class impacted, do not wait for the next cycle"},
    AlertPriority.CRITICAL: {"urgency": "urgent",    "sla": "within 24h"},
    AlertPriority.WARNING:  {"urgency": "normal",     "sla": "next working cycle"},
    AlertPriority.INFO:     {"urgency": "low",         "sla": "no deadline -- audit trail only"},
}
_URGENCY_RANK = {"immediate": 3, "urgent": 2, "normal": 1, "low": 0}


@dataclass
class RemediationAction:
    alert: Alert                      # full traceability to the source alert
    action_type: RemediationType
    description: str                  # taken from Alert.recommended_action
    automatable: bool
    requires_approval: bool
    urgency: str = "normal"           # "immediate" | "urgent" | "normal" | "low" -- derived from Alert.priority
    sla: str = ""                     # human-readable deadline matching `urgency`
    is_primary: bool = True           # False when superseded by a more specific action (see _mark_superseded_actions)
    superseded_by: Optional[str] = None  # title of the superseding action's source alert, if any
    payload: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"           # "pending" | "approved" | "executed" | "skipped"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_alert_title": self.alert.title,
            "source_agent": self.alert.source_agent,
            "priority": self.alert.priority.value,
            "action_type": self.action_type.value,
            "description": self.description,
            "automatable": self.automatable,
            "requires_approval": self.requires_approval,
            "urgency": self.urgency,
            "sla": self.sla,
            "is_primary": self.is_primary,
            "superseded_by": self.superseded_by,
            "payload": self.payload,
            "status": self.status,
            "timestamp": self.timestamp,
        }

    def approve(self) -> None:
        """Call explicitly from the approval workflow (human or automated
        with its own guardrails) before dispatch."""
        self.status = "approved"


# ---------------------------------------------------------------------- #
# Pluggable handlers, one per RemediationType. Defaults don't mutate
# anything -- they just log what a real handler should trigger. Replace via
# RemediationDispatcher.register() without touching RemediationAgent.
# ---------------------------------------------------------------------- #

def _default_retrain_handler(action: RemediationAction) -> None:
    logger.warning(
        f"[REMEDIATION] RETRAIN suggested for '{action.alert.title}' (urgency={action.urgency}, sla={action.sla}) "
        f"-- no retraining pipeline wired in. Manual trigger required. Details: {action.description}"
    )


def _default_recalibrate_handler(action: RemediationAction) -> None:
    logger.warning(
        f"[REMEDIATION] RECALIBRATE suggested for '{action.alert.title}' (urgency={action.urgency}, sla={action.sla}) "
        f"-- no calibration service wired in. Details: {action.description}"
    )


def _default_ticket_handler(action: RemediationAction) -> None:
    logger.info(
        f"[REMEDIATION] {action.action_type.value} for '{action.alert.title}' (urgency={action.urgency}, sla={action.sla}) "
        f"-- ticket to open (no tracker wired in). Details: {action.description}"
    )


def _default_noop_handler(action: RemediationAction) -> None:
    logger.debug(f"[REMEDIATION] {action.action_type.value} for '{action.alert.title}': no action needed.")


_DEFAULT_HANDLERS: Dict[RemediationType, Callable[[RemediationAction], None]] = {
    RemediationType.RETRAIN: _default_retrain_handler,
    RemediationType.RECALIBRATE: _default_recalibrate_handler,
    RemediationType.DATA_PIPELINE_INVESTIGATION: _default_ticket_handler,
    RemediationType.FEATURE_INVESTIGATION: _default_ticket_handler,
    RemediationType.ESCALATE: _default_ticket_handler,
    RemediationType.MONITOR: _default_noop_handler,
    RemediationType.NONE: _default_noop_handler,
}


class RemediationDispatcher:
    """Handler registry per RemediationType, same principle as
    AlertDispatcher: wire real calls (trigger an Airflow/Kubeflow pipeline,
    open a Jira ticket, call a recalibration service...) without touching
    RemediationAgent's logic."""

    def __init__(self):
        self._handlers: Dict[RemediationType, Callable[[RemediationAction], None]] = dict(_DEFAULT_HANDLERS)

    def register(self, action_type: RemediationType, handler: Callable[[RemediationAction], None]) -> None:
        self._handlers[action_type] = handler

    def dispatch(self, actions: List[RemediationAction]) -> None:
        # Highest urgency first: when a cycle produces several actions
        # (routine + a PAGE-level one), the immediate one must be decided
        # on / logged before the queue works through lower-urgency items,
        # not in whatever order alerts happened to be generated.
        ordered = sorted(actions, key=lambda a: _URGENCY_RANK.get(a.urgency, 0), reverse=True)

        for action in ordered:
            if not action.is_primary:
                # Superseded by a more specific action (see
                # _mark_superseded_actions): kept in the returned list for
                # traceability/count parity, but deliberately never
                # executed independently -- avoids two uncoordinated
                # retrain triggers for what is one incident.
                action.status = "skipped"
                logger.info(
                    f"[REMEDIATION] {action.action_type.value} for '{action.alert.title}' superseded by "
                    f"'{action.superseded_by}' -- not executed independently, see the superseding action "
                    "for the adequate response."
                )
                continue

            if action.requires_approval and action.status != "approved":
                log_fn = logger.warning if action.urgency == "immediate" else logger.info
                log_fn(
                    f"[REMEDIATION] {action.action_type.value} for '{action.alert.title}' "
                    f"(urgency={action.urgency}, sla={action.sla}) pending human approval -- "
                    "not auto-executed (see RemediationAction.approve())."
                )
                continue
            handler = self._handlers.get(action.action_type, _default_noop_handler)
            try:
                handler(action)
                action.status = "executed"
                if action.urgency == "immediate":
                    # AUDIT FIX (visibility): an immediate-urgency action
                    # executing should be as loud as the PAGE alert that
                    # triggered it -- not just another INFO line among
                    # routine remediations.
                    logger.critical(
                        f"[REMEDIATION] IMMEDIATE action executed: {action.action_type.value} for "
                        f"'{action.alert.title}' (sla={action.sla})."
                    )
            except Exception as exc:
                logger.error(f"Remediation {action.action_type.value} failed for '{action.alert.title}': {exc}")
                action.status = "skipped"


# ---------------------------------------------------------------------- #
# Agent
# ---------------------------------------------------------------------- #

def _mark_superseded_actions(actions: List["RemediationAction"]) -> None:
    """When a `root_cause` alert and a generic performance/data-drift alert
    both resolve to RETRAIN in the same cycle, the root_cause one is
    strictly more specific -- it names the suspected feature(s) and the
    causal link (see alert_agent.py::process_root_cause), which only fires
    once performance is already CRITICAL with a model_weights cause. Acting
    on both independently means two uncoordinated retrain requests for what
    is, in practice, one incident.

    The generic action is NEVER dropped from the returned list (that would
    break the count-parity invariant covered by
    tests/test_remediation_survives_the_graph.py, and would silently
    discard an Alert the operator already saw) -- it's flagged
    `is_primary=False` / `superseded_by=<root cause alert title>` instead,
    so exactly one action is presented as the one to actually act on."""
    root_cause_actions = [
        a for a in actions if a.action_type == RemediationType.RETRAIN and a.alert.source_agent == "root_cause"
    ]
    if not root_cause_actions:
        return
    primary = root_cause_actions[0]  # process_root_cause emits at most one alert per cycle
    for a in actions:
        if a is primary:
            continue
        if a.action_type == RemediationType.RETRAIN and a.alert.source_agent in ("performance_drift", "data_drift"):
            a.is_primary = False
            a.superseded_by = primary.alert.title


class RemediationAgent:
    """Translates Alerts (already classified by AlertAgent) into
    RemediationAction items. Deliberately thin: case-by-case classification
    is already done upstream in AlertAgent; here we apply the
    automatable/requires_approval policy, attach an urgency/SLA derived
    from the source alert's priority, resolve which action is primary when
    several point at the same incident, and route."""

    def __init__(
        self,
        dispatcher: Optional[RemediationDispatcher] = None,
        policy: Optional[Dict[RemediationType, Dict[str, bool]]] = None,
    ):
        self.dispatcher = dispatcher or RemediationDispatcher()
        self.policy = policy or _DEFAULT_POLICY

    def process(self, alerts: List[Alert]) -> List[RemediationAction]:
        actions: List[RemediationAction] = []
        for alert in alerts:
            if alert.remediation_type == RemediationType.NONE:
                continue
            rule = self.policy.get(alert.remediation_type, {"automatable": False, "requires_approval": True})
            urgency_info = _URGENCY_BY_PRIORITY.get(alert.priority, _URGENCY_BY_PRIORITY[AlertPriority.WARNING])
            actions.append(RemediationAction(
                alert=alert,
                action_type=alert.remediation_type,
                description=alert.recommended_action or alert.message,
                automatable=rule["automatable"],
                requires_approval=rule["requires_approval"],
                urgency=urgency_info["urgency"],
                sla=urgency_info["sla"],
                payload={"source_agent": alert.source_agent, "priority": alert.priority.value, **alert.context},
            ))

        _mark_superseded_actions(actions)

        if actions:
            self.dispatcher.dispatch(actions)
        return actions


# ---------------------------------------------------------------------- #
# LangGraph integration -- optional node, wire AFTER "alerting"
# (see the integration note in alert_agent.py)
# ---------------------------------------------------------------------- #

def remediation_node(state: Dict[str, Any], agent: Optional[RemediationAgent] = None) -> Dict[str, Any]:
    """Reads `state["_alert_objects"]` (raw Alert objects set by
    alert_node), generates + dispatches RemediationAction items, returns
    enriched state.

    `state["pending_approval"]` enables extra conditional routing (e.g. ->
    a "human_review" node that notifies a human and waits for approval
    before this node is called again with the approved actions)."""
    agent = agent or RemediationAgent()
    alert_objects: List[Alert] = state.get("_alert_objects", [])
    actions = agent.process(alert_objects)

    # Cap, same reason as "alerts" in orchestrator.py: without it,
    # remediation_actions grows by one entry per cycle indefinitely in the
    # checkpointed State.
    _MAX_HISTORY = 200
    existing = state.get("remediation_actions", [])
    capped = (existing + [a.to_dict() for a in actions])[-_MAX_HISTORY:]

    return {
        **state,
        "remediation_actions": capped,
        # Raw RemediationAction objects for THIS CYCLE ONLY (never
        # accumulated/capped): a downstream consumer
        # (diagnostic_report.py::report_node) needs the typed objects for
        # the current cycle, not the serialized history.
        "_remediation_action_objects": actions,
        "pending_approval": any(a.requires_approval and a.status != "approved" for a in actions),
    }
