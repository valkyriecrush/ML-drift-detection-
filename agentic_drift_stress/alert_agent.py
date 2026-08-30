"""
Alert Agent (4th agent).

Consumes the 3 detection agents' outputs:
- DriftDetector.detect_drift()        -> DriftReport            (data/feature drift)
- TargetDriftAgent.detect()           -> TargetDriftResult       (target drift)
- PerformanceDriftAgent.evaluate(...) -> PerformanceDriftResult  (performance drift)

and turns each result into 0..N alerts, routed case-by-case (each agent has
its own semantics and deserves a tailored message + channel, not a generic
severity->alert mapping).

Designed as a LangGraph node: `alert_node(state)` reads shared state,
produces alerts, dispatches them, and returns enriched state (alerts +
escalation flags) for downstream conditional routing.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from drift_detector import DriftReport, DriftSeverity, FeatureType
from target_drift_agent import TargetDriftResult, LabelStatus
from performance_drift_agent import PerformanceDriftResult, PerformanceSeverity, LabelAvailability
from feature_intelligence import FeatureIntelligenceContext, FeatureRiskAssessor, RiskLevel

logger = logging.getLogger(__name__)


class AlertPriority(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    PAGE = "page"  # above CRITICAL: wakes someone up (e.g. a critical class collapsed)


class AlertChannel(Enum):
    LOG = "log"
    SLACK = "slack"
    EMAIL = "email"
    PAGER = "pager"


class RemediationType(Enum):
    """Machine-actionable classification of the suggested action, alongside
    the free-text `recommended_action`. AlertAgent only SETS this type -- it
    never triggers anything itself; RemediationAgent routes on it downstream."""
    RETRAIN = "retrain"
    RECALIBRATE = "recalibrate"                                  # recalibrate probabilities (Platt/Isotonic)
    DATA_PIPELINE_INVESTIGATION = "data_pipeline_investigation"   # upstream issue, not the model itself
    FEATURE_INVESTIGATION = "feature_investigation"               # case-by-case look at drifted features
    ESCALATE = "escalate"                                         # immediate human escalation (PAGE)
    MONITOR = "monitor"                                           # no action, just watch next cycles
    NONE = "none"


# Default priority -> channel routing. Overridable when instantiating AlertAgent.
DEFAULT_ROUTING: Dict[AlertPriority, List[AlertChannel]] = {
    AlertPriority.INFO: [AlertChannel.LOG],
    AlertPriority.WARNING: [AlertChannel.LOG, AlertChannel.SLACK],
    AlertPriority.CRITICAL: [AlertChannel.LOG, AlertChannel.SLACK, AlertChannel.EMAIL],
    AlertPriority.PAGE: [AlertChannel.LOG, AlertChannel.SLACK, AlertChannel.EMAIL, AlertChannel.PAGER],
}


@dataclass
class Alert:
    source_agent: str          # "data_drift" | "target_drift" | "performance_drift"
    priority: AlertPriority
    title: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    recommended_action: Optional[str] = None
    remediation_type: RemediationType = RemediationType.NONE
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_agent": self.source_agent,
            "priority": self.priority.value,
            "title": self.title,
            "message": self.message,
            "context": self.context,
            "recommended_action": self.recommended_action,
            "remediation_type": self.remediation_type.value,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------- #
# Dispatch: pluggable channels (each just a callable(Alert) -> None)
# ---------------------------------------------------------------------- #

def _log_sender(alert: Alert) -> None:
    level = {
        AlertPriority.INFO: logging.INFO,
        AlertPriority.WARNING: logging.WARNING,
        AlertPriority.CRITICAL: logging.ERROR,
        AlertPriority.PAGE: logging.CRITICAL,
    }[alert.priority]
    logger.log(level, f"[{alert.source_agent}] {alert.title} :: {alert.message}")


# Priorities where a failed send should retry with backoff instead of just
# logging and dropping the alert -- PAGE/CRITICAL exist precisely to wake
# someone up, so silent loss here is the worst-case failure mode.
_RETRYABLE_PRIORITIES = {AlertPriority.CRITICAL, AlertPriority.PAGE}
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_BACKOFF_BASE_SECONDS = 0.5
_DEFAULT_DEAD_LETTER_PATH = "alert_dead_letter.jsonl"


class AlertDispatcher:
    """Channel registry. Swap in real senders (Slack webhook, SMTP,
    PagerDuty API...) without touching the detection logic.

    For CRITICAL/PAGE (`_RETRYABLE_PRIORITIES`): each sender retries with
    exponential backoff (`max_retries`, `backoff_base_seconds`); if every
    channel still fails, the alert is written to a local dead-letter queue
    (`dead_letter_path`, JSONL, append-only) instead of being lost; and a
    distinct CRITICAL "alerting itself is down" log is emitted, separate
    from the original alert."""

    def __init__(
        self,
        routing: Optional[Dict[AlertPriority, List[AlertChannel]]] = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        backoff_base_seconds: float = _DEFAULT_BACKOFF_BASE_SECONDS,
        dead_letter_path: str = _DEFAULT_DEAD_LETTER_PATH,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.routing = routing or DEFAULT_ROUTING
        self._senders: Dict[AlertChannel, Callable[[Alert], None]] = {AlertChannel.LOG: _log_sender}
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.dead_letter_path = dead_letter_path
        self._sleep = sleep_fn
        self._async_threads: List[threading.Thread] = []

    def register(self, channel: AlertChannel, sender: Callable[[Alert], None]) -> None:
        self._senders[channel] = sender

    def _send_with_retry(self, sender: Callable[[Alert], None], alert: Alert, channel: AlertChannel) -> Optional[Exception]:
        """Try `sender(alert)` with exponential backoff if the priority is
        retryable. Returns None on success, else the last exception seen."""
        retryable = alert.priority in _RETRYABLE_PRIORITIES
        attempts = self.max_retries if retryable else 1
        last_exc: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                sender(alert)
                return None
            except Exception as exc:  # noqa: BLE001 -- an external sender (SMTP/webhook) can raise anything
                last_exc = exc
                if attempt < attempts:
                    delay = self.backoff_base_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        f"Failed to send alert on {channel.value} (attempt {attempt}/{attempts}): "
                        f"{exc}. Retrying in {delay:.1f}s."
                    )
                    self._sleep(delay)
        return last_exc

    def _write_dead_letter(self, alert: Alert, failures: Dict[str, str]) -> None:
        """Append-only: never overwrites previous entries. A write failure
        here is itself logged as CRITICAL."""
        entry = {
            "alert": alert.to_dict(),
            "channel_failures": failures,
            "dead_lettered_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            with open(self.dead_letter_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.critical(
                f"ALERTING DOWN: could not write the dead-letter queue "
                f"('{self.dead_letter_path}'), alert may be lost despite retries "
                f"-- {exc}. Original alert: {alert.to_dict()}"
            )

    def dispatch(self, alerts: List[Alert]) -> None:
        for alert in alerts:
            failures: Dict[str, str] = {}
            channels = self.routing.get(alert.priority, [AlertChannel.LOG])
            for channel in channels:
                sender = self._senders.get(channel)
                if sender is None:
                    logger.warning(f"No sender registered for channel {channel.value}, alert not sent on this channel.")
                    failures[channel.value] = "no_sender_registered"
                    continue
                exc = self._send_with_retry(sender, alert, channel)
                if exc is not None:
                    logger.error(f"Alert delivery failed on {channel.value} after retries: {exc}")
                    failures[channel.value] = str(exc)
                else:
                    # AUDIT FIX (missing delivery confirmation): success on a
                    # channel used to be entirely silent -- only failures were
                    # logged. For an on-call engineer, "no log line" and "it
                    # went out fine" are indistinguishable without this: you
                    # cannot tell a quiet cycle from a channel that stopped
                    # confirming delivery. One explicit, structured line per
                    # successful send, loud enough to grep on CRITICAL/PAGE.
                    log_fn = logger.warning if alert.priority in _RETRYABLE_PRIORITIES else logger.info
                    log_fn(
                        f"[ALERT DELIVERED] channel={channel.value} priority={alert.priority.value.upper()} "
                        f"source={alert.source_agent} title='{alert.title}'"
                    )

            if failures and alert.priority in _RETRYABLE_PRIORITIES and len(failures) == len(channels):
                # Every channel failed for a CRITICAL/PAGE alert -- write it
                # to the dead-letter queue so it isn't lost silently.
                logger.critical(
                    f"ALERTING DOWN: {alert.priority.value.upper()} alert "
                    f"'{alert.title}' could not be delivered on ANY channel "
                    f"({list(failures.keys())}) -- writing to dead-letter queue."
                )
                self._write_dead_letter(alert, failures)

    def dispatch_async(self, alerts: List[Alert]) -> None:
        """Runs dispatch() (retries + dead-letter included) in a separate
        daemon thread, so the caller (a fast-loop graph node) never blocks
        on a slow/down channel's retries. Same delivery guarantees as
        dispatch(), just off the critical path."""
        thread = threading.Thread(target=self.dispatch, args=(alerts,), daemon=True)
        self._async_threads.append(thread)
        thread.start()

    def wait_for_pending_dispatches(self, timeout: Optional[float] = None) -> None:
        """For tests / clean shutdown: waits for dispatch_async() threads to finish."""
        for thread in self._async_threads:
            thread.join(timeout=timeout)
        self._async_threads = [t for t in self._async_threads if t.is_alive()]


# ---------------------------------------------------------------------- #
# Agent: case-by-case translation of detection results into alert(s)
# ---------------------------------------------------------------------- #

_QUALITY_ISSUE_MISSING_RATE_THRESHOLD = 0.05  # 5% missing on a drifted feature is not noise


def _diagnose_data_quality(group: List[Any]) -> Dict[str, Any]:
    """Turns the per-feature quality signal now computed by DriftDetector
    (missing_rate, new_categories -- see drift_detector.py::_quality_diagnostics)
    into a FINAL verdict instead of asking a human to go check: either the
    batch itself is unhealthy (pipeline issue, evidence-backed) or it is
    clean (well-formed values, no new categories) and the drift is then a
    genuine population shift that retraining is the right response to.

    `group` is a list of (FeatureDriftResult, rationale, risk_level) tuples,
    same shape already built in process_data_drift."""
    dirty: List[str] = []
    reasons: List[str] = []
    for r, _rationale, _risk_level in group:
        missing_rate = r.details.get("missing_rate")
        new_categories = r.details.get("new_categories") or []
        if missing_rate is not None and missing_rate >= _QUALITY_ISSUE_MISSING_RATE_THRESHOLD:
            dirty.append(r.feature_name)
            reasons.append(f"{r.feature_name}: {missing_rate*100:.1f}% missing values this batch")
        if new_categories:
            dirty.append(r.feature_name)
            preview = ", ".join(new_categories[:3])
            reasons.append(f"{r.feature_name}: new unseen categor{'y' if len(new_categories)==1 else 'ies'} ({preview})")

    return {"is_dirty": bool(dirty), "dirty_features": sorted(set(dirty)), "reasons": reasons}


class AlertAgent:
    """
    Each detection agent produces alerts per its own semantics:
    - data drift:    overall severity + individual critical features
    - target drift:  PROXY (to confirm) vs CONFIRMED (reliable)
    - performance:   data_pipeline / model_weights / calibration, plus an
                      absolute floor (critical-class recall) that pages directly
    """

    def __init__(
        self,
        dispatcher: Optional[AlertDispatcher] = None,
        feature_intelligence: Optional[FeatureIntelligenceContext] = None,
    ):
        self.dispatcher = dispatcher or AlertDispatcher()
        # Default SHAP/VIF context; can also be passed per-cycle to
        # process_data_drift() if importance/correlation is recomputed each run.
        self.feature_intelligence = feature_intelligence

    # ------------------------------------------------------------------ #
    # 1. Data / feature drift -- weighted by actual model impact
    # ------------------------------------------------------------------ #

    # PSI alone says "the distribution moved". SHAP/VIF says "does it matter
    # to the model". Final priority is the product of both, not PSI alone.
    _ESCALATION_TABLE = {
        # (psi_severity, risk_level) -> adjusted priority
        (DriftSeverity.CRITICAL, RiskLevel.DIRECT_IMPACT): AlertPriority.CRITICAL,
        (DriftSeverity.CRITICAL, RiskLevel.INDIRECT_CORRELATION): AlertPriority.CRITICAL,
        (DriftSeverity.CRITICAL, RiskLevel.LOW_RISK): AlertPriority.WARNING,       # downgrade: strong PSI, unlikely model impact
        (DriftSeverity.CRITICAL, RiskLevel.UNKNOWN): AlertPriority.CRITICAL,       # no evidence to downgrade -> stay cautious
        (DriftSeverity.WARNING, RiskLevel.DIRECT_IMPACT): AlertPriority.CRITICAL,  # upgrade: moderate PSI but a key model feature
        (DriftSeverity.WARNING, RiskLevel.INDIRECT_CORRELATION): AlertPriority.WARNING,
        (DriftSeverity.WARNING, RiskLevel.LOW_RISK): AlertPriority.INFO,           # near-silent: statistical noise, no impact
        (DriftSeverity.WARNING, RiskLevel.UNKNOWN): AlertPriority.WARNING,
    }

    def process_data_drift(
        self, report: DriftReport, feature_intelligence: Optional[FeatureIntelligenceContext] = None
    ) -> List[Alert]:
        alerts: List[Alert] = []

        if report.overall_severity == DriftSeverity.NONE:
            return alerts  # nothing to report, avoid alert noise

        candidates = [r for r in report.feature_results if r.severity in (DriftSeverity.CRITICAL, DriftSeverity.WARNING)]
        if not candidates:
            return alerts

        intelligence = feature_intelligence or self.feature_intelligence
        assessor = FeatureRiskAssessor(intelligence) if intelligence is not None else None

        # Re-evaluate each feature: adjusted priority + statistical rationale + risk_level
        # (risk_level kept as the actual enum, not re-derived from `rationale`'s free
        # text later -- that text never literally contains "DIRECT_IMPACT").
        decisions = []  # (feature_result, adjusted_priority, rationale, risk_level)
        for r in candidates:
            if assessor is None:
                decisions.append((r, AlertPriority(r.severity.value) if r.severity.value in ("critical", "warning") else AlertPriority.WARNING, "No SHAP/VIF available: decision based on PSI only.", None))
                continue
            assessment = assessor.assess(r.feature_name)
            adjusted = self._ESCALATION_TABLE[(r.severity, assessment.risk_level)]
            decisions.append((r, adjusted, assessment.rationale, assessment.risk_level))

        by_priority: Dict[AlertPriority, list] = {}
        for r, priority, rationale, risk_level in decisions:
            by_priority.setdefault(priority, []).append((r, rationale, risk_level))

        for priority in (AlertPriority.CRITICAL, AlertPriority.WARNING, AlertPriority.INFO):
            group = by_priority.get(priority)
            if not group:
                continue

            names = ", ".join(r.feature_name for r, _, _ in group[:5])
            rationale_lines = [f"- {r.feature_name}: {rationale}" for r, rationale, _ in group[:5]]

            if priority == AlertPriority.CRITICAL:
                title = f"High model-impact drift on {len(group)} feature(s)"
                direct_impact_names = [r.feature_name for r, _, risk_level in group if risk_level == RiskLevel.DIRECT_IMPACT]
                quality = _diagnose_data_quality(group)

                if quality["is_dirty"]:
                    action = (
                        "FINAL VERDICT -- pipeline issue confirmed, do NOT retrain yet: "
                        + "; ".join(quality["reasons"]) + ". "
                        f"Fix ingestion for {', '.join(quality['dirty_features'])} first "
                        "(schema, upstream source, mapping of new categories), then re-run this "
                        "cycle on clean data before deciding whether real drift remains."
                    )
                    rtype = RemediationType.DATA_PIPELINE_INVESTIGATION
                else:
                    action = (
                        "FINAL VERDICT -- retrain: batch is clean (no missing-value spike, no "
                        "new category) on the drifted feature(s), so this is a genuine population "
                        "shift, not a data-quality artifact. "
                        + (f"Prioritize {', '.join(direct_impact_names)} (direct model impact per SHAP). "
                           if direct_impact_names else "")
                        + "Check for a root_cause alert this cycle before scheduling the retrain."
                    )
                    rtype = RemediationType.RETRAIN
            elif priority == AlertPriority.WARNING:
                title = f"Drift to monitor on {len(group)} feature(s)"
                action = (
                    "No immediate action: risk is indirect (correlated with a key "
                    "feature) or undetermined (missing SHAP/correlation data). "
                    "Re-evaluate next cycle."
                )
                rtype = RemediationType.MONITOR
            else:
                title = f"Statistical drift, likely no model impact on {len(group)} feature(s)"
                action = "No priority action: high PSI but low-risk per SHAP/correlation. Kept for audit trail only."
                rtype = RemediationType.NONE

            alerts.append(Alert(
                source_agent="data_drift",
                priority=priority,
                title=title,
                message=f"Affected features: {names}.\n" + "\n".join(rationale_lines),
                context={
                    "drift_percentage": report.drift_percentage,
                    "features": [r.feature_name for r, _, _ in group],
                    "psi_scores": {r.feature_name: r.psi_score for r, _, _ in group},
                    "shap_adjusted": assessor is not None,
                },
                recommended_action=action,
                remediation_type=rtype,
            ))

        return alerts

    # ------------------------------------------------------------------ #
    # 2. Target drift
    # ------------------------------------------------------------------ #

    def process_target_drift(self, result: TargetDriftResult) -> List[Alert]:
        alerts: List[Alert] = []

        if result.status == LabelStatus.PENDING or result.drift is None:
            return alerts  # normal state, waiting on labels/predictions

        if not result.drift.is_drifted:
            return alerts

        is_categorical = result.drift.feature_type == FeatureType.CATEGORICAL
        chi2_note = result.drift.details.get("chi2_note") if is_categorical else None
        chi2_pvalue = result.drift.details.get("chi2_pvalue") if is_categorical else None

        if result.status == LabelStatus.PROXY:
            # Early signal on predictions, not yet confirmed by true labels.
            alerts.append(Alert(
                source_agent="target_drift",
                priority=AlertPriority.WARNING,
                title="Early target-drift signal (proxy, unconfirmed)",
                message=(
                    f"PSI on predictions ({result.drift.psi_score:.3f}) exceeds the drift "
                    "threshold. True labels not yet available: needs confirmation."
                ),
                context={
                    "n_samples": result.n_samples,
                    "psi_score": result.drift.psi_score,
                    "severity": result.drift.severity.value,
                },
                recommended_action=(
                    "Don't retrain on this signal alone -- it's drift on "
                    "PREDICTIONS, not ground truth. Wait for the next labeled "
                    "cycle to confirm; investigate manually if it persists "
                    "unconfirmed for 2-3 cycles."
                ),
                remediation_type=RemediationType.MONITOR,
            ))
        else:  # CONFIRMED
            priority = (
                AlertPriority.CRITICAL if result.drift.severity == DriftSeverity.CRITICAL else AlertPriority.WARNING
            )
            extra = ""
            if chi2_note:
                extra = f" (chi2 test skipped: {chi2_note})"
            elif chi2_pvalue is not None:
                extra = f" (chi2 p-value={chi2_pvalue:.4f})"

            alerts.append(Alert(
                source_agent="target_drift",
                priority=priority,
                title="Target drift confirmed (true labels)",
                message=(
                    f"The real target distribution has drifted (PSI={result.drift.psi_score:.3f}, "
                    f"severity={result.drift.severity.value}){extra}."
                ),
                context={
                    "n_samples": result.n_samples,
                    "psi_score": result.drift.psi_score,
                    "norm_wasserstein_distance": result.drift.norm_wasserstein_dist,
                },
                recommended_action=(
                    "1) Check for a business/exogenous cause (seasonality, policy "
                    "change, external event) before retraining. 2) If the cause is "
                    "structural, plan a retrain on recent data. 3) Document the "
                    "cause to refine future drift detection."
                    if priority == AlertPriority.CRITICAL
                    else "Monitor next cycles; no retrain yet, PSI stays below the critical threshold."
                ),
                remediation_type=RemediationType.RETRAIN if priority == AlertPriority.CRITICAL else RemediationType.MONITOR,
            ))

        return alerts

    # ------------------------------------------------------------------ #
    # 3. Performance drift
    # ------------------------------------------------------------------ #

    def process_performance_drift(self, result: PerformanceDriftResult) -> List[Alert]:
        alerts: List[Alert] = []

        if result.status == LabelAvailability.PENDING or result.severity is None:
            return alerts
        if result.severity == PerformanceSeverity.NONE:
            return alerts

        # Case 1: absolute floor on a critical class -> page directly,
        # regardless of remediation_type, since it's a business safety signal.
        floor_breach = [m for m in result.metrics if m.name.startswith("recall_class_") and m.degraded]
        for m in floor_breach:
            alerts.append(Alert(
                source_agent="performance_drift",
                priority=AlertPriority.PAGE,
                title=f"Recall collapse on critical class ({m.name})",
                message=f"{m.name} = {m.current:.3f} (baseline {m.baseline:.3f}): below the absolute floor.",
                context={"window_size": result.window_size, "metric": m.to_dict() if hasattr(m, "to_dict") else vars(m)},
                recommended_action=(
                    "Escalate immediately, outside the standard cycle: possible "
                    "impact on a sensitive business class. Check whether a "
                    "rollback is viable, notify the business owner, don't wait "
                    "for the next scheduled retrain cycle."
                ),
                remediation_type=RemediationType.ESCALATE,
            ))

        # Case 2: remediation_type = data_pipeline (AUC uncomputable -> loss of diversity)
        if result.remediation_type == "data_pipeline":
            alerts.append(Alert(
                source_agent="performance_drift",
                priority=AlertPriority.CRITICAL,
                title="Class diversity lost in the window",
                message="AUC not computable: likely an upstream data issue, not a model-weight problem.",
                context={"proba_coverage": result.proba_coverage, "notes": result.notes},
                recommended_action=(
                    "Check ingestion/the data pipeline before retraining. An "
                    "uncomputable AUC typically points to a loss of input class "
                    "diversity (an overly strict upstream filter, a partial "
                    "outage, a missing traffic segment), not model weights. "
                    "Retraining now would train on already-biased data."
                ),
                remediation_type=RemediationType.DATA_PIPELINE_INVESTIGATION,
            ))

        # Case 3: real prediction degradation
        elif result.remediation_type == "model_weights":
            priority = AlertPriority.CRITICAL if result.severity == PerformanceSeverity.CRITICAL else AlertPriority.WARNING
            degraded_names = ", ".join(m.name for m in result.metrics if m.degraded)
            alerts.append(Alert(
                source_agent="performance_drift",
                priority=priority,
                title="Real model performance degradation",
                message=f"Degraded metrics this window: {degraded_names}.",
                context={"window_size": result.window_size, "metrics": [vars(m) for m in result.metrics if m.degraded]},
                recommended_action=(
                    "Retrain recommended. Check first whether a root_cause alert "
                    "was raised this cycle: if so, fix that feature/pipeline "
                    "before retraining, or the same issue will recur."
                    if priority == AlertPriority.CRITICAL
                    else "Monitor; retrain only if degradation persists (not yet critical)."
                ),
                remediation_type=RemediationType.RETRAIN if priority == AlertPriority.CRITICAL else RemediationType.MONITOR,
            ))

        # Case 4: classification is sound but poorly calibrated
        elif result.remediation_type == "calibration":
            alerts.append(Alert(
                source_agent="performance_drift",
                priority=AlertPriority.WARNING,
                title="Calibration drift",
                message="Predictions remain correct but output probabilities are poorly calibrated.",
                context={"calibration_metrics": [vars(m) for m in result.calibration_metrics if m.degraded]},
                recommended_action=(
                    "Recalibrate (Platt scaling or isotonic regression) rather "
                    "than a full retrain: ranking/discrimination is fine, only "
                    "the output probabilities are off."
                ),
                remediation_type=RemediationType.RECALIBRATE,
            ))

        return alerts

    # ------------------------------------------------------------------ #
    # 4. Cross-agent synthesis: corroborating a probable root cause
    # ------------------------------------------------------------------ #

    def process_root_cause(
        self,
        data_drift_report: Optional[DriftReport],
        performance_result: Optional[PerformanceDriftResult],
        feature_intelligence: Optional[FeatureIntelligenceContext] = None,
    ) -> List[Alert]:
        """If real performance degradation (model_weights) coincides with a
        drifted feature that has direct SHAP impact, the probable cause is
        identified with high confidence -- distinct from two independent
        alerts merely coinciding in time."""
        alerts: List[Alert] = []
        if data_drift_report is None or performance_result is None:
            return alerts
        if performance_result.remediation_type != "model_weights":
            return alerts
        if performance_result.severity != PerformanceSeverity.CRITICAL:
            return alerts

        intelligence = feature_intelligence or self.feature_intelligence
        if intelligence is None:
            return alerts

        assessor = FeatureRiskAssessor(intelligence)
        drifted = [r for r in data_drift_report.feature_results if r.severity in (DriftSeverity.CRITICAL, DriftSeverity.WARNING)]
        direct_impact_causes = [
            r.feature_name for r in drifted
            if assessor.assess(r.feature_name).risk_level == RiskLevel.DIRECT_IMPACT
        ]

        if direct_impact_causes:
            alerts.append(Alert(
                source_agent="root_cause",
                priority=AlertPriority.CRITICAL,
                title="Probable root cause identified for the performance degradation",
                message=(
                    f"Performance degradation (model_weights) coincides with confirmed drift on "
                    f"{', '.join(direct_impact_causes)}, feature(s) with direct SHAP impact: likely causal link."
                ),
                context={"suspected_features": direct_impact_causes},
                recommended_action=(
                    f"Prioritize fixing (data or pipeline) {', '.join(direct_impact_causes)} "
                    "before any blind retrain -- retraining without fixing the "
                    "cause would reproduce the same problem next cycle."
                ),
                remediation_type=RemediationType.RETRAIN,
            ))
        return alerts

    def process_all(
        self,
        data_drift_report: Optional[DriftReport] = None,
        target_drift_result: Optional[TargetDriftResult] = None,
        performance_drift_result: Optional[PerformanceDriftResult] = None,
        feature_intelligence: Optional[FeatureIntelligenceContext] = None,
    ) -> List[Alert]:
        alerts: List[Alert] = []
        if data_drift_report is not None:
            alerts.extend(self.process_data_drift(data_drift_report, feature_intelligence=feature_intelligence))
        if target_drift_result is not None:
            alerts.extend(self.process_target_drift(target_drift_result))
        if performance_drift_result is not None:
            alerts.extend(self.process_performance_drift(performance_drift_result))
        alerts.extend(self.process_root_cause(data_drift_report, performance_drift_result, feature_intelligence))

        if alerts:
            self.dispatcher.dispatch(alerts)
        return alerts


# ---------------------------------------------------------------------- #
# LangGraph integration
# ---------------------------------------------------------------------- #
#
# This agent is stateless across cycles: it expects the graph's shared state
# to already contain the other 3 agents' outputs under these keys.
#
#   class MonitoringState(TypedDict, total=False):
#       data_drift_report: DriftReport
#       target_drift_result: TargetDriftResult
#       performance_drift_result: PerformanceDriftResult
#       alerts: List[dict]
#       escalate: bool
#
# Example wiring:
#
#   from langgraph.graph import StateGraph, END
#   _alert_agent = AlertAgent()
#   graph = StateGraph(MonitoringState)
#   graph.add_node("data_drift", data_drift_node)
#   graph.add_node("target_drift", target_drift_node)
#   graph.add_node("performance_drift", performance_drift_node)
#   graph.add_node("alerting", alert_node)
#   graph.add_edge("data_drift", "alerting")
#   graph.add_edge("target_drift", "alerting")
#   graph.add_edge("performance_drift", "alerting")
#   graph.add_conditional_edges(
#       "alerting",
#       lambda state: "retrain" if state.get("escalate") else END,
#       {"retrain": "retrain", END: END},
#   )
#
# Optional 5th agent -- remediation_agent.py -- translates each
# Alert.remediation_type into a structured RemediationAction. Wire it AFTER
# "alerting":
#   from remediation_agent import remediation_node
#   graph.add_node("remediation", remediation_node)
#   graph.add_edge("alerting", "remediation")

def alert_node(state: Dict[str, Any], agent: Optional[AlertAgent] = None) -> Dict[str, Any]:
    """LangGraph node: reads state, generates + dispatches alerts, returns
    enriched state. `agent` is injectable (keep one shared instance/
    dispatcher rather than recreating one per call).

    `state["feature_intelligence"]` (optional): a FeatureIntelligenceContext
    recomputed upstream. Without it, the agent falls back to a PSI-only
    decision (degraded but explicit, never silent)."""
    agent = agent or AlertAgent()

    alerts = agent.process_all(
        data_drift_report=state.get("data_drift_report"),
        target_drift_result=state.get("target_drift_result"),
        performance_drift_result=state.get("performance_drift_result"),
        feature_intelligence=state.get("feature_intelligence"),
    )

    escalate = any(a.priority in (AlertPriority.CRITICAL, AlertPriority.PAGE) for a in alerts)

    return {
        **state,
        "alerts": state.get("alerts", []) + [a.to_dict() for a in alerts],
        # Raw Alert objects (not serialized dicts): consumed by remediation_node.
        "_alert_objects": state.get("_alert_objects", []) + alerts,
        "escalate": escalate,
    }
