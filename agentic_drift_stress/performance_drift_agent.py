"""
Performance Drift Agent (v3 -- post-MLOps-audit).

Compares real model metrics (computed on a sliding window of labeled
predictions) against a reference baseline, and detects significant,
persistent degradation.

Key changes from the audit:
- average="macro" instead of "weighted" for precision/recall/f1, plus
  optional per-critical-class recall tracking (e.g. fraud) -- "weighted"
  fully hid a minority class's collapse (Recall 0.80->0.00 on a 5%-prevalence
  class still showed 0.95 overall Recall).
- Thresholds standardized by the baseline's bootstrap std, replacing fixed
  absolute/relative thresholds that were either hypersensitive (baseline
  near 0) or insensitive (baseline already degraded). warning_std/
  critical_std are expressed in standard deviations, like a DDM concept-
  drift detector.
- has_proba is no longer all-or-nothing: a window partially missing y_proba
  (e.g. partial timeouts) still computes AUC/calibration on the valid
  subset, as long as it's at least min_proba_ratio of the window.
- remediation_type (renamed from primary_issue) explicitly distinguishes
  "data_pipeline" (AUC uncomputable = class-diversity loss, an upstream
  data issue) from "model_weights" (real prediction degradation) and
  "calibration".

Not addressed here (documented as tech debt):
- Full seasonality (hourly/weekday baseline): needs a time index on
  observations, out of scope for a targeted patch. refresh_baseline() is a
  partial mitigation (a manually refreshable rolling baseline), not an
  automatic solution.
- Pre-allocated NumPy circular buffer: an optimization worth it only at very
  large scale (tens of thousands of predictions/minute), not a priority at
  window_size~500.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Deque, Dict, List, Optional, Sequence, Tuple
import logging

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


class TaskType(Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class LabelAvailability(Enum):
    LABELED = "labeled"
    PENDING = "pending"


class PerformanceSeverity(Enum):
    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"


# Direction only: the threshold is now standardized by the baseline's
# bootstrap std, so there's no more absolute/relative distinction here.
# contributes=False -> computed and reported but never used for severity.
_METRIC_DIRECTION = {
    "accuracy": ("higher_better", True),
    "precision": ("higher_better", True),
    "recall": ("higher_better", True),
    "f1": ("higher_better", True),
    "auc_roc": ("higher_better", True),
    "auc_pr": ("higher_better", True),
    "mae": ("lower_better", True),
    "rmse": ("lower_better", True),
    # Median Absolute Error: robust to outliers, unlike MAE/RMSE (means).
    # Diagnostic guard: if RMSE spikes but medae stays stable, that points
    # to an isolated outlier rather than real systemic degradation.
    "medae": ("lower_better", True),
    "mape": ("lower_better", False),   # blows up near y_true=0, informative only
    "r2": ("higher_better", False),    # sensitive to variance of y, informative only
    "log_loss": ("lower_better", True),
    "brier_score": ("lower_better", True),
    "ece": ("lower_better", True),
}


@dataclass
class PerformanceMetricResult:
    name: str
    baseline: float
    current: float
    delta: float
    std_score: float  # delta normalized by the baseline's std
    degraded: bool
    contributes_to_severity: bool


@dataclass
class PerformanceDriftResult:
    status: LabelAvailability
    severity: Optional[PerformanceSeverity]
    window_size: int
    metrics: List[PerformanceMetricResult] = field(default_factory=list)
    calibration_metrics: List[PerformanceMetricResult] = field(default_factory=list)
    remediation_type: Optional[str] = None  # "data_pipeline" | "model_weights" | "calibration" | None
    proba_coverage: Optional[float] = None  # share of the window with valid y_proba
    notes: List[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict:
        def _m(m: PerformanceMetricResult) -> dict:
            return {
                "name": m.name,
                "baseline": round(m.baseline, 4),
                "current": round(m.current, 4),
                "delta": round(m.delta, 4),
                "std_score": round(m.std_score, 3),
                "degraded": m.degraded,
            }

        return {
            "status": self.status.value,
            "severity": self.severity.value if self.severity else None,
            "remediation_type": self.remediation_type,
            "window_size": self.window_size,
            "proba_coverage": round(self.proba_coverage, 3) if self.proba_coverage is not None else None,
            "metrics": [_m(m) for m in self.metrics],
            "calibration_metrics": [_m(m) for m in self.calibration_metrics],
            "notes": self.notes,
            "timestamp": self.timestamp,
        }


class PerformanceDriftAgent:
    """Sliding-window performance-degradation detector, with thresholds
    standardized by the baseline's std."""

    def __init__(
        self,
        task_type: TaskType,
        window_size: int = 500,
        min_samples: int = 30,
        warning_std: float = 2.0,
        critical_std: float = 3.0,
        use_calibration: bool = False,
        critical_classes: Optional[Sequence] = None,
        critical_recall_floor: float = 0.5,
        min_proba_ratio: float = 0.8,
        degraded_proba_ratio: float = 0.5,
        n_bootstrap: int = 30,
    ):
        if warning_std >= critical_std:
            raise ValueError("warning_std must be strictly less than critical_std.")
        if not (0.0 < degraded_proba_ratio < min_proba_ratio <= 1.0):
            raise ValueError("must have 0 < degraded_proba_ratio < min_proba_ratio <= 1.")

        self.task_type = task_type
        self.window_size = window_size
        self.min_samples = min_samples
        self.warning_std = warning_std
        self.critical_std = critical_std
        self.use_calibration = use_calibration and task_type == TaskType.CLASSIFICATION
        self.critical_classes = list(critical_classes) if critical_classes else []
        # Absolute floor for critical classes: a recall_class_c below this
        # forces CRITICAL immediately, regardless of std_score -- bootstrap
        # can artificially inflate the std of a very rare class (resamples
        # with zero members of that class -> recall fallback=1.0), which
        # would otherwise dilute a real partial collapse.
        self.critical_recall_floor = critical_recall_floor
        self.min_proba_ratio = min_proba_ratio
        # Between degraded_proba_ratio and min_proba_ratio: AUC/calibration
        # are still computed but severity is capped at WARNING (never
        # CRITICAL), reflecting reduced confidence.
        self.degraded_proba_ratio = degraded_proba_ratio
        self.n_bootstrap = n_bootstrap

        self._window: Deque[Tuple] = deque(maxlen=window_size)
        self.baseline_metrics: Dict[str, float] = {}
        self.baseline_std: Dict[str, float] = {}
        self.baseline_calibration: Dict[str, float] = {}
        self.baseline_calibration_std: Dict[str, float] = {}
        self._baseline_set = False
        self._classes: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Baseline
    # ------------------------------------------------------------------ #

    def set_baseline(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> None:
        """Set the performance baseline AND its std (via bootstrap), used to
        standardize all downstream detection thresholds."""
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        y_proba = np.asarray(y_proba) if y_proba is not None else None

        if self.task_type == TaskType.CLASSIFICATION:
            self.baseline_metrics = self._classification_metrics(y_true, y_pred, self.critical_classes)
            self.baseline_std = self._bootstrap_std(
                lambda yt, yp: self._classification_metrics(yt, yp, self.critical_classes), y_true, y_pred
            )
            if y_proba is not None:
                proba_metrics = self._probabilistic_metrics(y_true, y_proba)
                self.baseline_metrics.update(proba_metrics)
                self.baseline_std.update(self._bootstrap_std(self._probabilistic_metrics, y_true, y_proba))
        else:
            self.baseline_metrics = self._regression_metrics(y_true, y_pred)
            self.baseline_std = self._bootstrap_std(self._regression_metrics, y_true, y_pred)

        if self.use_calibration:
            if y_proba is None:
                raise ValueError("use_calibration=True requires y_proba for the baseline.")
            # Class order is fixed HERE, permanently, never re-derived from a
            # current window that might not contain every class.
            self._classes = np.unique(y_true)
            self.baseline_calibration = self._calibration_metrics(y_true, y_proba, self._classes)
            self.baseline_calibration_std = self._bootstrap_std(
                lambda yt, ypr: self._calibration_metrics(yt, ypr, self._classes), y_true, y_proba
            )

        self._baseline_set = True
        logger.info(f"Baseline set ({self.task_type.value}): {self.baseline_metrics}")

    def refresh_baseline(self, y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> None:
        """Partial mitigation for static-baseline drift: lets the baseline be
        recomputed periodically (e.g. weekly) on confirmed-healthy recent
        data, without restarting the agent. Does NOT model seasonality
        (day/hour) -- a simple rolling baseline, not a calendar decomposition."""
        self.set_baseline(y_true, y_pred, y_proba)
        logger.info("Baseline refreshed (rolling baseline, no seasonal decomposition).")

    def _bootstrap_std(self, metric_fn, *arrays, seed: int = 123) -> Dict[str, float]:
        """Estimate each metric's dispersion via bootstrap on the baseline
        data, using MAD (Median Absolute Deviation) rather than the classic
        std -- robust to outliers and heterogeneous baselines that would
        otherwise inflate a classic std and dull the score's sensitivity."""
        rng = np.random.default_rng(seed)
        n = len(arrays[0])
        samples = []
        for _ in range(self.n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            resampled = [a[idx] for a in arrays]
            samples.append(metric_fn(*resampled))
        keys = samples[0].keys()
        result = {}
        for k in keys:
            values = np.array([s[k] for s in samples])
            median = np.median(values)
            mad = np.median(np.abs(values - median))
            result[k] = float(1.4826 * mad)  # consistency factor to approximate a std
        return result

    # ------------------------------------------------------------------ #
    # Update / detection
    # ------------------------------------------------------------------ #

    def update(
        self,
        y_true: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
        y_proba: Optional[np.ndarray] = None,
    ) -> PerformanceDriftResult:
        """Add a labeled batch to the sliding window and evaluate severity.

        If y_true is absent (labels not arrived yet), returns PENDING
        without modifying the window -- a normal state, not an error."""
        if not self._baseline_set:
            raise ValueError("Baseline not set. Call set_baseline() first.")

        timestamp = datetime.now(timezone.utc).isoformat()

        if y_true is None or y_pred is None:
            return PerformanceDriftResult(
                status=LabelAvailability.PENDING, severity=None, window_size=len(self._window), timestamp=timestamp
            )

        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        for i in range(len(y_true)):
            proba_i = y_proba[i] if y_proba is not None else None
            self._window.append((y_true[i], y_pred[i], proba_i))

        if len(self._window) < self.min_samples:
            logger.warning(f"Window too small ({len(self._window)}/{self.min_samples}), cycle skipped.")
            return PerformanceDriftResult(
                status=LabelAvailability.PENDING, severity=None, window_size=len(self._window), timestamp=timestamp
            )

        # Single pass over the window (instead of 3 separate comprehensions).
        w_true_list, w_pred_list, w_proba_list = [], [], []
        for t, p, pr in self._window:
            w_true_list.append(t)
            w_pred_list.append(p)
            w_proba_list.append(pr)
        w_true = np.array(w_true_list)
        w_pred = np.array(w_pred_list)

        # has_proba has 3 tiers instead of all-or-nothing:
        # >= min_proba_ratio: full confidence, AUC/calibration count normally
        # [degraded_proba_ratio, min_proba_ratio[: still computed, but
        #   severity capped at WARNING (never CRITICAL) -- reduced confidence
        # < degraded_proba_ratio: ignored entirely, too unreliable
        valid_idx = [i for i, pr in enumerate(w_proba_list) if pr is not None]
        proba_coverage = len(valid_idx) / len(w_proba_list) if w_proba_list else 0.0
        has_proba = proba_coverage >= self.degraded_proba_ratio
        proba_confidence_reduced = self.degraded_proba_ratio <= proba_coverage < self.min_proba_ratio

        notes: List[str] = []
        if proba_confidence_reduced:
            notes.append(
                f"y_proba available on {proba_coverage:.0%} of the window (between "
                f"{self.degraded_proba_ratio:.0%} and {self.min_proba_ratio:.0%}): "
                f"AUC/calibration computed but severity capped at WARNING (reduced confidence)."
            )
        elif not has_proba and len(valid_idx) > 0:
            notes.append(
                f"y_proba available on only {proba_coverage:.0%} of the window "
                f"(< {self.degraded_proba_ratio:.0%}): AUC/calibration skipped this cycle."
            )

        diversity_collapsed = False

        if self.task_type == TaskType.CLASSIFICATION:
            current_metrics = self._classification_metrics(w_true, w_pred, self.critical_classes)
            if has_proba:
                w_true_proba = np.array([w_true_list[i] for i in valid_idx])
                w_proba = np.array([w_proba_list[i] for i in valid_idx])
                current_metrics.update(self._probabilistic_metrics(w_true_proba, w_proba))
        else:
            current_metrics = self._regression_metrics(w_true, w_pred)

        metric_results = self._compare(self.baseline_metrics, current_metrics, self.baseline_std)

        # An uncomputable AUC/AUC-PR (NaN) signals class-diversity loss --
        # always treated as an explicit CRITICAL signal, never "no degradation".
        for m in metric_results:
            if m.name in ("auc_roc", "auc_pr") and np.isnan(m.current):
                m.degraded = True
                diversity_collapsed = True

        # Probabilistic metrics (AUC/AUC-PR) are capped at WARNING if proba
        # coverage is reduced -- separated from pure classification metrics
        # for aggregation.
        proba_metric_results = [m for m in metric_results if m.name in ("auc_roc", "auc_pr")]
        core_metric_results = [m for m in metric_results if m.name not in ("auc_roc", "auc_pr")]

        severity = self._aggregate_severity(core_metric_results)
        if proba_metric_results:
            proba_severity = self._aggregate_severity(proba_metric_results, cap_at_warning=proba_confidence_reduced)
            if self._severity_rank(proba_severity) > self._severity_rank(severity):
                severity = proba_severity

        calibration_results: List[PerformanceMetricResult] = []
        if self.use_calibration and has_proba:
            # self._classes is fixed at baseline time (see set_baseline): if
            # the current window has a class absent from the baseline
            # (target drift on y_true itself, e.g. a new fraud category),
            # log_loss(labels=self._classes) raises a ValueError. Excluding
            # those samples from the calibration calc (never silently -- the
            # note says so) avoids crashing the cycle.
            known_mask = np.isin(w_true_proba, self._classes)
            n_unknown = int((~known_mask).sum())
            if n_unknown > 0:
                unknown_classes = sorted(set(np.unique(w_true_proba[~known_mask]).tolist()))
                notes.append(
                    f"{n_unknown} sample(s) in the window belong to class(es) absent from the "
                    f"baseline ({unknown_classes}): excluded from the calibration calc. This can "
                    "itself indicate target drift (new class) -- see target_drift_agent."
                )

            if known_mask.sum() >= self.min_samples:
                current_calibration = self._calibration_metrics(
                    w_true_proba[known_mask], w_proba[known_mask], self._classes
                )
                calibration_results = self._compare(
                    self.baseline_calibration, current_calibration, self.baseline_calibration_std
                )
                calib_severity = self._aggregate_severity(
                    calibration_results, cap_at_warning=proba_confidence_reduced
                )
                if self._severity_rank(calib_severity) > self._severity_rank(severity):
                    severity = calib_severity
            else:
                notes.append(
                    "Too few known-class samples to compute calibration this cycle "
                    f"({int(known_mask.sum())}/{self.min_samples})."
                )

        remediation_type = self._determine_remediation(metric_results, calibration_results, diversity_collapsed)

        return PerformanceDriftResult(
            status=LabelAvailability.LABELED,
            severity=severity,
            window_size=len(self._window),
            metrics=metric_results,
            calibration_metrics=calibration_results,
            remediation_type=remediation_type,
            proba_coverage=proba_coverage,
            notes=notes,
            timestamp=timestamp,
        )

    # ------------------------------------------------------------------ #
    # Metric computation
    # ------------------------------------------------------------------ #

    @staticmethod
    def _classification_metrics(y_true, y_pred, critical_classes: Sequence = ()) -> Dict[str, float]:
        """average="macro" (not "weighted"): each class gets equal weight,
        so a minority class's collapse is no longer drowned out by the
        majority class. Per-critical-class recall is tracked explicitly on
        top for an unambiguous diagnosis (macro alone can still dilute the
        signal when there are many classes)."""
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }
        for c in critical_classes:
            mask = y_true == c
            if mask.sum() > 0:
                metrics[f"recall_class_{c}"] = float((y_pred[mask] == c).sum() / mask.sum())
            else:
                metrics[f"recall_class_{c}"] = 1.0  # class absent from this batch: nothing to flag
        return metrics

    @staticmethod
    def _probabilistic_metrics(y_true, y_proba) -> Dict[str, float]:
        """AUC-ROC / AUC-PR, kept separate from pure classification metrics
        so they can be computed on a subset of the window (see the tolerant
        has_proba handling)."""
        metrics: Dict[str, float] = {}
        try:
            n_classes = y_proba.shape[1] if y_proba.ndim > 1 else 2
            proba_for_auc = y_proba[:, 1] if (y_proba.ndim > 1 and n_classes == 2) else y_proba
            multi_class = "ovr" if n_classes > 2 else "raise"
            metrics["auc_roc"] = float(roc_auc_score(y_true, proba_for_auc, multi_class=multi_class))
            if n_classes == 2:
                metrics["auc_pr"] = float(average_precision_score(y_true, proba_for_auc))
        except ValueError as exc:
            logger.warning(f"AUC not computable on this batch (possible loss of class diversity?): {exc}")
            metrics["auc_roc"] = float("nan")
            metrics["auc_pr"] = float("nan")
        return metrics

    @staticmethod
    def _regression_metrics(y_true, y_pred) -> Dict[str, float]:
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        medae = float(median_absolute_error(y_true, y_pred))
        nonzero = np.abs(y_true) > 1e-8
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100) if nonzero.any() else 0.0
        r2 = float(r2_score(y_true, y_pred))
        return {"mae": mae, "rmse": rmse, "medae": medae, "mape": mape, "r2": r2}

    @staticmethod
    def _calibration_metrics(y_true, y_proba, classes: np.ndarray) -> Dict[str, float]:
        y_true = np.asarray(y_true)
        proba = np.asarray(y_proba)

        if proba.ndim == 1:
            proba_matrix = np.column_stack([1 - proba, proba])
        else:
            proba_matrix = proba

        class_to_idx = {c: i for i, c in enumerate(classes)}
        ll = float(log_loss(y_true, proba_matrix, labels=classes))

        one_hot = np.zeros_like(proba_matrix)
        for i, y in enumerate(y_true):
            if y in class_to_idx:
                one_hot[i, class_to_idx[y]] = 1.0
        brier = float(np.mean(np.sum((proba_matrix - one_hot) ** 2, axis=1)))

        confidence = proba_matrix.max(axis=1)
        predicted_idx = proba_matrix.argmax(axis=1)
        idx_to_class = {i: c for c, i in class_to_idx.items()}
        predicted_class = np.array([idx_to_class.get(i, classes[0]) for i in predicted_idx])
        correct = (predicted_class == y_true).astype(float)

        n_bins = 10
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (confidence > lo) & (confidence <= hi) if i > 0 else (confidence >= lo) & (confidence <= hi)
            if mask.sum() == 0:
                continue
            bin_acc = correct[mask].mean()
            bin_conf = confidence[mask].mean()
            ece += (mask.sum() / len(confidence)) * abs(bin_acc - bin_conf)

        return {"log_loss": ll, "brier_score": brier, "ece": float(ece)}

    # ------------------------------------------------------------------ #
    # Comparison and severity (standardized by the baseline's std)
    # ------------------------------------------------------------------ #

    def _effective_std(self, name: str, baseline_value: float, std_value: float) -> float:
        """A safety floor proportional to each metric's own scale -- fixes a
        prior bug where a fixed absolute floor (1e-3) dominated for
        small-scale metrics (e.g. MAE=0.001, where 1e-3 exceeded the
        baseline itself and drowned any real signal). The floor is now
        purely relative to the baseline, plus a tiny numerical epsilon to
        avoid strict division by zero."""
        floor_rel = 0.01 * abs(baseline_value)
        numerical_epsilon = 1e-9
        return max(std_value, floor_rel, numerical_epsilon)

    def _compare(
        self, baseline: Dict[str, float], current: Dict[str, float], baseline_std: Dict[str, float]
    ) -> List[PerformanceMetricResult]:
        results = []
        for name, current_value in current.items():
            if name not in baseline:
                continue
            baseline_value = baseline[name]
            direction, contributes = _METRIC_DIRECTION.get(name, ("higher_better", True))
            delta = current_value - baseline_value

            std_value = baseline_std.get(name, 0.0)
            eff_std = self._effective_std(name, baseline_value, std_value)
            std_score = delta / eff_std

            if direction == "higher_better":
                degraded = std_score < -self.warning_std
            else:
                degraded = std_score > self.warning_std

            results.append(
                PerformanceMetricResult(
                    name=name,
                    baseline=baseline_value,
                    current=current_value,
                    delta=delta,
                    std_score=std_score,
                    degraded=bool(degraded and contributes),
                    contributes_to_severity=contributes,
                )
            )
        return results

    def _aggregate_severity(self, metric_results: List[PerformanceMetricResult], cap_at_warning: bool = False) -> PerformanceSeverity:
        worst = PerformanceSeverity.NONE
        for m in metric_results:
            if not m.contributes_to_severity:
                continue

            # Absolute floor for a critical class: regardless of std_score, a
            # recall below critical_recall_floor is ALWAYS critical.
            if m.name.startswith("recall_class_") and m.current < self.critical_recall_floor:
                return PerformanceSeverity.CRITICAL

            direction, _ = _METRIC_DIRECTION.get(m.name, ("higher_better", True))
            magnitude = -m.std_score if direction == "higher_better" else m.std_score

            if magnitude > self.critical_std and not cap_at_warning:
                return PerformanceSeverity.CRITICAL
            elif magnitude > self.warning_std and self._severity_rank(worst) < self._severity_rank(
                PerformanceSeverity.WARNING
            ):
                worst = PerformanceSeverity.WARNING
        return worst

    @staticmethod
    def _severity_rank(severity: PerformanceSeverity) -> int:
        return [PerformanceSeverity.NONE, PerformanceSeverity.WARNING, PerformanceSeverity.CRITICAL].index(severity)

    @staticmethod
    def _determine_remediation(
        metric_results: List[PerformanceMetricResult],
        calibration_results: List[PerformanceMetricResult],
        diversity_collapsed: bool,
    ) -> Optional[str]:
        """Points the monitoring agent to the right corrective action:
        - "data_pipeline": AUC uncomputable = class-diversity loss in the
          window -- an upstream data issue (target drift, ingestion), NOT a
          model-weight problem. Top priority: retraining fixes nothing.
        - "model_weights": real prediction degradation on valid data.
        - "calibration": classification is sound, but probabilities are
          poorly calibrated.
        """
        if diversity_collapsed:
            return "data_pipeline"

        perf_degraded = any(m.degraded for m in metric_results)
        calib_degraded = any(m.degraded for m in calibration_results)

        if perf_degraded:
            return "model_weights"
        if calib_degraded:
            return "calibration"
        return None
