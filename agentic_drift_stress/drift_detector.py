"""
Data Drift Detection Module (production-ready & optimized).

Detects drift on numeric and categorical features via:
- PSI (Population Stability Index), quantile binning (numeric) or
  proportions (categorical)
- Wasserstein distance normalized by a robust dispersion measure (IQR/Std)

The KS-Test is intentionally omitted: it produces too many false positives
on large samples.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance

# psi_common.py lives at the repo root, shared with
# drift_simulator/baseline/baseline_calculator.py, so both use the same PSI formula.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from psi_common import psi_from_counts  # noqa: E402

logger = logging.getLogger(__name__)


class DriftSeverity(Enum):
    NONE = "none"
    LOW = "low"
    WARNING = "warning"
    CRITICAL = "critical"


class FeatureType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"


@dataclass
class FeatureDriftResult:
    feature_name: str
    feature_type: FeatureType
    psi_score: float
    severity: DriftSeverity
    is_drifted: bool
    norm_wasserstein_dist: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftReport:
    total_features: int
    drifted_features: int
    feature_results: List[FeatureDriftResult]
    overall_severity: DriftSeverity
    drift_percentage: float
    recommendations: List[str]
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_features": self.total_features,
            "drifted_features": self.drifted_features,
            "drift_percentage": round(self.drift_percentage, 2),
            "overall_severity": self.overall_severity.value,
            "feature_results": [
                {
                    "feature": r.feature_name,
                    "type": r.feature_type.value,
                    "psi_score": round(r.psi_score, 4),
                    "norm_wasserstein_distance": round(r.norm_wasserstein_dist, 4)
                    if r.norm_wasserstein_dist is not None
                    else None,
                    "severity": r.severity.value,
                    "is_drifted": r.is_drifted,
                    "missing_rate": r.details.get("missing_rate"),
                    "new_categories": r.details.get("new_categories") or [],
                }
                for r in self.feature_results
            ],
            "recommendations": self.recommendations,
            "timestamp": self.timestamp,
        }


class DriftDetector:
    """Configurable univariate drift detector for numeric and categorical variables."""

    def __init__(
        self,
        psi_low: float = 0.05,
        psi_warning: float = 0.1,
        psi_critical: float = 0.25,
        wasserstein_threshold: float = 0.2,
        n_bins: int = 10,
        max_reference_samples: int = 5000,
        min_samples: int = 10,
        random_state: Optional[int] = None,
    ):
        self.psi_low = psi_low
        self.psi_warning = psi_warning
        self.psi_critical = psi_critical
        self.wasserstein_threshold = wasserstein_threshold
        self.n_bins = n_bins
        self.max_reference_samples = max_reference_samples
        self.min_samples = min_samples
        self._rng = np.random.default_rng(random_state)

        self.reference_stats: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Reference setup
    # ------------------------------------------------------------------ #

    def set_reference(self, data: pd.DataFrame) -> None:
        """Compute and store the reference population's statistics."""
        self.reference_stats.clear()

        # Consistent global subsampling at the DataFrame level
        if len(data) > self.max_reference_samples:
            idx = self._rng.choice(len(data), size=self.max_reference_samples, replace=False)
            sampled_data = data.iloc[idx]
        else:
            sampled_data = data

        for col in sampled_data.columns:
            series = self._clean(sampled_data[col])
            if series.empty:
                logger.warning(f"Column '{col}' empty after cleaning, skipped.")
                continue

            if pd.api.types.is_numeric_dtype(series):
                values = series.to_numpy()
                iqr = float(stats.iqr(values))
                std = float(np.std(values))

                # Guard against division by zero / near-zero values
                dispersion = iqr if iqr > 1e-6 else (std if std > 1e-6 else 1.0)

                self.reference_stats[col] = {
                    "type": FeatureType.NUMERIC,
                    "mean": float(np.mean(values)),
                    "std": std,
                    "dispersion": dispersion,
                    "sample": values,
                }
            else:
                str_series = series.astype(str)
                self.reference_stats[col] = {
                    "type": FeatureType.CATEGORICAL,
                    "proportions": str_series.value_counts(normalize=True).to_dict(),
                    "total_count": len(str_series),
                }

        logger.info(f"Reference configured: {len(self.reference_stats)} features.")

    @staticmethod
    def _clean(series: pd.Series) -> pd.Series:
        """Drop NaNs and infinite values."""
        series = series.dropna()
        if pd.api.types.is_numeric_dtype(series):
            series = series[np.isfinite(series)]
        return series

    # ------------------------------------------------------------------ #
    # Metric computation
    # ------------------------------------------------------------------ #

    def _psi_numeric(self, reference: np.ndarray, current: np.ndarray) -> float:
        """Numeric PSI via safe quantile binning. The smoothing + final
        formula is delegated to psi_common.psi_from_counts, shared with
        baseline_calculator.compute_psi() to avoid divergent results
        between the two sub-projects. Bin selection (by reference
        quantiles) stays local to this method."""
        if len(reference) == 0 or len(current) == 0:
            return 0.0

        quantiles = np.linspace(0, 100, self.n_bins + 1)
        bins = np.unique(np.percentile(reference, quantiles))

        if len(bins) < 2:
            return 0.0

        bins[0], bins[-1] = -np.inf, np.inf

        ref_counts, _ = np.histogram(reference, bins=bins)
        cur_counts, _ = np.histogram(current, bins=bins)

        return psi_from_counts(ref_counts, cur_counts, epsilon=1e-4)

    def _psi_categorical(self, ref_props: Dict[str, float], current: pd.Series) -> float:
        """PSI for categories (with smoothing for unseen categories)."""
        cur_props = current.value_counts(normalize=True).to_dict()
        categories = set(ref_props) | set(cur_props)
        eps = 1e-4
        psi = 0.0

        for cat in categories:
            p_ref = ref_props.get(cat, eps)
            p_cur = cur_props.get(cat, eps)
            psi += (p_cur - p_ref) * np.log(p_cur / p_ref)

        return float(psi)

    # ------------------------------------------------------------------ #
    # Per-feature detection
    # ------------------------------------------------------------------ #

    def detect_feature_drift(self, feature_name: str, current_series: pd.Series) -> FeatureDriftResult:
        """Detect drift for a single feature."""
        if feature_name not in self.reference_stats:
            raise ValueError(f"Feature '{feature_name}' not in reference.")

        ref = self.reference_stats[feature_name]
        current_clean = self._clean(current_series)

        # Data-quality signal computed on the RAW series (before dropna) --
        # this is exactly the information _clean() silently discards today,
        # and the piece missing to turn "check the pipeline" into a final,
        # evidence-based verdict (see AlertAgent._diagnose_data_quality).
        quality = self._quality_diagnostics(feature_name, current_series, ref)

        current_is_numeric = pd.api.types.is_numeric_dtype(current_clean)
        ref_is_numeric = ref["type"] == FeatureType.NUMERIC

        if current_is_numeric != ref_is_numeric:
            raise TypeError(
                f"Type mismatch for '{feature_name}': reference="
                f"{'numeric' if ref_is_numeric else 'categorical'}, "
                f"current={'numeric' if current_is_numeric else 'categorical'}."
            )

        if ref_is_numeric:
            result = self._detect_numeric(feature_name, ref, current_clean.to_numpy())
        else:
            result = self._detect_categorical(feature_name, ref, current_clean.astype(str))

        result.details.update(quality)
        return result

    @staticmethod
    def _quality_diagnostics(feature_name: str, raw_series: pd.Series, ref: Dict[str, Any]) -> Dict[str, Any]:
        """Data-quality checks on the RAW (uncleaned) current batch:
        missing_rate (NaN/inf) and, for categorical features, new
        categories never seen in the reference distribution. Both are
        strong evidence to tell apart an upstream pipeline problem
        (values suddenly missing / a new unmapped category) from a
        genuine population shift (values present, well-formed, just
        distributed differently) -- exactly the distinction a human is
        asked to make today by the generic "check the pipeline" message."""
        n_total = len(raw_series)
        if n_total == 0:
            return {"missing_rate": None, "n_missing": 0, "n_total": 0, "new_categories": []}

        is_numeric = pd.api.types.is_numeric_dtype(raw_series)
        n_na = int(raw_series.isna().sum())
        if is_numeric:
            non_null = raw_series.dropna()
            n_inf = int((~np.isfinite(non_null.to_numpy(dtype=float))).sum()) if len(non_null) else 0
            n_missing = n_na + n_inf
        else:
            n_missing = n_na

        new_categories: List[str] = []
        if ref["type"] == FeatureType.CATEGORICAL:
            known = set(ref.get("proportions", {}).keys())
            observed = set(raw_series.dropna().astype(str).unique())
            new_categories = sorted(observed - known)

        return {
            "missing_rate": round(n_missing / n_total, 4),
            "n_missing": n_missing,
            "n_total": n_total,
            "new_categories": new_categories,
        }

    def _detect_numeric(self, name: str, ref: Dict[str, Any], current: np.ndarray) -> FeatureDriftResult:
        ref_sample = ref["sample"]

        psi_score = self._psi_numeric(ref_sample, current)
        norm_wasserstein = wasserstein_distance(ref_sample, current) / ref["dispersion"]

        wasserstein_flag = norm_wasserstein > self.wasserstein_threshold

        if psi_score >= self.psi_critical or (psi_score >= self.psi_warning and wasserstein_flag):
            severity = DriftSeverity.CRITICAL
            is_drifted = True
        elif psi_score >= self.psi_warning or wasserstein_flag:
            severity = DriftSeverity.WARNING
            is_drifted = True
        elif psi_score >= self.psi_low:
            severity = DriftSeverity.LOW
            is_drifted = False
        else:
            severity = DriftSeverity.NONE
            is_drifted = False

        return FeatureDriftResult(
            feature_name=name,
            feature_type=FeatureType.NUMERIC,
            psi_score=psi_score,
            norm_wasserstein_dist=float(norm_wasserstein),
            severity=severity,
            is_drifted=is_drifted,
            details={
                "reference_mean": ref["mean"],
                "reference_std": ref["std"],
                "current_mean": float(np.mean(current)),
                "current_std": float(np.std(current)),
            },
        )

    def _detect_categorical(self, name: str, ref: Dict[str, Any], current: pd.Series) -> FeatureDriftResult:
        psi_score = self._psi_categorical(ref["proportions"], current)

        if psi_score >= self.psi_critical:
            severity = DriftSeverity.CRITICAL
            is_drifted = True
        elif psi_score >= self.psi_warning:
            severity = DriftSeverity.WARNING
            is_drifted = True
        elif psi_score >= self.psi_low:
            severity = DriftSeverity.LOW
            is_drifted = False
        else:
            severity = DriftSeverity.NONE
            is_drifted = False

        return FeatureDriftResult(
            feature_name=name,
            feature_type=FeatureType.CATEGORICAL,
            psi_score=psi_score,
            severity=severity,
            is_drifted=is_drifted,
        )

    # ------------------------------------------------------------------ #
    # Full report
    # ------------------------------------------------------------------ #

    def detect_drift(self, current_data: pd.DataFrame, features: Optional[List[str]] = None) -> DriftReport:
        """Generate the full drift-detection report on the current dataset."""
        if not self.reference_stats:
            raise ValueError("Reference not set. Call set_reference() first.")

        if features is None:
            features = [c for c in current_data.columns if c in self.reference_stats]

        severity_order = [DriftSeverity.NONE, DriftSeverity.LOW, DriftSeverity.WARNING, DriftSeverity.CRITICAL]
        results: List[FeatureDriftResult] = []
        max_severity = DriftSeverity.NONE

        for feature in features:
            if feature not in current_data.columns:
                logger.warning(f"'{feature}' missing from current data, skipped.")
                continue

            clean = self._clean(current_data[feature])
            if len(clean) < self.min_samples:
                logger.warning(f"'{feature}' has too few samples ({len(clean)}), skipped.")
                continue

            result = self.detect_feature_drift(feature, current_data[feature])
            results.append(result)

            if severity_order.index(result.severity) > severity_order.index(max_severity):
                max_severity = result.severity

        drifted_count = sum(r.is_drifted for r in results)

        return DriftReport(
            total_features=len(results),
            drifted_features=drifted_count,
            feature_results=results,
            overall_severity=max_severity,
            drift_percentage=(drifted_count / len(results) * 100) if results else 0.0,
            recommendations=self._recommendations(results, max_severity),
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def _recommendations(self, results: List[FeatureDriftResult], severity: DriftSeverity) -> List[str]:
        recs = []

        if severity == DriftSeverity.CRITICAL:
            recs.append("CRITICAL: model retraining recommended.")
            recs.append("Check upstream data pipelines (schema or value changes).")
        elif severity == DriftSeverity.WARNING:
            recs.append("WARNING: moderate drift detected. Increase monitoring frequency.")

        critical = [r.feature_name for r in results if r.severity == DriftSeverity.CRITICAL]
        if critical:
            recs.append(f"Critical drift on: {', '.join(critical[:5])}")

        high_psi = [r.feature_name for r in results if r.psi_score >= self.psi_warning]
        if high_psi:
            recs.append(f"{len(high_psi)} feature(s) have PSI >= {self.psi_warning}")

        if not recs:
            recs.append("No significant drift detected. Model health is stable.")

        return recs
