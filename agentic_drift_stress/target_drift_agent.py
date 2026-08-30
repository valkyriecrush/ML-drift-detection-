"""
Target Drift Agent.

Detects drift on the target variable (y) by reusing drift_detector.py's
PSI/Wasserstein engine, handling the timing gap specific to target drift:
- true labels available          -> CONFIRMED drift (reliable)
- true labels not yet available,
  but model predictions are      -> PROXY drift (early signal, needs confirmation)
- neither available              -> PENDING (nothing to compute this cycle)

Works for both a continuous target (regression: PSI + Wasserstein) and a
categorical one (classification: PSI on proportions) -- type is
auto-detected by DriftDetector from the column dtype.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from drift_detector import DriftDetector, FeatureDriftResult, FeatureType

logger = logging.getLogger(__name__)

TARGET_COLUMN_NAME = "target"


class LabelStatus(Enum):
    CONFIRMED = "confirmed"  # true labels available
    PROXY = "proxy"          # predictions used while waiting for true labels
    PENDING = "pending"      # nothing usable this cycle


@dataclass
class TargetDriftResult:
    status: LabelStatus
    drift: Optional[FeatureDriftResult]
    n_samples: int
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "n_samples": self.n_samples,
            "timestamp": self.timestamp,
            "drift": (
                {
                    "psi_score": round(self.drift.psi_score, 4),
                    "severity": self.drift.severity.value,
                    "is_drifted": self.drift.is_drifted,
                    "norm_wasserstein_distance": (
                        round(self.drift.norm_wasserstein_dist, 4)
                        if self.drift.norm_wasserstein_dist is not None
                        else None
                    ),
                    "details": self.drift.details,
                }
                if self.drift is not None
                else None
            ),
        }


class TargetDriftAgent:
    """Drift detection agent for the target variable (y)."""

    def __init__(self, detector: Optional[DriftDetector] = None, min_samples: int = 30):
        self.min_samples = min_samples
        self.detector = detector or DriftDetector(min_samples=min_samples)
        self._reference_set = False

    def set_reference(self, y_reference: Union[pd.Series, np.ndarray]) -> None:
        """Set the target's reference distribution (historical true labels)."""
        y_ref = self._to_series(y_reference)
        if y_ref.dropna().empty:
            raise ValueError("Target reference is empty.")

        ref_df = pd.DataFrame({TARGET_COLUMN_NAME: y_ref})
        self.detector.set_reference(ref_df)
        self._reference_set = True
        logger.info(f"Target reference set with {len(y_ref)} samples.")

    def detect(
        self,
        y_true: Optional[Union[pd.Series, np.ndarray]] = None,
        y_pred: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> TargetDriftResult:
        """Detect target drift for one monitoring cycle. Prefers y_true; if
        absent, falls back to y_pred in PROXY mode. If neither is provided,
        returns PENDING without raising (normal state while waiting on labels)."""
        if not self._reference_set:
            raise ValueError("Reference not set. Call set_reference() first.")

        timestamp = datetime.now(timezone.utc).isoformat()

        if y_true is not None:
            values = self._to_series(y_true)
            status = LabelStatus.CONFIRMED
        elif y_pred is not None:
            values = self._to_series(y_pred)
            status = LabelStatus.PROXY
            logger.warning("True labels unavailable: using predictions as a proxy.")
        else:
            return TargetDriftResult(status=LabelStatus.PENDING, drift=None, n_samples=0, timestamp=timestamp)

        clean = values.dropna()
        if len(clean) < self.min_samples:
            logger.warning(f"Too few samples ({len(clean)}) for a reliable computation, cycle skipped.")
            return TargetDriftResult(
                status=LabelStatus.PENDING, drift=None, n_samples=len(clean), timestamp=timestamp
            )

        drift_result = self.detector.detect_feature_drift(TARGET_COLUMN_NAME, clean)

        # Chi2 alongside PSI, categorical target only
        ref_info = self.detector.reference_stats[TARGET_COLUMN_NAME]
        if ref_info["type"] == FeatureType.CATEGORICAL:
            clean_categorical = clean.astype(str)
            chi2_stat, chi2_pvalue, skip_reason = self._chi_square_categorical(
                ref_info["proportions"], clean_categorical
            )
            drift_result.details["chi2_statistic"] = chi2_stat
            drift_result.details["chi2_pvalue"] = chi2_pvalue
            if skip_reason:
                drift_result.details["chi2_note"] = skip_reason

        if status == LabelStatus.PROXY and drift_result.is_drifted:
            drift_result.details["note"] = "Proxy signal (predictions), needs confirmation with true labels."

        return TargetDriftResult(status=status, drift=drift_result, n_samples=len(clean), timestamp=timestamp)

    @staticmethod
    def _chi_square_categorical(
        ref_proportions: Dict[str, float], current: pd.Series
    ) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """Chi2 goodness-of-fit test between reference proportions and current
        counts. Returns (statistic, p_value, skip_reason). Skipped if any
        expected count is < 5."""
        n_current = len(current)
        if n_current == 0:
            return None, None, "Current sample is empty."

        current_counts = current.value_counts()
        categories = sorted(set(ref_proportions) | set(current_counts.index))

        f_obs = np.array([current_counts.get(cat, 0) for cat in categories], dtype=float)
        f_exp = np.array([ref_proportions.get(cat, 0.0) * n_current for cat in categories], dtype=float)

        if np.any(f_exp < 5):
            return None, None, "Chi2 skipped: at least one category has expected count < 5."

        # Safety normalization for minor float imprecision
        f_exp = f_exp * (np.sum(f_obs) / np.sum(f_exp))

        statistic, p_value = stats.chisquare(f_obs=f_obs, f_exp=f_exp)
        return float(statistic), float(p_value), None

    @staticmethod
    def _to_series(values: Union[pd.Series, np.ndarray]) -> pd.Series:
        return values if isinstance(values, pd.Series) else pd.Series(values)
