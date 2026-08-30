# Purpose: abstract base class shared by all drift simulators (reset, model loading, preprocessing, PSI checks, JSON export).

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Generator, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from baseline.baseline_calculator import BaselineCalculator, _to_native
from src.eda import grab_col_names
from src.feature_engineering import data_prep, feature_extraction, label_encoder, one_hot_encoder
from src.preprocessing import (
    ZERO_AS_MISSING_COLS,
    compute_global_medians,
    replace_with_thresholds,
    zeros_to_missing,
    zeros_to_missing_infer,
)

# Hidden column carrying the "true y" (before any label noise / drifted rule
# threshold) across every resampling/reindexing step.
HIDDEN_TRUE_LABEL_COL = "_y_true_uncorrupted"


# =============================================================================
# Generic drift status (OK / WARNING / CRITICAL)
# =============================================================================
class DriftStatus(str, Enum):
    OK = "OK"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class DriftStatusResult:
    """
    Classification result of a reference PSI against low/high thresholds
    supplied by the subclass (only the subclass knows its scenario's
    business semantics — "moderate", "severe", etc.).
    """

    status: str
    psi_reference: float
    reference_feature: Optional[str]
    threshold_warning_low: float
    threshold_critical_high: Optional[float]
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return _to_native(
            {
                "status": self.status,
                "psi_reference": self.psi_reference,
                "reference_feature": self.reference_feature,
                "threshold_warning_low": self.threshold_warning_low,
                "threshold_critical_high": self.threshold_critical_high,
                "message": self.message,
            }
        )


# =============================================================================
# Gradual drift utility (generator) — shared, stateless
# =============================================================================
def generate_gradual_drift_stream(
    df: pd.DataFrame,
    feature: str,
    total_delta: float,
    n_batches: int,
    rng: np.random.RandomState,
) -> Generator[pd.DataFrame, None, None]:
    """
    Generates n_batches cumulative states of gradual drift on `feature`,
    starting from `df` (already modified by previous steps -> the caller
    must pass the current `current_df`, not the raw baseline).
    Only the LAST batch's state should be kept by the caller.
    """
    state = df.copy()
    per_batch_delta = total_delta / n_batches
    for _ in range(n_batches):
        noise = rng.normal(loc=0.0, scale=abs(per_batch_delta) * 0.05, size=len(state))
        state = state.copy()
        state[feature] = state[feature] + per_batch_delta + noise
        yield state


class BaseDriftSimulator(ABC):
    """
    Abstract base class. Do not instantiate directly: use a concrete
    subclass (e.g. `NormalDriftSimulator`) that implements `run()`.
    """

    def __init__(self, baseline_calc: BaselineCalculator) -> None:
        self.baseline_calc = baseline_calc
        self.baseline_df: pd.DataFrame = baseline_calc.baseline_df.copy()
        self.current_df: Optional[pd.DataFrame] = None
        self.y_true_uncorrupted: Optional[pd.Series] = None
        self.metadata: Dict[str, Any] = {}
        self.model: Any = None
        # Persistent scaler, fit once on baseline_df, never reset by reset().
        self._persistent_scaler: Optional[RobustScaler] = None
        self._persistent_scaler_cols: Optional[List[str]] = None
        # Frozen global (label-free) medians for zero-imputation at
        # inference, computed once on the baseline (see
        # `_ensure_persistent_scaler()` / `_prep_infer()`).
        self._persistent_global_medians: Optional[Dict[str, float]] = None

        self.reset(self.baseline_df)

    # ------------------------------------------------------------------
    # 0. Idempotence & traceability
    # ------------------------------------------------------------------
    def reset(self, baseline_df: pd.DataFrame) -> None:
        """
        Destroys any leftover state from a previous run that is MUTABLE
        per-run (current_df, y_true_uncorrupted, metadata). Deliberately
        does NOT touch the persistent scaler: it's an artifact fit once on
        the baseline, same as `baseline_calc.bin_edges`.
        """
        self.current_df = baseline_df.copy()
        self.y_true_uncorrupted = None
        self.metadata = {
            "psi_checkpoints": {},
            "flags": {},
            "scenario_type": None,
            "composite_drift": False,
            "resampling_contamination_warning": False,
        }

    def _load_model(self, model_path: str = "models/lgbm_model.pkl") -> None:
        # Confirmed on the real repo: notebooks/04_modeling.ipynb persists
        # the model with joblib.dump(lgbm_model, "../models/lgbm_model.pkl").
        import joblib

        self.model = joblib.load(model_path)

    def _expected_columns(self) -> Optional[List[str]]:
        """
        Source of truth for the expected column order out of `data_prep`,
        in priority order:
        1. `config/performance.yaml` (booster_info.feature_names), via
           `BaselineCalculator.expected_model_columns()`.
        2. `model.feature_name_` (fallback, LightGBM-specific).
        """
        cols = self.baseline_calc.expected_model_columns()
        if cols:
            return cols
        if self.model is not None and hasattr(self.model, "feature_name_"):
            return list(self.model.feature_name_)
        return None

    # ------------------------------------------------------------------
    # Preprocessing pipeline (faithful to src.feature_engineering.data_prep,
    # plus a persistent scaler)
    # ------------------------------------------------------------------
    def _prep_with_persistent_scaler(
        self, X: pd.DataFrame, y: pd.Series, fit_scaler: bool
    ) -> pd.DataFrame:
        target_col = y.name or "Outcome"
        index = X.index
        dataframe = X.merge(y.to_frame(), left_index=True, right_index=True).set_index(index)

        cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, print_results=False)

        for col in num_cols:
            replace_with_thresholds(dataframe, col)

        # Zero -> missing, then median imputation per class: reuse the
        # canonical src.preprocessing.zeros_to_missing() instead of
        # re-deriving the same np.where/fillna logic here.
        # NOTE: baseline_df is already cleaned upstream by
        # BaselineCalculator.load_or_compute() (same zeros_to_missing()
        # applied BEFORE any drift primitive runs), so this call is normally
        # a no-op here. It's kept as a safety net for zeros that drift itself
        # could reintroduce (e.g. a feature_step delta landing exactly on 0).
        present_zero_cols = [c for c in ZERO_AS_MISSING_COLS if c in dataframe.columns]
        if present_zero_cols:
            dataframe = zeros_to_missing(dataframe, columns=present_zero_cols)

        feature_extraction(dataframe)

        if fit_scaler:
            self._persistent_scaler = RobustScaler().fit(dataframe[num_cols])
            self._persistent_scaler_cols = list(num_cols)
            dataframe[num_cols] = self._persistent_scaler.transform(dataframe[num_cols])
        else:
            if self._persistent_scaler is None:
                raise RuntimeError(
                    "Persistent scaler not initialized: call "
                    "_prep_with_persistent_scaler(baseline_X, baseline_y, fit_scaler=True) first."
                )
            scaler_cols = [c for c in self._persistent_scaler_cols if c in dataframe.columns]
            missing_cols = [c for c in self._persistent_scaler_cols if c not in dataframe.columns]
            if missing_cols:
                self.metadata.setdefault("flags", {})["scaler_missing_columns"] = missing_cols
            dataframe[scaler_cols] = self._persistent_scaler.transform(dataframe[scaler_cols])
            extra_cols = [c for c in num_cols if c not in self._persistent_scaler_cols]
            if extra_cols:
                self.metadata.setdefault("flags", {})["scaler_unscaled_new_columns"] = extra_cols

        binary_cols = [
            col
            for col in dataframe.columns
            if dataframe[col].dtype not in ["int64", "float64"] and dataframe[col].nunique() == 2
        ]
        for col in binary_cols:
            label_encoder(dataframe, col)

        ohe_cols = [col for col in dataframe.columns if 12 >= dataframe[col].nunique() > 2]
        dataframe = one_hot_encoder(dataframe, ohe_cols, drop_first=True)

        return dataframe.drop([target_col], axis=1)

    def _ensure_persistent_scaler(self, target_col: str) -> None:
        """Fit the persistent scaler on baseline_df, once (no-op afterward).

        Also freezes the global (label-free) per-feature medians used at
        inference (`self._persistent_global_medians`, via
        `compute_global_medians()`), so that scoring a live batch later
        never needs to (re)compute anything from that batch's label.
        """
        if self._persistent_scaler is not None:
            return
        X_baseline = self.baseline_df.drop(columns=[target_col])
        y_baseline = self.baseline_df[target_col]
        zero_cols_present = [c for c in ZERO_AS_MISSING_COLS if c in X_baseline.columns]
        self._persistent_global_medians = compute_global_medians(X_baseline, columns=zero_cols_present)
        self._prep_with_persistent_scaler(X_baseline, y_baseline, fit_scaler=True)

    def _prep_infer(self, X: pd.DataFrame, fit_scaler: bool) -> pd.DataFrame:
        """
        Inference-safe counterpart of `_prep_with_persistent_scaler()`:
        NEVER receives or needs `y`. Zeros are imputed with the medians
        frozen once on the baseline (`self._persistent_global_medians`,
        via `zeros_to_missing_infer()`) instead of the per-class median
        used at training time (`zeros_to_missing()`, label-conditioned)
        -- see the AUDIT P0 FIX note on `_run_pipeline()`.
        """
        dataframe = X.copy()

        cat_cols, num_cols, cat_but_car = grab_col_names(dataframe, print_results=False)

        for col in num_cols:
            replace_with_thresholds(dataframe, col)

        present_zero_cols = [c for c in ZERO_AS_MISSING_COLS if c in dataframe.columns]
        if present_zero_cols:
            dataframe = zeros_to_missing_infer(
                dataframe, self._persistent_global_medians or {}, columns=present_zero_cols
            )

        feature_extraction(dataframe)

        if fit_scaler:
            self._persistent_scaler = RobustScaler().fit(dataframe[num_cols])
            self._persistent_scaler_cols = list(num_cols)
            dataframe[num_cols] = self._persistent_scaler.transform(dataframe[num_cols])
        else:
            if self._persistent_scaler is None:
                raise RuntimeError(
                    "Persistent scaler not initialized: call "
                    "_ensure_persistent_scaler(target_col) first."
                )
            scaler_cols = [c for c in self._persistent_scaler_cols if c in dataframe.columns]
            missing_cols = [c for c in self._persistent_scaler_cols if c not in dataframe.columns]
            if missing_cols:
                self.metadata.setdefault("flags", {})["scaler_missing_columns"] = missing_cols
            dataframe[scaler_cols] = self._persistent_scaler.transform(dataframe[scaler_cols])
            extra_cols = [c for c in num_cols if c not in self._persistent_scaler_cols]
            if extra_cols:
                self.metadata.setdefault("flags", {})["scaler_unscaled_new_columns"] = extra_cols

        binary_cols = [
            col
            for col in dataframe.columns
            if dataframe[col].dtype not in ["int64", "float64"] and dataframe[col].nunique() == 2
        ]
        for col in binary_cols:
            label_encoder(dataframe, col)

        ohe_cols = [col for col in dataframe.columns if 12 >= dataframe[col].nunique() > 2]
        dataframe = one_hot_encoder(dataframe, ohe_cols, drop_first=True)

        return dataframe

    def _run_pipeline(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        scaler_mode: str = "persistent_baseline_fit",
    ) -> pd.DataFrame:
        """
        Integration point with the agentic repo's preprocessing.

        AUDIT P0 FIX (label leakage): `scaler_mode="persistent_baseline_fit"`
        (default, the PRODUCTION scoring path) no longer requires `y` at
        all -- biologically-impossible zeros are imputed with the medians
        frozen once on the baseline (`_prep_infer` / `zeros_to_missing_infer`),
        never conditioned on the label. `scaler_mode="refit_per_batch"`
        reproduces the repo's ORIGINAL per-batch refit (label-conditioned
        median imputation via `data_prep`) for training-fidelity comparison
        only, and explicitly refuses to run without `y` rather than
        silently accepting `y=None` and failing later.

        In both cases, one-hot columns unstable under drift are realigned
        via `_expected_columns()` (missing -> 0, extra -> dropped).
        """
        if scaler_mode == "persistent_baseline_fit":
            target_col = (y.name if y is not None and y.name else None) or self.baseline_calc.config[
                "baseline"
            ]["target_col"]
            self._ensure_persistent_scaler(target_col)
            X_processed = self._prep_infer(X, fit_scaler=False)
        elif scaler_mode == "refit_per_batch":
            if y is None:
                raise ValueError(
                    "scaler_mode='refit_per_batch' reproduces the ORIGINAL "
                    "training pipeline (label-conditioned median imputation "
                    "via data_prep) and therefore requires `y` -- it is not "
                    "a safe inference path. Use the default "
                    "scaler_mode='persistent_baseline_fit' to score a batch "
                    "without the label."
                )
            X_processed, _ = data_prep(X, y)
        else:
            raise ValueError(f"Unknown scaler_mode: {scaler_mode!r}")

        expected_cols = self._expected_columns()
        if expected_cols:
            missing = [c for c in expected_cols if c not in X_processed.columns]
            for c in missing:
                X_processed[c] = 0
            X_processed = X_processed[expected_cols]

        return X_processed

    # ------------------------------------------------------------------
    # 1. Feature Drift (Covariate Shift)
    # ------------------------------------------------------------------
    def feature_step(self, feature: str, delta: float, mask_ratio: float, step_id: int) -> None:
        rng = np.random.RandomState(42 + step_id)
        mask = rng.random_sample(len(self.current_df)) < mask_ratio
        self.current_df = self.current_df.copy()
        # Pre-existing bug (unrelated to config): on pandas>=2, writing a
        # float delta into an int64 column via .loc raises LossySetitemError
        # instead of silently upcasting. Upcast to float64 first so a
        # non-integer `delta` (e.g. std_multiplier * sigma from normal_drift)
        # never breaks the assignment.
        if pd.api.types.is_integer_dtype(self.current_df[feature]):
            self.current_df[feature] = self.current_df[feature].astype(float)
        self.current_df.loc[mask, feature] = self.current_df.loc[mask, feature] + delta

    def feature_variance_shift(self, feature: str, factor: float, step_id: int) -> None:
        rng = np.random.RandomState(42 + step_id)  # reserved for future stochastic extensions
        col = self.current_df[feature]
        mean = col.mean()
        self.current_df = self.current_df.copy()
        self.current_df[feature] = mean + (col - mean) * factor

    def run_gradual_drift(self, feature: str, total_delta: float, n_batches: int, step_id: int) -> None:
        rng = np.random.RandomState(42 + step_id)
        last_state = None
        for batch_state in generate_gradual_drift_stream(
            self.current_df, feature, total_delta, n_batches, rng
        ):
            last_state = batch_state
        if last_state is not None:
            self.current_df = last_state

    # ------------------------------------------------------------------
    # 2. Concept Drift P(Y|X)
    # ------------------------------------------------------------------
    def concept_drift_threshold(
        self,
        feature: str,
        rule_threshold_drifted: float,
        target_col: str,
        glucose_already_shifted: bool,
    ) -> None:
        self.current_df = self.current_df.copy()
        self.current_df[target_col] = (self.current_df[feature] >= rule_threshold_drifted).astype(int)
        if glucose_already_shifted:
            self.metadata["composite_drift"] = True
            self.metadata["flags"]["composite_drift_note"] = (
                f"Rule threshold applied on '{feature}' which was already shifted by the "
                f"feature drift step -> degradation due to the threshold vs. the shift cannot "
                f"be isolated without recomputing on an undrifted '{feature}'."
            )

    def concept_drift_label_noise(self, flip_ratio: float, target_col: str, step_id: int) -> None:
        rng = np.random.RandomState(42 + step_id)
        self.current_df = self.current_df.copy()
        n_flip = int(len(self.current_df) * flip_ratio)
        flip_idx = rng.choice(self.current_df.index, size=n_flip, replace=False)
        self.current_df.loc[flip_idx, target_col] = 1 - self.current_df.loc[flip_idx, target_col]

        realism_threshold = 0.20
        self.metadata["scenario_type"] = (
            "technical_stress_test" if flip_ratio > realism_threshold else "clinical_realistic"
        )

    # ------------------------------------------------------------------
    # 3. Target / Prior Shift P(Y)
    # ------------------------------------------------------------------
    def target_drift(
        self,
        new_prevalence: float,
        target_col: str,
        resampling_method: str,
        step_id: int,
    ) -> None:
        rng = np.random.RandomState(42 + step_id)
        df = self.current_df
        pos = df[df[target_col] == 1]
        neg = df[df[target_col] == 0]

        n_total = len(df)
        n_pos_target = int(round(n_total * new_prevalence))
        n_neg_target = n_total - n_pos_target

        if resampling_method == "oversample_duplicate":
            pos_idx = rng.choice(pos.index, size=n_pos_target, replace=True)
            neg_idx = rng.choice(neg.index, size=n_neg_target, replace=True)
            self.metadata["resampling_contamination_warning"] = True
        elif resampling_method == "undersample":
            n_pos_target = min(n_pos_target, len(pos))
            n_neg_target = min(n_neg_target, len(neg))
            pos_idx = rng.choice(pos.index, size=n_pos_target, replace=False)
            neg_idx = rng.choice(neg.index, size=n_neg_target, replace=False)
        else:  # "smote" -> not implemented here, requires the repo's real pipeline
            raise NotImplementedError(
                "resampling_method='smote' must be wired to the repo's real pipeline "
                "(e.g. imblearn.over_sampling.SMOTE) — not stubbed here."
            )

        self.current_df = pd.concat([df.loc[pos_idx], df.loc[neg_idx]], axis=0).reset_index(drop=True)
        self.metadata["flags"]["resampling_method"] = resampling_method

    # ------------------------------------------------------------------
    # 4. Data Quality / Anomalies
    # ------------------------------------------------------------------
    def inject_outliers(
        self,
        feature: str,
        pct: float,
        step_id: int,
        outlier_target: str = "pre_pipeline",
        value: float = 999.0,
    ) -> None:
        rng = np.random.RandomState(42 + step_id)
        self.current_df = self.current_df.copy()
        n_outliers = int(len(self.current_df) * pct)
        idx = rng.choice(self.current_df.index, size=n_outliers, replace=False)

        if outlier_target == "post_pipeline":
            stats = self.baseline_calc.get_stats(feature)
            calibrated_value = stats["q3"] + 1.4 * stats["iqr"]
            self.current_df.loc[idx, feature] = calibrated_value
        else:
            self.current_df.loc[idx, feature] = value

    def inject_missing_values(self, feature: str, pct: float, step_id: int) -> None:
        rng = np.random.RandomState(42 + step_id)
        self.current_df = self.current_df.copy()
        n_missing = int(len(self.current_df) * pct)
        idx = rng.choice(self.current_df.index, size=n_missing, replace=False)
        self.current_df.loc[idx, feature] = np.nan

    # ------------------------------------------------------------------
    # PSI: generic checkpoint + generic status classification
    # ------------------------------------------------------------------
    def compute_psi_checkpoint(self, df: pd.DataFrame, features: List[str]) -> Dict[str, float]:
        """
        Computes PSI per feature on `df`, always delegating to
        `BaselineCalculator.compute_psi` (fixed bin grid + baseline stats,
        never recomputed on drifted data).
        """
        return {
            f: self.baseline_calc.compute_psi(df[f], f)
            for f in features
            if f in df.columns
        }

    @staticmethod
    def classify_drift_status(
        psi_checkpoint: Dict[str, float],
        threshold_warning_low: float,
        threshold_critical_high: Optional[float] = None,
    ) -> DriftStatusResult:
        """
        Generic OK / WARNING / CRITICAL classification from a multi-feature
        PSI checkpoint, thresholds supplied by the subclass (only it knows
        its scenario's semantics).

        - PSI < threshold_warning_low                             -> OK
        - threshold_warning_low <= PSI < threshold_critical_high   -> WARNING
        - PSI >= threshold_critical_high (if provided)              -> CRITICAL

        The reference feature is the one with the highest PSI (worst case ->
        the most severe status observed on the checkpoint), consistent with
        `config/baseline_config.yml -> psi.critical_threshold`.
        """
        if not psi_checkpoint:
            return DriftStatusResult(
                status=DriftStatus.OK.value,
                psi_reference=0.0,
                reference_feature=None,
                threshold_warning_low=threshold_warning_low,
                threshold_critical_high=threshold_critical_high,
                message="No feature to evaluate in this PSI checkpoint.",
            )

        reference_feature, psi_reference = max(psi_checkpoint.items(), key=lambda kv: kv[1])

        if threshold_critical_high is not None and psi_reference >= threshold_critical_high:
            status = DriftStatus.CRITICAL
        elif psi_reference >= threshold_warning_low:
            status = DriftStatus.WARNING
        else:
            status = DriftStatus.OK

        message = (
            f"Highest PSI in the checkpoint = {psi_reference:.4f} (feature '{reference_feature}'), "
            f"WARNING threshold >= {threshold_warning_low}"
            + (f", CRITICAL threshold >= {threshold_critical_high}" if threshold_critical_high is not None else "")
            + f" -> status {status.value}."
        )

        return DriftStatusResult(
            status=status.value,
            psi_reference=float(psi_reference),
            reference_feature=reference_feature,
            threshold_warning_low=threshold_warning_low,
            threshold_critical_high=threshold_critical_high,
            message=message,
        )

    # ------------------------------------------------------------------
    # Generic JSON export
    # ------------------------------------------------------------------
    def export_report(self, report: Dict[str, Any], output_path: str) -> str:
        """
        Exports `report` as JSON to `output_path`, creating parent
        directories as needed. Purges any residual `np.generic` /
        `np.ndarray` via `_to_native` before serialization.
        """
        parent_dir = os.path.dirname(output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        native_report = _to_native(report)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(native_report, f, indent=2, ensure_ascii=False)

        return output_path

    # ------------------------------------------------------------------
    # Abstract entry point
    # ------------------------------------------------------------------
    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Chains the drift primitives specific to the subclass's scenario and
        returns the report (JSON-native dict). Every subclass MUST start
        with `self.reset(self.baseline_df)` (strict idempotence).
        """
        raise NotImplementedError
