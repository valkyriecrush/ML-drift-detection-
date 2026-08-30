# Purpose: fit ONCE (at materialize time) and persist the stateful preprocessing artifacts (scaler, outlier thresholds, imputation medians, binary encoders, expected column order) needed to turn a single new record into a model-ready row, without ever refitting at serving time.
#
# This is the piece a plain offline feature table doesn't give you: applying
# src.feature_engineering.data_prep as-is to a single online record would
# refit a RobustScaler on n=1 (meaningless), pick "binary" vs "one-hot"
# columns from a nunique() computed on n=1, and impute zeros with a
# per-Outcome-class median that requires knowing the label being predicted.
# All of that must instead be fit once on the training distribution and
# reused unchanged at serving time -> that's what this module persists.

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler

from src.feature_engineering import one_hot_encoder
from src.preprocessing import ZERO_AS_MISSING_COLS, outlier_thresholds


@dataclass
class FittedArtifacts:
    """Everything needed to turn one raw record into a model-ready row,
    fit once on the materialized offline data and never refit per-request."""

    num_cols: List[str]
    scaler: RobustScaler
    outlier_bounds: Dict[str, Tuple[float, float]]
    impute_medians: Dict[str, float]
    binary_encoders: Dict[str, LabelEncoder]
    model_columns: Optional[List[str]]

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FittedArtifacts":
        return joblib.load(path)


def fit_artifacts(
    df_raw: pd.DataFrame,
    df_offline_features: pd.DataFrame,
    num_cols: List[str],
    target_col: str,
    model_columns: Optional[List[str]],
) -> FittedArtifacts:
    """Fit every stateful artifact from the materialized data.

    Parameters
    ----------
    df_raw : pd.DataFrame
        The untouched source data (straight from ``data/diabetes.csv``),
        used to fit outlier thresholds and imputation medians on the same
        distribution `replace_with_thresholds` / `zeros_to_missing` would
        see in the original pipeline.
    df_offline_features : pd.DataFrame
        The feature-engineered offline table (clinical bands added,
        outliers capped, zeros imputed) — UNENCODED (categorical band
        columns still as category/object dtype). Binary/one-hot encoding
        is fit HERE, once, on this real training distribution, rather
        than trusting a `nunique() == 2` check performed on data that's
        already been encoded elsewhere (that would silently see only 0/1
        values and detect nothing).
    num_cols : List[str]
        The raw numeric columns to scale (`raw_measurements` group).
    target_col : str
        Name of the label column, excluded from every artifact.
    model_columns : list[str], optional
        Fixed output column order (from config/performance.yaml). If
        None, falls back to the column order produced by encoding
        `df_offline_features` here.
    """
    # --- outlier thresholds (IQR, fit on raw values, persisted) ---------
    outlier_bounds: Dict[str, Tuple[float, float]] = {}
    for col in num_cols:
        low, up = outlier_thresholds(df_raw, col)
        outlier_bounds[col] = (float(low), float(up))

    # --- imputation medians ---------------------------------------------
    # NOTE: the offline table (df_offline_features) was imputed with the
    # per-Outcome-class median, faithfully replicating data_prep for
    # training/batch use. That strategy needs the label, which is exactly
    # what's unknown at inference time. For online serving we fall back to
    # the overall median (computed after zero -> NaN, before any
    # class-conditional fill) -> a deliberate, documented simplification,
    # not an oversight.
    impute_medians: Dict[str, float] = {}
    df_for_median = df_raw.copy()
    for col in ZERO_AS_MISSING_COLS:
        if col in df_for_median.columns:
            df_for_median[col] = df_for_median[col].replace(0, pd.NA)
            impute_medians[col] = float(pd.to_numeric(df_for_median[col]).median())

    # --- scaler, fit once on the offline (capped, imputed, unscaled) data
    scaler = RobustScaler()
    scaler.fit(df_offline_features[num_cols])

    # --- encoding, fit once here on the UNENCODED categorical columns ---
    work = df_offline_features.drop(columns=[target_col]).copy()
    work[num_cols] = work[num_cols].astype(float)

    binary_cols = [
        col for col in work.columns
        if col not in num_cols
        and work[col].dtype not in ("int64", "float64")
        and work[col].nunique() == 2
    ]
    binary_encoders: Dict[str, LabelEncoder] = {}
    for col in binary_cols:
        le = LabelEncoder()
        le.fit(work[col].astype(str))
        binary_encoders[col] = le
        work[col] = le.transform(work[col].astype(str))

    ohe_cols = [col for col in work.columns if 12 >= work[col].nunique() > 2]
    work = one_hot_encoder(work, ohe_cols, drop_first=True)

    columns = list(model_columns) if model_columns else list(work.columns)

    return FittedArtifacts(
        num_cols=list(num_cols),
        scaler=scaler,
        outlier_bounds=outlier_bounds,
        impute_medians=impute_medians,
        binary_encoders=binary_encoders,
        model_columns=columns,
    )
