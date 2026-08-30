# Purpose: validates incoming data against the reference schema/config files and exposes the model's expected column order.

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class SchemaValidator:
    schema_path: str = "config/diabetes_schema.yaml"
    performance_path: str = "config/performance.yaml"
    cleaned_reference_path: Optional[str] = "config/diabetes_cleaned.yaml"
    features_reference_path: Optional[str] = "config/diabetes_features.yaml"

    schema: Dict[str, Any] = field(default_factory=dict, init=False)
    performance: Dict[str, Any] = field(default_factory=dict, init=False)
    cleaned_reference: Dict[str, Any] = field(default_factory=dict, init=False)
    features_reference: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.schema = _load_yaml(self.schema_path)
        self.performance = _load_yaml(self.performance_path)
        self.cleaned_reference = (
            _load_yaml(self.cleaned_reference_path)
            if self.cleaned_reference_path and os.path.exists(self.cleaned_reference_path)
            else {}
        )
        self.features_reference = (
            _load_yaml(self.features_reference_path)
            if self.features_reference_path and os.path.exists(self.features_reference_path)
            else {}
        )

    # ------------------------------------------------------------------
    # 1. Validation of raw numeric columns
    # ------------------------------------------------------------------
    def _numeric_feature_specs(self) -> List[Dict[str, Any]]:
        return self.schema.get("schema", {}).get("features", {}).get("numeric", [])

    def validate_raw(self, df: pd.DataFrame) -> List[str]:
        """
        Compares `df` against the bounds documented in diabetes_schema.yaml.

        Returns a list of violations (readable strings), never an
        exception: in a drift stress-test, going outside baseline bounds
        is the EXPECTED behavior, not an application error.
        """
        violations: List[str] = []
        for spec in self._numeric_feature_specs():
            name = spec["name"]
            if name not in df.columns:
                violations.append(f"Expected column missing from the DataFrame: '{name}'")
                continue

            series = df[name].dropna()
            if series.empty:
                continue

            lo, hi = spec.get("min"), spec.get("max")
            if lo is not None and series.min() < lo:
                violations.append(
                    f"{name}: observed min {series.min():.3f} < schema min {lo} "
                    f"(drift expected in 'severe'/'extreme' scenario)"
                )
            if hi is not None and series.max() > hi:
                violations.append(
                    f"{name}: observed max {series.max():.3f} > schema max {hi} "
                    f"(drift expected in 'severe'/'extreme' scenario)"
                )

            if spec.get("zero_is_missing") and (df[name] == 0).any():
                n_zero = int((df[name] == 0).sum())
                violations.append(
                    f"{name}: {n_zero} value(s) at 0 while zero_is_missing=true "
                    f"in the schema (should be imputed upstream by data_prep)"
                )

        return violations

    def baseline_prevalence(self) -> Optional[float]:
        """Documented reference prevalence (schema + cleaned_reference must agree)."""
        return self.schema.get("schema", {}).get("target", {}).get("baseline_prevalence")

    # ------------------------------------------------------------------
    # 2. Column order expected by the model (post data_prep)
    # ------------------------------------------------------------------
    def expected_model_columns(self) -> Optional[List[str]]:
        """
        Exact order of columns expected out of `data_prep`, as documented
        by `booster_info.feature_names` from the LightGBM block in
        performance.yaml. Returns None if the information is not present
        (the caller must then fall back to `model.feature_name_`).
        """
        estimators = self.performance.get("voting_classifier", {}).get("estimators", [])
        for est in estimators:
            booster_info = est.get("booster_info")
            if booster_info and "feature_names" in booster_info:
                return list(booster_info["feature_names"])
        return None

    def expected_n_features(self) -> Optional[int]:
        estimators = self.performance.get("voting_classifier", {}).get("estimators", [])
        for est in estimators:
            booster_info = est.get("booster_info")
            if booster_info and "max_feature_idx" in booster_info:
                return int(booster_info["max_feature_idx"]) + 1
        return None
