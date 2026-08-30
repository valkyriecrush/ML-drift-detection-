"""
feature_importance.py -- Computes a REAL FeatureIntelligenceContext (SHAP +
correlation) 
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict

import numpy as np
import shap

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "agentic_drift_stress", ROOT / "bridge"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from feature_intelligence import FeatureIntelligenceContext, compute_correlation_matrix  # noqa: E402
from real_scenario_runner import _cwd, DRIFT_SIMULATOR_DIR  # noqa: E402


def _raw_feature_for_column(col: str, tracked: list) -> str:
    if col in tracked:
        return col
    for feat in tracked:
        if col.startswith(feat + "_CAT"):
            return feat
    return ""  # unmapped column (e.g. Life_Level_CAT, handled separately)


def compute_feature_intelligence_context(bc, cfg) -> FeatureIntelligenceContext:
    tracked = cfg["baseline"]["tracked_features"]
    target_col = cfg["baseline"]["target_col"]

    with _cwd():
        from drift.no_drift.no_drift_simulator import NoDriftSimulator

        sim = NoDriftSimulator(baseline_calc=bc)
        sim._load_model(cfg["baseline"]["model_path"])

        X_baseline = bc.baseline_df.drop(columns=[target_col])
        y_baseline = bc.baseline_df[target_col]
        scaler_mode = cfg.get("preprocessing", {}).get("scaler_mode", "persistent_baseline_fit")
        X_processed = sim._run_pipeline(X_baseline, y_baseline, scaler_mode=scaler_mode)

        explainer = shap.TreeExplainer(sim.model)
        raw_shap = explainer.shap_values(X_processed)

    # LightGBM sklearn binary classifier: shap_values can be a 2D array
    # (positive class) or a [neg, pos] list depending on the shap version --
    # always take the positive class (Outcome=1).
    if isinstance(raw_shap, list):
        shap_values = raw_shap[1]
    elif isinstance(raw_shap, np.ndarray) and raw_shap.ndim == 3:
        shap_values = raw_shap[:, :, 1]
    else:
        shap_values = raw_shap

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    per_column = dict(zip(X_processed.columns, mean_abs_shap.tolist()))

    raw_importance: Dict[str, float] = defaultdict(float)
    life_level_shap = 0.0
    for col, val in per_column.items():
        if col == "Life_Level_CAT":
            life_level_shap = val
            continue
        raw_feat = _raw_feature_for_column(col, tracked)
        if raw_feat:
            raw_importance[raw_feat] += val

    # Life_Level_CAT = composite flag of Age, BloodPressure, BMI -> split equally
    for feat in ("Age", "BloodPressure", "BMI"):
        if feat in tracked:
            raw_importance[feat] += life_level_shap / 3.0

    shap_importances = {f: float(raw_importance.get(f, 0.0)) for f in tracked}

    correlation_matrix = compute_correlation_matrix(bc.baseline_df[tracked], method="spearman")

    return FeatureIntelligenceContext(
        shap_importances=shap_importances,
        correlation_matrix=correlation_matrix,
        precomputed_vif=None,  # let FeatureRiskAssessor approximate it via the correlation matrix
    )
