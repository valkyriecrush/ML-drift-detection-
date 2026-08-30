

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
DRIFT_SIMULATOR_DIR = ROOT / "drift_simulator"

if str(DRIFT_SIMULATOR_DIR) not in sys.path:
    sys.path.insert(0, str(DRIFT_SIMULATOR_DIR))

from baseline.baseline_calculator import BaselineCalculator  # noqa: E402
from drift.no_drift.no_drift_simulator import NoDriftSimulator  # noqa: E402
from drift.normal_drift.normal_drift_simulator import NormalDriftSimulator  # noqa: E402
from drift.severe_drift.severe_drift_simulator import SevereDriftSimulator  # noqa: E402


@dataclass
class RealScenarioBatch:
    name: str
    features_df: pd.DataFrame     # tracked_features columns, RAW data (pre-pipeline)
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    report: Dict[str, Any]        # simulator's native report (PSI, statuses, etc.)


class _cwd:
    """Context manager: baseline_config.yml's paths (config/, data/,
    models/, baseline/.cache) are relative to CWD -- exactly like running
    'python run_drift.py' from "drift_simulator/"."""

    def __enter__(self):
        self._prev = os.getcwd()
        os.chdir(DRIFT_SIMULATOR_DIR)
        return self

    def __exit__(self, *exc):
        os.chdir(self._prev)


def _load_config() -> Dict[str, Any]:
    with open(DRIFT_SIMULATOR_DIR / "config" / "baseline_config.yml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _predict(sim, X: pd.DataFrame, scaler_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    """Reuses the simulator's real pipeline AS-IS (sim._run_pipeline,
    sim.model) -- the same code NoDriftSimulator / NormalDriftSimulator /
    SevereDriftSimulator use internally to compute prediction_change_rate /
    f1_degradation, applied here to get PER-SAMPLE predictions.

    No longer takes `y` as a parameter (label-leakage fix): the label used
    to be passed all the way to missing-zero imputation (median conditioned
    on Outcome) -- an untenable logical circle for real scoring (you'd need
    the diagnosis to predict it). With `scaler_mode="persistent_baseline_fit"`
    (the config default), `sim._run_pipeline` no longer uses the label at
    all: this function demonstrates that by not even receiving it."""
    X_processed = sim._run_pipeline(X, scaler_mode=scaler_mode)
    y_pred = sim.model.predict(X_processed)
    y_proba = sim.model.predict_proba(X_processed)[:, 1]
    return np.asarray(y_pred), np.asarray(y_proba)


def build_baseline_calculator() -> Tuple[BaselineCalculator, Dict[str, Any]]:
    with _cwd():
        cfg = _load_config()
        bc = BaselineCalculator(config_path="config/baseline_config.yml")
        bc.load_or_compute()
    return bc, cfg


def get_baseline_predictions(bc: BaselineCalculator, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Real model predictions on the baseline itself (needed for
    bootstrap_agents: y_true_baseline / y_pred_baseline / y_proba_baseline)."""
    with _cwd():
        target_col = cfg["baseline"]["target_col"]
        scaler_mode = cfg.get("preprocessing", {}).get("scaler_mode", "persistent_baseline_fit")

        sim = NoDriftSimulator(baseline_calc=bc)  # throwaway instance, just for its pipeline+model
        sim._load_model(cfg["baseline"]["model_path"])

        X_baseline = bc.baseline_df.drop(columns=[target_col])
        y_baseline = bc.baseline_df[target_col]
        # y_baseline is only RETURNED (to evaluate the prediction after the
        # fact) -- no longer passed to _predict/_run_pipeline, which don't
        # need it (see _predict()).
        y_pred, y_proba = _predict(sim, X_baseline, scaler_mode)
        return y_baseline.to_numpy(), y_pred, y_proba


def run_real_scenario(name: str, bc: BaselineCalculator, cfg: Dict[str, Any], severity: str = "severe") -> RealScenarioBatch:
    """name in {'no_drift', 'normal_drift', 'severe_drift'}."""
    with _cwd():
        target_col = cfg["baseline"]["target_col"]
        tracked = cfg["baseline"]["tracked_features"]
        scaler_mode = cfg.get("preprocessing", {}).get("scaler_mode", "persistent_baseline_fit")
        model_path = cfg["baseline"]["model_path"]

        if name == "no_drift":
            sim = NoDriftSimulator(baseline_calc=bc)
            report = sim.simulate_no_drift_scenario(config=cfg, model_path=model_path, export=False)
        elif name == "normal_drift":
            sim = NormalDriftSimulator(baseline_calc=bc)
            report = sim.run(config=cfg, model_path=model_path, export=False)
        elif name == "severe_drift":
            sim = SevereDriftSimulator(baseline_calc=bc)
            report = sim.simulate_severe_scenario(severity=severity, config=cfg, model_path=model_path, export=False)
        else:
            raise ValueError(f"Unknown scenario: {name}")

        X_drifted = sim.current_df.drop(columns=[target_col])
        y_drifted = sim.current_df[target_col]

        # y_drifted is kept to EVALUATE the prediction afterwards (the
        # batch's y_true, see RealScenarioBatch) -- a separate step, after
        # scoring. It's no longer passed to _predict()/_run_pipeline(),
        # which now score X_drifted without ever seeing the label, as a
        # real production flow would.
        y_pred, y_proba = _predict(sim, X_drifted, scaler_mode)
        features_df = X_drifted[tracked].reset_index(drop=True)

        return RealScenarioBatch(
            name=name, features_df=features_df, y_true=y_drifted.to_numpy(),
            y_pred=y_pred, y_proba=y_proba, report=report,
        )
