"""
Regression tests for the label-leakage fix in the inference pipeline.
`base_drift_simulator._run_pipeline` and
`bridge/real_scenario_runner._predict` used to require `y` (the true label)
to impute biologically impossible zeros -- an untenable logical circle for
real scoring (the label is unknown by definition).

These tests check that scoring now works WITHOUT the label, and that
`src.preprocessing` explicitly refuses the old unsafe usage.
"""

from __future__ import annotations

import pandas as pd
import pytest


def test_zeros_to_missing_infer_never_needs_the_label():
    from src.preprocessing import compute_global_medians, zeros_to_missing_infer

    df = pd.DataFrame({"Glucose": [0, 120, 0, 140], "BMI": [0, 25.0, 30.0, 0]})
    medians = compute_global_medians(df, columns=["Glucose", "BMI"])

    # No Outcome/label column exists in `df` or `medians` --
    # zeros_to_missing_infer() must neither require nor access it.
    assert "Outcome" not in df.columns
    out = zeros_to_missing_infer(df, medians, columns=["Glucose", "BMI"])

    assert (out["Glucose"] != 0).all()
    assert (out["BMI"] != 0).all()


def test_zeros_to_missing_infer_missing_frozen_stats_raises():
    from src.preprocessing import zeros_to_missing_infer

    df = pd.DataFrame({"Glucose": [0, 120]})
    with pytest.raises(KeyError):
        zeros_to_missing_infer(df, medians={}, columns=["Glucose"])


def test_zeros_to_missing_train_requires_label_column():
    """The original function stays usable at training time (label known),
    but now EXPLICITLY refuses to run without the label column -- instead
    of silently producing NaNs, as it did before the fix."""
    from src.preprocessing import zeros_to_missing

    df_without_label = pd.DataFrame({"Glucose": [0, 120, 130]})
    with pytest.raises(KeyError):
        zeros_to_missing(df_without_label)


def test_zeros_to_missing_train_still_works_with_label():
    from src.preprocessing import zeros_to_missing

    df = pd.DataFrame({"Glucose": [0, 120, 0, 140], "Outcome": [0, 0, 1, 1]})
    out = zeros_to_missing(df, columns=["Glucose"])
    assert (out["Glucose"] != 0).all()


@pytest.mark.usefixtures("chdir_drift_simulator")
def test_run_pipeline_scores_without_the_label(chdir_drift_simulator):
    """The key regression test: instantiate a real simulator and score a
    batch WITHOUT ever providing y to _run_pipeline -- exactly the
    production scenario (label unknown at scoring time)."""
    yaml = pytest.importorskip("yaml")
    from baseline.baseline_calculator import BaselineCalculator
    from drift.no_drift.no_drift_simulator import NoDriftSimulator

    with open("config/baseline_config.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    bc = BaselineCalculator(config_path="config/baseline_config.yml")
    bc.load_or_compute()

    sim = NoDriftSimulator(baseline_calc=bc)
    sim._load_model(cfg["baseline"]["model_path"])

    target_col = cfg["baseline"]["target_col"]
    X = bc.baseline_df.drop(columns=[target_col]).head(20)

    # No `y` at all: proof that production scoring no longer needs the
    # label to work.
    X_processed = sim._run_pipeline(X, scaler_mode="persistent_baseline_fit")
    preds = sim.model.predict(X_processed)

    assert len(preds) == len(X)


@pytest.mark.usefixtures("chdir_drift_simulator")
def test_refit_per_batch_mode_refuses_to_run_without_label(chdir_drift_simulator):
    """`scaler_mode="refit_per_batch"` reproduces the TRAINING pipeline
    (label-conditioned imputation): it must stay reserved for offline replay
    and explicitly refuse to run without y, rather than silently accepting
    y=None that would break further down the line."""
    yaml = pytest.importorskip("yaml")
    from baseline.baseline_calculator import BaselineCalculator
    from drift.no_drift.no_drift_simulator import NoDriftSimulator

    with open("config/baseline_config.yml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    bc = BaselineCalculator(config_path="config/baseline_config.yml")
    bc.load_or_compute()

    sim = NoDriftSimulator(baseline_calc=bc)
    target_col = cfg["baseline"]["target_col"]
    X = bc.baseline_df.drop(columns=[target_col]).head(5)

    with pytest.raises(ValueError):
        sim._run_pipeline(X, scaler_mode="refit_per_batch")
