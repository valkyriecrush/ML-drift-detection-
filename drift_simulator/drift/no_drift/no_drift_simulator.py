# Purpose: "no drift" scenario — runs the pipeline on unchanged baseline data to check for false positives (expected status: OK).

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from baseline.baseline_calculator import _to_native
from drift.base_drift_simulator import BaseDriftSimulator, DriftStatus

logger = logging.getLogger(__name__)

# NOTE: no hardcoded _DEFAULT_TRACKED_FEATURES / _DEFAULT_DRIFT_STATUS_THRESHOLDS
# / _DEFAULT_OUTPUT_PATH here on purpose. `config/baseline_config.yml` is the
# single source of truth (baseline.tracked_features, no_drift.drift_status_thresholds,
# no_drift.output.report_path) — falling back to metrics/features baked into this
# module would let the simulator silently diverge from the real config/database.


@dataclass
class NoDriftReport:
    """Typed report of the "No Drift" scenario -> exported JSON shape."""

    severity: str
    scenario_type: Optional[str]
    baseline_reference: Dict[str, str]
    schema_warnings: List[str]
    psi_checkpoints: Dict[str, Dict[str, float]]
    overall_drift_status: Dict[str, Any]
    target_status_check: Dict[str, Any]
    prediction_change_rate: float
    f1_degradation: float
    flags: Dict[str, Any]
    generated_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.severity, str):
            raise TypeError(f"severity must be a str, got {type(self.severity)}")
        if not isinstance(self.prediction_change_rate, (int, float)):
            raise TypeError(
                f"prediction_change_rate must be numeric, got {type(self.prediction_change_rate)}"
            )
        if not isinstance(self.f1_degradation, (int, float)):
            raise TypeError(f"f1_degradation must be numeric, got {type(self.f1_degradation)}")

    def to_dict(self) -> Dict[str, Any]:
        return _to_native(asdict(self))


class NoDriftSimulator(BaseDriftSimulator):
    """
    Negative control / pure baseline: no drift primitive is executed.
    Serves as a non-regression check for the MLOps pipeline and the
    LightGBM model — a `NoDriftSimulator` that doesn't return `OK` reveals
    a false drift positive (a pipeline bug, not a data issue).
    """

    # ------------------------------------------------------------------
    # Abstract entry point (BaseDriftSimulator.run): simple delegation to
    # `simulate_no_drift_scenario`.
    # ------------------------------------------------------------------
    def run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.simulate_no_drift_scenario(*args, **kwargs)

    def simulate_no_drift_scenario(
        self,
        config: Dict[str, Any],
        apply_pre_processing: bool = True,
        model_path: str = "models/lgbm_model.pkl",
        export: bool = True,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        # --- Mandatory line 1: idempotence — current_df starts as a
        # strictly intact copy of baseline_df, no feature altered, no label
        # noise, no target resampling, no anomaly.
        self.reset(self.baseline_df)

        # `config` is now REQUIRED and must come straight from
        # config/baseline_config.yml (yaml.safe_load) — no silent fallback to
        # hardcoded metrics/features baked into this module. Same contract as
        # SevereDriftSimulator.simulate_severe_scenario.
        if not config:
            raise ValueError(
                "NoDriftSimulator.run()/simulate_no_drift_scenario() requires "
                "`config` loaded from config/baseline_config.yml "
                "(e.g. yaml.safe_load(open('config/baseline_config.yml'))) — "
                "no default metrics/features are used."
            )
        try:
            target_col = config["baseline"]["target_col"]
            tracked_features = config["baseline"]["tracked_features"]
            thresholds = config["no_drift"]["drift_status_thresholds"]
        except KeyError as exc:
            raise KeyError(
                f"Missing key {exc} in config — check baseline.target_col, "
                "baseline.tracked_features and no_drift.drift_status_thresholds "
                "in config/baseline_config.yml."
            ) from exc
        cfg = config

        # --- Baseline audit (incl. non-blocking schema warnings) ---
        self.metadata["baseline_reference"] = self.baseline_calc.get_baseline_reference()
        self.metadata["schema_warnings"] = list(self.baseline_calc.schema_warnings)

        # current_df == baseline_df (intact copy set by reset() above): no
        # drift primitive is called here, by definition of this scenario.
        X_current = self.current_df.drop(columns=[target_col])
        y_current = self.current_df[target_col]

        scaler_mode = cfg.get("preprocessing", {}).get("scaler_mode", "persistent_baseline_fit")
        prediction_change_rate = 0.0
        f1_degradation = 0.0
        psi_final: Dict[str, float] = {}

        if apply_pre_processing:
            n_before = X_current.shape[0]
            try:
                if self.model is None:
                    self._load_model(model_path)
            except Exception:
                pass  # model unavailable -> model metrics skipped, PSI still available

            # current_df goes through _run_pipeline with the persistent
            # "persistent_baseline_fit" scaler, exactly like the baseline
            # that fit it -> expected PSI ~ 0.0.
            X_processed = self._run_pipeline(X_current, y_current, scaler_mode=scaler_mode)
            assert X_processed.shape[0] == n_before, (
                f"Pipeline error: preprocessing altered the row count! "
                f"Original: {n_before}, Processed: {X_processed.shape[0]}"
            )

            # PSI must be computed in the ORIGINAL feature space: bin_edges
            # were built from raw baseline_df (see BaselineCalculator).
            # X_processed has already been through the persistent RobustScaler
            # (+ encoding), so its columns are on a completely different
            # scale than bin_edges -> comparing them there produced a false
            # drift positive on every feature, in every scenario. Use
            # X_current (raw, pre-pipeline) instead, matching how
            # psi_post_feature_drift / psi_post_target_drift are computed in
            # the other simulators.
            psi_final = self.compute_psi_checkpoint(X_current, tracked_features)
            self.metadata["psi_checkpoints"]["psi_final"] = psi_final

            try:
                if self.model is None:
                    raise FileNotFoundError(model_path)

                baseline_X_processed = self._run_pipeline(
                    self.baseline_df.drop(columns=[target_col]),
                    self.baseline_df[target_col],
                    scaler_mode=scaler_mode,
                )
                baseline_preds = self.model.predict(baseline_X_processed)
                current_preds = self.model.predict(X_processed)

                n_common = min(len(baseline_preds), len(current_preds))
                prediction_change_rate = float(
                    np.mean(baseline_preds[:n_common] != current_preds[:n_common])
                )

                from sklearn.metrics import f1_score

                f1_baseline = f1_score(self.baseline_df[target_col], baseline_preds)
                f1_current = f1_score(y_current, current_preds)
                f1_degradation = float(f1_baseline - f1_current)
                self.metadata["flags"]["f1_model_degradation"] = f1_degradation
            except FileNotFoundError:
                self.metadata["flags"]["model_load_error"] = (
                    f"Model not found at '{model_path}' — wire up the real repo path."
                )

        # ================= Drift status (OK expected) =================
        overall_status = self.classify_drift_status(
            psi_final, thresholds["warning_low"], thresholds["critical_high"]
        )

        if overall_status.status != DriftStatus.OK.value:
            logger.warning(
                "NoDriftSimulator: false drift positive detected on unaltered "
                "data (status %s, feature '%s') — %s",
                overall_status.status,
                overall_status.reference_feature,
                overall_status.message,
            )

        matches_expectation = (
            overall_status.status == DriftStatus.OK.value
            and prediction_change_rate == 0.0
            and f1_degradation == 0.0
        )

        target_status_check = {
            "expected_status": DriftStatus.OK.value,
            "actual_status": overall_status.status,
            "matches_expectation": matches_expectation,
        }

        # ================= Report =================
        report = NoDriftReport(
            severity="none",
            scenario_type=self.metadata["scenario_type"],
            baseline_reference=self.metadata["baseline_reference"],
            schema_warnings=self.metadata["schema_warnings"],
            psi_checkpoints=dict(self.metadata["psi_checkpoints"]),
            overall_drift_status=overall_status.to_dict(),
            target_status_check=target_status_check,
            prediction_change_rate=prediction_change_rate,
            f1_degradation=f1_degradation,
            flags=self.metadata["flags"],
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

        report_dict = report.to_dict()

        if export:
            final_output_path = output_path or config["no_drift"]["output"]["report_path"]
            self.export_report(report_dict, final_output_path)
            report_dict["_exported_to"] = final_output_path

        return report_dict
