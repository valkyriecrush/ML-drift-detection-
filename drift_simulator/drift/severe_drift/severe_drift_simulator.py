# Purpose: "severe drift" scenario — combines multiple drift types to trigger a CRITICAL status and check it against an SLA.

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from baseline.baseline_calculator import _to_native
from drift.base_drift_simulator import BaseDriftSimulator, HIDDEN_TRUE_LABEL_COL


# =============================================================================
# Typed report (dataclasses + light validation)
# =============================================================================
@dataclass
class PSICheckpoints:
    psi_post_feature_drift: Dict[str, float] = field(default_factory=dict)
    psi_post_target_drift: Dict[str, float] = field(default_factory=dict)
    psi_post_data_quality: Dict[str, float] = field(default_factory=dict)
    psi_final: Dict[str, float] = field(default_factory=dict)


@dataclass
class SLACheckResult:
    meets_expected_severity: bool
    criteria_failed: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not isinstance(self.meets_expected_severity, bool):
            raise TypeError(
                f"SLACheckResult.meets_expected_severity must be a bool, "
                f"got {type(self.meets_expected_severity)}"
            )
        if not all(isinstance(c, str) for c in self.criteria_failed):
            raise TypeError("SLACheckResult.criteria_failed must be a list of str")


@dataclass
class StressTestReport:
    """Typed structure of `simulate_severe_scenario`'s report."""

    severity: str
    scenario_type: Optional[str]
    composite_drift: bool
    resampling_contamination_warning: bool
    baseline_reference: Dict[str, str]
    schema_warnings: List[str]
    psi_checkpoints: PSICheckpoints
    prediction_change_rate: float
    f1_degradation: float
    flags: Dict[str, Any]
    sla_check: SLACheckResult
    generated_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.severity, str):
            raise TypeError(f"severity must be a str, got {type(self.severity)}")
        if self.scenario_type is not None and not isinstance(self.scenario_type, str):
            raise TypeError(f"scenario_type must be a str or None, got {type(self.scenario_type)}")
        if not isinstance(self.composite_drift, bool):
            raise TypeError(f"composite_drift must be a bool, got {type(self.composite_drift)}")
        if not isinstance(self.resampling_contamination_warning, bool):
            raise TypeError(
                f"resampling_contamination_warning must be a bool, "
                f"got {type(self.resampling_contamination_warning)}"
            )
        if not isinstance(self.prediction_change_rate, (int, float)):
            raise TypeError(
                f"prediction_change_rate must be numeric, got {type(self.prediction_change_rate)}"
            )
        if not isinstance(self.f1_degradation, (int, float)):
            raise TypeError(f"f1_degradation must be numeric, got {type(self.f1_degradation)}")
        if not isinstance(self.psi_checkpoints, PSICheckpoints):
            raise TypeError("psi_checkpoints must be a PSICheckpoints instance")
        if not isinstance(self.sla_check, SLACheckResult):
            raise TypeError("sla_check must be a SLACheckResult instance")

    def to_dict(self) -> Dict[str, Any]:
        return _to_native(asdict(self))


class SevereDriftSimulator(BaseDriftSimulator):
    """
    "SEVERE_DRIFT" scenario: extreme/composite drift, expected to trigger a
    `CRITICAL` status. Inherits `BaseDriftSimulator` for all shared
    mechanics (idempotence, preprocessing pipeline, drift primitives, PSI
    computation, JSON export).
    """

    # ------------------------------------------------------------------
    # Abstract entry point (BaseDriftSimulator.run): simple delegation to
    # `simulate_severe_scenario`, which remains the public entry point.
    # ------------------------------------------------------------------
    def run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return self.simulate_severe_scenario(*args, **kwargs)

    # ------------------------------------------------------------------
    # SLA
    # ------------------------------------------------------------------
    def _sla_check(
        self,
        psi_post_feature_drift: Dict[str, float],
        prediction_change_rate: float,
        f1_degradation: float,
        thresholds: Dict[str, float],
    ) -> Dict[str, Any]:
        criteria_failed: List[str] = []

        glucose_psi = psi_post_feature_drift.get("Glucose", 0.0)
        if glucose_psi <= thresholds["psi_post_feature_drift_glucose_min"]:
            criteria_failed.append(
                f"PSI Glucose post-feature-drift = {glucose_psi:.4f} "
                f"<= threshold {thresholds['psi_post_feature_drift_glucose_min']}"
            )
        if prediction_change_rate < thresholds["prediction_change_rate_min"]:
            criteria_failed.append(
                f"Prediction Change Rate = {prediction_change_rate:.4f} "
                f"< threshold {thresholds['prediction_change_rate_min']}"
            )
        if f1_degradation <= thresholds["f1_degradation_min"]:
            criteria_failed.append(
                f"F1 degradation = {f1_degradation:.4f} "
                f"<= threshold {thresholds['f1_degradation_min']}"
            )

        return {
            "meets_expected_severity": len(criteria_failed) == 0,
            "criteria_failed": criteria_failed,
        }

    # ------------------------------------------------------------------
    # Main orchestrator
    # ------------------------------------------------------------------
    def simulate_severe_scenario(
        self,
        severity: str = "severe",
        config: Optional[Dict[str, Any]] = None,
        apply_pre_processing: bool = True,
        model_path: str = "models/lgbm_model.pkl",
        export: bool = True,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        # --- Mandatory line 1: idempotence ---
        self.reset(self.baseline_df)

        cfg = config or {}
        preset = cfg["severity_presets"][severity]
        target_col = cfg["baseline"]["target_col"]
        tracked_features = cfg["baseline"]["tracked_features"]
        sla_thresholds = cfg["sla_check"]
        step_ids = cfg["random_state"]["step_ids"]

        # --- Baseline audit (incl. non-blocking schema warnings) ---
        self.metadata["baseline_reference"] = self.baseline_calc.get_baseline_reference()
        self.metadata["schema_warnings"] = list(self.baseline_calc.schema_warnings)

        # ================= 1. Feature Drift =================
        glu = preset["feature_step"]["Glucose"]
        self.feature_step("Glucose", glu["delta"], glu["mask_ratio"], step_ids["feature_step_glucose"])

        bmi = preset["feature_step"]["BMI"]
        self.feature_step("BMI", bmi["delta"], bmi["mask_ratio"], step_ids["feature_step_bmi"])

        bp = preset["feature_variance_shift"]["BloodPressure"]
        self.feature_variance_shift("BloodPressure", bp["factor"], step_ids["feature_variance_shift_bp"])

        ins = preset["gradual_drift"]["Insulin"]
        self.run_gradual_drift(
            "Insulin", ins["total_delta"], ins["n_batches"], step_ids["gradual_drift_insulin"]
        )

        psi_post_feature_drift = self.compute_psi_checkpoint(self.current_df, tracked_features)
        self.metadata["psi_checkpoints"]["psi_post_feature_drift"] = psi_post_feature_drift

        # ================= 2. Concept Drift =================
        # The "true y" (before the drifted rule threshold and label noise) is
        # carried as a HIDDEN column of current_df, not a Series captured
        # separately. That way, when target_drift (step 3) resamples +
        # resets current_df's index, this column is resampled/reindexed the
        # exact same way as (X, y_drifted) -> no misalignment possible
        # between y_true_uncorrupted and the predictions computed below.
        self.current_df = self.current_df.copy()
        self.current_df[HIDDEN_TRUE_LABEL_COL] = self.current_df[target_col].copy()

        cd = preset["concept_drift"]
        self.concept_drift_threshold(
            feature="Glucose",
            rule_threshold_drifted=cd["rule_threshold_drifted"],
            target_col=target_col,
            glucose_already_shifted=True,  # Glucose was already shifted in step 1
        )
        self.concept_drift_label_noise(
            flip_ratio=cd["label_noise_flip_ratio"],
            target_col=target_col,
            step_id=step_ids["concept_drift_label_noise"],
        )

        # ================= 3. Target / Prior Shift =================
        td = preset["target_drift"]
        self.target_drift(
            new_prevalence=td["new_prevalence"],
            target_col=target_col,
            resampling_method=td["resampling_method"],
            step_id=step_ids["target_drift_resampling"],
        )
        psi_post_target_drift = self.compute_psi_checkpoint(self.current_df, tracked_features)
        self.metadata["psi_checkpoints"]["psi_post_target_drift"] = psi_post_target_drift

        # ================= 4. Data Quality / Anomalies =================
        dq = preset["data_quality"]
        self.inject_outliers(
            feature=dq["outliers"]["feature"],
            pct=dq["outliers"]["pct"],
            step_id=step_ids["inject_outliers"],
            outlier_target="pre_pipeline",
            value=dq["outliers"]["value"],
        )
        self.inject_missing_values(
            feature=dq["missing"]["feature"],
            pct=dq["missing"]["pct"],
            step_id=step_ids["inject_missing_values"],
        )
        psi_post_data_quality = self.compute_psi_checkpoint(self.current_df, tracked_features)
        self.metadata["psi_checkpoints"]["psi_post_data_quality"] = psi_post_data_quality

        # ================= 5. Pipeline + guardrail =================
        # Synchronized extraction of the "true y" AFTER all resampling/
        # reindexing steps: at this point, y_true_uncorrupted has exactly
        # the same index and row order as X_drifted/y_drifted, including
        # duplicates introduced by oversample_duplicate.
        self.y_true_uncorrupted = self.current_df[HIDDEN_TRUE_LABEL_COL].copy()
        X_drifted = self.current_df.drop(columns=[target_col, HIDDEN_TRUE_LABEL_COL])
        y_drifted = self.current_df[target_col]

        scaler_mode = cfg.get("preprocessing", {}).get("scaler_mode", "persistent_baseline_fit")

        prediction_change_rate = 0.0
        f1_degradation = 0.0

        if apply_pre_processing:
            n_before = X_drifted.shape[0]

            # The model must be loaded BEFORE the first call to
            # _run_pipeline to allow falling back to model.feature_name_ if
            # config/performance.yaml is absent (see _expected_columns).
            try:
                if self.model is None:
                    self._load_model(model_path)
            except Exception:
                pass  # model unavailable -> column alignment disabled below

            X_processed = self._run_pipeline(X_drifted, y_drifted, scaler_mode=scaler_mode)
            assert X_processed.shape[0] == n_before, (
                f"Pipeline error: preprocessing altered the row count! "
                f"Original: {n_before}, Processed: {X_processed.shape[0]}"
            )

            # PSI must be computed in the ORIGINAL feature space: bin_edges
            # were built from raw baseline_df (see BaselineCalculator).
            # X_processed has already been through the persistent RobustScaler
            # (+ encoding), so its columns are on a completely different
            # scale than bin_edges -> comparing them there produced a false
            # drift positive on every feature, masking the real (correctly
            # escalating) drift already captured by psi_post_feature_drift /
            # psi_post_target_drift / psi_post_data_quality above. Use
            # X_drifted (raw, pre-pipeline) instead.
            psi_final = self.compute_psi_checkpoint(X_drifted, tracked_features)
            self.metadata["psi_checkpoints"]["psi_final"] = psi_final

            # ---- Baseline vs. drifted predictions (if model available) ----
            try:
                baseline_X_processed = self._run_pipeline(
                    self.baseline_df.drop(columns=[target_col]),
                    self.baseline_df[target_col],
                    scaler_mode=scaler_mode,
                )
                baseline_preds = self.model.predict(baseline_X_processed)
                drifted_preds = self.model.predict(X_processed)

                # baseline_preds and drifted_preds cover two populations of
                # different sizes (target_drift resampled): the positional
                # comparison truncated to n_common remains necessarily
                # approximate for prediction_change_rate, unlike the F1
                # computation below which is exact.
                n_common = min(len(baseline_preds), len(drifted_preds))
                prediction_change_rate = float(
                    np.mean(baseline_preds[:n_common] != drifted_preds[:n_common])
                )

                from sklearn.metrics import f1_score
                f1_baseline = f1_score(self.baseline_df[target_col], baseline_preds)

                # y_true_uncorrupted and y_drifted now have EXACTLY the same
                # index/order/length as X_processed (carried as current_df
                # columns through the whole resampling) -> no positional
                # truncation or iloc needed to "align" lengths that were
                # previously not guaranteed to match.
                assert len(self.y_true_uncorrupted) == len(drifted_preds) == len(y_drifted), (
                    "Unexpected misalignment between y_true_uncorrupted, y_drifted and "
                    "drifted_preds — resampling was not propagated correctly."
                )
                f1_true_degradation_source = f1_score(self.y_true_uncorrupted, drifted_preds)
                f1_corrupted_test_source = f1_score(y_drifted, drifted_preds)

                f1_degradation = float(f1_baseline - f1_true_degradation_source)
                self.metadata["flags"]["f1_model_degradation"] = f1_degradation
                self.metadata["flags"]["f1_test_set_corruption_delta"] = float(
                    f1_true_degradation_source - f1_corrupted_test_source
                )
            except FileNotFoundError:
                self.metadata["flags"]["model_load_error"] = (
                    f"Model not found at '{model_path}' — wire up the real repo path."
                )

        # ================= SLA =================
        sla_result = self._sla_check(
            psi_post_feature_drift=psi_post_feature_drift,
            prediction_change_rate=prediction_change_rate,
            f1_degradation=f1_degradation,
            thresholds=sla_thresholds,
        )

        report = StressTestReport(
            severity=severity,
            scenario_type=self.metadata["scenario_type"],
            composite_drift=self.metadata["composite_drift"],
            resampling_contamination_warning=self.metadata["resampling_contamination_warning"],
            baseline_reference=self.metadata["baseline_reference"],
            schema_warnings=self.metadata["schema_warnings"],
            psi_checkpoints=PSICheckpoints(**self.metadata["psi_checkpoints"]),
            prediction_change_rate=prediction_change_rate,
            f1_degradation=f1_degradation,
            flags=self.metadata["flags"],
            sla_check=SLACheckResult(**sla_result),
            generated_at=datetime.now(timezone.utc).isoformat(),
        )
        report_dict = report.to_dict()

        if export:
            try:
                report_path_template = cfg["severe_drift"]["output"]["report_path_template"]
            except KeyError as exc:
                raise KeyError(
                    f"Missing key {exc} in config — check severe_drift.output.report_path_template "
                    "in config/baseline_config.yml."
                ) from exc
            final_output_path = output_path or report_path_template.format(severity=severity)
            self.export_report(report_dict, final_output_path)
            report_dict["_exported_to"] = final_output_path

        return report_dict
