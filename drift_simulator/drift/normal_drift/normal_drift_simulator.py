# Purpose: "normal drift" scenario — injects mild, realistic drift to typically produce a WARNING status.

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from baseline.baseline_calculator import _to_native
from drift.base_drift_simulator import (
    HIDDEN_TRUE_LABEL_COL,
    BaseDriftSimulator,
    DriftStatus,
    DriftStatusResult,
)

logger = logging.getLogger(__name__)

# NOTE: no hardcoded _DEFAULT_NORMAL_DRIFT_CONFIG / _DEFAULT_TRACKED_FEATURES
# here on purpose. `config/baseline_config.yml` (baseline.* + normal_drift.*)
# is the single source of truth — falling back to metrics/features baked into
# this module would let the simulator silently diverge from the real
# config/database instead of failing loudly when the config is missing.


@dataclass
class NormalDriftReport:
    """Typed report of the "Normal Drift" scenario -> exported JSON shape."""

    scenario: str
    scenario_type: Optional[str]
    baseline_reference: Dict[str, str]
    schema_warnings: List[str]
    parameters: Dict[str, Any]
    psi_checkpoints: Dict[str, Dict[str, float]]
    drift_status_by_checkpoint: Dict[str, Dict[str, Any]]
    overall_drift_status: Dict[str, Any]
    target_status_check: Dict[str, Any]
    prediction_change_rate: float
    f1_degradation: float
    flags: Dict[str, Any]
    generated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return _to_native(asdict(self))


class NormalDriftSimulator(BaseDriftSimulator):
    """
    Moderate drift simulator ("degraded business as usual"), meant to
    trigger a monitoring `WARNING` rather than a critical alert. Inherits
    `BaseDriftSimulator` for all shared mechanics (idempotence,
    preprocessing pipeline, drift primitives, PSI computation, JSON
    export).
    """

    def run(
        self,
        config: Dict[str, Any],
        apply_pre_processing: bool = True,
        model_path: str = "models/lgbm_model.pkl",
        export: bool = True,
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        # --- Mandatory line 1: idempotence (same contract as the other scenarios) ---
        self.reset(self.baseline_df)

        # `config` is now REQUIRED and must come straight from
        # config/baseline_config.yml (yaml.safe_load) — no silent fallback to
        # hardcoded metrics/features baked into this module. Same contract as
        # SevereDriftSimulator.simulate_severe_scenario.
        if not config:
            raise ValueError(
                "NormalDriftSimulator.run() requires `config` loaded from "
                "config/baseline_config.yml "
                "(e.g. yaml.safe_load(open('config/baseline_config.yml'))) — "
                "no default metrics/features are used."
            )
        try:
            cfg = config
            nd_cfg = cfg["normal_drift"]
            target_col = cfg["baseline"]["target_col"]
            tracked_features = cfg["baseline"]["tracked_features"]
            step_ids = nd_cfg["step_ids"]
        except KeyError as exc:
            raise KeyError(
                f"Missing key {exc} in config — check baseline.target_col, "
                "baseline.tracked_features and the full normal_drift section "
                "in config/baseline_config.yml."
            ) from exc

        # --- Baseline audit (incl. non-blocking schema warnings) ---
        self.metadata["baseline_reference"] = self.baseline_calc.get_baseline_reference()
        self.metadata["schema_warnings"] = list(self.baseline_calc.schema_warnings)

        # ================= 1. Mild Feature Drift (Delta ~= 0.5 * sigma) =================
        feature_shift_params: Dict[str, Any] = {}
        for feature, params in nd_cfg["features"].items():
            std_multiplier = params["std_multiplier"]
            mask_ratio = params["mask_ratio"]
            sigma = self.baseline_calc.get_stats(feature)["std"]
            delta = std_multiplier * sigma

            step_id_key = f"feature_step_{feature.lower()}"
            step_id = step_ids.get(step_id_key, 100 + hash(feature) % 100)
            self.feature_step(feature, delta, mask_ratio, step_id)

            feature_shift_params[feature] = {
                "std_multiplier": std_multiplier,
                "sigma": float(sigma),
                "delta": float(delta),
                "mask_ratio": mask_ratio,
            }

        psi_post_feature_drift = self.compute_psi_checkpoint(self.current_df, tracked_features)
        self.metadata["psi_checkpoints"]["psi_post_feature_drift"] = psi_post_feature_drift

        # ================= 2. Label Noise (r ~= 3-5%) =================
        # Hidden column with the "true y" before noise -> lets us measure
        # the F1 degradation attributable to the label noise itself (same
        # convention as SevereDriftSimulator).
        self.current_df = self.current_df.copy()
        self.current_df[HIDDEN_TRUE_LABEL_COL] = self.current_df[target_col].copy()

        cd_cfg = nd_cfg["concept_drift"]
        flip_min, flip_max = cd_cfg["label_noise_flip_ratio_min"], cd_cfg["label_noise_flip_ratio_max"]
        flip_ratio = cd_cfg.get("label_noise_flip_ratio", (flip_min + flip_max) / 2)
        self.concept_drift_label_noise(
            flip_ratio=flip_ratio,
            target_col=target_col,
            step_id=step_ids["concept_drift_label_noise"],
        )

        # ================= 3. Target / Prior Shift (prevalence +/- 5 pts) =================
        td_cfg = nd_cfg["target_drift"]
        baseline_prevalence = float(self.baseline_df[target_col].mean())
        direction = 1.0 if td_cfg["direction"] == "up" else -1.0
        new_prevalence = min(max(baseline_prevalence + direction * td_cfg["prevalence_delta"], 0.0), 1.0)

        self.target_drift(
            new_prevalence=new_prevalence,
            target_col=target_col,
            resampling_method=td_cfg["resampling_method"],
            step_id=step_ids["target_drift_resampling"],
        )
        psi_post_target_drift = self.compute_psi_checkpoint(self.current_df, tracked_features)
        self.metadata["psi_checkpoints"]["psi_post_target_drift"] = psi_post_target_drift

        # ================= 4. Pipeline + model evaluation (optional) =================
        # No "Data Quality" step for this scenario (see module docstring):
        # sync the "true y" directly before the pipeline.
        self.y_true_uncorrupted = self.current_df[HIDDEN_TRUE_LABEL_COL].copy()
        X_drifted = self.current_df.drop(columns=[target_col, HIDDEN_TRUE_LABEL_COL])
        y_drifted = self.current_df[target_col]

        scaler_mode = cfg.get("preprocessing", {}).get("scaler_mode", "persistent_baseline_fit")
        prediction_change_rate = 0.0
        f1_degradation = 0.0
        psi_final: Dict[str, float] = {}

        if apply_pre_processing:
            n_before = X_drifted.shape[0]
            try:
                if self.model is None:
                    self._load_model(model_path)
            except Exception:
                pass  # model unavailable -> model metrics skipped, PSI still available

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
            # mild) drift already captured by psi_post_feature_drift /
            # psi_post_target_drift above. Use X_drifted (raw, pre-pipeline)
            # instead.
            psi_final = self.compute_psi_checkpoint(X_drifted, tracked_features)
            self.metadata["psi_checkpoints"]["psi_final"] = psi_final

            try:
                if self.model is None:
                    # `_load_model` failed silently above (`except Exception:
                    # pass`) -> raise explicitly here to fall into the same
                    # `except` block as the "file missing" case, rather than
                    # an uncaught `AttributeError` on `.predict()`.
                    raise FileNotFoundError(model_path)

                baseline_X_processed = self._run_pipeline(
                    self.baseline_df.drop(columns=[target_col]),
                    self.baseline_df[target_col],
                    scaler_mode=scaler_mode,
                )
                baseline_preds = self.model.predict(baseline_X_processed)
                drifted_preds = self.model.predict(X_processed)

                n_common = min(len(baseline_preds), len(drifted_preds))
                import numpy as np  # local import: avoids a hard dependency if model is absent

                prediction_change_rate = float(
                    np.mean(baseline_preds[:n_common] != drifted_preds[:n_common])
                )

                from sklearn.metrics import f1_score

                f1_baseline = f1_score(self.baseline_df[target_col], baseline_preds)
                assert len(self.y_true_uncorrupted) == len(drifted_preds) == len(y_drifted), (
                    "Unexpected misalignment between y_true_uncorrupted, y_drifted and "
                    "drifted_preds — resampling was not propagated correctly."
                )
                f1_true_degradation_source = f1_score(self.y_true_uncorrupted, drifted_preds)
                f1_degradation = float(f1_baseline - f1_true_degradation_source)
                self.metadata["flags"]["f1_model_degradation"] = f1_degradation
            except FileNotFoundError:
                self.metadata["flags"]["model_load_error"] = (
                    f"Model not found at '{model_path}' — wire up the real repo path."
                )

        # ================= 5. Drift status (OK / WARNING / CRITICAL) =================
        thresholds = nd_cfg["drift_status_thresholds"]
        drift_status_by_checkpoint: Dict[str, DriftStatusResult] = {
            "psi_post_feature_drift": self.classify_drift_status(
                psi_post_feature_drift, thresholds["warning_low"], thresholds["critical_high"]
            ),
            "psi_post_target_drift": self.classify_drift_status(
                psi_post_target_drift, thresholds["warning_low"], thresholds["critical_high"]
            ),
        }
        if psi_final:
            drift_status_by_checkpoint["psi_final"] = self.classify_drift_status(
                psi_final, thresholds["warning_low"], thresholds["critical_high"]
            )

        # Overall status = the post-pipeline checkpoint if available (most
        # representative of what actually reaches the model in production),
        # otherwise the last checkpoint computed.
        overall_status = drift_status_by_checkpoint.get(
            "psi_final", drift_status_by_checkpoint["psi_post_target_drift"]
        )

        if overall_status.status == DriftStatus.WARNING.value:
            logger.warning(
                "NormalDriftSimulator: moderate drift detected (%s) — %s",
                overall_status.reference_feature,
                overall_status.message,
            )
        elif overall_status.status == DriftStatus.CRITICAL.value:
            logger.warning(
                "NormalDriftSimulator: drift beyond the expected moderate band "
                "(CRITICAL status, feature '%s') — this scenario's parameters "
                "are calibrated for a WARNING status; revisit std_multiplier / "
                "flip_ratio / prevalence_delta if this recurs.",
                overall_status.reference_feature,
            )

        target_status_check = {
            "expected_status": DriftStatus.WARNING.value,
            "actual_status": overall_status.status,
            "matches_expectation": overall_status.status == DriftStatus.WARNING.value,
        }

        # ================= Report =================
        report = NormalDriftReport(
            scenario="normal_drift",
            scenario_type=self.metadata["scenario_type"],
            baseline_reference=self.metadata["baseline_reference"],
            schema_warnings=self.metadata["schema_warnings"],
            parameters={
                "feature_shift": feature_shift_params,
                "label_noise_flip_ratio": flip_ratio,
                "prevalence": {
                    "baseline": baseline_prevalence,
                    "target": new_prevalence,
                    "delta": td_cfg["prevalence_delta"],
                    "direction": td_cfg["direction"],
                    "resampling_method": td_cfg["resampling_method"],
                },
            },
            psi_checkpoints=dict(self.metadata["psi_checkpoints"]),
            drift_status_by_checkpoint={
                stage: result.to_dict() for stage, result in drift_status_by_checkpoint.items()
            },
            overall_drift_status=overall_status.to_dict(),
            target_status_check=target_status_check,
            prediction_change_rate=prediction_change_rate,
            f1_degradation=f1_degradation,
            flags=self.metadata["flags"],
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

        report_dict = report.to_dict()

        if export:
            final_output_path = output_path or nd_cfg["output"]["report_path"]
            self.export_report(report_dict, final_output_path)  # nd_cfg["output"]["report_path"] comes straight from the yml
            report_dict["_exported_to"] = final_output_path

        return report_dict
