# Purpose: FeatureStore — materializes the offline feature table (with sha256-based cache invalidation, same pattern as baseline/baseline_calculator.py) and serves both historical (training/batch) and online (single-record, train/serve-consistent) feature retrieval.

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from baseline.schema_validator import SchemaValidator
from feature_store.registry import FeatureRegistry
from feature_store.transformers import FittedArtifacts, fit_artifacts
from src.feature_engineering import feature_extraction, one_hot_encoder
from src.preprocessing import ZERO_AS_MISSING_COLS, replace_with_thresholds


def _sha256_of_df(df: pd.DataFrame) -> str:
    """Stable SHA256 fingerprint (sorted column order) of a dataframe —
    same fingerprinting approach as BaselineCalculator._sha256_of_df, so a
    change in the source data invalidates the offline cache automatically."""
    payload = df[sorted(df.columns)].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class FeatureStore:
    """Lightweight, config-driven feature store.

    Two entry points, matching the classic offline/online split:

    - `get_historical_features()`: the full, human-readable (unscaled)
      offline feature table — for training or batch scoring. This dataset
      has no event timestamp in the source CSV, so there is no
      point-in-time / time-travel correctness here; it's a plain batch
      offline store, not a temporal one.
    - `get_online_features(record)`: a single model-ready row for a NEW
      patient, computed with the SAME fitted scaler / thresholds / medians
      / column order persisted at `materialize()` time, so it can never
      silently drift from what the model was trained on.

    `PatientID` (the entity key) is a synthetic, stable row index assigned
    at materialize() time — the Pima Indians Diabetes Database has no real
    patient identifier in the source data. That's a documented limitation,
    not a hidden one.
    """

    config_path: str = "config/feature_store.yml"

    config: Dict[str, Any] = field(default_factory=dict, init=False)
    registry: Optional[FeatureRegistry] = field(default=None, init=False)
    schema_validator: Optional[SchemaValidator] = field(default=None, init=False)

    offline_df: Optional[pd.DataFrame] = field(default=None, init=False)
    artifacts: Optional[FittedArtifacts] = field(default=None, init=False)
    manifest: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.registry = FeatureRegistry(self.config)

        fs = self.config["feature_store"]
        schema_path = fs.get("schema_path")
        performance_path = fs.get("model_feature_view")
        if schema_path and performance_path and os.path.exists(schema_path) and os.path.exists(performance_path):
            self.schema_validator = SchemaValidator(
                schema_path=schema_path, performance_path=performance_path
            )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def _offline_paths(self) -> Dict[str, str]:
        cfg = self.config["feature_store"]["offline_store"]
        cache_dir = cfg["cache_dir"]
        return {
            "dir": cache_dir,
            "features": os.path.join(cache_dir, cfg["features_file"]),
            "manifest": os.path.join(cache_dir, cfg["manifest_file"]),
        }

    def _artifacts_path(self) -> str:
        cfg = self.config["feature_store"]["artifacts"]
        return os.path.join(cfg["dir"], cfg["file"])

    def _expected_model_columns(self) -> Optional[List[str]]:
        if self.schema_validator is None:
            return None
        return self.schema_validator.expected_model_columns()

    # ------------------------------------------------------------------
    # Materialization (offline)
    # ------------------------------------------------------------------
    def materialize(self, force_recompute: bool = False) -> "FeatureStore":
        """Build (or load from cache) the offline feature table and the
        fitted online-serving artifacts.

        Cache is invalidated by a sha256 hash of the raw source file, same
        approach as `BaselineCalculator.load_or_compute` — recomputation
        only happens when `data/diabetes.csv` actually changes, or when
        `force_recompute=True` is passed explicitly.
        """
        fs_cfg = self.config["feature_store"]
        entity_key = self.registry.entity_key
        target_col = self.registry.target_col
        source_path = fs_cfg["source"]["raw_data_path"]

        df_raw = pd.read_csv(source_path)
        current_hash = _sha256_of_df(df_raw)

        paths = self._offline_paths()
        artifacts_path = self._artifacts_path()

        if not force_recompute and os.path.exists(paths["manifest"]) and os.path.exists(paths["features"]):
            with open(paths["manifest"], "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if manifest.get("source_hash") == current_hash and os.path.exists(artifacts_path):
                self.offline_df = pd.read_csv(paths["features"])
                self.artifacts = FittedArtifacts.load(artifacts_path)
                self.manifest = manifest
                return self

        self._build_and_persist(df_raw, entity_key, target_col, current_hash, source_path, paths, artifacts_path)
        return self

    def _build_and_persist(
        self,
        df_raw: pd.DataFrame,
        entity_key: str,
        target_col: str,
        current_hash: str,
        source_path: str,
        paths: Dict[str, str],
        artifacts_path: str,
    ) -> None:
        num_cols = self.registry.raw_numeric_features()

        # 1. Assign the synthetic, materialize-time-stable entity key.
        df = df_raw.reset_index(drop=True).copy()
        df.insert(0, entity_key, np.arange(len(df)))

        # 2. Outlier capping — same IQR method as src.preprocessing /
        #    src.feature_engineering.data_prep, applied to the raw numeric
        #    columns before the zero -> missing step, exactly like data_prep.
        for col in num_cols:
            replace_with_thresholds(df, col)

        # 3. Zero -> missing, then per-Outcome-class median imputation
        #    (faithful to src.feature_engineering.data_prep, valid here
        #    because the label IS available for the offline/training table).
        for col in ZERO_AS_MISSING_COLS:
            if col in df.columns:
                df[col] = np.where(df[col] == 0, np.nan, df[col])
        df[num_cols] = df[num_cols].fillna(df.groupby(target_col)[num_cols].transform("median"))

        # 4. Clinical risk bands (deterministic, online-safe by construction).
        feature_extraction(df)

        # `offline_df`: human-readable, UNSCALED — this is the reusable
        # artifact a feature store should hand out for training / batch
        # scoring / EDA. Scaling is applied downstream, per-consumer.
        self.offline_df = df.copy()

        # 5. Fit every stateful online-serving artifact (scaler, binary
        #    encoders, outlier thresholds, imputation medians, final
        #    column order) on this same unencoded offline table.
        model_columns = self._expected_model_columns()
        self.artifacts = fit_artifacts(
            df_raw=df_raw,
            df_offline_features=df.drop(columns=[entity_key]),
            num_cols=num_cols,
            target_col=target_col,
            model_columns=model_columns,
        )

        # 6. Persist.
        os.makedirs(paths["dir"], exist_ok=True)
        self.offline_df.to_csv(paths["features"], index=False)
        self.artifacts.save(artifacts_path)

        self.manifest = {
            "source_path": source_path,
            "source_hash": current_hash,
            "entity_key": entity_key,
            "n_rows": int(len(self.offline_df)),
            "n_features": int(len(self.offline_df.columns) - 2),  # minus entity_key, target_col
            "feature_groups": self.registry.list_groups(),
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(paths["manifest"], "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2)

    # ------------------------------------------------------------------
    # Historical (offline / training) retrieval
    # ------------------------------------------------------------------
    def get_historical_features(
        self,
        feature_names: Optional[List[str]] = None,
        entity_ids: Optional[List[int]] = None,
        with_target: bool = True,
    ) -> pd.DataFrame:
        """Return the offline feature table, optionally filtered to
        specific features and/or entity ids. Unscaled, human-readable —
        callers that need model-ready input should scale/encode themselves
        (see `src.feature_engineering.data_prep`) or use
        `get_online_features` row-by-row.
        """
        if self.offline_df is None:
            raise RuntimeError("Call materialize() before get_historical_features().")

        df = self.offline_df
        entity_key = self.registry.entity_key
        target_col = self.registry.target_col

        if entity_ids is not None:
            df = df[df[entity_key].isin(entity_ids)]

        if feature_names is not None:
            cols = [entity_key] + list(feature_names)
            if with_target:
                cols.append(target_col)
            df = df[cols]
        elif not with_target:
            df = df.drop(columns=[target_col])

        return df.reset_index(drop=True).copy()

    # ------------------------------------------------------------------
    # Online (single-record, serving-time) retrieval
    # ------------------------------------------------------------------
    def get_online_features(self, record: Dict[str, Any]) -> pd.DataFrame:
        """Compute a single model-ready row for a NEW patient record.

        `record` must contain the 8 raw clinical measurements (the
        `raw_measurements` feature group). Applies, IN ORDER, the exact
        steps `materialize()` fit and persisted: outlier capping (fixed
        thresholds), zero -> missing imputation (fixed overall medians —
        see `feature_store.transformers.fit_artifacts` for why this differs
        from the per-class median used for the offline table), clinical
        risk bands, scaling (`scaler.transform`, never refit), and encoding
        — then reindexes to the exact column set/order the model expects,
        filling any one-hot column absent from this single row with 0.

        Returns a single-row DataFrame ready for `model.predict(...)`.
        """
        if self.artifacts is None:
            raise RuntimeError("Call materialize() before get_online_features().")

        num_cols = self.artifacts.num_cols
        missing = [c for c in num_cols if c not in record]
        if missing:
            raise ValueError(f"Missing required raw features in record: {missing}")

        row = pd.DataFrame([{col: record[col] for col in num_cols}])

        # 1. Outlier capping with the PERSISTED thresholds (not recomputed).
        for col in num_cols:
            low, up = self.artifacts.outlier_bounds[col]
            row[col] = row[col].clip(lower=low, upper=up)

        # 2. Zero -> missing, imputed with the PERSISTED overall median.
        for col in ZERO_AS_MISSING_COLS:
            if col in row.columns and row.loc[0, col] == 0:
                row.loc[0, col] = self.artifacts.impute_medians[col]

        # 3. Clinical risk bands — same deterministic function as offline.
        feature_extraction(row)

        # 4. Scale numeric columns with the PERSISTED, already-fit scaler.
        row[num_cols] = self.artifacts.scaler.transform(row[num_cols])

        # 5. Encode: binary columns via the PERSISTED LabelEncoders, then
        #    one-hot the remaining categorical (band) columns.
        for col, le in self.artifacts.binary_encoders.items():
            if col in row.columns:
                row[col] = le.transform(row[col].astype(str))

        cat_cols = [c for c in row.columns if row[c].dtype.name in ("category", "object")]
        row = one_hot_encoder(row, cat_cols, drop_first=True) if cat_cols else row

        # 6. Reindex to the exact training-time column set/order —
        #    guarantees a single row (which can't itself produce every
        #    one-hot category) still matches the model's expected input.
        if self.artifacts.model_columns:
            row = row.reindex(columns=self.artifacts.model_columns, fill_value=0)

        return row

    # ------------------------------------------------------------------
    def describe(self) -> str:
        lines = [self.registry.describe()]
        if self.manifest:
            lines.append(
                f"Offline table: {self.manifest['n_rows']} rows materialized "
                f"{self.manifest['computed_at']} (source hash "
                f"{self.manifest['source_hash'][:12]}...)"
            )
        return "\n".join(lines)
