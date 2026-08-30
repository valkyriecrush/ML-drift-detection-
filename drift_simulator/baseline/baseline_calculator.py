# Purpose: computes and caches baseline statistics (PSI bins, descriptive stats, hash) used as the reference for drift detection.

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from baseline.schema_validator import SchemaValidator
from src.preprocessing import zeros_to_missing

# psi_common.py lives at the repo root, shared with
# agentic_drift_stress/drift_detector.py -- same sys.path bootstrap as that
# module, so this stays importable whether via tests (conftest already adds
# ROOT), via run_drift.py (CWD=drift_simulator/, ROOT isn't automatically on
# the path), or via bridge/.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from psi_common import psi_from_counts  # noqa: E402

logger = logging.getLogger(__name__)


def _to_native(obj: Any) -> Any:
    """Recursive purge of any np.generic / np.ndarray -> native JSON types."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    return obj


@dataclass
class BaselineReference:
    path: str
    hash: str
    computed_at: str

    def as_dict(self) -> Dict[str, str]:
        return {"path": self.path, "hash": self.hash, "computed_at": self.computed_at}


@dataclass
class BaselineCalculator:
    config_path: str = "config/baseline_config.yml"
    config: Dict[str, Any] = field(default_factory=dict, init=False)
    baseline_df: Optional[pd.DataFrame] = field(default=None, init=False)
    baseline_stats: Dict[str, Dict[str, float]] = field(default_factory=dict, init=False)
    bin_edges: Dict[str, List[float]] = field(default_factory=dict, init=False)
    baseline_reference: Optional[BaselineReference] = field(default=None, init=False)
    schema_validator: Optional[SchemaValidator] = field(default=None, init=False)
    schema_warnings: List[str] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self._init_schema_validator()

    def _init_schema_validator(self) -> None:
        """
        Instantiates SchemaValidator from the paths declared under
        `reference_files` in baseline_config.yml. Non-blocking: if the
        reference files are absent (minimal deployment without
        config/diabetes_schema.yaml etc.), continues without validation.
        """
        ref = self.config.get("reference_files", {})
        schema_path = ref.get("schema_path")
        performance_path = ref.get("performance_path")
        if schema_path and performance_path and os.path.exists(schema_path) and os.path.exists(performance_path):
            self.schema_validator = SchemaValidator(
                schema_path=schema_path,
                performance_path=performance_path,
                cleaned_reference_path=ref.get("cleaned_reference_path"),
                features_reference_path=ref.get("features_reference_path"),
            )

    # ------------------------------------------------------------------
    # Hashing / cache
    # ------------------------------------------------------------------
    @staticmethod
    def _sha256_of_df(df: pd.DataFrame) -> str:
        """Stable SHA256 fingerprint (sorted column order) of the baseline DataFrame."""
        payload = df[sorted(df.columns)].to_csv(index=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _cache_path(self) -> str:
        cache_dir = self.config["baseline"]["cache_dir"]
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, self.config["baseline"]["cache_file"])

    # ------------------------------------------------------------------
    # Baseline loading / computation (once, then disk cache)
    # ------------------------------------------------------------------
    def load_or_compute(self, source_path: Optional[str] = None, force_recompute: bool = False) -> "BaselineCalculator":
        """
        Loads the baseline from cache if it exists and matches the current
        source file's hash; otherwise computes it once and persists it.

        NOTE: `force_recompute=True` must remain exceptional — the goal is
        to NOT recompute the baseline on every drift simulation.
        """
        source_path = source_path or self.config["baseline"]["source_data_path"]
        cache_path = self._cache_path()

        # Confirmed on the real repo (agentic/main.py): there is no dedicated
        # loader in src/preprocessing.py, `main.py` itself reads the raw CSV
        # with pd.read_csv(). We stay faithful to that existing behavior.
        df_raw = pd.read_csv(source_path)

        # Documentary validation (never blocking) against diabetes_schema.yaml,
        # done on the RAW df (before cleaning) to faithfully report the
        # number of biologically impossible zeros actually present in the
        # source file.
        if self.schema_validator is not None:
            self.schema_warnings = self.schema_validator.validate_raw(df_raw)

        # IMPORTANT: src/preprocessing.zeros_to_missing() is applied HERE,
        # BEFORE any baseline stat / PSI grid computation and BEFORE any
        # drift injection. A 0 in Glucose/BloodPressure/SkinThickness/
        # Insulin/BMI is not a physiologically valid value -> it MUST be
        # treated as missing and then imputed (median per Outcome class)
        # before baseline_df serves as the reference for everything else.
        # `Pregnancies` is deliberately NOT in ZERO_AS_MISSING_COLS: a 0
        # there is a legitimate baseline characteristic (never pregnant),
        # not a missing value -> zeros_to_missing() does not touch it.
        # Without this upstream cleanup, get_stats()/compute_psi() and the
        # drift deltas (std_multiplier * sigma, etc.) would be computed on
        # a distribution polluted by these artificial zeros.
        df = zeros_to_missing(df_raw)

        # The hash (and thus cache invalidation) is based on the data AFTER
        # cleaning: it's this data that determines baseline_stats and
        # bin_edges, so its content is what must invalidate the cache if
        # the source file (or the cleaning logic) changes.
        current_hash = self._sha256_of_df(df)

        if not force_recompute and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached_hash = cached.get("baseline_reference", {}).get("hash")
            if cached_hash == current_hash:
                self._hydrate_from_cache(cached, df)
                return self

            # Baseline-provenance guard: a cache hash mismatch used to
            # silently regenerate baseline_stats_v2.json without
            # distinguishing "same data, hash differs for a benign reason
            # (e.g. row order)" from "the baseline actually changed" -- a
            # potentially critical divergence (drift in the reference
            # itself) went completely unnoticed. We compute the new
            # baseline first, CLASSIFY it against the old one, log
            # accordingly, then persist (with the updated hash) plus the
            # detail in hash_mismatch_report.jsonl.
            self._compute_fresh(df, source_path, current_hash)
            classification = self._classify_hash_mismatch(cached)
            level = (
                logging.WARNING
                if classification == "statistically_equivalent_mirror"
                else logging.CRITICAL
            )
            logger.log(
                level,
                "Baseline hash differs for '%s' (cache=%s, recomputed=%s) "
                "-- classification=%s",
                source_path,
                cached_hash,
                current_hash,
                classification,
            )
            self._persist_hash_mismatch_report(cache_path, cached_hash, current_hash, classification)
            self._persist_cache(cache_path)
            return self

        self._compute_fresh(df, source_path, current_hash)
        self._persist_cache(cache_path)
        return self

    def _classify_hash_mismatch(self, cached: Dict[str, Any]) -> str:
        """
        Compares the freshly computed `self.baseline_stats` (post
        `_compute_fresh`) against the cached ones to decide whether a hash
        mismatch reflects a statistically equivalent mirror (same data,
        different hash -- e.g. a different row order in the source file)
        or a materially different baseline (a real change worth a CRITICAL
        alert, since every downstream PSI/bin_edges decision depends on
        this baseline being the trusted reference).
        """
        cached_stats = cached.get("baseline_stats", {})
        for feature, stats in self.baseline_stats.items():
            cached_feature_stats = cached_stats.get(feature)
            if cached_feature_stats is None:
                return "materially_different_baseline"
            for key, value in stats.items():
                cached_value = cached_feature_stats.get(key)
                if cached_value is None or not np.isclose(
                    value, cached_value, rtol=1e-6, atol=1e-9
                ):
                    return "materially_different_baseline"
        return "statistically_equivalent_mirror"

    def _persist_hash_mismatch_report(
        self, cache_path: str, cached_hash: Optional[str], current_hash: str, classification: str
    ) -> None:
        report_path = os.path.join(os.path.dirname(cache_path), "hash_mismatch_report.jsonl")
        entry = {
            "detected_at": datetime.now(timezone.utc).isoformat(),
            "cached_hash": cached_hash,
            "recomputed_hash": current_hash,
            "classified_as": classification,
        }
        with open(report_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(_to_native(entry)) + "\n")

    def _hydrate_from_cache(self, cached: Dict[str, Any], df: pd.DataFrame) -> None:
        self.baseline_df = df
        self.baseline_stats = cached["baseline_stats"]
        self.bin_edges = cached["bin_edges"]
        self.baseline_reference = BaselineReference(**cached["baseline_reference"])

    def _compute_fresh(self, df: pd.DataFrame, source_path: str, current_hash: str) -> None:
        self.baseline_df = df
        n_bins = self.config["baseline"]["psi"]["n_bins"]
        tracked = self.config["baseline"]["tracked_features"]

        stats: Dict[str, Dict[str, float]] = {}
        edges: Dict[str, List[float]] = {}
        for feature in tracked:
            if feature not in df.columns:
                continue
            series = df[feature].dropna().astype(float)
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            stats[feature] = _to_native(
                {
                    "mean": series.mean(),
                    "std": series.std(),
                    "min": series.min(),
                    "max": series.max(),
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                }
            )
            # Bin grid fixed from the baseline's quantiles -> never
            # recomputed on drifted data (implementation constraint).
            quantile_edges = np.unique(
                np.quantile(series, np.linspace(0, 1, n_bins + 1))
            )
            # open bounds to capture any shift outside the baseline range
            quantile_edges[0] = -np.inf
            quantile_edges[-1] = np.inf
            edges[feature] = _to_native(quantile_edges)

        self.baseline_stats = stats
        self.bin_edges = edges
        self.baseline_reference = BaselineReference(
            path=source_path,
            hash=current_hash,
            computed_at=datetime.now(timezone.utc).isoformat(),
        )

    def _persist_cache(self, cache_path: str) -> None:
        payload = {
            "baseline_stats": self.baseline_stats,
            "bin_edges": self.bin_edges,
            "baseline_reference": self.baseline_reference.as_dict(),
        }
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(_to_native(payload), f, indent=2)

    # ------------------------------------------------------------------
    # PSI with fixed bin_edges + Laplace smoothing
    # ------------------------------------------------------------------
    def compute_psi(self, actual: pd.Series, feature: str) -> float:
        """
        PSI(feature) between the (fixed) baseline distribution and `actual`,
        binning on self.bin_edges[feature] (NEVER recomputed), with Laplace
        epsilon smoothing to avoid division by zero / log(0).
        """
        if feature not in self.bin_edges:
            raise KeyError(f"No baseline bin grid for '{feature}'. "
                            f"Call load_or_compute() first.")

        epsilon = float(self.config["baseline"]["psi"]["laplace_epsilon"])
        edges = np.array(self.bin_edges[feature], dtype=float)

        expected_counts, _ = np.histogram(
            self.baseline_df[feature].dropna().astype(float), bins=edges
        )
        actual_counts, _ = np.histogram(actual.dropna().astype(float), bins=edges)

        # Delegates to the shared calculation in psi_common.py, the SAME
        # formula used by agentic_drift_stress/drift_detector.py.
        return psi_from_counts(expected_counts, actual_counts, epsilon=epsilon)

    def get_stats(self, feature: str) -> Dict[str, float]:
        return dict(self.baseline_stats[feature])

    def get_baseline_reference(self) -> Dict[str, str]:
        return self.baseline_reference.as_dict()

    # ------------------------------------------------------------------
    # Columns expected by the model (from config/performance.yaml)
    # ------------------------------------------------------------------
    def expected_model_columns(self) -> Optional[List[str]]:
        """
        Fixed order of columns expected out of `data_prep`, read from
        `config/performance.yaml` (booster_info.feature_names from the
        LightGBM block). Returns None if no SchemaValidator could be
        instantiated (reference files absent) — the caller must then fall
        back to `model.feature_name_`.
        """
        if self.schema_validator is None:
            return None
        return self.schema_validator.expected_model_columns()
