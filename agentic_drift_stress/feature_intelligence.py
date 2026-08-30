"""
Feature Intelligence.

Statistical layer that answers one question PSI alone can't: "does this
drift actually matter to the model?"

Two-level logic:
1. SHAP importance -> does the feature directly drive predictions? High
   importance means the drift matters regardless of raw PSI.
2. If SHAP importance is low/absent -> is the feature an indirect proxy for
   an important one? VIF (via correlation-matrix inversion: VIF_i =
   (R^-1)_ii) and direct correlation with high-SHAP features detect
   collinearity: a feature with no importance of its own can still carry a
   redundant signal that reaches the model through a key feature.
   If neither importance nor correlation is significant, the drift is
   statistically isolated from the model's behavior -> low business risk.

Without a correlation matrix or SHAP values, the assessor falls back to
UNKNOWN rather than inventing a risk level.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


class RiskLevel(Enum):
    DIRECT_IMPACT = "direct_impact"                 # high SHAP importance: confirmed impact
    INDIRECT_CORRELATION = "indirect_correlation"    # unimportant alone, but collinear with a key feature
    LOW_RISK = "low_risk"                            # neither important nor correlated
    UNKNOWN = "unknown"                              # insufficient SHAP/correlation data


@dataclass
class FeatureRiskAssessment:
    feature: str
    shap_importance: Optional[float]
    shap_percentile: Optional[float]  # 0-100, rank among tracked features
    vif: Optional[float]
    correlated_important_feature: Optional[str]
    correlation_with_important: Optional[float]
    risk_level: RiskLevel
    rationale: str


@dataclass
class FeatureIntelligenceContext:
    """External statistical inputs (computed outside this agent, e.g. via
    shap.Explainer on a recent batch, or at the last training run)."""

    shap_importances: Dict[str, float]                  # e.g. mean(|shap_value|) per feature, raw
    correlation_matrix: Optional[pd.DataFrame] = None    # feature x feature, on the reference set
    precomputed_vif: Optional[Dict[str, float]] = None   # "true" VIF (full regression), takes priority over the correlation-matrix approximation
    shap_high_percentile: float = 75.0   # >= this percentile -> DIRECT_IMPACT
    vif_high_threshold: float = 5.0      # common econometrics threshold (5-10 by convention)
    correlation_relevant_threshold: float = 0.6
    correlation_method: str = "pearson"  # "pearson" (linear) or "spearman" (rank, catches non-linear
                                          # monotonic dependence). Doesn't affect the computation here
                                          # (the matrix is already precomputed), just reported in the
                                          # rationale: VIF_i = (R^-1)_ii is a rigorous variance-explained
                                          # measure only under Pearson; under Spearman it's a rank-based
                                          # collinearity proxy, more robust but less directly interpretable.


def compute_correlation_matrix(data: pd.DataFrame, method: str = "spearman") -> pd.DataFrame:
    """Helper to build correlation_matrix upstream, defaulting to Spearman
    rather than Pearson.

    Why Spearman by default: Pearson only captures linear dependence. Two
    features linked by a non-linear monotonic relation (e.g. quadratic, log)
    can show r_Pearson near 0, understating VIF via (R^-1)_ii and missing a
    match in _best_correlated_important_feature -- a real INDIRECT_CORRELATION
    risk wrongly classified as LOW_RISK. Spearman stays sensitive to any
    monotonic relation, at the cost of a less direct link between VIF and
    classic linear-regression variance explained (see correlation_method).

    For non-monotonic dependencies (e.g. U-shaped), neither Pearson nor
    Spearman is enough; a distance correlation (dcor) would be needed, out
    of scope here.
    """
    return data.corr(method=method)


class FeatureRiskAssessor:
    """Turns a FeatureIntelligenceContext into a per-feature verdict, caching
    the (expensive) correlation-matrix inversion."""

    def __init__(self, context: FeatureIntelligenceContext):
        self.context = context
        self._vif_cache: Optional[Dict[str, float]] = None
        self._percentile_cache: Dict[str, float] = self._precompute_shap_percentiles()

    # ------------------------------------------------------------------ #
    # SHAP: relative importance
    # ------------------------------------------------------------------ #

    def _precompute_shap_percentiles(self) -> Dict[str, float]:
        """Compute all percentiles in one vectorized pass at init time,
        instead of recomputing on every assess() call (assess() may call
        this several times per feature, directly and via
        _best_correlated_important_feature)."""
        importances = self.context.shap_importances
        if not importances:
            return {}
        features = list(importances.keys())
        values = np.array([importances[f] for f in features], dtype=float)
        # matrix[i, j] = values[i] >= values[j]  <=>  values[j] <= values[i]
        # percentile_i = share of j such that values[j] <= values[i]
        matrix = values[:, None] >= values[None, :]
        percentiles = matrix.mean(axis=1) * 100.0
        return dict(zip(features, percentiles.tolist()))

    def _shap_percentile(self, feature: str) -> Optional[float]:
        return self._percentile_cache.get(feature)

    # ------------------------------------------------------------------ #
    # VIF via correlation-matrix inversion: VIF_i = (R^-1)_ii -- exactly
    # equivalent to regression-based VIF when R is the full correlation
    # matrix of standardized predictors, without a per-feature regression.
    # ------------------------------------------------------------------ #

    def _all_vif(self) -> Dict[str, float]:
        if self._vif_cache is not None:
            return self._vif_cache

        corr = self.context.correlation_matrix
        if corr is None or corr.empty:
            self._vif_cache = {}
            return self._vif_cache

        try:
            inv = np.linalg.pinv(corr.values)  # pseudo-inverse: robust if near-singular
            diag = np.diag(inv)
            # VIF is >= 1 by definition; clip to guard against numerical artifacts.
            diag = np.clip(diag, 1.0, None)
            self._vif_cache = {col: float(v) for col, v in zip(corr.columns, diag)}
        except np.linalg.LinAlgError:
            self._vif_cache = {}

        return self._vif_cache

    def _vif(self, feature: str) -> Optional[float]:
        # Precomputed VIF (full regression on the one-hot design matrix)
        # takes priority over the correlation-matrix approximation.
        if self.context.precomputed_vif and feature in self.context.precomputed_vif:
            return self.context.precomputed_vif[feature]
        return self._all_vif().get(feature)

    # ------------------------------------------------------------------ #
    # Direct correlation with the most SHAP-important feature
    # ------------------------------------------------------------------ #

    def _best_correlated_important_feature(self, feature: str):
        corr = self.context.correlation_matrix
        importances = self.context.shap_importances
        if corr is None or feature not in corr.columns or not importances:
            return None, None

        important_features = [
            f for f in importances
            if f != feature and f in corr.columns
            and (self._shap_percentile(f) or 0) >= self.context.shap_high_percentile
        ]
        if not important_features:
            return None, None

        row = corr.loc[feature, important_features].abs()
        best_feature = row.idxmax()
        best_corr = float(corr.loc[feature, best_feature])
        return best_feature, best_corr

    # ------------------------------------------------------------------ #
    # Verdict
    # ------------------------------------------------------------------ #

    def assess(self, feature: str) -> FeatureRiskAssessment:
        percentile = self._shap_percentile(feature)
        importance = self.context.shap_importances.get(feature)
        vif = self._vif(feature)
        corr_feature, corr_value = self._best_correlated_important_feature(feature)

        if percentile is None:
            # No SHAP for this feature: still try the indirect signal
            if vif is not None and vif >= self.context.vif_high_threshold:
                return FeatureRiskAssessment(
                    feature=feature, shap_importance=None, shap_percentile=None,
                    vif=vif, correlated_important_feature=corr_feature, correlation_with_important=corr_value,
                    risk_level=RiskLevel.INDIRECT_CORRELATION,
                    rationale=(
                        f"SHAP unavailable for '{feature}', but VIF={vif:.1f} (>= {self.context.vif_high_threshold}) "
                        "indicates strong collinearity with other predictors: indirect risk, not to be ignored."
                    ),
                )
            return FeatureRiskAssessment(
                feature=feature, shap_importance=None, shap_percentile=None,
                vif=vif, correlated_important_feature=None, correlation_with_important=None,
                risk_level=RiskLevel.UNKNOWN,
                rationale=f"No usable SHAP/correlation data for '{feature}': decision based on PSI alone.",
            )

        if percentile >= self.context.shap_high_percentile:
            return FeatureRiskAssessment(
                feature=feature, shap_importance=importance, shap_percentile=percentile,
                vif=vif, correlated_important_feature=None, correlation_with_important=None,
                risk_level=RiskLevel.DIRECT_IMPACT,
                rationale=(
                    f"'{feature}' is at the {percentile:.0f}th percentile of SHAP importance among tracked "
                    f"features (threshold={self.context.shap_high_percentile:.0f}th, raw importance={importance:.4f}): "
                    "the drift directly affects predictions."
                ),
            )

        # Low SHAP importance: look for indirect risk via collinearity
        vif_flag = vif is not None and vif >= self.context.vif_high_threshold
        corr_flag = corr_value is not None and abs(corr_value) >= self.context.correlation_relevant_threshold

        if vif_flag or corr_flag:
            detail = []
            if vif_flag:
                detail.append(f"VIF={vif:.1f}")
            if corr_flag:
                detail.append(f"correlation r={corr_value:.2f} with '{corr_feature}' (high-SHAP feature)")
            return FeatureRiskAssessment(
                feature=feature, shap_importance=importance, shap_percentile=percentile,
                vif=vif, correlated_important_feature=corr_feature, correlation_with_important=corr_value,
                risk_level=RiskLevel.INDIRECT_CORRELATION,
                rationale=(
                    f"'{feature}' has low SHAP importance of its own (percentile {percentile:.0f}), but "
                    f"{' and '.join(detail)}: the drift can still reach the model indirectly via this redundancy."
                ),
            )

        return FeatureRiskAssessment(
            feature=feature, shap_importance=importance, shap_percentile=percentile,
            vif=vif, correlated_important_feature=None, correlation_with_important=None,
            risk_level=RiskLevel.LOW_RISK,
            rationale=(
                f"'{feature}' has low SHAP importance (percentile {percentile:.0f}) and isn't correlated with "
                "an influential feature: likely negligible model impact despite the statistical drift."
            ),
        )

    # ------------------------------------------------------------------ #
    # Merge with the PSI drift result: RiskLevel alone doesn't say whether
    # the observed drift is significant, and PSI alone doesn't say whether
    # it matters to the model. assess() stays independent of drift
    # (testable in isolation); this method combines both into the final
    # actionable verdict.
    # ------------------------------------------------------------------ #

    def assess_with_drift(
        self,
        feature: str,
        drift_result=None,
        psi: Optional[float] = None,
        psi_high_threshold: float = 0.2,
    ) -> "CombinedFeatureVerdict":
        """Combine assess(feature) with an external drift signal.

        drift_result: any duck-typed object exposing `.psi` (e.g. a
        FeatureDriftResult) -- falls back to the explicit `psi` param.
        Deliberately no direct import of FeatureDriftResult, to avoid
        coupling this module to the PSI module."""
        resolved_psi = psi
        if resolved_psi is None and drift_result is not None:
            resolved_psi = getattr(drift_result, "psi", None)

        risk = self.assess(feature)
        psi_significant = resolved_psi is not None and resolved_psi >= psi_high_threshold

        if resolved_psi is None:
            priority = "monitor"
            note = "no PSI provided: verdict based only on intrinsic model risk."
        elif risk.risk_level in (RiskLevel.DIRECT_IMPACT, RiskLevel.INDIRECT_CORRELATION):
            if psi_significant:
                priority = "critical"
                note = f"PSI={resolved_psi:.3f} (>= {psi_high_threshold}) on a confirmed model-risk feature: priority action."
            else:
                priority = "monitor"
                note = f"PSI={resolved_psi:.3f} below threshold, but model-risk feature: watch if drift increases."
        elif risk.risk_level is RiskLevel.LOW_RISK:
            if psi_significant:
                priority = "ignore"
                note = f"PSI={resolved_psi:.3f} high but feature statistically isolated from the model: no expected business impact."
            else:
                priority = "ignore"
                note = "PSI below threshold and no model impact: no action."
        else:  # UNKNOWN
            priority = "investigate" if psi_significant else "monitor"
            note = (
                f"PSI={resolved_psi:.3f}: insufficient SHAP/correlation data to resolve model risk, "
                "manual check recommended." if psi_significant else
                "PSI below threshold and model risk undetermined: passive monitoring is enough."
            )

        return CombinedFeatureVerdict(
            feature=feature,
            psi=resolved_psi,
            risk=risk,
            priority=priority,
            rationale=f"{risk.rationale} | {note}",
        )


@dataclass
class CombinedFeatureVerdict:
    """Final actionable verdict: merges RiskLevel (model impact) and PSI
    (drift magnitude). This object, not FeatureRiskAssessment alone, should
    drive downstream alerting decisions."""
    feature: str
    psi: Optional[float]
    risk: FeatureRiskAssessment
    priority: str  # "critical" | "investigate" | "monitor" | "ignore"
    rationale: str


# ---------------------------------------------------------------------- #
# Loading from the JSON report generated by generate_feature_intelligence_report.py
# ---------------------------------------------------------------------- #

def load_feature_intelligence_context(json_path: Union[str, Path]) -> FeatureIntelligenceContext:
    """Read the JSON report (shap_importances / vif / correlation_matrix)
    and build the context expected by AlertAgent.process_data_drift()."""
    with open(json_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    shap_importances = report["shap_importances"]
    corr_dict = report.get("correlation_matrix")
    correlation_matrix = pd.DataFrame(corr_dict) if corr_dict else None
    precomputed_vif = report.get("vif")

    thresholds = report.get("thresholds", {})
    return FeatureIntelligenceContext(
        shap_importances=shap_importances,
        correlation_matrix=correlation_matrix,
        precomputed_vif=precomputed_vif,
        shap_high_percentile=thresholds.get("shap_high_percentile", 75.0),
        vif_high_threshold=thresholds.get("vif_high_threshold", 5.0),
        correlation_relevant_threshold=thresholds.get("correlation_relevant_threshold", 0.6),
        correlation_method=report.get("correlation_method", "pearson"),
    )
