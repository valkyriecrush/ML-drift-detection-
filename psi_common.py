"""
psi_common.py -- SINGLE, shared implementation of the PSI (Population
Stability Index) calculation, used by both
`agentic_drift_stress/drift_detector.py` and
`drift_simulator/baseline/baseline_calculator.py`.

Background: prior to this module, each sub-project had its own PSI function
with two different smoothing conventions to avoid division by zero / log(0)
(epsilon added after normalization vs. before). Both are individually
legitimate, but produced numerically DIFFERENT PSI values on the same data
and bins -- to the point that the same scenario (`normal_drift`) was
classified WARNING by one and CRITICAL by the other.

This module centralizes the PSI math (one smoothing convention: additive
Laplace smoothing on raw counts before renormalization -- the most standard
convention, stable even when a bin is empty on both sides). Each sub-project
keeps its own bin-SELECTION logic (legitimately different between the two
tools: `BaselineCalculator` uses a bin grid fixed on the full, cached
baseline; `DriftDetector` resamples and rebins its own reference on every
`set_reference()`) -- only the PSI formula itself is now shared and
identical everywhere.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def psi_from_counts(
    reference_counts: Sequence[float],
    current_counts: Sequence[float],
    epsilon: float = 1e-4,
) -> float:
    """Compute PSI from two count histograms (same bins).

    Smoothing convention: additive Laplace smoothing on raw counts, before
    renormalizing to proportions -- see module docstring for background.

    Parameters
    ----------
    reference_counts, current_counts : array-like of float
        Per-bin counts (same bins, same length) for the reference and
        current populations.
    epsilon : float, default 1e-4
        Additive smoothing term, applied to counts before renormalization.
        Must match between two calculations meant to be comparable (e.g.
        across sub-projects).

    Returns
    -------
    float
    """
    ref_counts = np.asarray(reference_counts, dtype=float)
    cur_counts = np.asarray(current_counts, dtype=float)

    if ref_counts.shape != cur_counts.shape:
        raise ValueError(
            f"reference_counts and current_counts must have the same shape "
            f"(same bins): {ref_counts.shape} != {cur_counts.shape}"
        )

    n_bins = ref_counts.shape[0]
    ref_total = ref_counts.sum()
    cur_total = cur_counts.sum()

    ref_props = (ref_counts + epsilon) / (ref_total + epsilon * n_bins)
    cur_props = (cur_counts + epsilon) / (cur_total + epsilon * n_bins)

    return float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))


def psi_from_samples(
    reference: np.ndarray,
    current: np.ndarray,
    bin_edges: Sequence[float],
    epsilon: float = 1e-4,
) -> float:
    """Compute PSI between two raw samples, binned per `bin_edges` (the
    binning grid is decided by the caller -- this module makes NO binning
    decisions, only the final PSI calculation).

    Returns 0.0 if either sample is empty (not enough signal for a
    meaningful PSI), matching both original implementations.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    if len(reference) == 0 or len(current) == 0:
        return 0.0

    edges = np.asarray(bin_edges, dtype=float)
    if len(edges) < 2:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    return psi_from_counts(ref_counts, cur_counts, epsilon=epsilon)
