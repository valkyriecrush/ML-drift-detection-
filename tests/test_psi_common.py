"""
Regression tests for psi_common.py (unifying the two previously divergent
PSI implementations -- see the module docstring).
"""

import numpy as np
import pytest

from psi_common import psi_from_counts, psi_from_samples


def test_psi_zero_when_distributions_identical():
    counts = [100, 200, 150, 50]
    assert psi_from_counts(counts, counts) == pytest.approx(0.0, abs=1e-9)


def test_psi_positive_when_distributions_differ():
    ref = [100, 100, 100, 100]
    cur = [10, 190, 10, 190]
    psi = psi_from_counts(ref, cur)
    assert psi > 0


def test_psi_known_value_two_bins():
    # Two bins, a simple proportion-mass shift: checks the formula itself
    # (not just its sign), with a negligible epsilon.
    ref = [500, 500]
    cur = [900, 100]
    eps = 1e-4
    ref_props = np.array([(500 + eps) / (1000 + 2 * eps)] * 2)
    cur_props = np.array([(900 + eps) / (1000 + 2 * eps), (100 + eps) / (1000 + 2 * eps)])
    expected = float(np.sum((cur_props - ref_props) * np.log(cur_props / ref_props)))
    assert psi_from_counts(ref, cur, epsilon=eps) == pytest.approx(expected, rel=1e-9)


def test_psi_mismatched_shapes_raises():
    with pytest.raises(ValueError):
        psi_from_counts([1, 2, 3], [1, 2])


def test_psi_from_samples_empty_returns_zero():
    assert psi_from_samples(np.array([]), np.array([1.0, 2.0]), bin_edges=[-np.inf, 0, np.inf]) == 0.0
    assert psi_from_samples(np.array([1.0]), np.array([]), bin_edges=[-np.inf, 0, np.inf]) == 0.0


def test_psi_from_samples_matches_psi_from_counts():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=500)
    current = rng.normal(loc=0.5, size=500)
    edges = np.quantile(reference, np.linspace(0, 1, 11))
    edges[0], edges[-1] = -np.inf, np.inf

    via_samples = psi_from_samples(reference, current, bin_edges=edges)

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    via_counts = psi_from_counts(ref_counts, cur_counts)

    assert via_samples == pytest.approx(via_counts, rel=1e-9)


def test_baseline_calculator_and_drift_detector_agree_on_same_bins_and_counts():
    """Key regression check: before the fix, DriftDetector and
    BaselineCalculator produced DIFFERENT PSI values for the same counts
    (two distinct epsilon conventions). This test checks both now delegate
    to the same formula."""
    from baseline.baseline_calculator import BaselineCalculator  # noqa: F401  (import path sanity)
    import inspect
    import drift_detector

    # Both modules must import the SAME psi_from_counts function (object
    # identity, not just a numeric result that happens to match) --
    # guarantees a future epsilon-convention change propagates to both.
    import baseline.baseline_calculator as bc_module

    assert bc_module.psi_from_counts is drift_detector.psi_from_counts
