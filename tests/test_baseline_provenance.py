"""
Regression tests for the baseline-provenance fix (cache silently
regenerated on a hash mismatch).
"""

import json

import pytest


@pytest.mark.usefixtures("chdir_drift_simulator")
def test_hash_mismatch_with_equivalent_stats_is_classified_and_logged(chdir_drift_simulator, caplog):
    from baseline.baseline_calculator import BaselineCalculator

    cache_path = chdir_drift_simulator / "baseline" / ".cache" / "baseline_stats_v2.json"
    report_path = chdir_drift_simulator / "baseline" / ".cache" / "hash_mismatch_report.jsonl"
    report_path.unlink(missing_ok=True)

    original_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    original_hash = original_cache["baseline_reference"]["hash"]

    tampered = dict(original_cache)
    tampered["baseline_reference"] = dict(original_cache["baseline_reference"])
    tampered["baseline_reference"]["hash"] = "not_the_real_hash"
    cache_path.write_text(json.dumps(tampered), encoding="utf-8")

    try:
        import logging

        with caplog.at_level(logging.WARNING):
            bc = BaselineCalculator(config_path="config/baseline_config.yml")
            bc.load_or_compute()

        assert any("Baseline hash differs" in rec.message for rec in caplog.records)
        assert report_path.exists()
        entry = json.loads(report_path.read_text(encoding="utf-8").strip().splitlines()[-1])
        assert entry["classified_as"] == "statistically_equivalent_mirror"

        # Self-healing: the cache must be rewritten with the REAL hash,
        # not stay stuck on the tampered one.
        healed_cache = json.loads(cache_path.read_text(encoding="utf-8"))
        assert healed_cache["baseline_reference"]["hash"] == original_hash
    finally:
        report_path.unlink(missing_ok=True)


@pytest.mark.usefixtures("chdir_drift_simulator")
def test_matching_hash_loads_from_cache_without_warning(chdir_drift_simulator, caplog):
    from baseline.baseline_calculator import BaselineCalculator
    import logging

    with caplog.at_level(logging.WARNING):
        bc = BaselineCalculator(config_path="config/baseline_config.yml")
        bc.load_or_compute()

    assert not any("Baseline hash differs" in rec.message for rec in caplog.records)
