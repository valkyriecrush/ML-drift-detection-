"""
test_no_drift.py -- "no_drift" scenario via the REAL NoDriftSimulator from
"drift_simulator" (real LightGBM model, real pipeline). Expected status on
our agent: data drift severity = NONE/LOW, no alerts.

Usage: python3 test_no_drift.py
"""

import logging

logging.basicConfig(level=logging.WARNING)

from run_scenario import run_scenario  # noqa: E402

if __name__ == "__main__":
    result = run_scenario("no_drift", model_id="pima_no_drift")

    print("\n--- Check (expected vs actual) ---")
    ok = result["data_drift_severity"] in ("DriftSeverity.NONE", "DriftSeverity.LOW") and result["n_alerts"] == 0
    print("Status:", "OK (matches expectation)" if ok else "MISMATCH -- see details above")
