"""
test_severe_drift.py -- "severe_drift" scenario via the REAL
SevereDriftSimulator from "drift_simulator" (real LightGBM model, real
pipeline). Expected status on our agent: data drift severity = CRITICAL,
data_drift + performance_drift alerts, email sent (AlertPriority.CRITICAL
routes to EMAIL by default).

Usage:
    python3 test_severe_drift.py
    # To also exercise the mock HTTP path (email + slack):
    #   uvicorn mock_alert_server:app --port 8000   (separate terminal)
    #   python3 test_severe_drift.py
"""

import logging

logging.basicConfig(level=logging.WARNING)

from run_scenario import run_scenario  # noqa: E402

if __name__ == "__main__":
    result = run_scenario("severe_drift", model_id="pima_severe_drift", severity="severe")

    print("\n--- Check (expected vs actual) ---")
    ok = result["data_drift_severity"] == "DriftSeverity.CRITICAL" and result["n_alerts"] > 0
    print("Status:", "OK (matches expectation)" if ok else "MISMATCH -- see details above")
    print("CRITICAL alerts -> should appear in email_alert_log.jsonl (project root).")
