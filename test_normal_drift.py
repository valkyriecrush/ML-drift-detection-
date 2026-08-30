"""
test_normal_drift.py -- "normal_drift" scenario via the REAL
NormalDriftSimulator from "drift_simulator" (real LightGBM model, real
pipeline). Note: the simulator classifies this scenario as "WARNING" under
ITS OWN PSI convention (warning_low=0.05/critical_high=0.25) -- our agent
(agentic_drift_stress) has its OWN PSI implementation and severity
thresholds, so the resulting classification can legitimately differ (both
statuses are printed below for comparison).

Usage: python3 test_normal_drift.py
"""

import logging

logging.basicConfig(level=logging.WARNING)

from run_scenario import run_scenario  # noqa: E402

if __name__ == "__main__":
    result = run_scenario("normal_drift", model_id="pima_normal_drift")

    print("\n--- Check (expected vs actual) ---")
    print(f"drift_simulator native status: {result['native_simulator_status']} (expected: WARNING)")
    print(f"Our agent's severity          : {result['data_drift_severity']}")
