"""
conftest.py -- shared fixtures for the pytest suite.

Before this, the repo's only "tests" were 3 smoke scripts
(`test_no_drift.py` etc., 20-27 lines, a single informal assertion, no
framework). This conftest provides the sys.path bootstrap needed to test
each sub-project IN ISOLATION (without depending on `run_scenario.py`'s
import order), with `tmp_path` fixtures so no global state is shared
between runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
AGENTIC_DIR = ROOT / "agentic_drift_stress"
DRIFT_SIMULATOR_DIR = ROOT / "drift_simulator"
BRIDGE_DIR = ROOT / "bridge"

for p in (ROOT, AGENTIC_DIR, DRIFT_SIMULATOR_DIR, BRIDGE_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture
def chdir_drift_simulator(monkeypatch):
    """Many drift_simulator modules (config/, data/, models/,
    baseline/.cache) use CWD-relative paths -- exactly like running
    `python run_drift.py` from `drift_simulator/`."""
    monkeypatch.chdir(DRIFT_SIMULATOR_DIR)
    yield DRIFT_SIMULATOR_DIR
