# Purpose: CLI entry point that runs the no_drift / normal_drift / severe_drift scenarios.

from __future__ import annotations

import argparse
import json
import sys

import yaml

from baseline.baseline_calculator import BaselineCalculator
from drift.no_drift.no_drift_simulator import NoDriftSimulator
from drift.normal_drift.normal_drift_simulator import NormalDriftSimulator
from drift.severe_drift.severe_drift_simulator import SevereDriftSimulator


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError(f"'{config_path}' is empty or invalid YAML.")
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="config/baseline_config.yml",
        help="Path to the .yml config (single source of truth for features/metrics).",
    )
    parser.add_argument(
        "--scenario",
        choices=["no_drift", "normal", "severe", "all"],
        default="all",
        help="Which scenario to run (default: all).",
    )
    parser.add_argument(
        "--severity",
        choices=["severe", "extreme"],
        default="severe",
        help="Severity preset for the 'severe' scenario (config/baseline_config.yml -> severity_presets).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_path = cfg["baseline"]["model_path"]

    bc = BaselineCalculator(config_path=args.config)
    bc.load_or_compute()
    if bc.schema_warnings:
        print(f"[schema_warnings] {bc.schema_warnings}", file=sys.stderr)

    results = {}

    if args.scenario in ("no_drift", "all"):
        sim = NoDriftSimulator(baseline_calc=bc)
        results["no_drift"] = sim.simulate_no_drift_scenario(config=cfg, model_path=model_path)
        print(f"[no_drift]  status={results['no_drift']['overall_drift_status']['status']}")

    if args.scenario in ("normal", "all"):
        sim = NormalDriftSimulator(baseline_calc=bc)
        results["normal_drift"] = sim.run(config=cfg, model_path=model_path)
        print(f"[normal]    status={results['normal_drift']['overall_drift_status']['status']}")

    if args.scenario in ("severe", "all"):
        sim = SevereDriftSimulator(baseline_calc=bc)
        results["severe_drift"] = sim.simulate_severe_scenario(
            severity=args.severity, config=cfg, model_path=model_path
        )
        print(f"[severe]    sla_met={results['severe_drift']['sla_check']['meets_expected_severity']}")

    print(json.dumps({k: v.get("_exported_to", "(not exported)") for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
