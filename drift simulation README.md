# Drift Simulator — MLOps Stress-Testing Framework

A modular Python framework designed to simulate and evaluate **Data Drift**, **Target Drift**, and **Concept Drift** against a production-trained machine learning baseline.

It provides reproducible stress‑testing pipelines using fixed baseline bins, schema validation, and configurable drift scenarios to ensure models meet strict SLA limits before deployment.

---

## 🏗️ Repository Architecture

```plaintext
drift_simulator/
├── config/                         # Configuration & Data Schemas
│   ├── baseline_config.yml          # Core drift thresholds, seeds, & pipeline parameters
│   ├── diabetes_schema.yaml         # Raw dataset schema validation rules
│   ├── diabetes_cleaned.yaml        # Cleaned dataset specifications
│   ├── diabetes_features.yaml       # Feature engineering schema definitions
│   ├── performance.yaml             # Model performance baseline & feature alignment
│   ├── X_test.yaml                  # Feature test distribution schemas
│   └── Y_test.yaml                  # Target test distribution schemas
│
├── baseline/                       # Baseline Computing & Schema Engine
│   ├── baseline_calculator.py       # Computes fixed PSI bin edges & statistical baselines
│   ├── schema_validator.py         # Validates data batches against config schemas
│   └── .cache/                     # Local disk cache for computed baselines
│       └── baseline_stats_v2.json  # Cached baseline stats & fixed bin grids
│
├── drift/                          # Drift Simulation Engines
│   ├── base_drift_simulator.py     # Base class (OOP) handling pipelines & PSI metrics
│   ├── no_drift/                   # Control group simulator
│   │   └── no_drift_simulator.py
│   ├── normal_drift/               # Moderate / Expected drift simulator
│   │   └── normal_drift_simulator.py
│   └── severe_drift/               # Extreme / Stress-test drift simulator
│       └── severe_drift_simulator.py
│
├── run_drift.py                    # Entry point CLI script to execute simulations
├── requirements.txt                # Third-party dependencies
└── *_results.json                  # Output evaluation reports (No Drift, Normal, Severe)

text

---

## 💡 Core Concepts & System Design

- **Fixed Bin Grid (Baseline Cache)**: Baseline statistics and PSI bin boundaries are computed once and stored in `.cache/baseline_stats_v2.json`. This ensures PSI calculations across all drift runs are evaluated against identical reference bins.

- **Modular Drift Simulators**: Extends `BaseDriftSimulator` into three distinct simulation strategies:
  - **No Drift** – Evaluates model stability on uncorrupted baseline distributions.
  - **Normal Drift** – Injects mild, realistic feature and target shifts to verify system tolerance (`0.10 ≤ PSI < 0.25`).
  - **Severe Drift** – Applies heavy statistical perturbations, noise, and missing values to stress‑test system SLA boundaries (`PSI ≥ 0.25`).

- **Schema Validation Engine**: `SchemaValidator` enforces non‑blocking schema checks across pipeline steps against rules defined in `config/*.yaml`.

- **Reproducible Perturbations**: All drift primitives use deterministic, step‑specific random seeds (`RandomState`) to guarantee complete test reproducibility.

---

## ⚡ Quickstart

### 1. Installation

```bash
git clone <repository-url>


python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
2. Running Simulations
Execute the full drift suite via run_drift.py:

bash
# Run all drift scenarios (No Drift, Normal Drift, Severe Drift)
python run_drift.py
⚙️ Configuration (config/baseline_config.yml)
Adjust simulation severity, feature multipliers, and PSI SLA thresholds directly in the configuration file:

yaml
normal_drift:
  features:
    Glucose: { std_multiplier: 0.55, mask_ratio: 0.5 }
    BMI: { std_multiplier: 0.55, mask_ratio: 0.5 }
  drift_status_thresholds:
    warning_low: 0.10
    critical_high: 0.25
  output:
    report_path: "normal_drift_results.json"
📊 Evaluation Output
Simulation executions output JSON reports containing:

PSI Checkpoints – Evaluated at feature drift, target drift, data quality, and post‑pipeline stages.

SLA Validation – Status check determining whether drift metrics met expected severity thresholds.

Schema Warnings – Structural or type mismatches encountered during pipeline execution.



