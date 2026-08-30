# Diagnostic report -- `pima_no_drift`

**Overall status: 🟢 HEALTHY**
Generated at 2026-08-30T18:06:20.144535+00:00 | Cycle (most recent signal): 2026-08-30T18:06:20.140535+00:00

## Executive summary

Data drift: 0/8 feature(s) drifted (severity none). Target drift (confirmed): PSI=0.000, severity none. Performance: severity none. No alerts raised this cycle.

## 1. Data / Feature Drift

Overall severity: **none** -- 0/8 feature(s) drifted (0.0%).

| Feature | Type | PSI | Wasserstein (norm) | Severity | Drifted |
|---|---|---|---|---|---|
| Pregnancies | numeric | 0.0000 | 0.0000 | none | no |
| Glucose | numeric | 0.0000 | 0.0000 | none | no |
| BloodPressure | numeric | 0.0000 | 0.0000 | none | no |
| SkinThickness | numeric | 0.0000 | 0.0000 | none | no |
| Insulin | numeric | 0.0000 | 0.0000 | none | no |
| BMI | numeric | 0.0000 | 0.0000 | none | no |
| DiabetesPedigreeFunction | numeric | 0.0000 | 0.0000 | none | no |
| Age | numeric | 0.0000 | 0.0000 | none | no |

## 2. Target Drift

Status: **confirmed** | n_samples = 768

PSI = 0.0000 | severity = **none** | drifted = no
Wasserstein distance (normalized) = 0.0000
Reference mean = 0.3490 -> current mean = 0.3490 (reference std = 0.4766 -> current std = 0.4766)

## 3. Performance Drift

Status: **labeled** | severity: **none** | remediation_type: `None` | window = 500 samples
y_proba coverage over the window: 100.0%

### Performance metrics

| Metric | Baseline | Current | Delta | Std score | Degraded | Contributes to severity |
|---|---|---|---|---|---|---|
| accuracy | 0.9349 | 0.9400 | +0.0051 | +0.55 | no | yes |
| precision | 0.9290 | 0.9303 | +0.0013 | +0.13 | no | yes |
| recall | 0.9275 | 0.9354 | +0.0079 | +0.82 | no | yes |
| f1 | 0.9282 | 0.9328 | +0.0045 | +0.49 | no | yes |
| recall_class_1 | 0.9030 | 0.9217 | +0.0187 | +1.46 | no | yes |
| auc_roc | 0.9829 | 0.9846 | +0.0017 | +0.17 | no | yes |
| auc_pr | 0.9701 | 0.9701 | -0.0000 | -0.00 | no | yes |

### Calibration metrics

| Metric | Baseline | Current | Delta | Std score | Degraded | Contributes to severity |
|---|---|---|---|---|---|---|
| log_loss | 0.1954 | 0.1916 | -0.0038 | -0.37 | no | yes |
| brier_score | 0.1041 | 0.1006 | -0.0035 | -0.53 | no | yes |
| ece | 0.0591 | 0.0664 | +0.0072 | +1.32 | no | yes |

## Alerts raised this cycle

None.

## Remediation actions

None generated this cycle.
