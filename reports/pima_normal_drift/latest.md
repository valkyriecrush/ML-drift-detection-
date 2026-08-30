# Diagnostic report -- `pima_normal_drift`

**Overall status: 🔴 CRITICAL**
Generated at 2026-08-30T18:05:39.492985+00:00 | Cycle (most recent signal): 2026-08-30T18:05:39.485989+00:00

## Executive summary

Data drift: 2/8 feature(s) drifted (severity critical). Target drift (confirmed): PSI=0.000, severity none. Performance: severity critical, cause = model_weights. Alerts: 3 critical, 1 warning. 3 remediation action(s) pending human approval.

## 🔴 Probable root cause identified (cross-agent corroboration)

**Probable root cause identified for the performance degradation**

Performance degradation (model_weights) coincides with confirmed drift on Glucose, feature(s) with direct SHAP impact: likely causal link.

**Recommended action:** Prioritize fixing (data or pipeline) Glucose before any blind retrain -- retraining without fixing the cause would reproduce the same problem next cycle.

## 1. Data / Feature Drift

Overall severity: **critical** -- 2/8 feature(s) drifted (25.0%).

| Feature | Type | PSI | Wasserstein (norm) | Severity | Drifted |
|---|---|---|---|---|---|
| Pregnancies | numeric | 0.0005 | 0.0052 | none | no |
| Glucose | numeric | 0.1022 | 0.2253 | critical | yes |
| BloodPressure | numeric | 0.0003 | 0.0095 | none | no |
| SkinThickness | numeric | 0.0003 | 0.0080 | none | no |
| Insulin | numeric | 0.0004 | 0.0276 | none | no |
| BMI | numeric | 0.1182 | 0.2406 | critical | yes |
| DiabetesPedigreeFunction | numeric | 0.0004 | 0.0097 | none | no |
| Age | numeric | 0.0005 | 0.0025 | none | no |

## 2. Target Drift

Status: **confirmed** | n_samples = 740

PSI = 0.0000 | severity = **none** | drifted = no
Wasserstein distance (normalized) = 0.0267
Reference mean = 0.3490 -> current mean = 0.3757 (reference std = 0.4766 -> current std = 0.4843)

## 3. Performance Drift

Status: **labeled** | severity: **critical** | remediation_type: `model_weights` | window = 500 samples
y_proba coverage over the window: 100.0%

### Performance metrics

| Metric | Baseline | Current | Delta | Std score | Degraded | Contributes to severity |
|---|---|---|---|---|---|---|
| accuracy | 0.9349 | 0.9060 | -0.0289 | -3.09 | yes | yes |
| precision | 0.9290 | 0.7180 | -0.2110 | -21.21 | yes | yes |
| recall | 0.9275 | 0.9129 | -0.0146 | -1.51 | no | yes |
| f1 | 0.9282 | 0.7725 | -0.1557 | -16.77 | yes | yes |
| recall_class_1 | 0.9030 | 0.9211 | +0.0181 | +1.41 | no | yes |
| auc_roc | 0.9829 | 0.9469 | -0.0361 | -3.67 | yes | yes |
| auc_pr | 0.9701 | 0.6857 | -0.2844 | -29.32 | yes | yes |

### Calibration metrics

| Metric | Baseline | Current | Delta | Std score | Degraded | Contributes to severity |
|---|---|---|---|---|---|---|
| log_loss | 0.1954 | 0.2695 | +0.0741 | +7.08 | yes | yes |
| brier_score | 0.1041 | 0.1527 | +0.0487 | +7.46 | yes | yes |
| ece | 0.0591 | 0.0405 | -0.0187 | -3.40 | no | yes |

## Alerts raised this cycle (4)

### 🔴 [CRITICAL] High model-impact drift on 1 feature(s) _(source: data_drift)_

Affected features: Glucose.  
- Glucose: 'Glucose' is at the 88th percentile of SHAP importance among tracked features (threshold=75th, raw importance=0.5135): the drift directly affects predictions.

**Recommended action (expert):** FINAL VERDICT -- retrain: batch is clean (no missing-value spike, no new category) on the drifted feature(s), so this is a genuine population shift, not a data-quality artifact. Prioritize Glucose (direct model impact per SHAP). Check for a root_cause alert this cycle before scheduling the retrain.

_Remediation type: `retrain`_

### 🔴 [CRITICAL] Real model performance degradation _(source: performance_drift)_

Degraded metrics this window: accuracy, precision, f1, auc_roc, auc_pr.

**Recommended action (expert):** Retrain recommended. Check first whether a root_cause alert was raised this cycle: if so, fix that feature/pipeline before retraining, or the same issue will recur.

_Remediation type: `retrain`_

### 🔴 [CRITICAL] Probable root cause identified for the performance degradation _(source: root_cause)_

Performance degradation (model_weights) coincides with confirmed drift on Glucose, feature(s) with direct SHAP impact: likely causal link.

**Recommended action (expert):** Prioritize fixing (data or pipeline) Glucose before any blind retrain -- retraining without fixing the cause would reproduce the same problem next cycle.

_Remediation type: `retrain`_

### 🟡 [WARNING] Drift to monitor on 1 feature(s) _(source: data_drift)_

Affected features: BMI.  
- BMI: 'BMI' has low SHAP importance (percentile 50) and isn't correlated with an influential feature: likely negligible model impact despite the statistical drift.

**Recommended action (expert):** No immediate action: risk is indirect (correlated with a key feature) or undetermined (missing SHAP/correlation data). Re-evaluate next cycle.

_Remediation type: `monitor`_

## Remediation actions (4)

| Type | Urgency (SLA) | Status | Approval required | Automatable | Source alert |
|---|---|---|---|---|---|
| retrain | urgent (within 24h) -- superseded by 'Probable root cause identified for the performance degradation' | skipped | yes | yes | High model-impact drift on 1 feature(s) |
| monitor | normal (next working cycle) | executed | no | yes | Drift to monitor on 1 feature(s) |
| retrain | urgent (within 24h) -- superseded by 'Probable root cause identified for the performance degradation' | skipped | yes | yes | Real model performance degradation |
| retrain | urgent (within 24h) | pending | yes | yes | Probable root cause identified for the performance degradation |
