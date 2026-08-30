# Diagnostic report -- `pima_severe_drift`

**Overall status: 🚨 PAGE**
Generated at 2026-08-30T18:04:08.799241+00:00 | Cycle (most recent signal): 2026-08-30T18:04:08.789240+00:00

## Executive summary

Data drift: 5/8 feature(s) drifted (severity critical). Target drift (confirmed): PSI=0.000, severity warning. Performance: severity critical, cause = model_weights. Alerts: 3 critical, 1 page, 2 warning. 2 remediation action(s) pending human approval.

## 🔴 Probable root cause identified (cross-agent corroboration)

**Probable root cause identified for the performance degradation**

Performance degradation (model_weights) coincides with confirmed drift on Glucose, SkinThickness, Insulin, feature(s) with direct SHAP impact: likely causal link.

**Recommended action:** Prioritize fixing (data or pipeline) Glucose, SkinThickness, Insulin before any blind retrain -- retraining without fixing the cause would reproduce the same problem next cycle.

## 1. Data / Feature Drift

Overall severity: **critical** -- 5/8 feature(s) drifted (62.5%).

| Feature | Type | PSI | Wasserstein (norm) | Severity | Drifted | Quality |
|---|---|---|---|---|---|---|
| Pregnancies | numeric | 0.0090 | 0.0174 | none | no | ok |
| Glucose | numeric | 2.0510 | 1.5349 | critical | yes | ok |
| BloodPressure | numeric | 3.7542 | 0.8923 | critical | yes | ok |
| SkinThickness | numeric | 0.0339 | 6.9195 | warning | yes | ok |
| Insulin | numeric | 10.6260 | 2.3314 | critical | yes | ⚠️ 29.9% missing |
| BMI | numeric | 1.0948 | 0.9875 | critical | yes | ok |
| DiabetesPedigreeFunction | numeric | 0.0074 | 0.0273 | none | no | ok |
| Age | numeric | 0.0122 | 0.0200 | none | no | ok |

## 2. Target Drift

Status: **confirmed** | n_samples = 768

PSI = 0.0000 | severity = **warning** | drifted = yes
Wasserstein distance (normalized) = 0.3008
Reference mean = 0.3490 -> current mean = 0.6497 (reference std = 0.4766 -> current std = 0.4771)

## 3. Performance Drift

Status: **labeled** | severity: **critical** | remediation_type: `model_weights` | window = 500 samples
y_proba coverage over the window: 100.0%

### Performance metrics

| Metric | Baseline | Current | Delta | Std score | Degraded | Contributes to severity |
|---|---|---|---|---|---|---|
| accuracy | 0.9349 | 0.5860 | -0.3489 | -37.32 | yes | yes |
| precision | 0.9290 | 0.5888 | -0.3402 | -34.20 | yes | yes |
| recall | 0.9275 | 0.5889 | -0.3385 | -35.02 | yes | yes |
| f1 | 0.9282 | 0.5860 | -0.3422 | -36.87 | yes | yes |
| recall_class_1 | 0.9030 | 0.6277 | -0.2753 | -21.46 | yes | yes |
| auc_roc | 0.9829 | 0.6531 | -0.3298 | -33.55 | yes | yes |
| auc_pr | 0.9701 | 0.5978 | -0.3723 | -38.38 | yes | yes |

### Calibration metrics

| Metric | Baseline | Current | Delta | Std score | Degraded | Contributes to severity |
|---|---|---|---|---|---|---|
| log_loss | 0.1954 | 0.7391 | +0.5436 | +51.91 | yes | yes |
| brier_score | 0.1041 | 0.5238 | +0.4197 | +64.31 | yes | yes |
| ece | 0.0591 | 0.1784 | +0.1193 | +21.71 | yes | yes |

## Alerts raised this cycle (6)

### 🚨 [PAGE] Recall collapse on critical class (recall_class_1) _(source: performance_drift)_

recall_class_1 = 0.628 (baseline 0.903): below the absolute floor.

**Recommended action (expert):** Escalate immediately, outside the standard cycle: possible impact on a sensitive business class. Check whether a rollback is viable, notify the business owner, don't wait for the next scheduled retrain cycle.

_Remediation type: `escalate`_

### 🔴 [CRITICAL] High model-impact drift on 3 feature(s) _(source: data_drift)_

Affected features: Glucose, SkinThickness, Insulin.  
- Glucose: 'Glucose' is at the 88th percentile of SHAP importance among tracked features (threshold=75th, raw importance=0.5135): the drift directly affects predictions.  
- SkinThickness: 'SkinThickness' is at the 75th percentile of SHAP importance among tracked features (threshold=75th, raw importance=0.2962): the drift directly affects predictions.  
- Insulin: 'Insulin' is at the 100th percentile of SHAP importance among tracked features (threshold=75th, raw importance=1.5079): the drift directly affects predictions.

**Recommended action (expert):** FINAL VERDICT -- pipeline issue confirmed, do NOT retrain yet: Insulin: 29.9% missing values this batch. Fix ingestion for Insulin first (schema, upstream source, mapping of new categories), then re-run this cycle on clean data before deciding whether real drift remains.

_Remediation type: `data_pipeline_investigation`_

### 🔴 [CRITICAL] Real model performance degradation _(source: performance_drift)_

Degraded metrics this window: accuracy, precision, recall, f1, recall_class_1, auc_roc, auc_pr.

**Recommended action (expert):** Retrain recommended. Check first whether a root_cause alert was raised this cycle: if so, fix that feature/pipeline before retraining, or the same issue will recur.

_Remediation type: `retrain`_

### 🔴 [CRITICAL] Probable root cause identified for the performance degradation _(source: root_cause)_

Performance degradation (model_weights) coincides with confirmed drift on Glucose, SkinThickness, Insulin, feature(s) with direct SHAP impact: likely causal link.

**Recommended action (expert):** Prioritize fixing (data or pipeline) Glucose, SkinThickness, Insulin before any blind retrain -- retraining without fixing the cause would reproduce the same problem next cycle.

_Remediation type: `retrain`_

### 🟡 [WARNING] Drift to monitor on 2 feature(s) _(source: data_drift)_

Affected features: BloodPressure, BMI.  
- BloodPressure: 'BloodPressure' has low SHAP importance (percentile 25) and isn't correlated with an influential feature: likely negligible model impact despite the statistical drift.  
- BMI: 'BMI' has low SHAP importance (percentile 50) and isn't correlated with an influential feature: likely negligible model impact despite the statistical drift.

**Recommended action (expert):** No immediate action: risk is indirect (correlated with a key feature) or undetermined (missing SHAP/correlation data). Re-evaluate next cycle.

_Remediation type: `monitor`_

### 🟡 [WARNING] Target drift confirmed (true labels) _(source: target_drift)_

The real target distribution has drifted (PSI=0.000, severity=warning).

**Recommended action (expert):** Monitor next cycles; no retrain yet, PSI stays below the critical threshold.

_Remediation type: `monitor`_

## Remediation actions (6)

| Type | Urgency (SLA) | Status | Approval required | Automatable | Source alert |
|---|---|---|---|---|---|
| data_pipeline_investigation | urgent (within 24h) | executed | no | yes | High model-impact drift on 3 feature(s) |
| monitor | normal (next working cycle) | executed | no | yes | Drift to monitor on 2 feature(s) |
| monitor | normal (next working cycle) | executed | no | yes | Target drift confirmed (true labels) |
| escalate | immediate (act now -- business-critical class impacted, do not wait for the next cycle) | executed | no | yes | Recall collapse on critical class (recall_class_1) |
| retrain | urgent (within 24h) -- superseded by 'Probable root cause identified for the performance degradation' | skipped | yes | yes | Real model performance degradation |
| retrain | urgent (within 24h) | pending | yes | yes | Probable root cause identified for the performance degradation |
