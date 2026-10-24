# Phase 2 model evaluation

**Synthetic research evaluation only.** All metrics below were computed from held-out predictions. Threshold is fixed at 0.50; no test-driven threshold or hyperparameter tuning.

Model A: L2 logistic regression with sigmoid calibration; multi-output ridge trajectory regression. Model B: sigmoid-calibrated histogram gradient boosting; eight separate histogram gradient boosting regressors.

## Table 1 — patient-level split sensitivity

| Split | Model | Fit/cal/test patients | Test windows | AUROC | AUPRC (AP) | Brier | Trajectory RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| 60-40 | A | 20/4/16 | 17,873 | 0.953 | 0.812 | 0.0572 | 8.51 |
| 60-40 | B | 20/4/16 | 17,873 | 0.993 | 0.964 | 0.0214 | 9.20 |
| 70-30 | A | 23/5/12 | 13,353 | 0.975 | 0.918 | 0.0424 | 9.12 |
| 70-30 | B | 23/5/12 | 13,353 | 0.989 | 0.956 | 0.0299 | 10.08 |
| 80-20 | A | 27/5/8 | 8,883 | 0.978 | 0.925 | 0.0675 | 9.38 |
| 80-20 | B | 27/5/8 | 8,883 | 0.993 | 0.972 | 0.0230 | 10.16 |
| 90-10 | A | 30/6/4 | 4,415 | 0.980 | 0.969 | 0.0469 | 9.72 |
| 90-10 | B | 30/6/4 | 4,415 | 0.993 | 0.986 | 0.0292 | 11.68 |

Fit/cal/test counts exclude calibration patients from estimator and scaler fitting. Outer training totals are 24, 28, 32 and 36. Calibration uses ceil(15% of outer training patients), so the exact fractions are 16.7%, 17.9%, 15.6% and 16.7%.

## Table 2 — detailed primary and four-archetype evaluation

| Split | Model | Recall | PPV | Specificity | FPR | F1 | AUROC | AUPRC (AP) | Brier | Mean lead (min) | RMSE | MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 80-20 | A | 0.441 | 0.998 | 1.000 | 0.000 | 0.612 | 0.978 | 0.925 | 0.0675 | 20.8 | 9.38 | 6.73 |
| 80-20 | B | 0.911 | 0.904 | 0.982 | 0.018 | 0.908 | 0.993 | 0.972 | 0.0230 | 47.5 | 10.16 | 7.08 |
| 90-10 | A | 0.803 | 0.983 | 0.994 | 0.006 | 0.884 | 0.980 | 0.969 | 0.0469 | 27.6 | 9.72 | 6.65 |
| 90-10 | B | 0.886 | 0.963 | 0.986 | 0.014 | 0.923 | 0.993 | 0.986 | 0.0292 | 28.8 | 11.68 | 8.06 |

Recall, PPV, specificity, FPR, F1, AUROC and AUPRC are fractions. AUPRC is scikit-learn average precision (non-interpolated AP), not trapezoidal PR area. RMSE/MAE pool eight horizons and are in mg/dL. Zero-division precision/recall/F1 are reported as 0; AUROC is unavailable for a single-class subgroup.

Mean lead time averages the earliest qualifying alert 15–120 minutes before each detected upward crossing, after the previous hyperglycemic reading. It excludes undetected crossings, so interpret it with event detection coverage below. No lead is assigned to already-high readings. Crossings use raw unsmoothed CGM and can include threshold jitter.

## Model selection without test reuse

Primary deployment regime: **80-20** (chosen before evaluation). Selected classifier: **Model B** by lowest internal-selection Brier. Selected trajectory regressor: **Model B** by lowest internal-selection RMSE. No claim is made that either is globally optimal.

For each regime, the outer training patients are partitioned into fit and calibration groups. An internal selection group is temporarily removed from the fit group. Candidate bases/scalers are trained on the remaining patients and calibrated on the calibration patients, then assessed on the independent selection patients. After selecting model families, bases are refitted on all fit patients and calibrated on the same untouched calibration group. Hold-outs never influence selection. The exported bundle uses the primary regime; 90:10 is only sensitivity analysis.

| Split | Family | Internal selection Brier | Internal selection RMSE |
|---|---|---:|---:|
| 60-40 | A | 0.0074 | 10.42 |
| 60-40 | B | 0.0022 | 7.66 |
| 70-30 | A | 0.0636 | 9.26 |
| 70-30 | B | 0.0363 | 8.85 |
| 80-20 | A | 0.0525 | 8.40 |
| 80-20 | B | 0.0494 | 7.97 |
| 90-10 | A | 0.0130 | 8.05 |
| 90-10 | B | 0.0117 | 7.83 |

## Crossings, post-meal subsets and persistence reference

| Split | Model | Test prevalence | Crossing detection | Mean lead | New-crossing recall | Post-meal recall | Persistence RMSE |
|---|---|---:|---:|---:|---:|---:|---:|
| 60-40 | A | 0.127 | 76/121 | 38.7 | 0.285 | 0.525 | 9.67 |
| 60-40 | B | 0.127 | 63/121 | 43.8 | 0.264 | 0.792 | 9.67 |
| 70-30 | A | 0.170 | 97/121 | 46.9 | 0.504 | 0.842 | 9.87 |
| 70-30 | B | 0.170 | 74/121 | 50.3 | 0.367 | 0.842 | 9.87 |
| 80-20 | A | 0.160 | 13/54 | 20.8 | 0.069 | 0.470 | 10.62 |
| 80-20 | B | 0.160 | 48/54 | 47.5 | 0.519 | 0.818 | 10.62 |
| 90-10 | A | 0.286 | 19/36 | 27.6 | 0.192 | 0.804 | 12.12 |
| 90-10 | B | 0.286 | 25/36 | 28.8 | 0.242 | 0.786 | 12.12 |

## Patient-cluster uncertainty (95% percentile intervals)

| Split | Model | Recall interval | Brier interval | RMSE interval |
|---|---|---|---|---|
| 80-20 | A | 0.113–0.504 | 0.011–0.148 | 8.104–10.743 |
| 80-20 | B | 0.579–0.998 | 0.002–0.051 | 7.624–12.918 |
| 90-10 | A | 0.503–0.854 | 0.000–0.093 | 7.756–11.536 |
| 90-10 | B | 0.536–0.945 | 0.000–0.061 | 7.257–15.764 |

300 resamples draw whole patients with replacement, retaining all their windows. Recall resamples without positives are omitted and their valid count is stored in metrics JSON. Eight or four test patients are far too few for dependable population confidence bounds; these intervals are descriptive and can be unstable or degenerate.

## Safety, causality and data quality

Input rows: 53,760. Causally usable predictor windows: 49,353. Complete clean future targets: 48,613. Jointly eligible training/evaluation windows: 44,684.

- Invalid status or CGM outside 40–400 mg/dL is never used as a glucose observation. Corrupt values are quarantined. A first missing packet may be extrapolated from two valid immediately preceding raw values, only within 40–400 mg/dL and with absolute prior slope ≤3 mg/dL/min. These engineering bounds are not clinical validation.
- No recursive extrapolation. A second consecutive invalid sample locks prediction immediately. The first estimate is never retroactively reclassified using a future dropout.
- A failed historical CGM sample blocks the whole 3-hour feature window until it ages out. Windows whose oldest two samples were imputed are also locked to ensure all underlying raw dependencies lie within the requested 180-minute lookback.
- Future labels require eight valid observed raw CGM samples. They are never interpolated or imputed. No model uses future meals, activity, sleep or patient IDs as inputs.
- All scaling is fitted only on estimator-fit patients. Medication categories are a predefined schema vocabulary. HGBT random-row early stopping is disabled; model complexity and threshold were fixed in advance.
- Guardrail tests check dropout progression/recovery, future perturbation invariance, bounded lookback, invalid target exclusion and the strict >180 boundary. Split membership and serialization checks are stored with the run.

## Feature count and schema

There are 26 clinical fields and 29 numerical model columns after one-hot encoding four nominal medication categories. Fasting insulin is represented through HOMA-IR, rather than adding a 27th raw field. The three requested interaction features are included. The full feature contract and units are in feature_contract.json. Clinical proxy indices are assumptions, not measured cortisol, insulin clearance or causal treatment effects.

## Important limits

- Data are generated from the Phase 1 simulator and are not empirically fitted to Synthea, MIMIC-IV or CGMacros. Patient hold-outs share the same simulation mechanism. High scores do not establish real-world safety, calibration or fairness.
- PT_001–PT_004 are deliberately in every test set. The 90:10 test set consists solely of these archetypes. Different split regimes have different test populations; their scores cannot establish a sample-efficiency gain. A separate fixed-test learning-curve experiment would be needed.
- Overall excursion performance includes persistent hyperglycemia, especially PT_003. Consult new-crossing, post-meal and per-patient metrics before interpreting early warning quality.
- No overfitting guarantee is possible. Complexity control, patient-level separation and bootstrap summaries mitigate some evaluation errors but do not replace external validation.
- Eight regressors provide point forecasts, not a coupled stochastic latent-state model. The code supplies causal state proxies; this phase does not demonstrate a validated personalized state-space digital twin.
- A plus/minus RMSE band is an error scale, not a calibrated prediction interval. The bundle provides calibration-patient per-horizon RMSE for a later display; test errors must not tune that band.
- Predictive feature edits do not identify counterfactual intervention benefit. Future walking or breathing prescriptions require an explicitly temporal intervention model and validation before clinical claims.
- No DPDP/CDSCO certification, clinical approval, device integration or physician-ready deployment is claimed. No dashboard or Phase 3 work was performed.

## API references

- Installed scikit-learn version: 1.8.0. Runtime signatures and fitted objects were checked locally.
- FrozenEstimator calibration: https://scikit-learn.org/stable/modules/calibration.html
- Histogram gradient boosting: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html
