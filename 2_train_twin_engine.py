#!/usr/bin/env python3
"""Phase 2: causal processing, patient-disjoint training and honest evaluation.

Run: python 2_train_twin_engine.py --data-dir data --output-dir outputs
Importable feature/inference functions support a later dashboard. No UI is built.
Synthetic research prototype. Not a clinical treatment or diagnostic system.
"""
from __future__ import annotations

import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import pickle
import platform
import time

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, mean_absolute_error, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

SEED = 2026
VERSION = 'synthetic-t2d-engine-2.0.0'
ALERT_THRESHOLD = .50
PRIMARY_SPLIT = '80-20'
ARCHETYPES = [f'PT_{i:03d}' for i in range(1, 5)]
MEDICATIONS = ['lifestyle_only', 'metformin', 'metformin_dpp4', 'basal_insulin']
HORIZONS = np.arange(15, 121, 15)
NUMERIC_FEATURES = [
    'current_cgm', 'velocity_15m', 'velocity_30m', 'acceleration_15m',
    'cgm_mean_1h', 'cgm_sd_1h', 'cgm_mean_3h', 'cgm_sd_3h',
    'carb_absorption_state_2h', 'rolling_steps_1h', 'rolling_active_kcal_1h',
    'autonomic_ratio', 'stress_state_flag', 'sleep_efficiency_prev_night',
    'sin_hour', 'cos_hour', 'baseline_hba1c', 'homa_ir', 'c_peptide_ng_ml',
    'bmi', 'egfr_ml_min', 'genetic_tcf7l2_risk', 'effective_glycemic_load',
    'insulin_independent_clearance', 'hepatic_cortisol_index',
]
FEATURE_NAMES = NUMERIC_FEATURES + ['med_regimen']
TARGET_NAMES = [f'glucose_plus_{m}m' for m in HORIZONS]
GI_MAP = {'None': 0, 'Low': 45, 'Medium': 60, 'High': 80}


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def safe_float(value):
    return float(value) if value is not None and np.isfinite(value) else None


def dump_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_inputs(data_dir):
    """Verify source hashes when the Phase 1 manifest is available."""
    data_dir = Path(data_dir)
    hashes = {}
    for name in ['patients_ehr.csv', 'wearable_timeseries.csv']:
        hashes[name] = hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
    manifest_path = data_dir / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for name, digest in hashes.items():
            require(digest == manifest['files'][name]['sha256'], f'Input checksum mismatch: {name}')
    ehr = pd.read_csv(data_dir / 'patients_ehr.csv', keep_default_na=False, na_values=[''])
    telemetry = pd.read_csv(data_dir / 'wearable_timeseries.csv', keep_default_na=False, na_values=[''])
    require(len(ehr) == 40 and ehr.patient_id.is_unique, 'Expected 40 unique EHR patients')
    require(len(telemetry) == 53760, 'Expected exactly 53,760 telemetry rows')
    require(set(ehr.patient_id) == set(telemetry.patient_id), 'Patient foreign key mismatch')
    require(not telemetry.duplicated(['patient_id', 'timestamp']).any(), 'Duplicate telemetry keys')
    require(ehr.med_regimen.isin(MEDICATIONS).all(), 'Unsupported medication category')
    require(np.isfinite(ehr[['baseline_hba1c','homa_ir','c_peptide_ng_ml','bmi','egfr_ml_min','genetic_tcf7l2_risk']]).all().all(), 'Invalid EHR value')
    require((ehr.bmi > 0).all(), 'Invalid BMI')
    telemetry['timestamp'] = pd.to_datetime(telemetry.timestamp)
    for pid, g in telemetry.groupby('patient_id', sort=True):
        require(len(g) == 1344, f'{pid}: wrong sample count')
        require(g.timestamp.is_monotonic_increasing, f'{pid}: time is not sorted')
        require(g.timestamp.diff().dropna().eq(pd.Timedelta(minutes=15)).all(), f'{pid}: broken cadence')
    return ehr, telemetry, hashes


def clean_cgm(group):
    """Past-only one-step linear extrapolation; never backfill or revise history.

    First missing sample can be repaired only from two immediately preceding
    valid raw observations. A second consecutive invalid sample is locked.
    Corrupt/out-of-range values are always locked. No recursive imputation.
    """
    raw = group.cgm_glucose_mgdl.to_numpy(dtype=float)
    status = group.sensor_status.to_numpy()
    raw_valid = (status == 'OK') & np.isfinite(raw) & (raw >= 40) & (raw <= 400)
    clean = np.full(len(group), np.nan)
    imputed = np.zeros(len(group), dtype=bool)
    reasons = np.full(len(group), 'OBSERVED_OK', dtype=object)
    run = 0
    for i in range(len(group)):
        if raw_valid[i]:
            clean[i] = raw[i]
            run = 0
            continue
        run += 1
        is_dropout = status[i] == 'DROPOUT_NAN' and not np.isfinite(raw[i])
        if run >= 2:
            reasons[i] = 'QUARANTINED_INSUFFICIENT_TELEMETRY'
        elif not is_dropout:
            reasons[i] = 'QUARANTINED_CORRUPT_TELEMETRY'
        elif i >= 2 and raw_valid[i-1] and raw_valid[i-2]:
            change = raw[i-1] - raw[i-2]
            estimate = raw[i-1] + change
            if abs(change / 15) <= 3 and 40 <= estimate <= 400:
                clean[i] = estimate
                imputed[i] = True
                reasons[i] = 'CAUSAL_EXTRAPOLATION_15M'
            else:
                reasons[i] = 'QUARANTINED_UNSAFE_EXTRAPOLATION'
        else:
            reasons[i] = 'QUARANTINED_INSUFFICIENT_PAST'
    return clean, raw_valid, imputed, reasons


def make_patient_features(group, ehr_row):
    """Produce exactly 26 clinical fields using observations no later than t.

    CGM means include endpoint observations: 5 samples for 1h and 13 for 3h.
    Activity uses four completed intervals; meals use eight bins in (t-2h,t].
    """
    g = group.reset_index(drop=True).copy()
    if not pd.api.types.is_datetime64_any_dtype(g.timestamp):
        g['timestamp'] = pd.to_datetime(g.timestamp)
    require(g.patient_id.nunique() == 1, 'Feature function requires one patient')
    require(g.timestamp.is_monotonic_increasing and not g.timestamp.duplicated().any(), 'Invalid timeline')
    require(g.timestamp.diff().dropna().eq(pd.Timedelta(minutes=15)).all(), 'Telemetry gap must be represented as a missing packet row')
    clean, raw_valid, imputed, current_reason = clean_cgm(g)
    cgm = pd.Series(clean)
    features = pd.DataFrame(index=g.index)
    features['current_cgm'] = cgm
    features['velocity_15m'] = cgm.diff() / 15
    features['velocity_30m'] = (cgm - cgm.shift(2)) / 30
    features['acceleration_15m'] = (cgm - 2*cgm.shift(1) + cgm.shift(2)) / 225
    for suffix, n in [('1h', 5), ('3h', 13)]:
        means = np.full(len(g), np.nan)
        deviations = np.full(len(g), np.nan)
        if len(g) >= n:
            windows = np.lib.stride_tricks.sliding_window_view(clean, n)
            means[n-1:] = windows.mean(axis=1)
            deviations[n-1:] = windows.std(axis=1, ddof=0)
        features['cgm_mean_' + suffix] = means
        features['cgm_sd_' + suffix] = deviations
    carbs = pd.to_numeric(g.meal_carbs_g, errors='coerce')
    gi = pd.to_numeric(g.meal_gi, errors='coerce')
    tau = g.meal_gi_tier.map({'High': 45., 'Medium': 60., 'Low': 90., 'None': 60.})
    absorption = pd.Series(0., index=g.index)
    for lag in range(8):
        absorption += (carbs.shift(lag) * (gi.shift(lag)/60) * np.exp(-15*lag/tau.shift(lag))).fillna(0)
    features['carb_absorption_state_2h'] = absorption
    features['rolling_steps_1h'] = g.steps_15m.rolling(4, min_periods=4).sum()
    features['rolling_active_kcal_1h'] = g.active_kcal_15m.rolling(4, min_periods=4).sum()
    features['autonomic_ratio'] = g.hr_bpm / g.hrv_rmssd_ms
    features['stress_state_flag'] = ((g.hr_bpm > 90) & (g.steps_15m < 25) & (g.hrv_rmssd_ms < 27)).astype(int)
    features['sleep_efficiency_prev_night'] = g.sleep_efficiency_prev_night
    hour = g.timestamp.dt.hour + g.timestamp.dt.minute / 60
    features['sin_hour'] = np.sin(2*np.pi*hour/24)
    features['cos_hour'] = np.cos(2*np.pi*hour/24)
    for col in ['baseline_hba1c','homa_ir','c_peptide_ng_ml','bmi','egfr_ml_min','genetic_tcf7l2_risk','med_regimen']:
        features[col] = ehr_row[col]
    rolling_carbs = carbs.rolling(8, min_periods=8).sum()
    gi_weighted_carbs = (carbs * gi / 60).rolling(8, min_periods=8).sum()
    gi_multiplier = gi_weighted_carbs.div(rolling_carbs.replace(0, np.nan)).fillna(1.)
    features['effective_glycemic_load'] = rolling_carbs * gi_multiplier * (ehr_row['homa_ir']/3) * (1+.07*ehr_row['genetic_tcf7l2_risk'])
    features['insulin_independent_clearance'] = (features.rolling_steps_1h/100) * (25/ehr_row['bmi'])
    features['hepatic_cortisol_index'] = (100-g.sleep_efficiency_prev_night) * features.autonomic_ratio
    features = features[FEATURE_NAMES]
    require(len(FEATURE_NAMES) == 26, 'Clinical feature contract must contain 26 fields')
    # An imputation at t-180 or t-165 depends on raw data before t-180.
    # Lock such windows to enforce the requested hard 180-minute raw lookback.
    edge_imputation = (pd.Series(imputed).shift(12, fill_value=False)
                       | pd.Series(imputed).shift(11, fill_value=False)).to_numpy()
    finite_features = np.isfinite(features[NUMERIC_FEATURES].to_numpy()).all(axis=1)
    aux_ok = (g.hr_bpm.between(35,220) & g.hrv_rmssd_ms.between(1,200)
              & g.steps_15m.between(0,2500) & g.active_kcal_15m.between(0,300)
              & g.sleep_efficiency_prev_night.between(0,100)
              & carbs.between(0,250) & gi.between(0,100)
              & g.meal_gi_tier.isin(GI_MAP) & g.meal_gi.eq(g.meal_gi_tier.map(GI_MAP)))
    auxiliary_window_valid = aux_ok.astype(int).rolling(13, min_periods=13).sum().eq(13).to_numpy()
    available = finite_features & ~edge_imputation & auxiliary_window_valid
    reasons = np.full(len(g), 'READY_OBSERVED', dtype=object)
    history_imputed = pd.Series(imputed).rolling(13, min_periods=1).max().astype(bool).to_numpy()
    reasons[history_imputed] = 'READY_WITH_CAUSAL_IMPUTATION'
    reasons[~finite_features] = 'QUARANTINED_CONTAMINATED_HISTORY'
    reasons[~auxiliary_window_valid] = 'QUARANTINED_AUXILIARY_TELEMETRY'
    reasons[edge_imputation] = 'QUARANTINED_IMPUTATION_OUTSIDE_LOOKBACK'
    reasons[~np.isfinite(clean)] = current_reason[~np.isfinite(clean)]
    reasons[:min(12, len(g))] = 'WARMUP_INSUFFICIENT_HISTORY'
    meta = pd.DataFrame(dict(patient_id=g.patient_id, timestamp=g.timestamp,
                            sample_index=np.arange(len(g)), raw_cgm=g.cgm_glucose_mgdl,
                            clean_cgm=clean, raw_valid=raw_valid, sensor_status=g.sensor_status,
                            imputed_current=imputed, history_contains_imputation=history_imputed,
                            prediction_eligible=available, safety_status=reasons,
                            meal_carbs_g=g.meal_carbs_g,
                            exercise_state_flag=((g.hr_bpm > 90) & (g.steps_15m >= 250)).astype(int)))
    return features, meta


def construct_targets(group):
    """Future values are used only for labels, never to repair predictor history."""
    n = len(group)
    glucose = group.cgm_glucose_mgdl.to_numpy(float)
    valid = (group.sensor_status.eq('OK').to_numpy() & np.isfinite(glucose)
             & (glucose >= 40) & (glucose <= 400))
    target = np.full((n,8), np.nan)
    available = np.zeros(n, dtype=bool)
    y = np.full(n, -1, dtype=int)
    if n >= 9:
        windows = np.lib.stride_tricks.sliding_window_view(glucose, 9)[:,1:]
        ok = np.lib.stride_tricks.sliding_window_view(valid, 9)[:,1:].all(axis=1)
        indices = np.flatnonzero(ok)
        target[indices] = windows[ok]
        available[indices] = True
        y[indices] = (windows[ok].max(axis=1) > 180).astype(int)
    return target, y, available


def build_dataset(ehr, telemetry):
    frames = []; metadata = []; targets = []; labels = []
    lookup = ehr.set_index('patient_id')
    for pid, group in telemetry.groupby('patient_id', sort=True):
        X, meta = make_patient_features(group, lookup.loc[pid])
        Y, y, available = construct_targets(group)
        meta['target_available'] = available
        meta['supervised_eligible'] = meta.prediction_eligible & available
        frames.append(X); metadata.append(meta); targets.append(Y); labels.append(y)
    X = pd.concat(frames, ignore_index=True)
    meta = pd.concat(metadata, ignore_index=True)
    meta['row_id'] = np.arange(len(meta))
    Y = np.concatenate(targets); y = np.concatenate(labels)
    ok = meta.supervised_eligible.to_numpy()
    require(np.isfinite(X.loc[ok, NUMERIC_FEATURES]).all().all(), 'Non-finite eligible predictors')
    require(np.isfinite(Y[ok]).all() and set(np.unique(y[ok])) == {0,1}, 'Invalid eligible targets')
    return X, Y, y, meta


def run_guardrail_tests(ehr, telemetry):
    group = telemetry[telemetry.patient_id == 'PT_004'].iloc[:50].reset_index(drop=True).copy()
    group['cgm_glucose_mgdl'] = 100. + np.arange(len(group))
    group['sensor_status'] = 'OK'
    row = ehr.set_index('patient_id').loc['PT_004']
    one = group.copy()
    one.loc[20, ['cgm_glucose_mgdl','sensor_status']] = [np.nan, 'DROPOUT_NAN']
    clean, _, imputed, _ = clean_cgm(one)
    require(clean[20] == 120 and imputed[20], 'One-step extrapolation failed')
    two = one.copy()
    two.loc[21, ['cgm_glucose_mgdl','sensor_status']] = [np.nan, 'DROPOUT_NAN']
    x2, m2 = make_patient_features(two, row)
    require(m2.prediction_eligible.iloc[20], 'First missing step should not know about the next dropout')
    require(not m2.prediction_eligible.iloc[21], 'Second missing step must be locked')
    require(not m2.prediction_eligible.iloc[33] and m2.prediction_eligible.iloc[34], 'History recovery boundary failed')
    prefix, _ = make_patient_features(two.iloc[:21], row)
    require(np.allclose(prefix[NUMERIC_FEATURES].iloc[-1], x2[NUMERIC_FEATURES].iloc[20]), 'Future dropout revised past features')
    corrupt = group.copy()
    corrupt.loc[20, ['cgm_glucose_mgdl','sensor_status']] = [510., 'OUT_OF_RANGE_ERROR']
    _, mc = make_patient_features(corrupt, row)
    require(not mc.prediction_eligible.iloc[20], 'Corrupt value must be locked')
    change = group.copy(); change.loc[31:, 'cgm_glucose_mgdl'] = 399.
    old, _ = make_patient_features(group, row); new, _ = make_patient_features(change, row)
    require(old.iloc[30].equals(new.iloc[30]), 'Features depend on future glucose')
    change = group.copy(); change.loc[:17, 'cgm_glucose_mgdl'] = 399.
    older, _ = make_patient_features(change, row)
    require(old.iloc[30].equals(older.iloc[30]), 'Features exceed 180-minute lookback')
    target_group = group.iloc[:9].copy(); target_group['cgm_glucose_mgdl'] = 180.
    _, y0, ok0 = construct_targets(target_group)
    require(ok0[0] and y0[0] == 0, '180 is not a positive excursion')
    target_group.loc[8,'cgm_glucose_mgdl'] = 180.01
    _, y1, ok1 = construct_targets(target_group)
    require(ok1[0] and y1[0] == 1, '>180 must be positive')
    target_group.loc[8,'sensor_status'] = 'OUT_OF_RANGE_ERROR'
    Y2, y2, ok2 = construct_targets(target_group)
    require(not ok2[0] and y2[0] == -1 and np.isnan(Y2[0]).all(), 'Bad future label not excluded')
    require(not ok1[1:].any(), 'Truncated horizons must not be labelled')
    return {'single_dropout_extrapolation': 'PASS', 'second_dropout_quarantine': 'PASS',
            'no_retroactive_future_dependency': 'PASS', 'corrupt_current_lock': 'PASS',
            'contaminated_history_recovery': 'PASS', 'future_perturbation_invariance': 'PASS',
            '180_minute_lookback_bound': 'PASS', 'strict_180_threshold': 'PASS',
            'invalid_future_exclusion': 'PASS', 'truncated_horizon_exclusion': 'PASS'}


def split_training_patients(ids, count, meta, y, seed):
    """Stratify training patients by whether they have any eligible positive row."""
    ids = sorted(ids)
    labels = []
    for pid in ids:
        mask = meta.patient_id.eq(pid).to_numpy() & meta.supervised_eligible.to_numpy()
        require(mask.any(), f'No usable windows for {pid}')
        labels.append(int((y[mask] == 1).any()))
    classes, frequencies = np.unique(labels, return_counts=True)
    strata = labels if len(classes) == 2 and min(frequencies) >= 2 and count >= 2 else None
    fit, holdout = train_test_split(sorted(ids), test_size=count, random_state=seed, stratify=strata)
    # Caller passes sorted ids, so strata and ids have the same order.
    return sorted(fit), sorted(holdout)


def patient_partitions(meta, y):
    all_ids = sorted(meta.patient_id.unique())
    rest = np.array([p for p in all_ids if p not in ARCHETYPES])
    order = np.random.default_rng(SEED).permutation(rest).tolist()
    partitions = {}
    for n_train, label in [(24,'60-40'),(28,'70-30'),(32,'80-20'),(36,'90-10')]:
        train_ids = sorted(order[:n_train]); test_ids = sorted(set(all_ids)-set(train_ids))
        n_cal = math.ceil(.15*n_train)
        fit_ids, cal_ids = split_training_patients(train_ids, n_cal, meta, y, SEED+n_train)
        n_select = math.ceil(.20*len(fit_ids))
        inner_fit_ids, selection_ids = split_training_patients(fit_ids, n_select, meta, y, SEED+100+n_train)
        require(not set(fit_ids)&set(cal_ids) and not set(train_ids)&set(test_ids), 'Patient overlap')
        require(set(ARCHETYPES).issubset(test_ids), 'Archetype escaped hold-out')
        require(not set(inner_fit_ids)&set(selection_ids), 'Selection overlap')
        partitions[label] = dict(outer_train=train_ids, fit=fit_ids, calibration=cal_ids,
                                 test=test_ids, inner_fit=inner_fit_ids, selection=selection_ids,
                                 actual_calibration_fraction=n_cal/n_train)
    return partitions


def row_mask(meta, ids):
    return meta.patient_id.isin(ids).to_numpy() & meta.supervised_eligible.to_numpy()


def preprocessing():
    return ColumnTransformer([
        ('numeric', StandardScaler(), NUMERIC_FEATURES),
        ('medication', OneHotEncoder(categories=[MEDICATIONS], handle_unknown='error', sparse_output=False), ['med_regimen'])
    ], remainder='drop', verbose_feature_names_out=True)


def model_pair(family):
    if family == 'A':
        classifier = LogisticRegression(C=.3, l1_ratio=0., solver='lbfgs', max_iter=1500, random_state=SEED)
        regressor = Ridge(alpha=100.)
    else:
        classifier = HistGradientBoostingClassifier(max_iter=120, learning_rate=.06,
            max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=5.,
            early_stopping=False, random_state=SEED)
        regressor = MultiOutputRegressor(HistGradientBoostingRegressor(max_iter=100,
            learning_rate=.06, max_leaf_nodes=15, min_samples_leaf=50,
            l2_regularization=5., early_stopping=False, random_state=SEED), n_jobs=1)
    return Pipeline([('preprocessor',preprocessing()),('estimator',classifier)]), Pipeline([
        ('preprocessor',preprocessing()),('estimator',regressor)])


def fit_pair(family, X, Y, y, fit_mask, cal_mask):
    require(set(np.unique(y[fit_mask])) == {0,1}, 'Fitting patients lack both classes')
    require(set(np.unique(y[cal_mask])) == {0,1}, 'Calibration patients lack both classes')
    base, trajectory = model_pair(family)
    base.fit(X.loc[fit_mask], y[fit_mask])
    calibrated = CalibratedClassifierCV(FrozenEstimator(base), method='sigmoid', ensemble=False)
    calibrated.fit(X.loc[cal_mask], y[cal_mask])
    trajectory.fit(X.loc[fit_mask], Y[fit_mask])
    return calibrated, trajectory, base


def classification_metrics(y, probabilities, threshold=ALERT_THRESHOLD):
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0,1]).ravel()
    both = len(np.unique(y)) == 2
    return dict(n_windows=int(len(y)), positives=int(y.sum()), prevalence=float(np.mean(y)),
                tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
                recall=float(recall_score(y,predictions,zero_division=0)),
                precision=float(precision_score(y,predictions,zero_division=0)),
                specificity=float(tn/(tn+fp)) if tn+fp else None,
                fpr=float(fp/(tn+fp)) if tn+fp else None,
                f1=float(f1_score(y,predictions,zero_division=0)),
                auroc=float(roc_auc_score(y,probabilities)) if both else None,
                auprc_ap=float(average_precision_score(y,probabilities)) if y.sum() else None,
                brier=float(brier_score_loss(y,probabilities)))


def trajectory_metrics(actual, prediction):
    errors = prediction-actual
    return dict(rmse=float(np.sqrt(np.mean(errors**2))), mae=float(np.mean(np.abs(errors))),
                horizon_rmse=np.sqrt(np.mean(errors**2, axis=0)).tolist(),
                horizon_mae=np.mean(np.abs(errors), axis=0).tolist(),
                horizon_bias=np.mean(errors, axis=0).tolist())


def clinical_lead_time(test_meta, probabilities, telemetry):
    """One record per observed upward crossing; detected-only mean + coverage.

    An alarm must be 15-120 minutes before a crossing, after the preceding
    hyperglycemic sample, and at a currently valid nonhyperglycemic observation.
    It must have an eligible complete supervised window in this evaluation.
    """
    events = []
    for pid, group in test_meta.groupby('patient_id', sort=True):
        raw = telemetry[telemetry.patient_id == pid].reset_index(drop=True)
        glucose = raw.cgm_glucose_mgdl.to_numpy()
        ok = raw.sensor_status.eq('OK').to_numpy() & np.isfinite(glucose) & (glucose>=40) & (glucose<=400)
        probability_map = dict(zip(group.sample_index.to_numpy(), probabilities[group.index.to_numpy()]))
        last_high = -1
        for k in range(1,len(raw)):
            if ok[k-1] and glucose[k-1] > 180:
                last_high = k-1
            if not (ok[k] and ok[k-1] and glucose[k-1] <= 180 < glucose[k]):
                continue
            candidates = [j for j in range(max(0,k-8,last_high+1),k)
                          if j in probability_map and ok[j] and glucose[j] <= 180]
            alarms = [j for j in candidates if probability_map[j] >= ALERT_THRESHOLD]
            events.append(dict(patient_id=pid,crossing_index=k,crossing_timestamp=raw.timestamp.iloc[k].isoformat(),
                               evaluable=bool(candidates), detected=bool(alarms),
                               lead_minutes=int((k-min(alarms))*15) if alarms else None))
    evaluated = [r for r in events if r['evaluable']]
    detected = [r for r in evaluated if r['detected']]
    return dict(observed_crossings=len(events), evaluable_crossings=len(evaluated),
                detected_crossings=len(detected),
                crossing_detection_fraction=len(detected)/len(evaluated) if evaluated else None,
                mean_lead_minutes=float(np.mean([r['lead_minutes'] for r in detected])) if detected else None), events


def patient_bootstrap(meta, y, p, actual, predicted, repetitions=300):
    """Cluster resampling preserves all correlated windows from each patient."""
    aggregates = []
    for pid in sorted(meta.patient_id.unique()):
        mask = meta.patient_id.eq(pid).to_numpy()
        yy, pp = y[mask], p[mask]
        aggregates.append([len(yy), ((pp-yy)**2).sum(),
                           np.logical_and(pp>=ALERT_THRESHOLD,yy==1).sum(), (yy==1).sum(),
                           ((predicted[mask]-actual[mask])**2).sum(), actual[mask].size])
    agg = np.asarray(aggregates, dtype=float)
    rng = np.random.default_rng(SEED)
    values = dict(recall=[], brier=[], rmse=[])
    for _ in range(repetitions):
        sums = agg[rng.integers(0,len(agg),len(agg))].sum(axis=0)
        if sums[3] > 0:
            values['recall'].append(float(sums[2]/sums[3]))
        values['brier'].append(float(sums[1]/sums[0]))
        values['rmse'].append(float(np.sqrt(sums[4]/sums[5])))
    return {name: dict(lower=float(np.quantile(v,.025)), upper=float(np.quantile(v,.975)),
                       valid_replicates=len(v), method='patient_cluster_percentile_300')
            for name,v in values.items() if len(v)}


def evaluate_pair(classifier, regressor, X, Y, y, meta, test_mask, telemetry):
    xt = X.loc[test_mask]; yt = y[test_mask]; actual = Y[test_mask]
    mt = meta.loc[test_mask].reset_index(drop=True)
    p = classifier.predict_proba(xt)[:,1]
    pred = regressor.predict(xt)
    require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(), 'Invalid predicted probability')
    require(pred.shape == actual.shape and np.isfinite(pred).all(), 'Invalid trajectory output')
    metrics = classification_metrics(yt,p)
    metrics.update(trajectory_metrics(actual,pred))
    lead, events = clinical_lead_time(mt,p,telemetry)
    metrics.update(lead)
    metrics['patient_cluster_ci95'] = patient_bootstrap(mt,yt,p,actual,pred)
    metrics['test_patients'] = int(mt.patient_id.nunique())
    metrics['patients_with_positive_windows'] = int(mt.loc[yt==1,'patient_id'].nunique())
    for name, mask in [('new_crossing', mt.raw_valid.to_numpy() & (mt.raw_cgm.to_numpy()<=180)),
                       ('post_meal', mt.raw_valid.to_numpy() & (mt.meal_carbs_g.to_numpy()>0)),
                       ('current_imputed', mt.imputed_current.to_numpy())]:
        metrics[name] = classification_metrics(yt[mask],p[mask]) if mask.any() else None
    persistence = np.repeat(mt.clean_cgm.to_numpy()[:,None],8,axis=1)
    metrics['persistence_reference'] = trajectory_metrics(actual,persistence)
    table = mt[['row_id','patient_id','timestamp','sample_index','raw_cgm','clean_cgm','imputed_current','history_contains_imputation','meal_carbs_g']].copy()
    table['target_excursion'] = yt
    table['predicted_probability'] = p
    table['alert_at_0_5'] = (p>=ALERT_THRESHOLD).astype(int)
    for i,h in enumerate(HORIZONS):
        table[f'actual_plus_{h}m'] = actual[:,i]
        table[f'predicted_plus_{h}m'] = pred[:,i]
    calibration = []
    for lower in np.linspace(0,.9,10):
        upper = lower+.1
        mask = (p>=lower) & ((p<upper) if upper<.999 else (p<=1))
        calibration.append(dict(bin_lower=round(float(lower),2),bin_upper=round(float(upper),2),
                                n=int(mask.sum()),mean_predicted=float(p[mask].mean()) if mask.any() else None,
                                observed_rate=float(yt[mask].mean()) if mask.any() else None))
    thresholds = []
    for threshold in np.arange(.30,.751,.05):
        thresholds.append({'threshold':round(float(threshold),2), **classification_metrics(yt,p,float(threshold))})
    patient_metrics = []
    for pid in sorted(mt.patient_id.unique()):
        mask = mt.patient_id.eq(pid).to_numpy()
        patient_metrics.append({'patient_id':pid, **classification_metrics(yt[mask],p[mask]),
                                **trajectory_metrics(actual[mask],pred[mask])})
    metrics['per_patient'] = patient_metrics
    metrics['calibration_bins'] = calibration
    metrics['threshold_sensitivity'] = thresholds
    return metrics, table, events


def feature_contract():
    units = dict(current_cgm='mg/dL',velocity_15m='mg/dL/min',velocity_30m='mg/dL/min',
        acceleration_15m='mg/dL/min^2',cgm_mean_1h='mg/dL',cgm_sd_1h='mg/dL',cgm_mean_3h='mg/dL',
        cgm_sd_3h='mg/dL',carb_absorption_state_2h='weighted g (proxy)',rolling_steps_1h='steps',
        rolling_active_kcal_1h='kcal',autonomic_ratio='(beats/min)/ms',stress_state_flag='binary',
        sleep_efficiency_prev_night='%',sin_hour='dimensionless',cos_hour='dimensionless',
        baseline_hba1c='%',homa_ir='dimensionless',c_peptide_ng_ml='ng/mL',bmi='kg/m^2',
        egfr_ml_min='mL/min/1.73m^2',genetic_tcf7l2_risk='allele count',effective_glycemic_load='weighted g (proxy)',
        insulin_independent_clearance='activity index (proxy)',hepatic_cortisol_index='autonomic index (proxy)',med_regimen='nominal category')
    return dict(version=VERSION,clinical_feature_count=26,model_column_count=29,
        feature_names=FEATURE_NAMES,numeric_features=NUMERIC_FEATURES,
        units=units,medication_categories=MEDICATIONS,forecast_horizons_minutes=HORIZONS.tolist(),
        raw_lookback_minutes=180,cgm_window_endpoint_samples={'1h':5,'3h':13},
        activity_bins_1h=4,meal_bins_2h=8,standard_deviation_ddof=0,
        insulin_representation='fasting_insulin_uiu_ml is represented through HOMA-IR, not an additional independent field',
        gi_multiplier='carb-weighted mean of GI/60 in (t-120min,t], or 1 if no carbs',
        genetic_multiplier='1 + 0.07 * synthetic TCF7L2 risk allele count; unvalidated assumption',
        carb_absorption='sum(carbs * GI/60 * exp(-lag_minutes/tau_GI)); tau=45/60/90min for high/medium/low',
        no_future_exogenous_inputs=True,labels_use_raw_observed_future_only=True,
        excluded_predictors=['patient_id','archetype','population_context','device_source','is_synthetic','future_targets'])


def predict_latest(bundle, ehr_row, patient_history):
    """Guarded inference for a later dashboard; accepts only history through t."""
    require(patient_history.patient_id.nunique() == 1, 'Inference needs a single patient')
    if hasattr(ehr_row, 'get') and ehr_row.get('patient_id') is not None:
        require(str(ehr_row['patient_id']) == str(patient_history.patient_id.iloc[-1]), 'EHR/history patient mismatch')
    features, meta = make_patient_features(patient_history, ehr_row)
    row = meta.iloc[-1]
    if not row.prediction_eligible:
        return dict(status=str(row.safety_status), probability=None, trajectory=None, features=None)
    x = features.iloc[[-1]][bundle['feature_names']]
    probability = float(bundle['classifier'].predict_proba(x)[0,1])
    curve = bundle['regressor'].predict(x)[0].tolist()
    return dict(status=str(row.safety_status),probability=probability,trajectory=curve,
                features=x.iloc[0].to_dict(),model_version=bundle['model_version'],
                horizon_minutes=bundle['horizon_minutes'])


def fmt(v, digits=3):
    return '—' if v is None else f'{v:.{digits}f}'


def report_tables(results, partitions):
    first = ['| Split | Model | Fit/cal/test patients | Test windows | AUROC | AUPRC (AP) | Brier | Trajectory RMSE |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    second = ['| Split | Model | Recall | PPV | Specificity | FPR | F1 | AUROC | AUPRC (AP) | Brier | Mean lead (min) | RMSE | MAE |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for regime, models in results.items():
        part = partitions[regime]
        for family, m in models.items():
            counts = f'{len(part["fit"])}/{len(part["calibration"])}/{len(part["test"])}'
            first.append(f'| {regime} | {family} | {counts} | {m["n_windows"]:,} | {fmt(m["auroc"])} | {fmt(m["auprc_ap"])} | {fmt(m["brier"],4)} | {fmt(m["rmse"],2)} |')
            if regime in ['80-20','90-10']:
                values = [fmt(m[k]) for k in ['recall','precision','specificity','fpr','f1','auroc','auprc_ap']]
                values += [fmt(m['brier'],4),fmt(m['mean_lead_minutes'],1),fmt(m['rmse'],2),fmt(m['mae'],2)]
                second.append('| '+ ' | '.join([regime,family]+values)+' |')
    return '\n'.join(first), '\n'.join(second)


def write_report(out, results, partitions, selection, selected, audit):
    table1, table2 = report_tables(results,partitions)
    report = ['# Phase 2 model evaluation', '', '**Synthetic research evaluation only.** All metrics below were computed from held-out predictions. Threshold is fixed at 0.50; no test-driven threshold or hyperparameter tuning.', '',
        'Model A: L2 logistic regression with sigmoid calibration; multi-output ridge trajectory regression. Model B: sigmoid-calibrated histogram gradient boosting; eight separate histogram gradient boosting regressors.', '',
        '## Table 1 — patient-level split sensitivity', '', table1, '',
        'Fit/cal/test counts exclude calibration patients from estimator and scaler fitting. Outer training totals are 24, 28, 32 and 36. Calibration uses ceil(15% of outer training patients), so the exact fractions are 16.7%, 17.9%, 15.6% and 16.7%.', '',
        '## Table 2 — detailed primary and four-archetype evaluation', '',table2,'',
        'Recall, PPV, specificity, FPR, F1, AUROC and AUPRC are fractions. AUPRC is scikit-learn average precision (non-interpolated AP), not trapezoidal PR area. RMSE/MAE pool eight horizons and are in mg/dL. Zero-division precision/recall/F1 are reported as 0; AUROC is unavailable for a single-class subgroup.', '',
        'Mean lead time averages the earliest qualifying alert 15–120 minutes before each detected upward crossing, after the previous hyperglycemic reading. It excludes undetected crossings, so interpret it with event detection coverage below. No lead is assigned to already-high readings. Crossings use raw unsmoothed CGM and can include threshold jitter.', '',
        '## Model selection without test reuse', '',
        f'Primary deployment regime: **{PRIMARY_SPLIT}** (chosen before evaluation). Selected classifier: **Model {selected["classifier_family"]}** by lowest internal-selection Brier. Selected trajectory regressor: **Model {selected["regressor_family"]}** by lowest internal-selection RMSE. No claim is made that either is globally optimal.', '',
        'For each regime, the outer training patients are partitioned into fit and calibration groups. An internal selection group is temporarily removed from the fit group. Candidate bases/scalers are trained on the remaining patients and calibrated on the calibration patients, then assessed on the independent selection patients. After selecting model families, bases are refitted on all fit patients and calibrated on the same untouched calibration group. Hold-outs never influence selection. The exported bundle uses the primary regime; 90:10 is only sensitivity analysis.', '',
        '| Split | Family | Internal selection Brier | Internal selection RMSE |', '|---|---|---:|---:|']
    for regime, family_scores in selection.items():
        for family, scores in family_scores.items():
            report.append(f'| {regime} | {family} | {scores["brier"]:.4f} | {scores["rmse"]:.2f} |')
    report += ['', '## Crossings, post-meal subsets and persistence reference', '',
               '| Split | Model | Test prevalence | Crossing detection | Mean lead | New-crossing recall | Post-meal recall | Persistence RMSE |',
               '|---|---|---:|---:|---:|---:|---:|---:|']
    for regime, models in results.items():
        for family,m in models.items():
            report.append(f'| {regime} | {family} | {m["prevalence"]:.3f} | {m["detected_crossings"]}/{m["evaluable_crossings"]} | {fmt(m["mean_lead_minutes"],1)} | {fmt(m["new_crossing"]["recall"])} | {fmt(m["post_meal"]["recall"])} | {m["persistence_reference"]["rmse"]:.2f} |')
    report += ['', '## Patient-cluster uncertainty (95% percentile intervals)', '',
               '| Split | Model | Recall interval | Brier interval | RMSE interval |', '|---|---|---|---|---|']
    for regime in ['80-20','90-10']:
        for family,m in results[regime].items():
            ci=m['patient_cluster_ci95']
            cols = [f'{ci[k]["lower"]:.3f}–{ci[k]["upper"]:.3f}' for k in ['recall','brier','rmse']]
            report.append('| '+' | '.join([regime,family]+cols)+' |')
    report += ['', '300 resamples draw whole patients with replacement, retaining all their windows. Recall resamples without positives are omitted and their valid count is stored in metrics JSON. Eight or four test patients are far too few for dependable population confidence bounds; these intervals are descriptive and can be unstable or degenerate.', '',
        '## Safety, causality and data quality', '',
        f'Input rows: {audit["input_rows"]:,}. Causally usable predictor windows: {audit["prediction_eligible_rows"]:,}. Complete clean future targets: {audit["target_available_rows"]:,}. Jointly eligible training/evaluation windows: {audit["supervised_eligible_rows"]:,}.', '',
        '- Invalid status or CGM outside 40–400 mg/dL is never used as a glucose observation. Corrupt values are quarantined. A first missing packet may be extrapolated from two valid immediately preceding raw values, only within 40–400 mg/dL and with absolute prior slope ≤3 mg/dL/min. These engineering bounds are not clinical validation.',
        '- No recursive extrapolation. A second consecutive invalid sample locks prediction immediately. The first estimate is never retroactively reclassified using a future dropout.',
        '- A failed historical CGM sample blocks the whole 3-hour feature window until it ages out. Windows whose oldest two samples were imputed are also locked to ensure all underlying raw dependencies lie within the requested 180-minute lookback.',
        '- Future labels require eight valid observed raw CGM samples. They are never interpolated or imputed. No model uses future meals, activity, sleep or patient IDs as inputs.',
        '- All scaling is fitted only on estimator-fit patients. Medication categories are a predefined schema vocabulary. HGBT random-row early stopping is disabled; model complexity and threshold were fixed in advance.',
        '- Guardrail tests check dropout progression/recovery, future perturbation invariance, bounded lookback, invalid target exclusion and the strict >180 boundary. Split membership and serialization checks are stored with the run.', '',
        '## Feature count and schema', '',
        'There are 26 clinical fields and 29 numerical model columns after one-hot encoding four nominal medication categories. Fasting insulin is represented through HOMA-IR, rather than adding a 27th raw field. The three requested interaction features are included. The full feature contract and units are in feature_contract.json. Clinical proxy indices are assumptions, not measured cortisol, insulin clearance or causal treatment effects.', '',
        '## Important limits', '',
        '- Data are generated from the Phase 1 simulator and are not empirically fitted to Synthea, MIMIC-IV or CGMacros. Patient hold-outs share the same simulation mechanism. High scores do not establish real-world safety, calibration or fairness.',
        '- PT_001–PT_004 are deliberately in every test set. The 90:10 test set consists solely of these archetypes. Different split regimes have different test populations; their scores cannot establish a sample-efficiency gain. A separate fixed-test learning-curve experiment would be needed.',
        '- Overall excursion performance includes persistent hyperglycemia, especially PT_003. Consult new-crossing, post-meal and per-patient metrics before interpreting early warning quality.',
        '- No overfitting guarantee is possible. Complexity control, patient-level separation and bootstrap summaries mitigate some evaluation errors but do not replace external validation.',
        '- Eight regressors provide point forecasts, not a coupled stochastic latent-state model. The code supplies causal state proxies; this phase does not demonstrate a validated personalized state-space digital twin.',
        '- A plus/minus RMSE band is an error scale, not a calibrated prediction interval. The bundle provides calibration-patient per-horizon RMSE for a later display; test errors must not tune that band.',
        '- Predictive feature edits do not identify counterfactual intervention benefit. Future walking or breathing prescriptions require an explicitly temporal intervention model and validation before clinical claims.',
        '- No DPDP/CDSCO certification, clinical approval, device integration or physician-ready deployment is claimed. No dashboard or Phase 3 work was performed.', '',
        '## API references', '',
        '- Installed scikit-learn version: '+sklearn.__version__+'. Runtime signatures and fitted objects were checked locally.',
        '- FrozenEstimator calibration: https://scikit-learn.org/stable/modules/calibration.html',
        '- Histogram gradient boosting: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html', '']
    (out/'evaluation_metrics_report.md').write_text('\n'.join(report),encoding='utf-8')
    (out/'chat_tables.md').write_text(table1+'\n\n'+table2+'\n',encoding='utf-8')
    print('\n'+table1+'\n\n'+table2, flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=Path(__file__).resolve().parent/'data')
    parser.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent/'outputs')
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    require(args.threads>=1,'threads must be positive')
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    started=time.time()
    ehr,telemetry,input_hashes=load_inputs(args.data_dir)
    guardrails=run_guardrail_tests(ehr,telemetry)
    X,Y,y,meta=build_dataset(ehr,telemetry)
    partitions=patient_partitions(meta,y)
    audit=dict(input_rows=len(meta),prediction_eligible_rows=int(meta.prediction_eligible.sum()),
               target_available_rows=int(meta.target_available.sum()),supervised_eligible_rows=int(meta.supervised_eligible.sum()),
               imputed_current_rows=int(meta.imputed_current.sum()),
               safety_status_counts={k:int(v) for k,v in meta.safety_status.value_counts().items()},
               guardrail_tests=guardrails)
    dump_json(out/'feature_contract.json',feature_contract())
    dump_json(out/'patient_partitions.json',partitions)
    dump_json(out/'guardrail_audit.json',audit)
    meta.to_csv(out/'window_eligibility_audit.csv.gz',index=False,compression='gzip')
    results={};selection={};chosen={};primary_models={};primary_predictions={}
    thresholds=[];calibration=[];all_patient_metrics=[];all_events=[]
    with threadpool_limits(limits=args.threads):
        for regime,part in partitions.items():
            print(f'[{regime}] patients fit={len(part["fit"])}, calibration={len(part["calibration"])}, test={len(part["test"])}',flush=True)
            masks={key:row_mask(meta,part[key]) for key in ['inner_fit','selection','fit','calibration','test']}
            require(not np.any(masks['fit']&masks['test']) and not np.any(masks['calibration']&masks['fit']), 'Row overlap')
            selection[regime]={}
            # Freeze model-family choices on internal validation before test evaluation.
            for family in ['A','B']:
                clf,reg,_=fit_pair(family,X,Y,y,masks['inner_fit'],masks['calibration'])
                p=clf.predict_proba(X.loc[masks['selection']])[:,1]
                curve=reg.predict(X.loc[masks['selection']])
                selection[regime][family]=dict(brier=float(brier_score_loss(y[masks['selection']],p)),
                    rmse=float(np.sqrt(np.mean((curve-Y[masks['selection']])**2))))
                print(f'[{regime}] Model {family} internal-selection scores: {selection[regime][family]}',flush=True)
            chosen[regime]=dict(classifier_family=min(selection[regime],key=lambda f:selection[regime][f]['brier']),
                                regressor_family=min(selection[regime],key=lambda f:selection[regime][f]['rmse']))
            results[regime]={}
            for family in ['A','B']:
                clf,reg,base=fit_pair(family,X,Y,y,masks['fit'],masks['calibration'])
                m,predictions,events=evaluate_pair(clf,reg,X,Y,y,meta,masks['test'],telemetry)
                m['fit_windows']=int(masks['fit'].sum());m['calibration_windows']=int(masks['calibration'].sum())
                results[regime][family]=m
                predictions.to_csv(out/f'predictions_{regime}_model_{family}.csv.gz',index=False,compression='gzip')
                thresholds.extend([{'split':regime,'model':family,**r} for r in m['threshold_sensitivity']])
                calibration.extend([{'split':regime,'model':family,**r} for r in m['calibration_bins']])
                all_patient_metrics.extend([{'split':regime,'model':family,**{k:v for k,v in r.items() if not isinstance(v,list)}} for r in m['per_patient']])
                all_events.extend([{'split':regime,'model':family,**r} for r in events])
                print(f'[{regime}] Model {family} test AUROC={m["auroc"]:.4f} Brier={m["brier"]:.4f} RMSE={m["rmse"]:.2f}',flush=True)
                if regime==PRIMARY_SPLIT:
                    primary_models[family]=(clf,reg,base)
                    primary_predictions[family]=predictions
            dump_json(out/'evaluation_metrics.json',results)
    selected=chosen[PRIMARY_SPLIT]
    cf=selected['classifier_family'];rf=selected['regressor_family']
    clf,_,base=primary_models[cf];_,reg,_=primary_models[rf]
    part=partitions[PRIMARY_SPLIT]
    cal_mask=row_mask(meta,part['calibration']);fit_mask=row_mask(meta,part['fit']);test_mask=row_mask(meta,part['test'])
    cal_prediction=reg.predict(X.loc[cal_mask])
    error_scale=np.sqrt(np.mean((cal_prediction-Y[cal_mask])**2,axis=0))
    preprocessing_fitted=base.named_steps['preprocessor']
    encoded_names=preprocessing_fitted.get_feature_names_out().tolist()
    require(len(encoded_names)==29,'Unexpected encoded column count')
    sampled=X.loc[fit_mask].sample(n=min(100,int(fit_mask.sum())),random_state=SEED).reset_index(drop=True)
    bundle=dict(bundle_format_version=1,model_version=f'{VERSION}-classifier-{cf}-trajectory-{rf}',
        classifier=clf,regressor=reg,feature_scaler=preprocessing_fitted.named_transformers_['numeric'],
        preprocessing=preprocessing_fitted,feature_names=FEATURE_NAMES,encoded_feature_names=encoded_names,
        feature_contract=feature_contract(),horizon_minutes=HORIZONS.tolist(),alert_threshold=ALERT_THRESHOLD,
        primary_split=PRIMARY_SPLIT,selected_families=selected,selection_rule='minimum internal-validation Brier and trajectory RMSE; independent of test metrics',
        selection_metrics=selection,evaluation_metrics=results,patient_partitions=partitions,
        deployment_fit_patient_ids=part['fit'],deployment_calibration_patient_ids=part['calibration'],
        deployment_test_patient_ids=part['test'],archetype_patient_ids=ARCHETYPES,
        calibration_horizon_rmse=error_scale.tolist(),calibration_error_band_type='plus/minus RMSE scale; NOT a calibrated prediction interval',
        explanation_background=sampled,input_sha256=input_hashes,
        inference_contract='classifier.predict_proba and regressor.predict accept raw 26-field DataFrames; pipelines perform scaling/one-hot encoding. Do not scale inputs twice.',
        safety_contract='Call predict_latest with time-bounded history. Suppress probabilities and trajectories whenever prediction_eligible is false.',
        environment=dict(python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__),
        limitations=['Synthetic shared simulator','No clinical validation','Noncausal feature-edit what-if limitations','Archetype-selected evaluation patients'])
    with (out/'twin_model_bundle.pkl').open('wb') as stream:
        pickle.dump(bundle,stream,protocol=5)
    with (out/'twin_model_bundle.pkl').open('rb') as stream:
        reloaded=pickle.load(stream)
    sample=X.loc[test_mask].iloc[:25]
    require(np.array_equal(clf.predict_proba(sample),reloaded['classifier'].predict_proba(sample)),'Classifier serialization changed predictions')
    require(np.array_equal(reg.predict(sample),reloaded['regressor'].predict(sample)),'Regressor serialization changed predictions')
    ehr4=ehr[ehr.patient_id.eq('PT_004')].iloc[0]
    history4=telemetry[telemetry.patient_id.eq('PT_004')].reset_index(drop=True)
    locked=predict_latest(reloaded,ehr4,history4.iloc[:342])
    require(locked['probability'] is None and locked['trajectory'] is None,'Guarded inference failed to lock double dropout')
    usable=np.flatnonzero(meta.patient_id.eq('PT_004').to_numpy() & meta.supervised_eligible.to_numpy())[0]
    index=int(meta.sample_index.iloc[usable])
    live=predict_latest(reloaded,ehr4,history4.iloc[:index+1])
    require(live['probability'] is not None and len(live['trajectory'])==8,'Guarded inference failed on valid history')
    audit['serialization_roundtrip']='PASS';audit['guarded_inference_lockout']='PASS';audit['guarded_inference_valid']='PASS'
    audit['patient_partition_disjointness']='PASS';audit['archetypes_held_out_all_regimes']='PASS'
    dump_json(out/'guardrail_audit.json',audit)
    dump_json(out/'internal_selection_metrics.json',dict(scores=selection,chosen_families=chosen))
    pd.DataFrame(thresholds).to_csv(out/'threshold_tradeoffs.csv',index=False)
    pd.DataFrame(calibration).to_csv(out/'calibration_bins.csv',index=False)
    pd.DataFrame(all_patient_metrics).to_csv(out/'per_patient_evaluation.csv',index=False)
    pd.DataFrame(all_events).to_csv(out/'crossing_event_evaluation.csv',index=False)
    write_report(out,results,partitions,selection,selected,audit)
    output_hashes={f.name:dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest())
                   for f in sorted(out.iterdir()) if f.is_file() and f.name!='run_manifest.json'}
    dump_json(out/'run_manifest.json',dict(version=VERSION,seed=SEED,environment=bundle['environment'],
        input_sha256=input_hashes,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        elapsed_seconds=round(time.time()-started,2),outputs=output_hashes))
    print(json.dumps({'status':'PASS','selected':selected,'primary_split':PRIMARY_SPLIT,
                      'eligible_rows':audit['supervised_eligible_rows'],'elapsed_seconds':round(time.time()-started,2)}),flush=True)


if __name__=='__main__':
    main()
