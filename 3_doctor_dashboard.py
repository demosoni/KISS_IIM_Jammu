#!/usr/bin/env python3
"""Physician-facing synthetic CDS replay and model-input scenario dashboard.

Run: streamlit run 3_doctor_dashboard.py
Use the accompanying package for the verified model, engine and input CSVs.
Every displayed probability, forecast and explanation is model-computed.
"""
from __future__ import annotations

import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

from datetime import datetime
import hashlib
import html
import importlib.util
import json
import math
from pathlib import Path
import pickle
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
IST = ZoneInfo('Asia/Kolkata')
APP_VERSION = 'glycemic-cds-replay-3.0.0'
GI_VALUES = {'None':0, 'Low':45, 'Medium':60, 'High':80}
DEFAULT_CONTROLS = {'walk_minutes':0, 'reduce_carbs_g':0, 'gi_tier':'Keep observed', 'breathing':False}
ARCHETYPE_NAMES = {
    'PT_001':'High Carb Spiker',
    'PT_002':'High Stress / Poor Sleep Executive',
    'PT_003':'Advanced T2D + CKD',
    'PT_004':'Sensor Dropout Safety Demo',
}
MEDICATION_LABELS = {'lifestyle_only':'Lifestyle only', 'metformin':'Metformin',
                     'metformin_dpp4':'Metformin + DPP-4', 'basal_insulin':'Basal insulin'}
EXPLANATION_GROUPS = {
    'Current glucose and trend':['current_cgm','velocity_15m','velocity_30m','acceleration_15m'],
    'Recent glucose history':['cgm_mean_1h','cgm_sd_1h','cgm_mean_3h','cgm_sd_3h'],
    'Carbohydrate absorption':['carb_absorption_state_2h'],
    'Recent steps and energy':['rolling_steps_1h','rolling_active_kcal_1h'],
    'Autonomic signals':['autonomic_ratio','stress_state_flag'],
    'Sleep and time of day':['sleep_efficiency_prev_night','sin_hour','cos_hour'],
    'Baseline EHR profile':['baseline_hba1c','homa_ir','c_peptide_ng_ml','bmi','egfr_ml_min','genetic_tcf7l2_risk','med_regimen'],
    'Glycemic load × HOMA-IR':['effective_glycemic_load'],
    'Activity × BMI':['insulin_independent_clearance'],
    'Sleep × autonomic ratio':['hepatic_cortisol_index'],
}
COLORS = dict(purple='#5B21B6', navy='#0F172A', slate='#64748B', red='#DC2626',
              green='#059669', cyan='#0891B2', pale='#F7F8FC')


def require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def asset_fingerprint(root):
    """Cache keys change when any verified dependency is replaced."""
    root=Path(root)
    manifest_path=root/'assets_manifest.json'
    require(manifest_path.is_file(), 'assets_manifest.json is missing; extract the complete Phase 3 package.')
    manifest=json.loads(manifest_path.read_text())
    info=[manifest_path.read_text()]
    for name in sorted(manifest):
        path=root/name
        require(path.is_file(), f'Required asset is missing: {name}')
        stat=path.stat()
        info.append(f'{name}:{stat.st_size}:{stat.st_mtime_ns}')
    return hashlib.sha256('\n'.join(info).encode()).hexdigest()


@st.cache_resource(show_spinner='Loading the verified model and synthetic patient histories…')
def load_resources(root_string, fingerprint):
    root=Path(root_string)
    manifest=json.loads((root/'assets_manifest.json').read_text())
    for name, expected in manifest.items():
        content=(root/name).read_bytes()
        require(hashlib.sha256(content).hexdigest()==expected['sha256'], f'Integrity check failed: {name}')
    model_path=root/'models/twin_model_bundle.pkl'
    # Only the locally supplied, checksum-verified bundle is unpickled.
    with model_path.open('rb') as handle:
        bundle=pickle.load(handle)
    require(bundle.get('bundle_format_version')==1, 'Unsupported model bundle format')
    require(bundle['primary_split']=='80-20', 'This app expects the evaluated primary 80:20 bundle')
    source=root/'2_train_twin_engine.py'
    spec=importlib.util.spec_from_file_location('twin_engine_phase3',source)
    engine=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    ehr,telemetry,input_hashes=engine.load_inputs(root/'data')
    require(input_hashes==bundle['input_sha256'], 'Model/data provenance mismatch')
    X,Y,y,meta=engine.build_dataset(ehr,telemetry)
    fit_mask=engine.row_mask(meta,bundle['deployment_fit_patient_ids'])
    numeric=bundle['feature_contract']['numeric_features']
    bounds={'minimum':X.loc[fit_mask,numeric].min().to_dict(), 'maximum':X.loc[fit_mask,numeric].max().to_dict()}
    allowed=bundle['deployment_test_patient_ids']
    require(set(ARCHETYPE_NAMES).issubset(allowed), 'Demonstration patients are not held out')
    require(not set(allowed)&set(bundle['deployment_fit_patient_ids']), 'Patient separation failure')
    records={}
    with threadpool_limits(limits=2):
        for pid in allowed:
            mask=meta.patient_id.eq(pid).to_numpy()
            raw=telemetry[telemetry.patient_id.eq(pid)].reset_index(drop=True)
            features=X.loc[mask].reset_index(drop=True)
            metadata=meta.loc[mask].reset_index(drop=True)
            # The UI is stricter than Phase 2: every invalid CURRENT packet locks.
            safe=(metadata.prediction_eligible & metadata.raw_valid).to_numpy()
            probabilities=np.full(len(raw),np.nan)
            probabilities[safe]=bundle['classifier'].predict_proba(features.loc[safe,bundle['feature_names']])[:,1]
            records[pid]={'raw':raw,'features':features,'meta':metadata,'safe':safe,'probabilities':probabilities}
        cohort_mask=engine.row_mask(meta,allowed)&meta.raw_valid.to_numpy()
        cohort_probability=bundle['classifier'].predict_proba(X.loc[cohort_mask,bundle['feature_names']])[:,1]
    return dict(bundle=bundle,engine=engine,ehr=ehr.set_index('patient_id',drop=False),records=records,
                model_sha=manifest['models/twin_model_bundle.pkl']['sha256'],bounds=bounds,
                cohort_probability=cohort_probability,cohort_y=y[cohort_mask],
                cohort_patient_count=len(allowed),fingerprint=fingerprint)


def recent_meal_index(history):
    start=max(0,len(history)-8)
    choices=np.flatnonzero(history.meal_carbs_g.to_numpy()[start:]>0)
    return int(start+choices[-1]) if len(choices) else None


def scenario_history(history, ehr_row, controls):
    """A hypothetical covariate edit, NOT a causal future intervention rollout.

    Observed CGM is deliberately held fixed. Meals and equivalent recent
    activity are edited in a separate copy, then passed through the same engine.
    """
    walk=int(controls['walk_minutes']);reduction=float(controls['reduce_carbs_g'])
    tier=controls['gi_tier'];breathing=bool(controls['breathing'])
    require(0<=walk<=45 and 0<=reduction<=50,'Scenario controls outside their supported range')
    require(tier in ['Keep observed','High','Medium','Low'],'Unknown GI tier')
    altered=history.reset_index(drop=True).copy(deep=True)
    applied={'walk_minutes_requested':walk,'walk_steps_added':0,'active_kcal_added':0.,
             'carbs_requested_g':reduction,'carbs_removed_g':0.,'meal_timestamp':None,
             'gi_before':None,'gi_after':None,'breathing_requested':breathing,'breathing_applied':False}
    meal=recent_meal_index(altered)
    if meal is not None:
        applied['meal_timestamp']=pd.Timestamp(altered.at[meal,'timestamp']).isoformat()
        applied['gi_before']=str(altered.at[meal,'meal_gi_tier'])
        removed=min(reduction,float(altered.at[meal,'meal_carbs_g']))
        altered.at[meal,'meal_carbs_g']=float(altered.at[meal,'meal_carbs_g'])-removed
        applied['carbs_removed_g']=removed
        if tier!='Keep observed':
            altered.at[meal,'meal_gi_tier']=tier
            altered.at[meal,'meal_gi']=GI_VALUES[tier]
        if altered.at[meal,'meal_carbs_g']==0:
            altered.at[meal,'meal_gi_tier']='None'
            altered.at[meal,'meal_gi']=0
        applied['gi_after']=str(altered.at[meal,'meal_gi_tier'])
    current=len(altered)-1
    stress=(altered.at[current,'hr_bpm']>90 and altered.at[current,'steps_15m']<25
            and altered.at[current,'hrv_rmssd_ms']<27)
    if breathing and stress:
        altered.at[current,'hr_bpm']=min(float(altered.at[current,'hr_bpm']),85.)
        altered.at[current,'hrv_rmssd_ms']=max(float(altered.at[current,'hrv_rmssd_ms']),40.)
        applied['breathing_applied']=True
    remaining=walk*100
    # A chosen assumption: 100 steps/min and at most 1,500 added steps per bin.
    for j in range(current,max(-1,current-4),-1):
        if remaining<=0:
            break
        capacity=max(0,2500-int(altered.at[j,'steps_15m']))
        addition=int(min(remaining,1500,capacity))
        altered.at[j,'steps_15m']=int(altered.at[j,'steps_15m'])+addition
        energy=addition*.039*float(ehr_row['bmi'])/25
        altered.at[j,'active_kcal_15m']=float(altered.at[j,'active_kcal_15m'])+energy
        if addition and altered.at[j,'steps_15m']>=250:
            altered.at[j,'hr_bpm']=max(float(altered.at[j,'hr_bpm']),100.)
            altered.at[j,'hrv_rmssd_ms']=max(float(altered.at[j,'hrv_rmssd_ms']),30.)
        remaining-=addition
        applied['walk_steps_added']+=addition
        applied['active_kcal_added']+=energy
    require(remaining==0,'Recent activity window cannot accommodate this scenario')
    require(altered.cgm_glucose_mgdl.equals(history.reset_index(drop=True).cgm_glucose_mgdl), 'Scenario altered observed CGM')
    return altered,applied


def outside_training_range(features, bounds):
    row=features.iloc[0]
    return [name for name,low in bounds['minimum'].items()
            if float(row[name])<low-1e-8 or float(row[name])>bounds['maximum'][name]+1e-8]


def score_snapshot(resources, pid, sample_index, controls=None):
    """No probability or trajectory is returned for an unsafe current window."""
    controls=dict(DEFAULT_CONTROLS if controls is None else controls)
    record=resources['records'][pid]
    require(0<=sample_index<len(record['raw']),'Replay index out of range')
    history=record['raw'].iloc[:sample_index+1].copy()
    metadata=record['meta'].iloc[sample_index]
    last=history.iloc[-1]
    if not bool(metadata.raw_valid):
        return dict(status='LOCKED',reason='CURRENT_PACKET_INVALID',baseline=None,scenario=None,applied=None)
    if not bool(metadata.prediction_eligible):
        return dict(status='LOCKED',reason=str(metadata.safety_status),baseline=None,scenario=None,applied=None)
    bundle=resources['bundle'];engine=resources['engine'];ehr_row=resources['ehr'].loc[pid]
    x=record['features'].iloc[[sample_index]][bundle['feature_names']].copy()
    with threadpool_limits(limits=2):
        baseline_p=float(bundle['classifier'].predict_proba(x)[0,1])
        baseline_curve=bundle['regressor'].predict(x)[0]
    baseline=dict(probability=baseline_p,trajectory=baseline_curve.tolist(),features=x,
                  outside_training_range=outside_training_range(x,resources['bounds']))
    altered,applied=scenario_history(history,ehr_row,controls)
    xf,mf=engine.make_patient_features(altered,ehr_row)
    if not bool(mf.prediction_eligible.iloc[-1]):
        return dict(status='READY',reason=str(metadata.safety_status),baseline=baseline,scenario=None,
                    scenario_reason=str(mf.safety_status.iloc[-1]),applied=applied)
    sx=xf.iloc[[-1]][bundle['feature_names']].copy()
    with threadpool_limits(limits=2):
        sp=float(bundle['classifier'].predict_proba(sx)[0,1])
        sc=bundle['regressor'].predict(sx)[0]
    require(np.isfinite([baseline_p,sp]).all() and np.isfinite(baseline_curve).all() and np.isfinite(sc).all(), 'Model produced non-finite outputs')
    scenario=dict(probability=sp,trajectory=sc.tolist(),features=sx,
                  outside_training_range=outside_training_range(sx,resources['bounds']))
    return dict(status='READY',reason=str(metadata.safety_status),baseline=baseline,scenario=scenario,applied=applied)


def select_event(record, kind):
    raw=record['raw'];safe=record['safe'];p=record['probabilities']
    if kind=='fault':
        bad=~record['meta'].raw_valid.to_numpy()
        choices=np.flatnonzero(bad & np.r_[False,bad[:-1]] & (np.arange(len(bad))>=12))
        if len(choices):
            return int(choices[0])
        choices=np.flatnonzero(bad & (np.arange(len(bad))>=12))
        return int(choices[0]) if len(choices) else None
    if kind=='meal':
        recent=raw.meal_carbs_g.rolling(8,min_periods=1).sum().to_numpy()>0
        eligible=safe & recent & (raw.cgm_glucose_mgdl.to_numpy()<=180)
        if not eligible.any():
            eligible=safe & recent
    elif kind=='dawn':
        hour=raw.timestamp.dt.hour.to_numpy()+raw.timestamp.dt.minute.to_numpy()/60
        eligible=safe & (hour>=4.5)&(hour<=8.5)&(raw.sleep_efficiency_prev_night.to_numpy()<65)
        if not eligible.any():
            eligible=safe & (hour>=4.5)&(hour<=8.5)
    elif kind=='stress':
        eligible=safe&(raw.hr_bpm.to_numpy()>90)&(raw.steps_15m.to_numpy()<25)&(raw.hrv_rmssd_ms.to_numpy()<27)
    else:
        raise ValueError('Unknown navigation event')
    choices=np.flatnonzero(eligible)
    return int(choices[np.nanargmax(p[choices])]) if len(choices) else None


def threshold_metrics(y, probabilities, threshold):
    alert=probabilities>=threshold
    tp=int(((y==1)&alert).sum());fn=int(((y==1)&~alert).sum())
    fp=int(((y==0)&alert).sum());tn=int(((y==0)&~alert).sum())
    return dict(n=len(y),recall=tp/(tp+fn) if tp+fn else None,
                precision=tp/(tp+fp) if tp+fp else None,
                fpr=fp/(fp+tn) if fp+tn else None,tp=tp,fn=fn,fp=fp,tn=tn)


def exact_grouped_shapley(bundle, features, reference_count=8):
    """Enumerate all 2^10 coalitions for an exact grouped interventional game.

    The reference expectation uses eight fixed estimator-fit rows. These are
    exact GROUP contributions for that finite reference, not per-field SHAP or
    causal effects. Correlated/mixed feature states can be out of distribution.
    """
    names=bundle['feature_names']
    grouped=[f for group in EXPLANATION_GROUPS.values() for f in group]
    require(len(grouped)==len(set(grouped)) and set(grouped)==set(names),'Explanation groups must partition all inputs')
    background=bundle['explanation_background'][names].sample(n=reference_count,random_state=2026).reset_index(drop=True)
    groups=list(EXPLANATION_GROUPS);m=len(groups);coalitions=1<<m
    cube=np.tile(background.to_numpy(dtype=object),(coalitions,1)).reshape(coalitions,reference_count,len(names))
    row=features[names].iloc[0]
    bits=np.arange(coalitions)
    for i,group in enumerate(groups):
        selected=(bits&(1<<i))!=0
        for feature in EXPLANATION_GROUPS[group]:
            cube[selected,:,names.index(feature)]=row[feature]
    batch=pd.DataFrame(cube.reshape(-1,len(names)),columns=names)
    numeric=bundle['feature_contract']['numeric_features']
    batch[numeric]=batch[numeric].astype(float)
    with threadpool_limits(limits=2):
        values=bundle['classifier'].predict_proba(batch)[:,1].reshape(coalitions,reference_count).mean(axis=1)
        actual=float(bundle['classifier'].predict_proba(features[names])[0,1])
    contributions=[]
    for i,group in enumerate(groups):
        contribution=0.
        for coalition in range(coalitions):
            if coalition&(1<<i):
                continue
            size=coalition.bit_count()
            weight=1/(m*math.comb(m-1,size))
            contribution+=weight*(values[coalition|(1<<i)]-values[coalition])
        contributions.append(dict(group=group,contribution=float(contribution),features=EXPLANATION_GROUPS[group]))
    residual=float(actual-values[0]-sum(r['contribution'] for r in contributions))
    require(abs(residual)<1e-9 and abs(values[-1]-actual)<1e-10,'Explanation additivity failed')
    return dict(reference_probability=float(values[0]),prediction_probability=actual,
                contributions=contributions,additivity_residual=residual,
                coalitions=coalitions,reference_rows=reference_count,
                method='Exact grouped Shapley values, finite-reference interventional game')


@st.cache_data(show_spinner=False,max_entries=96)
def explain_cached(feature_json, model_sha, _bundle):
    features=pd.DataFrame([json.loads(feature_json)],columns=_bundle['feature_names'])
    return exact_grouped_shapley(_bundle,features)


def trajectory_chart(record, sample_index, snapshot, bundle):
    history=record['raw'].iloc[max(0,sample_index-16):sample_index+1]
    # Plotly date traces drop timezone offsets. Use local-clock naive dates
    # consistently for traces, forecast dates, ranges and the vertical marker.
    # Audit/CSV timestamps remain timezone-aware ISO strings.
    local_times=history.timestamp.dt.tz_localize(None)
    now=local_times.iloc[-1]
    observed=history.cgm_glucose_mgdl.where(history.sensor_status.eq('OK')&history.cgm_glucose_mgdl.between(40,400))
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.08,
                      row_heights=[.76,.24],specs=[[{}],[{'secondary_y':True}]])
    fig.add_trace(go.Scatter(x=local_times,y=observed,mode='lines+markers',name='Observed CGM',
                            line=dict(color=COLORS['navy'],width=2.4),marker=dict(size=4),connectgaps=False),row=1,col=1)
    if snapshot['baseline'] is not None:
        future=[now+pd.Timedelta(minutes=m) for m in bundle['horizon_minutes']]
        baseline=np.array(snapshot['baseline']['trajectory'])
        spread=np.array(bundle['calibration_horizon_rmse'])
        fig.add_trace(go.Scatter(x=future,y=baseline-spread,mode='lines',line=dict(width=0),
                                showlegend=False,hoverinfo='skip'),row=1,col=1)
        fig.add_trace(go.Scatter(x=future,y=baseline+spread,mode='lines',line=dict(width=0),fill='tonexty',
                                fillcolor='rgba(220,38,38,0.11)',name='Baseline ± calibration RMSE',hoverinfo='skip'),row=1,col=1)
        anchor=float(history.cgm_glucose_mgdl.iloc[-1])
        fig.add_trace(go.Scatter(x=[now]+future,y=[anchor]+baseline.tolist(),mode='lines+markers',
                                name='Baseline forecast',line=dict(color=COLORS['red'],width=2.8,dash='dash'),
                                marker=dict(size=4)),row=1,col=1)
        if snapshot['scenario'] is not None:
            fig.add_trace(go.Scatter(x=[now]+future,y=[anchor]+snapshot['scenario']['trajectory'],mode='lines+markers',
                                    name='What-if feature scenario',line=dict(color=COLORS['green'],width=2.8),
                                    marker=dict(size=4)),row=1,col=1)
    fig.add_hline(y=180,line_dash='dot',line_color='#D97706',annotation_text='180 mg/dL',
                  annotation_position='top left',row=1,col=1)
    fig.add_hline(y=70,line_dash='dot',line_color='#0284C7',annotation_text='70 mg/dL',
                  annotation_position='bottom left',row=1,col=1)
    fig.add_vline(x=now.isoformat(),line_color='#94A3B8',line_dash='dot',row=1,col=1)
    fig.add_trace(go.Bar(x=local_times,y=history.meal_carbs_g,name='Logged carbs',
                        marker_color='#A78BFA',width=8*60*1000),row=2,col=1,secondary_y=False)
    fig.add_trace(go.Scatter(x=local_times,y=history.steps_15m,name='Observed steps / 15m',mode='markers',
                            marker=dict(color=COLORS['cyan'],symbol='diamond',size=6)),row=2,col=1,secondary_y=True)
    fig.update_yaxes(title_text='Glucose (mg/dL)',row=1,col=1)
    fig.update_yaxes(title_text='Carbs (g)',row=2,col=1,secondary_y=False,rangemode='tozero')
    fig.update_yaxes(title_text='Steps',row=2,col=1,secondary_y=True,rangemode='tozero')
    fig.update_xaxes(range=[now-pd.Timedelta(hours=4),now+pd.Timedelta(hours=2)],tickformat='%H:%M',showgrid=False)
    fig.update_xaxes(title_text='Local time (IST)',row=2,col=1)
    fig.update_layout(height=490,margin=dict(l=12,r=12,t=42,b=10),paper_bgcolor='white',plot_bgcolor='white',
                      font=dict(family='Arial',color=COLORS['navy']),hovermode='x unified',
                      legend=dict(orientation='h',y=1.13,x=0,font=dict(size=11)),barmode='overlay')
    fig.update_yaxes(gridcolor='#EEF2F7',zeroline=False)
    return fig


def explanation_chart(explanation):
    table=pd.DataFrame(explanation['contributions'])
    table=table.assign(magnitude=table.contribution.abs()).sort_values('magnitude')
    colors=[COLORS['red'] if x>=0 else COLORS['green'] for x in table.contribution]
    points=table.contribution*100
    span=max(float(points.max()-points.min()),1.)
    fig=go.Figure(go.Bar(x=points,y=table.group,orientation='h',marker_color=colors,
                        text=[f'{v:+.2f} pp' for v in points],textposition='outside',cliponaxis=False,
                        textfont=dict(size=11)))
    fig.add_vline(x=0,line_color='#CBD5E1',line_width=1)
    fig.update_layout(height=390,margin=dict(l=10,r=30,t=10,b=20),paper_bgcolor='white',plot_bgcolor='white',
                      xaxis_title='Change in model probability (percentage points)',font=dict(color=COLORS['navy']),
                      xaxis=dict(gridcolor='#EEF2F7',range=[min(0,float(points.min()))-.2*span,max(0,float(points.max()))+.28*span]),
                      yaxis=dict(tickfont=dict(size=11)))
    return fig


def patient_note(pid, timestamp, snapshot, threshold, controls, ehr_row, model_version):
    title=f'Synthetic replay — clinician-review draft\nPatient: {pid}\nObservation: {pd.Timestamp(timestamp).isoformat()}\nModel: {model_version}\n'
    if snapshot['baseline'] is None:
        return title+'\nPREDICTION LOCKED. Telemetry or the required history is unreliable. No glycemic risk, forecast or lifestyle recommendation is inferred from this window.\n'
    base=snapshot['baseline'];scenario=snapshot['scenario']
    flag='above' if base['probability']>=threshold else 'below'
    text=title+(f'\nBaseline model probability of any CGM >180 mg/dL in the next two hours: {base["probability"]:.1%} '
                f'({flag} the {threshold:.0%} alert threshold). Predicted peak: {max(base["trajectory"]):.1f} mg/dL.\n')
    if scenario is not None:
        text+=(f'What-if feature scenario probability: {scenario["probability"]:.1%}; predicted peak: {max(scenario["trajectory"]):.1f} mg/dL. '
               f'Controls: walk-equivalent {controls["walk_minutes"]} min, carbohydrate reduction requested {controls["reduce_carbs_g"]} g, '
               f'GI {controls["gi_tier"]}, breathing scenario {"on" if controls["breathing"] else "off"}.\n')
    text+='\nDiscussion prompt for the patient: Please review your recent meal, activity and sensor readings with your clinical team. Any lifestyle change should fit your existing care plan and be reviewed by your clinician.\n'
    if float(ehr_row['egfr_ml_min'])<60 or ehr_row['med_regimen']=='basal_insulin':
        text+='Clinician note: the synthetic record contains reduced eGFR and/or insulin treatment; individualized review is needed before suggesting activity or diet changes.\n'
    text+='\nThis is a synthetic research replay, not a diagnosis or a treatment recommendation. The scenario edits model inputs while keeping observed glucose fixed; it does not establish the benefit or safety of an intervention prescribed now. No medication change is recommended. This draft has not been sent.\n'
    return text


def session_audit_record(pid,timestamp,snapshot,threshold,controls,model_version,model_sha):
    def prediction_summary(value):
        if value is None:
            return None
        return {'probability':value['probability'],'trajectory_mgdl':value['trajectory'],
                'outside_training_range':value['outside_training_range'],
                'feature_sha256':hashlib.sha256(value['features'].to_json(orient='records').encode()).hexdigest()}
    payload=dict(app_version=APP_VERSION,model_version=model_version,model_sha256=model_sha,
                 patient_id=pid,observation_timestamp=pd.Timestamp(timestamp).isoformat(),
                 status=snapshot['status'],safety_reason=snapshot['reason'],threshold=threshold,controls=controls,
                 baseline=prediction_summary(snapshot['baseline']),scenario=prediction_summary(snapshot['scenario']),
                 applied_scenario=snapshot.get('applied'))
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()
    payload['event_sha256']=digest
    payload['computed_at']=datetime.now(IST).isoformat(timespec='seconds')
    return payload


def append_optional_audit(event):
    """Optional local JSONL audit; no network transmission or credentials."""
    destination=os.environ.get('TWIN_AUDIT_PATH')
    if not destination:
        return None
    path=Path(destination).expanduser()
    path.parent.mkdir(parents=True,exist_ok=True)
    encoded=(json.dumps(event,sort_keys=True,allow_nan=False)+'\n').encode()
    descriptor=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_APPEND,0o600)
    try:
        written=os.write(descriptor,encoded)
        require(written==len(encoded),'Incomplete audit write')
    finally:
        os.close(descriptor)
    return str(path)


def install_styles():
    st.markdown('''<style>
    .stApp { background:#FFFFFF; }
    .block-container { padding-top:1.7rem; padding-bottom:2.5rem; max-width:1440px; }
    h1,h2,h3 { color:#0F172A; letter-spacing:-0.03em; }
    h1 { font-size:2rem !important; }
    [data-testid="stSidebar"] { background:#F7F8FC; border-right:1px solid #E2E8F0; }
    [data-testid="stMetric"] { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:12px 14px; }
    [data-testid="stMetricLabel"] { color:#64748B; }
    [data-testid="stMetricValue"] { font-size:1.55rem; color:#0F172A; }
    [data-testid="stMetricValue"] > div { white-space:normal !important; overflow-wrap:anywhere; }
    .eyebrow { font-size:11px; font-weight:700; letter-spacing:.16em; color:#5B21B6; margin-bottom:6px; }
    .governance { padding:10px 14px; background:#F4F0FC; border:1px solid #E9D5FF; border-radius:8px; color:#4C1D95; font-size:12px; }
    .subtitle { color:#64748B; font-size:14px; margin-top:-8px; margin-bottom:16px; }
    .medication-card { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:12px 14px; min-height:88px; }
    .medication-label { font-size:14px; color:#64748B; margin-bottom:6px; }
    .medication-value { font-size:19px; line-height:1.3; color:#0F172A; white-space:normal; overflow-wrap:anywhere; }
    </style>''',unsafe_allow_html=True)


def percent(value):
    return 'N/A' if value is None else f'{value:.1%}'


def main():
    st.set_page_config(page_title='Glycemic Twin | Physician CDS',page_icon='🩺',layout='wide',initial_sidebar_state='auto')
    install_styles()
    try:
        resources=load_resources(str(ROOT),asset_fingerprint(ROOT))
    except Exception as error:
        st.error(f'Dashboard unavailable: {error}')
        st.info('Use the complete Phase 3 ZIP and install requirements_phase3.txt. No fallback scores are generated.')
        st.stop()
    bundle=resources['bundle']
    st.markdown('<div class="eyebrow">PERSONALIZED GLYCEMIC DIGITAL TWIN</div>',unsafe_allow_html=True)
    st.title('Physician decision-support workspace')
    st.markdown('<div class="subtitle">Explore a patient’s glucose forecast, inspect signal quality and compare hypothetical lifestyle inputs.</div>',unsafe_allow_html=True)
    st.markdown('<div class="governance"><b>Research CDS · non-diagnostic</b> &nbsp; | &nbsp; Fully synthetic records &nbsp; | &nbsp; Historical replay, not a live device feed &nbsp; | &nbsp; DPDP Act 2023 compliance not certified</div>',unsafe_allow_html=True)
    st.caption(f'Active model: {bundle["model_version"]} · App: {APP_VERSION}')
    with st.sidebar:
        st.markdown('### Patient digital twin')
        allowed=list(resources['records'])
        def display_pid(pid):
            return f'{pid} · {ARCHETYPE_NAMES.get(pid,"Held-out patient")}'
        pid=st.selectbox('Held-out patient',allowed,format_func=display_pid,key='patient_id')
        record=resources['records'][pid]
        if st.session_state.get('active_patient')!=pid:
            st.session_state['active_patient']=pid
            st.session_state['sample_index']=select_event(record,'meal') or 24
            for key,value in DEFAULT_CONTROLS.items():
                st.session_state[key]=value
        st.caption('All selectable patients were excluded from fitting and calibration in the exported model.')
        st.markdown('### Event navigator')
        b1,b2=st.columns(2)
        for column,label,kind,key in [(b1,'Meal risk','meal','jump_meal'),(b2,'Dawn','dawn','jump_dawn'),
                                      (b1,'Stress','stress','jump_stress'),(b2,'Sensor fault','fault','jump_fault')]:
            target=select_event(record,kind)
            if column.button(label,key=key,disabled=target is None,width='stretch'):
                st.session_state['sample_index']=target
                for control,value in DEFAULT_CONTROLS.items():
                    st.session_state[control]=value
        index=st.slider('Replay timestamp',0,len(record['raw'])-1,key='sample_index')
        selected=record['raw'].iloc[index]
        timestamp=selected.timestamp
        st.caption(timestamp.strftime('%d %b %Y · %H:%M IST'))
        st.caption('Event jumps use model risk and observed inputs, not future outcome labels.')
        st.divider()
        threshold=st.slider('Alert probability threshold',.30,.75,.50,.01,key='alert_threshold')
        tm=threshold_metrics(resources['cohort_y'],resources['cohort_probability'],threshold)
        st.caption(f'Synthetic hold-out replay · {resources["cohort_patient_count"]} patients · {tm["n"]:,} dashboard-eligible labelled windows')
        a,b=st.columns(2)
        a.metric('Recall',percent(tm['recall']));b.metric('Precision',percent(tm['precision']))
        st.caption(f'False-positive rate: {percent(tm["fpr"])}. These cohort metrics do not establish clinical performance.')
        with st.expander('Model and data details'):
            st.write(f'Primary split: {bundle["primary_split"]}; model families: classifier {bundle["selected_families"]["classifier_family"]}, trajectory {bundle["selected_families"]["regressor_family"]}.')
            st.write('Current-packet faults always lock this dashboard, including a single dropout. Phase 2 allowed limited single-step extrapolation; safe historical repairs can still appear in a valid current window.')
            st.write('26 clinical fields; 29 encoded model columns. Fasting insulin is represented through HOMA-IR.')
    ehr_row=resources['ehr'].loc[pid]
    meta=record['meta'].iloc[index]
    history=record['raw'].iloc[:index+1]
    st.subheader(f'{pid} · {ARCHETYPE_NAMES.get(pid,"Held-out T2D profile")}')
    st.caption(f'Observation: {timestamp.strftime("%d %B %Y, %H:%M IST")} · Synthetic age {int(ehr_row.age_years)} · T2D duration {ehr_row.t2d_duration_years:g} years')
    cols=st.columns(6)
    for column,label,value in zip(cols,['HbA1c','HOMA-IR proxy','BMI','eGFR','TCF7L2 alleles','Medication'],
        [f'{ehr_row.baseline_hba1c:.1f}%',f'{ehr_row.homa_ir:.2f}',f'{ehr_row.bmi:.1f}',f'{ehr_row.egfr_ml_min:.0f}',
         f'{int(ehr_row.genetic_tcf7l2_risk)} / 2',MEDICATION_LABELS[ehr_row.med_regimen]]):
        if label=='Medication':
            column.markdown(f'<div class="medication-card"><div class="medication-label">Medication</div><div class="medication-value">{html.escape(value)}</div></div>',unsafe_allow_html=True)
        else:
            column.metric(label,value)
    st.caption(f'BMI: kg/m² · eGFR: mL/min/1.73 m² · C-peptide: {ehr_row.c_peptide_ng_ml:.2f} ng/mL · Fasting insulin: {ehr_row.fasting_insulin_uiu_ml:.2f} µIU/mL. Genetic and metabolic effects are simulation assumptions.')
    current_valid=bool(meta.raw_valid)
    current=f'{selected.cgm_glucose_mgdl:.0f} mg/dL' if current_valid else 'Unavailable'
    velocity=record['features'].velocity_15m.iloc[index]
    arrow='↑' if np.isfinite(velocity) and velocity>.5 else ('↓' if np.isfinite(velocity) and velocity<-.5 else '→')
    trend=f'{arrow} {velocity:+.2f} mg/dL/min' if current_valid and np.isfinite(velocity) else 'No reliable rate of change'
    exercise=selected.hr_bpm>90 and selected.steps_15m>=250
    stress=selected.hr_bpm>90 and selected.steps_15m<25 and selected.hrv_rmssd_ms<27
    state='Exercise pattern' if exercise else ('Stress pattern' if stress else 'Neither rule triggered')
    live=st.columns(4)
    live[0].metric('Observed CGM',current);live[0].caption(trend)
    live[1].metric('Heart rate / HRV',f'{selected.hr_bpm:.0f} / {selected.hrv_rmssd_ms:.0f}');live[1].caption(f'bpm / ms · {state}')
    live[2].metric('Last completed-night sleep',f'{selected.sleep_efficiency_prev_night:.0f}%');live[2].caption(f'{selected.sleep_duration_prev_night_h:.1f} hours · efficiency')
    step_sum=history.steps_15m.iloc[-4:].sum()
    live[3].metric('Steps in recent hour',f'{int(step_sum):,}' if len(history)>=4 else 'Insufficient history')
    live[3].caption(f'{history.active_kcal_15m.iloc[-4:].sum():.1f} active kcal in available intervals')
    locked=not current_valid or not bool(meta.prediction_eligible)
    if locked:
        st.error('PREDICTION LOCKED: Telemetry signal is corrupted or missing, or the required safe history is unavailable. Do not infer clinical state.')
        st.caption(f'Packet status: {selected.sensor_status} · Guardrail: {meta.safety_status}. Two or more consecutive missing samples require quarantine; this interface also locks any invalid current packet.')
    elif bool(meta.history_contains_imputation):
        st.warning('A permitted single-step estimate is present in the recent history. The current reading is observed; interpret this research prediction with caution.')
    st.subheader('What-if lifestyle inputs')
    st.caption('Hypothetical model-input scenario, not a prescription: observed CGM stays fixed. A future walk or breathing exercise has not been causally modelled or clinically validated.')
    meal_index=recent_meal_index(history)
    c1,c2,c3,c4=st.columns(4)
    with c1:
        walk=st.slider('Brisk walk equivalent (minutes)',0,45,key='walk_minutes',disabled=locked)
    with c2:
        carbs=st.slider('Reduce recent meal carbs (g)',0,50,key='reduce_carbs_g',disabled=locked or meal_index is None)
    with c3:
        gi=st.selectbox('Meal GI scenario',['Keep observed','High','Medium','Low'],key='gi_tier',disabled=locked or meal_index is None)
    with c4:
        breathing=st.toggle('Paced-breathing scenario',key='breathing',disabled=locked)
        st.caption('Changes HR/HRV inputs only when the stress rule is active.')
    controls=dict(walk_minutes=walk,reduce_carbs_g=carbs,gi_tier=gi,breathing=breathing)
    if meal_index is None:
        st.caption('No carbohydrate-containing meal is logged in the previous two hours. Meal controls are unavailable.')
    else:
        meal=history.iloc[meal_index]
        st.caption(f'Meal being edited: {meal.timestamp.strftime("%H:%M IST")} · {meal.meal_carbs_g:.1f} g carbohydrates · {meal.meal_gi_tier} GI. The source record is never changed.')
    try:
        snapshot=score_snapshot(resources,pid,index,controls)
    except Exception as error:
        st.error(f'Prediction unavailable: {error}. No fallback estimates are shown.')
        st.stop()
    if snapshot['baseline'] is not None:
        base=snapshot['baseline'];scenario=snapshot['scenario']
        k1,k2,k3=st.columns(3)
        k1.metric('Baseline 2h excursion risk',f'{base["probability"]:.1%}')
        k2.metric('What-if scenario risk',f'{scenario["probability"]:.1%}' if scenario else 'Unavailable',
                  delta=f'{100*(scenario["probability"]-base["probability"]):+.1f} pp' if scenario and abs(scenario['probability']-base['probability'])>1e-10 else None,delta_color='inverse')
        difference=max(base['trajectory'])-max(scenario['trajectory']) if scenario else None
        k3.metric('Predicted peak reduction',f'{difference:+.1f} mg/dL' if difference is not None else 'Unavailable')
        if base['probability']>=threshold:
            st.warning(f'Baseline model alert: probability exceeds the {threshold:.0%} threshold. This includes persistent high glucose; it is not exclusively a new-onset alert.')
        else:
            st.info(f'Baseline probability is below the {threshold:.0%} alert threshold. A low model score is not an assurance of safety.')
        if scenario is None:
            st.warning(f'Scenario unavailable: {snapshot.get("scenario_reason","unsupported input window")}. The baseline forecast remains visible.')
        if base['outside_training_range'] or (scenario and scenario['outside_training_range']):
            with st.expander('Some inputs are outside the observed training range',expanded=True):
                st.write('Baseline: '+(', '.join(base['outside_training_range']) or 'none'))
                st.write('Scenario: '+(', '.join(scenario['outside_training_range']) if scenario and scenario['outside_training_range'] else 'none'))
                st.caption('Marginal range checks are not an uncertainty estimate or a full out-of-distribution detector.')
        disagreement=(base['probability']>=threshold)!=(max(base['trajectory'])>180)
        if disagreement:
            st.caption('The classifier alert and point-forecast peak disagree. They are separate models; the probability is not calculated from the plotted peak.')
        if min(base['trajectory'])<70 or (scenario and min(scenario['trajectory'])<70):
            st.warning('A model point forecast falls below 70 mg/dL. This prototype does not validate hypoglycemia risk or recommend corrective treatment.')
        with st.expander('Applied scenario assumptions'):
            st.json(snapshot['applied'])
            st.caption('Walk mapping: 100 steps/min, activity energy = added steps × 0.039 × BMI/25. HR ≥100 and HRV ≥30 represent activity. Breathing mapping: HR ≤85 and HRV ≥40 when the stress rule is active. These are unvalidated input assumptions, not measured intervention effects.')
    st.plotly_chart(trajectory_chart(record,index,snapshot,bundle),width='stretch',key='trajectory_chart')
    st.caption('Past four hours: observed CGM, meal logs and activity. Next two hours: model forecasts. Shading is ±1 calibration-patient RMSE per horizon, not a confidence or prediction interval. Invalid CGM packets are omitted; no future observed glucose is displayed.')
    if snapshot['baseline'] is not None:
        left,right=st.columns([1.2,1],gap='large')
        with left:
            st.subheader('Why this model prediction?')
            options=['Baseline']+(['Scenario'] if snapshot['scenario'] is not None else [])
            explain_target=st.radio('Explain prediction',options,horizontal=True,key='explain_target')
            chosen=snapshot['baseline'] if explain_target=='Baseline' else snapshot['scenario']
            payload=json.dumps(chosen['features'].iloc[0].to_dict(),sort_keys=True)
            try:
                with st.spinner('Computing exact grouped model contributions…'):
                    explanation=explain_cached(payload,resources['model_sha'],bundle)
                st.plotly_chart(explanation_chart(explanation),width='stretch',key='explanation_chart')
                st.caption(f'Reference {explanation["reference_probability"]:.1%} + grouped contributions = {explanation["prediction_probability"]:.1%}. Exact enumeration of {explanation["coalitions"]:,} coalitions over 8 training-reference rows.')
                with st.expander('Explanation method and feature groups'):
                    st.write('These are exact Shapley values for the displayed groups and finite reference distribution. They are not exact individual-field SHAP values or causal effects. Correlated input combinations may be unrealistic.')
                    st.dataframe(pd.DataFrame([{'Group':g,'Fields':', '.join(fields)} for g,fields in EXPLANATION_GROUPS.items()]),hide_index=True,width='stretch')
                    st.caption(f'Additivity residual: {explanation["additivity_residual"]:.2e}')
            except Exception as error:
                st.error(f'Explanation unavailable: {error}. No substitute attribution bars are shown.')
        with right:
            st.subheader('Clinician-review draft')
            note=patient_note(pid,timestamp,snapshot,threshold,controls,ehr_row,bundle['model_version'])
            edited_note=st.text_area('Patient discussion and clinician note',note,height=360,key=f'note_{pid}_{index}_{hashlib.sha256(note.encode()).hexdigest()[:8]}')
            st.download_button('Download draft note',edited_note,file_name=f'{pid}_clinician_review_{timestamp.strftime("%Y%m%d_%H%M")}.txt',mime='text/plain',key='download_note')
            st.caption('Edit the draft before exporting. Nothing is sent to a patient or clinician.')
            future=[timestamp+pd.Timedelta(minutes=m) for m in bundle['horizon_minutes']]
            forecast=pd.DataFrame({'timestamp':future,'baseline_mgdl':snapshot['baseline']['trajectory'],
                                   'calibration_rmse_mgdl':bundle['calibration_horizon_rmse']})
            if snapshot['scenario'] is not None:
                forecast['scenario_mgdl']=snapshot['scenario']['trajectory']
            st.download_button('Download forecast values',forecast.to_csv(index=False),file_name=f'{pid}_forecast.csv',mime='text/csv',key='download_forecast')
    event=session_audit_record(pid,timestamp,snapshot,threshold,controls,bundle['model_version'],resources['model_sha'])
    if 'audit_events' not in st.session_state:
        st.session_state['audit_events']=[]
    if not st.session_state['audit_events'] or st.session_state['audit_events'][-1]['event_sha256']!=event['event_sha256']:
        st.session_state['audit_events'].append(event)
        try:
            append_optional_audit(event)
        except Exception as error:
            st.error(f'Persistent audit write failed: {error}. Download the session audit below.')
    latest=st.session_state['audit_events'][-1]
    with st.expander('Timestamp audit trail'):
        st.caption(f'Last computation: {latest["computed_at"]} · {len(st.session_state["audit_events"])} recorded session decisions')
        st.dataframe(pd.DataFrame([{k:e[k] for k in ['computed_at','patient_id','observation_timestamp','status','threshold']} for e in st.session_state['audit_events'][-50:]]),hide_index=True,width='stretch')
        st.download_button('Download session audit JSON',json.dumps(st.session_state['audit_events'],indent=2,allow_nan=False),file_name='glycemic_twin_session_audit.json',mime='application/json',key='download_audit')
        st.caption('Session audit is lost when the session ends unless downloaded. Set TWIN_AUDIT_PATH for optional local JSONL persistence; it is not a tamper-proof clinical audit system.')
    st.caption('Synthetic research prototype. Clinical deployment would require independent validation, access controls, operational monitoring and governance review.')


if __name__=='__main__':
    main()
