# KISS_IIM_Jammu
KISS-Glycemic-Twin-Digital-Twin-Challenge-2026

# GLYCEMIC TWIN
### KISS | Happiest Health Digital Twin Challenge 2026

**A synthetic, research-stage digital twin proof of concept for short-horizon glucose excursion forecasting and physician decision support.**

> **Research prototype only.** This project uses synthetic patient records and simulated wearable data. It is not clinically validated, is not a medical device, and must not be used to diagnose, treat or make medication decisions for a real patient.

## 1. Team Details

- **Team name:** KISS
- **Team leader:** Shivam Soni (mbahc25075@iimj.ac.in | +91 7024364032)
- **Team member:** Somesh
- **Institution:** Indian Institute of Management Jammu

## 2. Problem Statement

Glucose excursions in people with Type 2 diabetes can be influenced by recent glucose trends, meals, physical activity, sleep, autonomic signals and baseline metabolic characteristics. A static medical record cannot, by itself, show how a patient's glucose may evolve over the next two hours.

GLYCEMIC TWIN demonstrates how historical electronic health record (EHR) information can be combined with timestamped wearable and continuous glucose monitoring (CGM) data to generate short-horizon glucose forecasts and flag possible future excursions.

The prototype focuses on one localized outcome: predicting whether observed CGM glucose may exceed 180 mg/dL during the following two hours.

## 3. Healthcare Use Case

The prototype provides a research-oriented physician decision-support workspace that allows a user to:

- Select a synthetic patient profile.
- Inspect historical glucose, activity, sleep and heart-rate-related signals.
- View a model-generated glucose trajectory for the next 15–120 minutes.
- Review the model's estimated probability of a future glucose excursion.
- Explore hypothetical changes to meal and recent activity inputs.
- Inspect grouped model-feature contributions and data-quality warnings.
- Review a draft clinician note and export selected results.

All patients and measurements are synthetic. No live medical device or hospital system is connected.

## 4. How the Digital Twin Works

The system has two principal data streams.

**Stream A — Static and historical EHR data**

Synthetic patient profiles include selected baseline characteristics such as HbA1c, BMI, HOMA-IR, C-peptide, eGFR, genetic-risk proxy and medication category.

**Stream B — Dynamic time-series data**

The simulated wearable dataset contains timestamped CGM, heart rate, heart-rate variability, steps, activity energy, meal logs and sleep-related variables.

The engine joins the streams using patient identifiers and timestamp-aligned historical windows. It derives model features, applies data-quality and prediction-eligibility checks, and supplies the eligible inputs to the fitted models.

The dashboard displays historical replay and model forecasts; it does not receive live wearable feeds.

## 5. Technical Stack

- Python
- pandas and NumPy for data processing
- SciPy and threadpoolctl for supporting computation
- scikit-learn for model training, calibration and evaluation
- Streamlit for the interactive dashboard
- Plotly for time-series and model-explanation visualizations

## 6. AI/ML Models

Two model families are compared.

**Model A**
- Logistic regression with probability calibration for excursion classification.
- Ridge regression for multi-horizon glucose forecasting.

**Model B**
- Histogram gradient boosting with probability calibration for excursion classification.
- Separate histogram gradient boosting regressors for eight forecast horizons.

The primary evaluation regime is an 80:20 patient-level split, with an additional calibration subset within the training patients. The model family is selected using internal validation rather than the final test results.

The training engine also evaluates 60:40, 70:30 and 90:10 split regimes as sensitivity analyses.

## 7. Synthetic Data and Evaluation

The supplied project describes:

- 40 synthetic patient profiles.
- 53,760 timestamped wearable packets.
- A nominal 15-minute sampling cadence across a simulated 14-day period.
- Patient-level separation between model fitting, calibration and testing.
- Four deliberately held-out patient archetypes for replay and demonstration.

The reported primary 80:20 Model B results are:

| Metric | Reported result |
|---|---:|
| AUROC | 0.993 |
| AUPRC (average precision) | 0.972 |
| Recall | 91.1% |
| Precision (PPV) | 90.4% |
| Brier score | 0.0230 |
| Glucose trajectory RMSE | 10.16 mg/dL |
| Glucose trajectory MAE | 7.08 mg/dL |
| New-crossing recall | 51.9% |

These figures are reported from the supplied synthetic-data evaluation. They must not be interpreted as estimates of clinical performance.

The new-crossing recall is particularly important: overall excursion recall can include readings from patients who are already experiencing persistent hyperglycemia. The ability to detect a new threshold crossing is a different and harder task.

**Reproducibility:** The reported figures should be treated as verified for submission only after the final source, input files, fitted bundle and evaluation outputs have been rerun together and reconciled.

## 8. Safety, Data Quality and Model Limitations

The prototype implements or documents the following safeguards:

- Validation of CGM values and sensor status.
- Restrictions on historical lookback and future information access.
- Handling of missing and corrupt sensor packets.
- Quarantine or suppression of predictions for ineligible current windows.
- Patient-level data partitioning.
- Serialization and prediction-reconciliation checks.
- Timestamped session decisions and model-input explanations.

These are software-level safeguards, not evidence of clinical safety certification.

The model has important limitations:

1. The dataset is synthetic and uses a shared simulation mechanism.
2. Held-out synthetic patients do not establish performance on real-world patients.
3. The model is not clinically validated and has not undergone external clinical validation.
4. The what-if controls alter model inputs; they do not establish causal treatment effects.
5. A displayed RMSE-based band is an error scale, not a calibrated prediction interval.
6. The prototype is not certified under the DPDP Act or by CDSCO, and no such certification is claimed.
7. The application is a historical replay, not a live medical-device integration.

## 9. Architecture Diagram

See architecture_diagram.pdf for the system architecture, covering the EHR and wearable streams, data-quality safeguards, feature engineering, model evaluation and physician dashboard.

## 10. Repository Structure

```text
KISS_IIM_Jammu/
├── README.md
├── LICENSE
├── requirements.txt
├── 2_train_twin_engine.py
├── 3_doctor_dashboard.py
├── assets_manifest.json
├── architecture_diagram.pdf
├── data/
│   ├── patients_ehr.csv
│   └── wearable_timeseries.csv
├── models/
│   └── twin_model_bundle.pkl
├── docs/
│   ├── evaluation_metrics_report.md
│   ├── phase1_quality_audit.md
│   └── demo_video_link.txt
└── KISS_IIM_Jammu_GLYCEMIC_TWIN.pdf (presentation)
```

## 11. Installation and Execution

Use Python 3.11 or 3.12 in a clean virtual environment. The training engine requires scikit-learn 1.6 or later.

### Step 1 — Install dependencies

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install the project dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2 — Train and evaluate the models

Ensure both synthetic CSV files are inside the `data/` directory, then run:

```bash
python 2_train_twin_engine.py --data-dir data --output-dir outputs
```

The engine should generate the trained model bundle, evaluation reports, data-quality audits and prediction outputs inside `outputs/`.

Check the final terminal status and inspect the generated evaluation and guardrail reports before proceeding.

### Step 3 — Prepare the dashboard model

The dashboard expects the model bundle at `models/twin_model_bundle.pkl`, whereas the training engine exports it to `outputs/twin_model_bundle.pkl`.

Create the model directory and copy the exported bundle:

```bash
mkdir -p models
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force models
Copy-Item outputs/twin_model_bundle.pkl models/twin_model_bundle.pkl
```

On macOS/Linux:

```bash
cp outputs/twin_model_bundle.pkl models/twin_model_bundle.pkl
```

### Step 4 — Generate the integrity manifest

Before launching the dashboard, generate `assets_manifest.json` containing SHA-256 hashes for the exact source, input and model files being used.

The manifest must include the relative paths expected by the dashboard, including:

- `2_train_twin_engine.py`
- `3_doctor_dashboard.py`
- `data/patients_ehr.csv`
- `data/wearable_timeseries.csv`
- `models/twin_model_bundle.pkl`

Each entry must contain its file's SHA-256 digest under the `sha256` key.

Do not edit or replace these files after generating the manifest. If any file changes, regenerate the manifest.

### Step 5 — Launch the doctor dashboard

```bash
streamlit run 3_doctor_dashboard.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

If the dashboard reports a missing file or integrity-check failure, stop and correct the package paths or regenerate the manifest. Do not bypass the checks.

## 12. Demonstration Video

**Video:** https://youtu.be/-CQVl9Om9N0 

The completed video should be at least 20 minutes long and demonstrate the project purpose, architecture, synthetic data, training and evaluation, patient selection, forecast display, what-if controls, safety lockout, explanation panel, limitations and actual execution of the prototype.

Ensure the link is accessible to the evaluation team.

## 13. Open-Source Licence

This project is intended to use the MIT License for the project code, subject to confirmation that all included code and dependencies may be redistributed under the selected licence.

The licence applies to the project code only; it does not grant rights to external datasets, third-party material or trademarks. The synthetic data and trained model should be distributed only with their provenance and limitations clearly documented.


## 14. Intended Use and Disclaimer

GLYCEMIC TWIN is an educational and research-stage proof of concept created for the Happiest Health Digital Twin Challenge 2026.

It is intended to demonstrate data-stream fusion, time-series prediction, historical patient replay and a physician-oriented interface.

It must not be used for clinical decisions, medication changes, emergency triage or patient treatment. All forecasts and hypothetical scenarios require independent validation before any clinical use can be considered.


## You can also run it in google colab by following the steps:
Step 1: Upload Your all Files to Google Colab.


Step 2: Run Phase 2 Training in Colab (Copy-Paste into Cell 1)

"
 1. Install/upgrade required libraries in Colab
!pip install -q "scikit-learn>=1.6.0" pandas numpy scipy threadpoolctl streamlit plotly

 2. Move the uploaded CSVs into the 'data/' folder expected by the script
import os, shutil
os.makedirs("data", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

if os.path.exists("patients_ehr.csv"):
    shutil.copy("patients_ehr.csv", "data/patients_ehr.csv")
if os.path.exists("wearable_timeseries.csv"):
    shutil.copy("wearable_timeseries.csv", "data/wearable_timeseries.csv")

 3. Run the Phase 2 Digital Twin Engine
!python 2_train_twin_engine.py --data-dir data --output-dir outputs
"


Step 3: Verify the "What-If" Digital Twin Simulation Inside Colab (Cell 2)
"
import pickle
import pandas as pd
from importlib.machinery import SourceFileLoader

 Load the functions from your script and the trained model bundle
eng = SourceFileLoader("eng", "2_train_twin_engine.py").load_module()
with open("outputs/twin_model_bundle.pkl", "rb") as f:
    bundle = pickle.load(f)

ehr = pd.read_csv("data/patients_ehr.csv", keep_default_na=False, na_values=[''])
ts = pd.read_csv("data/wearable_timeseries.csv", keep_default_na=False, na_values=[''])

 Select Archetype PT_001 (High Carb Spiker) at Lunch (Index 150: 117.65g High-GI meal)
pt1_ehr = ehr[ehr.patient_id == "PT_001"].iloc[0]
pt1_ts = ts[ts.patient_id == "PT_001"].reset_index(drop=True)
hist_baseline = pt1_ts.iloc[:151].copy()

 1. Baseline Prediction (No Intervention)
base_out = eng.predict_latest(bundle, pt1_ehr, hist_baseline)

 2. Counterfactual "What-If" Intervention (-40g Carbs, Swap to Low GI, +30m Brisk Walk)
hist_whatif = hist_baseline.copy()
hist_whatif.loc[hist_whatif.index[-1], "meal_carbs_g"] -= 40.0
hist_whatif.loc[hist_whatif.index[-1], "meal_gi_tier"] = "Low"
hist_whatif.loc[hist_whatif.index[-1], "meal_gi"] = 45
hist_whatif.loc[hist_whatif.index[-2:], "steps_15m"] += 450
hist_whatif.loc[hist_whatif.index[-2:], "active_kcal_15m"] += 22.0

whatif_out = eng.predict_latest(bundle, pt1_ehr, hist_whatif)

print(f"Current CGM: {hist_baseline.cgm_glucose_mgdl.iloc[-1]} mg/dL")
print(f"BASELINE (No Action) -> 2h Spike Risk: {base_out['probability']*100:.1f}% | Peak: {max(base_out['trajectory']):.1f} mg/dL")
print(f"WHAT-IF (Walk+LowGI) -> 2h Spike Risk: {whatif_out['probability']*100:.1f}% | Peak: {max(whatif_out['trajectory']):.1f} mg/dL")
"

Step 4: How to Run the Phase 3 Doctor Dashboard
Copy paste in next cell and run it.
"
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
!streamlit run 3_doctor_dashboard.py &>/dev/null &
!./cloudflared tunnel --url http://localhost:8501
"

What to Click Once the Link Appears:
1. Wait ~5 seconds after running the cell above until you see the box that says:Your quick Tunnel has been created! Visit it at:[https://xxxx-xxxx-xxxx.trycloudflare.com](https://xxxx-xxxx-xxxx.trycloudflare.com)   
2. Click that [https://...trycloudflare.com](https://...trycloudflare.com) link. (If it says "502 Bad Gateway" on the very first second, just refresh the page once after 5 seconds).   
3. Your Physician Decision-Support Workspace (3_doctor_dashboard.py) will load with:   
  - The 4 Archetype Patients (PT_001 to PT_004) in the left sidebar dropdown.   
  - The 4 Quick-Jump Event Buttons (Meal risk, Dawn, Stress, Sensor fault).   
  - The What-If Lifestyle Sliders (Brisk walk, Reduce meal carbs, Meal GI scenario, Paced-breathing).   
  - The Dual-Trajectory Plotly Chart + Exact Grouped Shapley Attribution Bars + Downloadable Clinician Note & Audit Trail.
