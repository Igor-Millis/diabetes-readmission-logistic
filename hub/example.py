"""Load the model from the Hub and score one encounter.

    pip install scikit-learn==1.9.* skops pandas huggingface_hub
    python example.py
"""
import sys

import pandas as pd
import skops.io as sio
from huggingface_hub import snapshot_download

path = snapshot_download("igorlimamillis/diabetes-readmission-logistic")
sys.path.insert(0, path)                      # makes `readmission.features.engineer` importable

file = f"{path}/lr_spline.skops"
untrusted = sio.get_untrusted_types(file=file)
print("types to review before trusting:", untrusted)   # readmission.features.engineer + scipy BSpline classes
model = sio.load(file, trusted=untrusted)

# One encounter in the raw UCI column format ("?" already converted to NaN).
row = {
    "race": "Caucasian", "gender": "Female", "age": "[70-80)", "admission_type_id": 1,
    "discharge_disposition_id": 3, "admission_source_id": 7, "time_in_hospital": 6, "payer_code": "MC",
    "medical_specialty": "InternalMedicine", "num_lab_procedures": 55, "num_procedures": 1, "num_medications": 18,
    "number_outpatient": 0, "number_emergency": 1, "number_inpatient": 2, "diag_1": "428", "diag_2": "250.02",
    "diag_3": "403", "number_diagnoses": 9, "max_glu_serum": None, "A1Cresult": ">8", "change": "Ch",
    "diabetesMed": "Yes",
}
drugs = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide", "glipizide",
         "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol", "troglitazone",
         "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
         "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]
row.update({d: "No" for d in drugs})
row["insulin"] = "Up"

p = model.predict_proba(pd.DataFrame([row]))[0, 1]
print(f"estimated 30-day readmission risk: {p:.3f}")
