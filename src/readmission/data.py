"""Loading and cohort definition.

Cohort rules (documented in the README):

* the target is readmission within 30 days (``readmitted == "<30"``); ``">30"`` and ``"NO"`` are negatives;
* encounters that ended in death or hospice are removed — those patients cannot (or are not expected to)
  come back, so keeping them would teach the model that dying "protects" against readmission;
* the three encounters with ``gender == "Unknown/Invalid"`` are removed;
* ``"?"`` is read as missing. Nothing is imputed here: missingness becomes its own category in features.py.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "diabetic_data.csv"

# discharge_disposition_id: 11 expired, 13/14 hospice, 19/20/21 expired (Medicaid hospice)
DEATH_OR_HOSPICE = {11, 13, 14, 19, 20, 21}

TARGET = "readmit30"
GROUP = "patient_nbr"


def load_raw(path: Path | str = RAW) -> pd.DataFrame:
    return pd.read_csv(path, na_values="?", low_memory=False)


def build_cohort(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Apply the cohort rules. Returns the cohort and a small audit trail of removed rows."""
    audit = {"raw_encounters": int(len(raw)), "raw_patients": int(raw[GROUP].nunique())}
    d = raw.copy()
    dead = d["discharge_disposition_id"].isin(DEATH_OR_HOSPICE)
    audit["removed_death_or_hospice"] = int(dead.sum())
    d = d[~dead]
    bad_sex = d["gender"].eq("Unknown/Invalid")
    audit["removed_unknown_gender"] = int(bad_sex.sum())
    d = d[~bad_sex]
    d[TARGET] = d["readmitted"].eq("<30").astype(int)
    d = d.drop(columns=["readmitted"]).reset_index(drop=True)
    audit["encounters"] = int(len(d))
    audit["patients"] = int(d[GROUP].nunique())
    audit["prevalence"] = float(d[TARGET].mean())
    return d, audit
