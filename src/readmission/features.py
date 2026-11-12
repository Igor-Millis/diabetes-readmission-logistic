"""Feature engineering.

``engineer`` is a pure, row-wise function (no statistics learned from data), so it can sit at the start
of a scikit-learn Pipeline without leaking anything across folds. Everything that *is* learned — scaling,
spline knots, rare-level pooling — happens inside the Pipeline, per fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, SplineTransformer, StandardScaler

# ---------------------------------------------------------------- vocabularies
COUNTS = ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
          "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses"]
NUMERIC = ["age_years", *COUNTS, "n_diabetes_drugs", "n_dose_changes"]

# The 23 medication columns: "No", "Steady", "Up", "Down".
DRUGS = ["metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride", "acetohexamide",
         "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose", "miglitol",
         "troglitazone", "tolazamide", "examide", "citoglipton", "insulin", "glyburide-metformin",
         "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone"]
# Drugs common enough to keep as their own categorical feature; the rest only enter through the counts.
MAIN_DRUGS = ["insulin", "metformin", "glipizide", "glyburide", "pioglitazone", "rosiglitazone", "glimepiride"]

CATEGORICAL = ["race", "gender", "admission_type", "discharge", "admission_source", "payer", "specialty",
               "diag_1_grp", "diag_2_grp", "diag_3_grp", "A1Cresult", "max_glu_serum", "change", "diabetesMed",
               *MAIN_DRUGS]

TOP_SPECIALTIES = ["InternalMedicine", "Emergency/Trauma", "Family/GeneralPractice", "Cardiology",
                   "Surgery-General", "Nephrology", "Orthopedics", "Orthopedics-Reconstructive", "Radiologist",
                   "Pulmonology", "Psychiatry", "Urology"]


def icd9_chapter(code: object) -> str:
    """Group an ICD-9-CM code into the 9 groups used by Strack et al. (2014), plus Missing."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "Missing"
    s = str(code).strip()
    if s[:1] in ("V", "E"):
        return "Other"
    try:
        x = float(s)
    except ValueError:
        return "Other"
    if 250 <= x < 251:
        return "Diabetes"
    if 390 <= x <= 459 or int(x) == 785:
        return "Circulatory"
    if 460 <= x <= 519 or int(x) == 786:
        return "Respiratory"
    if 520 <= x <= 579 or int(x) == 787:
        return "Digestive"
    if 800 <= x <= 999:
        return "Injury"
    if 710 <= x <= 739:
        return "Musculoskeletal"
    if 580 <= x <= 629 or int(x) == 788:
        return "Genitourinary"
    if 140 <= x <= 239:
        return "Neoplasms"
    return "Other"


def _admission_type(v: float) -> str:
    return {1: "Emergency", 2: "Urgent", 3: "Elective", 7: "Emergency"}.get(v, "Other/Unknown")


def _discharge(v: float) -> str:
    if v == 1:
        return "Home"
    if v in (6, 8):
        return "Home + health service"
    if v in (3, 4, 24):
        return "SNF/ICF"
    if v in (2, 9, 10, 23, 27, 28, 29, 30, 5, 15):
        return "Other inpatient facility"
    if v == 22:
        return "Rehab"
    if v == 7:
        return "Left against advice"
    return "Other/Unknown"


def _admission_source(v: float) -> str:
    if v == 7:
        return "Emergency room"
    if v in (1, 2, 3):
        return "Referral"
    if v in (4, 5, 6, 10, 18, 22, 25, 26):
        return "Transfer"
    return "Other/Unknown"


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Raw UCI columns -> modelling columns (NUMERIC + CATEGORICAL). Row-wise, no fitted state."""
    out = pd.DataFrame(index=df.index)
    # "[70-80)" -> 75
    out["age_years"] = df["age"].str.extract(r"\[(\d+)-")[0].astype(float) + 5
    for c in COUNTS:
        out[c] = df[c].astype(float)
    meds = df[DRUGS]
    out["n_diabetes_drugs"] = meds.ne("No").sum(axis=1).astype(float)
    out["n_dose_changes"] = meds.isin(["Up", "Down"]).sum(axis=1).astype(float)

    out["race"] = df["race"].fillna("Missing")
    out["gender"] = df["gender"]
    out["admission_type"] = df["admission_type_id"].map(_admission_type)
    out["discharge"] = df["discharge_disposition_id"].map(_discharge)
    out["admission_source"] = df["admission_source_id"].map(_admission_source)
    out["payer"] = df["payer_code"].fillna("Missing")
    spec = df["medical_specialty"]
    out["specialty"] = np.where(spec.isna(), "Missing", np.where(spec.isin(TOP_SPECIALTIES), spec, "Other"))
    for i in (1, 2, 3):
        out[f"diag_{i}_grp"] = df[f"diag_{i}"].map(icd9_chapter)
    # Not measured is the common case (83% / 95%): keep it as a level, it is informative.
    out["A1Cresult"] = df["A1Cresult"].fillna("None")
    out["max_glu_serum"] = df["max_glu_serum"].fillna("None")
    out["change"] = df["change"]
    out["diabetesMed"] = df["diabetesMed"]
    for c in MAIN_DRUGS:
        out[c] = df[c]
    return out[NUMERIC + CATEGORICAL]


# ---------------------------------------------------------------- preprocessors
def _onehot() -> OneHotEncoder:
    # Rare levels (< 50 encounters in the training fold) are pooled; unseen levels at predict time are ignored.
    return OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50, sparse_output=False)


def preprocessor(kind: str) -> ColumnTransformer:
    """
    ``linear``  — standardized numerics as-is (what most tutorials do).
    ``spline``  — log1p on counts, then cubic B-splines: lets a *linear-in-parameters* model bend.
    ``tree``    — numerics untouched, categoricals one-hot (trees do not need scaling).
    """
    if kind == "linear":
        num = StandardScaler()
    elif kind == "spline":
        num = make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                            SplineTransformer(n_knots=5, degree=3, knots="uniform", extrapolation="constant"),
                            StandardScaler())
    elif kind == "tree":
        num = "passthrough"
    else:
        raise ValueError(kind)
    return ColumnTransformer([("num", num, NUMERIC), ("cat", _onehot(), CATEGORICAL)],
                             verbose_feature_names_out=False)


ENGINEER = FunctionTransformer(engineer)


# ---------------------------------------------------------------- "closing the gap" variants
ICD3 = ["diag_1_3", "diag_2_3", "diag_3_3"]


def icd9_3digit(code: object) -> str:
    """'250.83' -> '250', 'V57' -> 'V57', 'E888' -> 'E888', NaN -> 'Missing'."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return "Missing"
    return str(code).split(".")[0]


def engineer_plus(df: pd.DataFrame) -> pd.DataFrame:
    """engineer() plus the 3-digit ICD-9 codes (high-cardinality: ~700 distinct primary diagnoses)."""
    out = engineer(df)
    for i in (1, 2, 3):
        out[f"diag_{i}_3"] = df[f"diag_{i}"].map(icd9_3digit).to_numpy()
    return out


ENGINEER_PLUS = FunctionTransformer(engineer_plus)

# Interactions suggested by the trees: prior utilisation x where the patient goes, age x utilisation, LOS x meds.
INTERACT_NUM = ["number_inpatient", "number_emergency", "number_outpatient", "time_in_hospital", "num_medications",
                "age_years", "number_diagnoses"]


def preprocessor_plus(icd3: bool, interactions: bool) -> ColumnTransformer:
    from sklearn.preprocessing import PolynomialFeatures
    spline = make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"),
                           SplineTransformer(n_knots=5, degree=3, knots="uniform", extrapolation="constant"),
                           StandardScaler())
    parts = [("num", spline, NUMERIC), ("cat", _onehot(), CATEGORICAL)]
    if icd3:
        parts.append(("icd3", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=100,
                                            sparse_output=False), ICD3))
    if interactions:
        # pairwise products of log-counts (21 terms) and log-counts x discharge destination / admission source
        logs = make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler(),
                             PolynomialFeatures(degree=2, interaction_only=True, include_bias=False), StandardScaler())
        parts.append(("pairs", logs, INTERACT_NUM))
        parts.append(("cross", make_pipeline(_CrossTransformer(), StandardScaler()),
                      ["number_inpatient", "number_emergency", "discharge", "admission_source"]))
    return ColumnTransformer(parts, verbose_feature_names_out=False)


from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402


class _CrossTransformer(TransformerMixin, BaseEstimator):
    """log1p(count) x one-hot(category) for two counts and two categorical columns."""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.levels_ = {c: [v for v, n in X[c].value_counts().items() if n >= 200]
                        for c in ("discharge", "admission_source")}
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        cols = []
        for cnt in ("number_inpatient", "number_emergency"):
            lc = np.log1p(X[cnt].to_numpy(float))
            for cat, levels in self.levels_.items():
                for lv in levels:
                    cols.append(lc * (X[cat].to_numpy() == lv))
        return np.column_stack(cols)
