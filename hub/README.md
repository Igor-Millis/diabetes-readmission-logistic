---
license: mit
library_name: sklearn
tags:
- tabular-classification
- sklearn
- skops
- logistic-regression
- healthcare
- calibration
datasets:
- uci/diabetes-130-us-hospitals
metrics:
- roc_auc
- average_precision
- brier_score
- log_loss
pipeline_tag: tabular-classification
model-index:
- name: diabetes-readmission-logistic
  results:
  - task:
      type: tabular-classification
      name: 30-day readmission
    dataset:
      name: Diabetes 130-US Hospitals for Years 1999-2008 (UCI)
      type: tabular
    metrics:
    - type: roc_auc
      value: 0.670
      name: ROC AUC (held-out patients)
    - type: average_precision
      value: 0.219
      name: PR AUC
    - type: brier_score
      value: 0.0962
      name: Brier score
    - type: log_loss
      value: 0.3347
      name: Log loss
---

# Diabetes 30-day readmission — splined logistic regression

A calibrated logistic regression that estimates the probability that a diabetic inpatient encounter is
followed by a readmission **within 30 days**. Trained on the UCI *Diabetes 130-US Hospitals for Years
1999–2008* dataset (Strack et al., 2014).

**Educational / portfolio model. Not for clinical use.**

- Code, experiment and every metric: [github.com/Igor-Millis/diabetes-readmission-logistic](https://github.com/Igor-Millis/diabetes-readmission-logistic)
- Write-up: [igor-millis.github.io/work/diabetes-readmission](https://igor-millis.github.io/work/diabetes-readmission/)

## Model

scikit-learn `Pipeline`:

1. `FunctionTransformer(readmission.features.engineer)` — row-wise mapping of the raw UCI columns
   (age bands → years, ICD-9 codes → 9 chapters, admission/discharge IDs → groups, medication counts,
   "not measured" kept as a level). No fitted state.
2. `ColumnTransformer` — counts: `log1p` → cubic B-splines (5 knots) → standardisation;
   categoricals: one-hot with rare levels (< 50) pooled.
3. `LogisticRegression(C=0.01)` (L2), C chosen by 5-fold patient-grouped CV log loss.

## Evaluation (19,773 encounters of 13,998 patients never seen in training)

| metric | value |
|---|---|
| ROC AUC | 0.670 [0.656, 0.683] (patient-level bootstrap) |
| PR AUC (prevalence 0.113) | 0.219 |
| Brier / log loss | 0.0962 / 0.3347 |
| calibration slope / intercept | 1.04 / −0.001 |
| ECE (risk deciles) / Hosmer–Lemeshow p | 0.007 / 0.077 |
| recall among top 20% risk | 38.8% (lift 1.94×) |
| at threshold 0.131 (F1-optimal out-of-fold) | recall 0.475 · precision 0.206 · F1 0.287 · MCC 0.174 |

For comparison, a tuned LightGBM reached ROC AUC 0.681 (paired ΔAUC +0.011 [+0.006, +0.016]); the logistic
model keeps ~94% of the lift over chance, is equally well calibrated, and wins with < 10k training encounters.
Full comparison of 40+ metrics across 8 models in `results.json`.

## Usage

```python
import sys, pandas as pd, skops.io as sio
from huggingface_hub import snapshot_download

path = snapshot_download("igorlimamillis/diabetes-readmission-logistic")
sys.path.insert(0, path)                         # the pipeline references readmission.features.engineer
untrusted = sio.get_untrusted_types(file=f"{path}/lr_spline.skops")
print(untrusted)                                 # review: readmission.features.engineer + scipy BSpline classes
model = sio.load(f"{path}/lr_spline.skops", trusted=untrusted)

df = pd.read_csv("diabetic_data.csv", na_values="?")   # raw UCI format
risk = model.predict_proba(df)[:, 1]
```

A complete single-row example is in `example.py`. Requires `scikit-learn==1.9.*`.

## Intended use and limitations

- Intended for learning and demonstrating model evaluation on tabular clinical data.
- 1999–2008 US data, ICD-9 coding; readmissions to other hospitals are not observed (label under-counts).
- Encounters ending in death or hospice were excluded from training; do not score them.
- Lower discrimination for patients aged 70+ (AUC 0.63) than under 50 (0.71).
- Coefficients describe associations, not causal effects.

## Citation (data)

Strack, B., DeShazo, J. P., Gennings, C., Olmo, J. L., Ventura, S., Cios, K. J., & Clore, J. N. (2014).
Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient
Records. *BioMed Research International*, 2014, 781670. Dataset: UCI ML Repository, doi:10.24432/C5230J (CC BY 4.0).
