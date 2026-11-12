"""End-to-end experiment. Writes results/results.json, models/ and figures/.

    python scripts/download_data.py
    python scripts/run_experiment.py            # ~10-15 min on a laptop CPU

Every number in the README comes from results/results.json produced by this script.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, GroupShuffleSplit, RandomizedSearchCV, StratifiedGroupKFold,
                                     StratifiedKFold, cross_val_predict, cross_val_score)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from readmission.data import GROUP, TARGET, build_cohort, load_raw  # noqa: E402
from readmission.features import CATEGORICAL, ENGINEER, NUMERIC, engineer, preprocessor  # noqa: E402
from readmission.irls import fit_logit  # noqa: E402
from readmission.metrics import (at_threshold, choose_thresholds, cluster_bootstrap, net_benefit,  # noqa: E402
                                 probabilistic, ranking, reliability_bins, summary, top_k)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
SEED = 42
RES = ROOT / "results"
FIG = ROOT / "figures"
MOD = ROOT / "models"
for d in (RES, FIG, MOD):
    d.mkdir(exist_ok=True)

t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


# ============================================================== 1. data & audit
raw = load_raw()
df, audit = build_cohort(raw)
miss = raw.isna().mean().sort_values(ascending=False)
audit["missing"] = {k: float(v) for k, v in miss[miss > 0].items()}
audit["encounters_per_patient"] = {str(k): int(v) for k, v in raw.groupby(GROUP).size().clip(upper=5).value_counts().sort_index().items()}
audit["share_encounters_repeat_patients"] = float((raw.groupby(GROUP)[GROUP].transform("size") > 1).mean())
audit["n_icd9_codes_diag1"] = int(raw["diag_1"].nunique())
audit["n_specialties"] = int(raw["medical_specialty"].nunique())
audit["readmitted_raw"] = {k: float(v) for k, v in raw["readmitted"].value_counts(normalize=True).items()}
log(f"cohort: {audit['encounters']} encounters, {audit['patients']} patients, prevalence {audit['prevalence']:.4f}")

y = df[TARGET].to_numpy()
groups = df[GROUP].to_numpy()
X = df.drop(columns=[TARGET])

# ============================================================== 2. patient-level split
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr, te = next(gss.split(X, y, groups))
Xtr, Xte, ytr, yte, gtr, gte = X.iloc[tr], X.iloc[te], y[tr], y[te], groups[tr], groups[te]
assert not set(gtr) & set(gte)
split = {"train": int(len(tr)), "test": int(len(te)), "train_patients": int(len(set(gtr))),
         "test_patients": int(len(set(gte))), "train_prev": float(ytr.mean()), "test_prev": float(yte.mean())}
log(f"split: {split}")
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)


def pipe(kind: str, model) -> Pipeline:
    return Pipeline([("engineer", ENGINEER), ("prep", preprocessor(kind)), ("model", model)])


def lr(**kw) -> LogisticRegression:
    return LogisticRegression(max_iter=3000, **kw)


C_GRID = list(np.logspace(-3, 1, 9))
specs: dict[str, tuple[Pipeline, dict | None, str]] = {
    "lr_one": (Pipeline([("engineer", ENGINEER),
                         ("prep", ColumnTransformer(
                             [("x", "passthrough", ["number_inpatient"])])),
                         ("model", lr(C=1e6))]), None, "grid"),
    "lr_linear": (pipe("linear", lr()), {"model__C": C_GRID}, "grid"),
    "lr_spline": (pipe("spline", lr()), {"model__C": C_GRID}, "grid"),
    "lr_enet": (pipe("spline", LogisticRegression(solver="saga", tol=1e-3, max_iter=500)),
                {"model__C": [0.003, 0.01, 0.03, 0.1], "model__l1_ratio": [0.5, 1.0]}, "grid"),
    "lr_balanced": (pipe("spline", lr(class_weight="balanced")), {"model__C": C_GRID}, "grid"),
    "rf": (pipe("tree", RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=SEED)),
           {"model__min_samples_leaf": [5, 20, 50], "model__max_features": ["sqrt", 0.3]}, "grid"),
    "lgbm": (pipe("tree", LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1)),
             {"model__n_estimators": [200, 400, 800], "model__learning_rate": [0.01, 0.03, 0.05],
              "model__num_leaves": [7, 15, 31, 63], "model__min_child_samples": [20, 50, 100, 200],
              "model__subsample": [0.7, 1.0], "model__subsample_freq": [1], "model__colsample_bytree": [0.5, 0.8, 1.0],
              "model__reg_lambda": [0.0, 1.0, 10.0]}, "random"),
}

fitted: dict[str, Pipeline] = {}
cv_info: dict[str, dict] = {}
for name, (est, grid, how) in specs.items():
    if grid is None:
        est.fit(Xtr, ytr)
        fitted[name] = est
        cv_info[name] = {"best_params": {}, "cv_logloss": None}
        log(f"{name}: fitted (no tuning)")
        continue
    if how == "grid":
        search = GridSearchCV(est, grid, scoring="neg_log_loss", cv=cv, n_jobs=-1 if name != "rf" else 1)
    else:
        search = RandomizedSearchCV(est, grid, n_iter=30, scoring="neg_log_loss", cv=cv, n_jobs=1, random_state=SEED)
    search.fit(Xtr, ytr, groups=gtr)
    fitted[name] = search.best_estimator_
    bp = {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()}
    cv_info[name] = {"best_params": bp, "cv_logloss": float(-search.best_score_)}
    log(f"{name}: best {bp}  cv logloss {-search.best_score_:.5f}")

# ============================================================== 3. evaluation — every metric
MODELS = list(fitted)
pred = {k: m.predict_proba(Xte)[:, 1] for k, m in fitted.items()}
prior = float(ytr.mean())
pred["prevalence"] = np.full(len(yte), prior)

# Out-of-fold predictions on the training patients: used ONLY to choose thresholds and as a CV view.
oof = {}
for k in MODELS:
    oof[k] = cross_val_predict(clone(fitted[k]), Xtr, ytr, groups=gtr, cv=cv, method="predict_proba",
                               n_jobs=1 if k in ("rf", "lgbm") else -1)[:, 1]
    log(f"oof {k} done")
thresholds = {k: choose_thresholds(ytr, oof[k]) for k in MODELS}

test, cv_metrics, thr_metrics = {}, {}, {}
for k in MODELS:
    test[k] = {**ranking(yte, pred[k]), **probabilistic(yte, pred[k], prior)}
    cv_metrics[k] = {**ranking(ytr, oof[k]), **probabilistic(ytr, oof[k], prior)}
    thr_metrics[k] = {pol: at_threshold(yte, pred[k], t) for pol, t in thresholds[k].items()}
test["prevalence"] = {"auc": 0.5, "gini": 0.0, "ks": 0.0, "ap": float(yte.mean()), "ap_lift": 1.0,
                      "brier": float(np.mean((yte - prior) ** 2)),
                      "logloss": float(-np.mean(yte * np.log(prior) + (1 - yte) * np.log(1 - prior))),
                      "brier_skill": 0.0, "logloss_skill": 0.0, "discrimination_slope": 0.0, "mean_p": prior,
                      "oe_ratio": float(yte.mean() / prior)}
for k in test:
    log(f"test {k:12s} auc={test[k]['auc']:.4f} ap={test[k]['ap']:.4f} brier={test[k]['brier']:.5f} "
        f"ll={test[k]['logloss']:.5f} slope={test[k].get('cal_slope', float('nan')):.3f}")
for k in MODELS:
    m = thr_metrics[k]["f1_opt"]
    log(f"thr {k:12s} t={m['threshold']:.3f} f1={m['f1']:.4f} mcc={m['mcc']:.4f} rec={m['recall']:.3f} prec={m['precision']:.3f}")

boots = cluster_bootstrap(yte, {k: pred[k] for k in MODELS}, gte, B=1000, seed=SEED)
ci = {k: {m: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for m, v in boots[k].items()} for k in MODELS}
auc_ci = {k: ci[k]["auc"] for k in MODELS}
paired = {}
for k in MODELS:
    if k == "lr_spline":
        continue
    paired[k] = {}
    for m in ("auc", "ap", "brier", "logloss"):
        diff = boots[k][m] - boots["lr_spline"][m]
        paired[k][m] = {"diff": float(test[k][m] - test["lr_spline"][m]),
                        "lo": float(np.percentile(diff, 2.5)), "hi": float(np.percentile(diff, 97.5))}
log(f"paired differences vs lr_spline (auc): { {k: v['auc'] for k, v in paired.items()} }")

reliab = {k: reliability_bins(yte, pred[k]) for k in ("lr_spline", "lgbm", "lr_balanced", "rf")}
ths = np.round(np.arange(0.04, 0.41, 0.01), 2)
dca = {"thresholds": ths.tolist(),
       "treat_all": net_benefit(yte, np.ones_like(yte, dtype=float), ths).tolist(),
       **{k: net_benefit(yte, pred[k], ths).tolist() for k in ("lr_spline", "lgbm", "lr_linear", "rf")}}
topk = {k: [top_k(yte, pred[k], f) for f in (0.1, 0.2, 0.3)] for k in ("lr_spline", "lgbm", "lr_linear", "rf")}

# ============================================================== 4. leakage: encounter split vs patient split
leak = {}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for k in ("lr_spline", "rf", "lgbm"):
    log(f"leakage check {k}")
    est = clone(fitted[k])
    a_enc = cross_val_score(est, Xtr, ytr, cv=skf, scoring="roc_auc", n_jobs=1 if k == "rf" else -1)
    a_pat = cross_val_score(est, Xtr, ytr, cv=cv, groups=gtr, scoring="roc_auc", n_jobs=1 if k == "rf" else -1)
    leak[k] = {"encounter_split": float(a_enc.mean()), "patient_split": float(a_pat.mean()), "test": test[k]["auc"]}
# A deliberately memorising forest (leaves of size 1) makes the mechanism visible.
deep = pipe("tree", RandomForestClassifier(n_estimators=300, min_samples_leaf=1, max_features="sqrt", n_jobs=-1,
                                           random_state=SEED))
leak["rf_deep"] = {"encounter_split": float(cross_val_score(deep, Xtr, ytr, cv=skf, scoring="roc_auc").mean()),
                   "patient_split": float(cross_val_score(deep, Xtr, ytr, cv=cv, groups=gtr, scoring="roc_auc").mean())}
deep.fit(Xtr, ytr)
leak["rf_deep"]["test"] = float(summary(yte, deep.predict_proba(Xte)[:, 1])["auc"])
log(f"leakage: {leak}")

# ============================================================== 5. learning curve (patients subsampled)
rng = np.random.default_rng(SEED)
upat = np.unique(gtr)
lc = {"frac": [0.02, 0.05, 0.1, 0.25, 0.5, 1.0], "lr_spline": [], "lgbm": [], "n": []}
for f in lc["frac"]:
    keep = set(rng.choice(upat, max(50, int(f * len(upat))), replace=False)) if f < 1 else set(upat)
    m = np.isin(gtr, list(keep))
    lc["n"].append(int(m.sum()))
    for k in ("lr_spline", "lgbm"):
        est = clone(fitted[k]).fit(Xtr[m], ytr[m])
        lc[k].append(float(summary(yte, est.predict_proba(Xte)[:, 1])["auc"]))
log(f"learning curve: {lc}")

# ============================================================== 6. subgroups (main model)
E = engineer(Xte)
sub = []
bands = pd.cut(E["age_years"], [0, 50, 70, 200], right=False, labels=["<50", "50–69", "70+"])
for var, s in (("age", bands), ("sex", E["gender"]), ("race", E["race"])):
    for lvl in s.dropna().unique():
        m = (s == lvl).to_numpy()
        if m.sum() < 300:
            continue
        from sklearn.metrics import roc_auc_score
        sub.append({"var": var, "level": str(lvl), "n": int(m.sum()), "prev": float(yte[m].mean()),
                    "mean_p": float(pred["lr_spline"][m].mean()), "auc": float(roc_auc_score(yte[m], pred["lr_spline"][m]))})
sub.sort(key=lambda r: (r["var"], -r["n"]))

# ============================================================== 7. inference: interpretable GLM, cluster-robust SEs
Etr = engineer(Xtr)
design = pd.DataFrame(index=Etr.index)
design["const"] = 1.0
inter = {"age_years": ("age (per 10 years)", 10.0), "time_in_hospital": ("length of stay (per day)", 1.0),
         "num_medications": ("medications (per 5)", 5.0), "num_lab_procedures": ("lab procedures (per 10)", 10.0),
         "number_diagnoses": ("diagnoses (per 1)", 1.0), "number_emergency": ("ER visits, prior year (per 1)", 1.0),
         "number_outpatient": ("outpatient visits, prior year (per 1)", 1.0),
         "number_inpatient": ("inpatient stays, prior year (per 1)", 1.0), "num_procedures": ("procedures (per 1)", 1.0)}
for c, (_, scale) in inter.items():
    design[c] = Etr[c] / scale
refs = {}
for c in ["discharge", "admission_source", "admission_type", "diag_1_grp", "A1Cresult", "insulin", "change",
          "diabetesMed", "gender", "race", "specialty", "payer"]:
    vc = Etr[c].value_counts()
    ref = vc.index[0]
    refs[c] = ref
    for lvl in vc.index[1:]:
        if vc[lvl] >= 200:
            design[f"{c}={lvl}"] = (Etr[c] == lvl).astype(float)
fit = fit_logit(design.to_numpy(), ytr, groups=gtr)
assert fit.converged
se_m, se_c = fit.se("model"), fit.se("cluster")
coefs = []
for j, name in enumerate(design.columns):
    if name == "const":
        continue
    label = inter[name][0] if name in inter else name
    b, s = fit.coef[j], se_c[j]
    coefs.append({"term": name, "label": label, "coef": float(b), "or": float(np.exp(b)),
                  "lo": float(np.exp(b - 1.96 * s)), "hi": float(np.exp(b + 1.96 * s)),
                  "se_model": float(se_m[j]), "se_cluster": float(s), "z": float(b / s)})
# verify the from-scratch IRLS against statsmodels
try:
    import statsmodels.api as sm
    smfit = sm.GLM(ytr, design.to_numpy(), family=sm.families.Binomial()).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(gtr)[0]})
    check = {"max_abs_coef_diff": float(np.max(np.abs(smfit.params - fit.coef))),
             "max_abs_cluster_se_diff": float(np.max(np.abs(smfit.bse - se_c)))}
except Exception as exc:  # pragma: no cover
    check = {"error": repr(exc)}
log(f"IRLS vs statsmodels: {check}")
inference = {"n": int(len(ytr)), "p": int(design.shape[1]), "refs": refs, "coefs": coefs, "check": check,
             "se_ratio_median": float(np.median(se_c[1:] / se_m[1:])), "loglik": fit.loglik}

# main-model coefficients (elastic net sparsity)
enet = fitted["lr_enet"].named_steps["model"]
enet_info = {"n_features": int(enet.coef_.size), "n_nonzero": int(np.sum(enet.coef_ != 0))}

# ============================================================== 8. save
model_names = {"prevalence": "Prevalence only", "lr_one": "LR · prior inpatient stays only",
               "lr_linear": "LR · linear terms", "lr_spline": "LR · log + splines (main)",
               "lr_enet": "LR · splines + elastic net", "lr_balanced": "LR · splines + class_weight",
               "rf": "Random forest", "lgbm": "LightGBM"}
results = {"audit": audit, "split": split, "cv": cv_info, "test": test, "cv_oof": cv_metrics, "ci": ci, "auc_ci": auc_ci,
           "thresholds": thresholds, "at_threshold": thr_metrics, "paired_vs_lr_spline": paired,
           "reliability": reliab, "dca": dca, "topk": topk, "leakage": leak, "learning_curve": lc, "subgroups": sub,
           "inference": inference, "enet": enet_info, "names": model_names,
           "versions": {"python": sys.version.split()[0], "sklearn": __import__("sklearn").__version__,
                        "lightgbm": __import__("lightgbm").__version__, "numpy": np.__version__, "pandas": pd.__version__}}
(RES / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
joblib.dump(fitted["lr_spline"], MOD / "lr_spline.joblib")
joblib.dump(fitted, MOD / "all_models.joblib")
try:
    import skops.io as sio
    sio.dump(fitted["lr_spline"], MOD / "lr_spline.skops")
except Exception as exc:  # pragma: no cover
    log(f"skops dump failed: {exc!r}")
np.savez_compressed(RES / "test_predictions.npz", y=yte, groups=gte, **pred)
log("done")
