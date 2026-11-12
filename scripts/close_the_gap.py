"""Can logistic regression catch up with gradient boosting?

Runs after run_experiment.py (re-uses its split and its saved test predictions) and fits three richer
logistic models: + 3-digit ICD-9 codes, + interactions, + both. Writes results/gap.json.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from readmission.data import GROUP, TARGET, build_cohort, load_raw  # noqa: E402
from readmission.features import ENGINEER_PLUS, preprocessor_plus  # noqa: E402
from readmission.metrics import cluster_bootstrap, probabilistic, ranking  # noqa: E402

warnings.filterwarnings("ignore")
SEED = 42
t0 = time.time()

df, _ = build_cohort(load_raw())
y = df[TARGET].to_numpy()
g = df[GROUP].to_numpy()
X = df.drop(columns=[TARGET])
tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(X, y, g))
Xtr, Xte, ytr, yte, gtr, gte = X.iloc[tr], X.iloc[te], y[tr], y[te], g[tr], g[te]
saved = np.load(ROOT / "results" / "test_predictions.npz")
assert np.array_equal(saved["y"], yte) and np.array_equal(saved["groups"], gte), "split differs from run_experiment.py"
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

variants = {"lr_icd3": (True, False), "lr_inter": (False, True), "lr_full": (True, True)}
pred, info = {}, {}
for name, (icd3, inter) in variants.items():
    pipe = Pipeline([("engineer", ENGINEER_PLUS), ("prep", preprocessor_plus(icd3, inter)),
                     ("model", LogisticRegression(max_iter=3000))])
    search = GridSearchCV(pipe, {"model__C": [0.003, 0.01, 0.03]}, scoring="neg_log_loss", cv=cv, n_jobs=-1)
    search.fit(Xtr, ytr, groups=gtr)
    best = search.best_estimator_
    n_feat = int(best.named_steps["model"].coef_.size)
    pred[name] = best.predict_proba(Xte)[:, 1]
    info[name] = {"C": float(search.best_params_["model__C"]), "cv_logloss": float(-search.best_score_), "n_features": n_feat}
    print(f"[{time.time() - t0:5.0f}s] {name}: {info[name]}", flush=True)

prior = float(ytr.mean())
test = {k: {**ranking(yte, p), **probabilistic(yte, p, prior)} for k, p in pred.items()}
allp = {**pred, "lr_spline": saved["lr_spline"], "lgbm": saved["lgbm"]}
boots = cluster_bootstrap(yte, allp, gte, B=1000, seed=SEED)
paired = {}
for k in pred:
    paired[k] = {}
    for ref in ("lr_spline", "lgbm"):
        diff = boots[k]["auc"] - boots[ref]["auc"]
        paired[k][ref] = {"diff": float(np.mean(diff)), "lo": float(np.percentile(diff, 2.5)), "hi": float(np.percentile(diff, 97.5))}
    print(k, {m: round(test[k][m], 4) for m in ("auc", "ap", "brier", "logloss", "cal_slope")}, paired[k], flush=True)
ci = {k: [float(np.percentile(boots[k]["auc"], 2.5)), float(np.percentile(boots[k]["auc"], 97.5))] for k in pred}
(ROOT / "results" / "gap.json").write_text(json.dumps({"info": info, "test": test, "auc_ci": ci, "paired": paired}, indent=1),
                                          encoding="utf-8")
print("done")
