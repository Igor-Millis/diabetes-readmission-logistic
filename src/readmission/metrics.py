"""Evaluation helpers: discrimination, calibration, decision curves and a patient-level bootstrap."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from .irls import fit_logit


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Logistic recalibration y ~ a + b * logit(p).

    slope  b: 1 is ideal; < 1 means predictions are too extreme (overfit), > 1 too timid.
    intercept: calibration-in-the-large, estimated with the slope fixed at 1 (offset model).
    """
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    slope = fit_logit(np.column_stack([np.ones_like(lp), lp]), y).coef[1]
    # intercept with offset lp: one-parameter Newton iterations
    a = 0.0
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(a + lp)))
        step = np.sum(y - mu) / np.sum(mu * (1 - mu))
        a += step
        if abs(step) < 1e-12:
            break
    return float(a), float(slope)


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error over equal-frequency bins (risk deciles)."""
    order = np.argsort(p)
    err = 0.0
    for idx in np.array_split(order, bins):
        err += len(idx) * abs(y[idx].mean() - p[idx].mean())
    return float(err / len(y))


def reliability_bins(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    order = np.argsort(p)
    return [{"mean_p": float(p[i].mean()), "obs": float(y[i].mean()), "n": int(len(i))}
            for i in np.array_split(order, bins)]


def summary(y: np.ndarray, p: np.ndarray) -> dict:
    a, b = calibration_intercept_slope(y, p)
    return {"auc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "logloss": float(log_loss(y, p)),
            "cal_intercept": a, "cal_slope": b, "ece": ece(y, p), "mean_p": float(p.mean())}


def top_k(y: np.ndarray, p: np.ndarray, frac: float) -> dict:
    """Flag the top `frac` of patients by risk: recall (share of readmissions caught) and precision."""
    k = int(round(frac * len(p)))
    idx = np.argsort(-p)[:k]
    tp = y[idx].sum()
    return {"frac": frac, "recall": float(tp / y.sum()), "precision": float(tp / k), "lift": float(tp / k / y.mean())}


def net_benefit(y: np.ndarray, p: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Decision-curve net benefit: TP/n - FP/n * t/(1-t)  (Vickers & Elkin, 2006)."""
    n = len(y)
    out = []
    for t in thresholds:
        flag = p >= t
        tp = np.sum(flag & (y == 1))
        fp = np.sum(flag & (y == 0))
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)


def cluster_bootstrap_auc(y: np.ndarray, preds: dict[str, np.ndarray], groups: np.ndarray,
                          B: int = 1000, seed: int = 0) -> dict[str, np.ndarray]:
    """Resample *patients* (not encounters) with replacement; returns B AUCs per model.

    Encounters of the same patient are correlated, so an encounter-level bootstrap would understate the
    uncertainty. All models are evaluated on the same resamples, which makes paired differences valid.
    """
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    rows_of = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    out = {k: np.empty(B) for k in preds}
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_of[g] for g in pick])
        for k, p in preds.items():
            out[k][b] = roc_auc_score(y[idx], p[idx])
    return out


# ====================================================================== full metric suite
def probabilistic(y: np.ndarray, p: np.ndarray, prior: float) -> dict:
    """Metrics of the probabilities themselves (proper scores, skill vs. the prevalence model, calibration)."""
    from scipy.stats import chi2, norm
    p = np.clip(p, 1e-6, 1 - 1e-6)
    base = summary(y, p)
    ll0 = -np.mean(y * np.log(prior) + (1 - y) * np.log(1 - prior))
    br0 = np.mean((y - prior) ** 2)
    # Spiegelhalter z: H0 = perfectly calibrated
    num_ = np.sum((y - p) * (1 - 2 * p))
    den = np.sqrt(np.sum((1 - 2 * p) ** 2 * p * (1 - p)))
    z = num_ / den
    # Hosmer-Lemeshow over risk deciles (10 groups, 8 df)
    order = np.argsort(p)
    hl = 0.0
    mce = 0.0
    for idx in np.array_split(order, 10):
        o, e, n = y[idx].sum(), p[idx].sum(), len(idx)
        hl += (o - e) ** 2 / (e * (1 - e / n))
        mce = max(mce, abs(o / n - e / n))
    base.update({
        "brier_skill": float(1 - base["brier"] / br0),
        "logloss_skill": float(1 - base["logloss"] / ll0),          # = McFadden-style R^2 on the test set
        "discrimination_slope": float(p[y == 1].mean() - p[y == 0].mean()),   # Yates / Tjur R^2
        "spiegelhalter_z": float(z), "spiegelhalter_p": float(2 * norm.sf(abs(z))),
        "hosmer_lemeshow": float(hl), "hosmer_lemeshow_p": float(chi2.sf(hl, 8)), "mce": float(mce),
        "oe_ratio": float(y.sum() / p.sum()),
    })
    return base


def ranking(y: np.ndarray, p: np.ndarray) -> dict:
    """Threshold-free ranking metrics."""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, p)
    auc = roc_auc_score(y, p)
    out = {"auc": float(auc), "gini": float(2 * auc - 1), "ks": float(np.max(tpr - fpr)),
           "ap": float(average_precision_score(y, p)), "ap_lift": float(average_precision_score(y, p) / y.mean())}
    for f in (0.05, 0.1, 0.2, 0.3):
        tk = top_k(y, p, f)
        out[f"recall@{int(f * 100)}%"] = tk["recall"]
        out[f"precision@{int(f * 100)}%"] = tk["precision"]
        out[f"lift@{int(f * 100)}%"] = tk["lift"]
    return out


def at_threshold(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    """Every confusion-matrix metric at threshold t (flag if p >= t)."""
    yhat = (p >= t).astype(int)
    tp = int(np.sum((yhat == 1) & (y == 1)))
    fp = int(np.sum((yhat == 1) & (y == 0)))
    fn = int(np.sum((yhat == 0) & (y == 1)))
    tn = int(np.sum((yhat == 0) & (y == 0)))
    n = tp + fp + fn + tn
    div = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
    sens, spec = div(tp, tp + fn), div(tn, tn + fp)
    prec, npv = div(tp, tp + fp), div(tn, tn + fn)
    f = lambda beta: div((1 + beta**2) * prec * sens, beta**2 * prec + sens) if tp else 0.0  # noqa: E731
    mcc_den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n**2
    return {
        "threshold": float(t), "flag_rate": div(tp + fp, n), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "accuracy": po, "balanced_accuracy": (sens + spec) / 2, "precision": prec, "recall": sens,
        "specificity": spec, "npv": npv, "f1": f(1), "f2": f(2), "f05": f(0.5),
        "mcc": float((tp * tn - fp * fn) / mcc_den) if mcc_den else 0.0,
        "kappa": float((po - pe) / (1 - pe)) if pe < 1 else 0.0, "youden": sens + spec - 1,
        "lr_pos": div(sens, 1 - spec), "lr_neg": div(1 - sens, spec),
        "dor": div(tp * tn, fp * fn),
    }


def choose_thresholds(y: np.ndarray, p: np.ndarray) -> dict:
    """Pick operating thresholds on *out-of-fold training* predictions (never on the test set)."""
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.995, 400)))
    f1 = [at_threshold(y, p, t)["f1"] for t in grid]
    yj = [at_threshold(y, p, t)["youden"] for t in grid]
    return {"f1_opt": float(grid[int(np.argmax(f1))]), "youden_opt": float(grid[int(np.argmax(yj))]),
            "top20": float(np.quantile(p, 0.8)), "default_0.5": 0.5}


def cluster_bootstrap(y: np.ndarray, preds: dict[str, np.ndarray], groups: np.ndarray, B: int = 1000,
                      seed: int = 0) -> dict[str, dict[str, np.ndarray]]:
    """Patient-level bootstrap of AUC, AP, Brier and log loss for several models on shared resamples."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    rows_of = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    names = ("auc", "ap", "brier", "logloss")
    out = {k: {m: np.empty(B) for m in names} for k in preds}
    for b in range(B):
        idx = np.concatenate([rows_of[g] for g in rng.integers(0, len(uniq), len(uniq))])
        yy = y[idx]
        for k, p in preds.items():
            pp = np.clip(p[idx], 1e-6, 1 - 1e-6)
            out[k]["auc"][b] = roc_auc_score(yy, pp)
            out[k]["ap"][b] = average_precision_score(yy, pp)
            out[k]["brier"][b] = np.mean((yy - pp) ** 2)
            out[k]["logloss"][b] = -np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))
    return out
