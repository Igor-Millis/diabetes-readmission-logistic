"""Render the README figures from results/results.json (matplotlib only)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "results" / "results.json").read_text(encoding="utf-8"))
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#c9c6bd", "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": "#ecebe6", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left", "legend.frameon": False,
})
N = R["names"]


def auc_ci() -> None:
    order = sorted(R["auc_ci"], key=lambda k: R["test"][k]["auc"])
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for i, k in enumerate(order):
        a = R["test"][k]["auc"]
        lo, hi = R["auc_ci"][k]
        c = BLUE if k == "lr_spline" else (ORANGE if k == "lgbm" else GRAY)
        ax.plot([lo, hi], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.plot(a, i, "o", color=c, ms=7, mec="white", mew=1.5)
        ax.text(hi + 0.003, i, f"{a:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(order)), [N[k] for k in order])
    ax.set_xlabel("ROC AUC on held-out patients (95% patient-level bootstrap CI)")
    ax.set_title("Boosting leads by ~0.01 AUC; logistic regression keeps ~94% of the lift over chance")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "auc_ci.png")


def calibration() -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    hi = 0.75
    ax.plot([0, hi], [0, hi], ls="--", color=GRAY, lw=1)
    for k, c, lab in (("lr_spline", BLUE, N["lr_spline"]), ("lgbm", ORANGE, N["lgbm"]),
                      ("lr_balanced", AQUA, N["lr_balanced"])):
        b = R["reliability"][k]
        ax.plot([q["mean_p"] for q in b], [q["obs"] for q in b], "-o", color=c, lw=2, ms=6, mec="white", mew=1.2,
                label=f"{lab}  (ECE {R['test'][k]['ece']:.3f})")
    ax.set_xlim(0, hi), ax.set_ylim(0, 0.35)
    ax.set_xlabel("mean predicted risk (risk deciles)")
    ax.set_ylabel("observed 30-day readmission rate")
    ax.set_title("Reweighting classes breaks calibration")
    ax.legend(loc="upper left", fontsize=8.5)
    fig.savefig(FIG / "calibration.png")


def decision_curve() -> None:
    d = R["dca"]
    t = np.array(d["thresholds"])
    fig, ax = plt.subplots(figsize=(6.6, 4))
    ax.axhline(0, color=GRAY, lw=1, label="treat none")
    ax.plot(t, d["treat_all"], color=GRAY, lw=1.5, ls="--", label="treat all")
    ax.plot(t, d["lgbm"], color=ORANGE, lw=2, label=N["lgbm"])
    ax.plot(t, d["lr_spline"], color=BLUE, lw=2, label=N["lr_spline"])
    ax.set_ylim(-0.01, max(d["lr_spline"]) * 1.12)
    ax.set_xlabel("risk threshold for a follow-up intervention")
    ax.set_ylabel("net benefit")
    ax.set_title("Both models beat treat-all only for low-cost follow-ups (threshold ≳ 0.05)")
    ax.legend(fontsize=8.5)
    fig.savefig(FIG / "decision_curve.png")


def odds_ratios() -> None:
    co = [c for c in R["inference"]["coefs"] if abs(c["z"]) > 3]
    co.sort(key=lambda c: c["or"])
    fig, ax = plt.subplots(figsize=(7.2, 0.28 * len(co) + 1.2))
    for i, c in enumerate(co):
        col = BLUE if c["or"] > 1 else ORANGE
        ax.plot([c["lo"], c["hi"]], [i, i], color=col, lw=2, solid_capstyle="round")
        ax.plot(c["or"], i, "s", color=col, ms=5)
    ax.axvline(1, color=GRAY, lw=1)
    ax.set_xscale("log")
    ticks = [0.5, 0.75, 1, 1.5, 2, 3, 4]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yticks(range(len(co)), [c["label"] for c in co], fontsize=8)
    ax.set_xlabel("odds ratio (log scale), 95% CI with patient-clustered standard errors")
    ax.set_title("Terms with |z| > 3 in the interpretable logistic model")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "odds_ratios.png")


def leakage() -> None:
    L = R["leakage"]
    keys = [k for k in ("lr_spline", "lgbm", "rf", "rf_deep") if k in L]
    lab = {**N, "rf_deep": "Random forest,\nleaf size 1", "lr_spline": "LR · log + splines\n(main)"}
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    for off, key, c, name in ((-0.27, "encounter_split", GRAY, "CV, random encounter split"),
                              (0, "patient_split", BLUE, "CV, split by patient"),
                              (0.27, "test", INK, "held-out test patients")):
        v = [L[k][key] for k in keys]
        ax.bar(x + off, v, width=0.25, color=c, label=name)
        for xi, vi in zip(x + off, v):
            ax.text(xi, vi + 0.003, f"{vi:.3f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, [lab[k] for k in keys], fontsize=8.5)
    ax.set_ylim(0.6, max(max(L[k].values()) for k in keys) + 0.02)
    ax.set_ylabel("ROC AUC")
    ax.set_title("Here, splitting encounters instead of patients barely moves CV AUC")
    ax.legend(fontsize=8, loc="upper left", ncol=3)
    ax.grid(axis="x", visible=False)
    fig.savefig(FIG / "leakage.png")


def learning_curve() -> None:
    lc = R["learning_curve"]
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(lc["n"], lc["lr_spline"], "-o", color=BLUE, lw=2, ms=6, mec="white", label=N["lr_spline"])
    ax.plot(lc["n"], lc["lgbm"], "-o", color=ORANGE, lw=2, ms=6, mec="white", label=N["lgbm"])
    ax.set_xscale("log")
    ax.set_xlabel("training encounters (patients subsampled)")
    ax.set_ylabel("test ROC AUC")
    ax.set_title("Logistic regression wins below ~10k encounters; boosting wins above")
    ax.legend(fontsize=8.5)
    fig.savefig(FIG / "learning_curve.png")


for f in (auc_ci, calibration, decision_curve, odds_ratios, leakage, learning_curve):
    f()
    plt.close("all")
print("figures ->", FIG)
