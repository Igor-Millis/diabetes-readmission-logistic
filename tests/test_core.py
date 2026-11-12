from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from readmission.data import build_cohort  # noqa: E402
from readmission.features import CATEGORICAL, NUMERIC, icd9_chapter  # noqa: E402
from readmission.irls import fit_logit  # noqa: E402
from readmission.metrics import calibration_intercept_slope, cluster_bootstrap_auc, ece, net_benefit  # noqa: E402


# ------------------------------------------------------------------ ICD-9 grouping
@pytest.mark.parametrize("code,group", [
    ("250.83", "Diabetes"), ("250", "Diabetes"), ("428", "Circulatory"), ("785", "Circulatory"),
    ("486", "Respiratory"), ("786", "Respiratory"), ("577", "Digestive"), ("996", "Injury"),
    ("715", "Musculoskeletal"), ("599", "Genitourinary"), ("174", "Neoplasms"), ("V57", "Other"),
    ("E888", "Other"), ("276", "Other"), (np.nan, "Missing"),
])
def test_icd9_chapter(code, group):
    assert icd9_chapter(code) == group


# ------------------------------------------------------------------ cohort rules
def _toy() -> pd.DataFrame:
    return pd.DataFrame({
        "patient_nbr": [1, 1, 2, 3, 4],
        "discharge_disposition_id": [1, 11, 13, 1, 1],
        "gender": ["Female", "Female", "Male", "Unknown/Invalid", "Male"],
        "readmitted": ["<30", ">30", "NO", "NO", "<30"],
    })


def test_cohort_removes_deaths_hospice_and_invalid_sex():
    d, audit = build_cohort(_toy())
    assert audit["removed_death_or_hospice"] == 2
    assert audit["removed_unknown_gender"] == 1
    assert d["readmit30"].tolist() == [1, 1]     # ">30" would be 0
    assert audit["patients"] == 2


# ------------------------------------------------------------------ IRLS
@pytest.fixture(scope="module")
def sim():
    rng = np.random.default_rng(1)
    n, G = 4000, 1500
    g = rng.integers(0, G, n)
    u = rng.normal(0, 1.0, G)[g]                 # shared patient effect -> within-cluster correlation
    X = np.column_stack([np.ones(n), rng.normal(size=n), rng.binomial(1, 0.3, n)])
    beta = np.array([-1.5, 0.8, -0.5])
    y = rng.binomial(1, 1 / (1 + np.exp(-(X @ beta + u))))
    return X, y, g


def test_irls_matches_sklearn_unpenalized(sim):
    X, y, _ = sim
    fit = fit_logit(X, y)
    sk = LogisticRegression(C=1e10, tol=1e-12, max_iter=10_000).fit(X[:, 1:], y)
    assert fit.converged
    np.testing.assert_allclose(fit.coef, np.r_[sk.intercept_, sk.coef_.ravel()], atol=1e-5)


def test_irls_matches_statsmodels_cluster_se(sim):
    sm = pytest.importorskip("statsmodels.api")
    X, y, g = sim
    fit = fit_logit(X, y, groups=g)
    ref = sm.GLM(y, X, family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": g})
    np.testing.assert_allclose(fit.coef, ref.params, atol=1e-8)
    np.testing.assert_allclose(fit.se("cluster"), ref.bse, rtol=1e-6)
    np.testing.assert_allclose(fit.se("model"), sm.GLM(y, X, family=sm.families.Binomial()).fit().bse, rtol=1e-6)


# ------------------------------------------------------------------ metrics
def test_perfectly_calibrated_scores():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.02, 0.6, 200_000)
    y = rng.binomial(1, p)
    a, b = calibration_intercept_slope(y, p)
    assert abs(a) < 0.02 and abs(b - 1) < 0.03
    assert ece(y, p) < 0.005


def test_overconfident_scores_have_slope_below_one():
    rng = np.random.default_rng(0)
    eta = rng.normal(-2, 1, 100_000)
    y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
    p_over = 1 / (1 + np.exp(-(2 * eta + 2)))     # logits stretched by 2
    _, b = calibration_intercept_slope(y, p_over)
    assert 0.45 < b < 0.55


def test_net_benefit_treat_all_formula():
    y = np.array([1, 0, 0, 0])
    nb = net_benefit(y, np.ones(4), np.array([0.2]))
    assert nb[0] == pytest.approx(0.25 - 0.75 * 0.25)


def test_cluster_bootstrap_shapes():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 500)
    g = rng.integers(0, 200, 500)
    out = cluster_bootstrap_auc(y, {"a": rng.uniform(size=500)}, g, B=20)
    assert out["a"].shape == (20,) and np.all((out["a"] > 0.3) & (out["a"] < 0.7))


def test_feature_lists_disjoint():
    assert not set(NUMERIC) & set(CATEGORICAL)
