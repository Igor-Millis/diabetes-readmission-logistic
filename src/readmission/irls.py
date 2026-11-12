"""Logistic regression from scratch: Newton–Raphson / IRLS, with model-based, sandwich and
cluster-robust (by patient) covariance. NumPy only.

The log-likelihood of a Bernoulli GLM with logit link is

    l(b) = sum_i [ y_i x_i'b - log(1 + exp(x_i'b)) ]

with score U(b) = X'(y - mu) and Fisher information I(b) = X'WX, W = diag(mu (1 - mu)).
Newton's step b <- b + I^{-1} U is the same as a weighted least-squares fit of the working response
z = Xb + W^{-1}(y - mu), hence "iteratively reweighted least squares".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * eta))  # numerically stable logistic


@dataclass
class LogitFit:
    coef: np.ndarray
    cov_model: np.ndarray       # I(b)^-1
    cov_robust: np.ndarray      # I^-1 (sum_i u_i u_i') I^-1  (Huber–White)
    cov_cluster: np.ndarray | None  # same, but scores summed within each cluster first
    loglik: float
    n_iter: int
    converged: bool

    def se(self, kind: str = "model") -> np.ndarray:
        cov = {"model": self.cov_model, "robust": self.cov_robust, "cluster": self.cov_cluster}[kind]
        return np.sqrt(np.diag(cov))


def fit_logit(X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None,
              tol: float = 1e-10, max_iter: int = 50) -> LogitFit:
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    b = np.zeros(X.shape[1])
    converged = False
    for it in range(1, max_iter + 1):
        mu = _sigmoid(X @ b)
        w = mu * (1 - mu)
        info = X.T @ (X * w[:, None])
        step = np.linalg.solve(info, X.T @ (y - mu))
        b = b + step
        if np.max(np.abs(step)) < tol:
            converged = True
            break
    mu = _sigmoid(X @ b)
    w = mu * (1 - mu)
    info_inv = np.linalg.inv(X.T @ (X * w[:, None]))
    scores = X * (y - mu)[:, None]                     # u_i = x_i (y_i - mu_i)
    meat = scores.T @ scores
    cov_cluster = None
    if groups is not None:
        _, inv = np.unique(groups, return_inverse=True)
        g_scores = np.zeros((inv.max() + 1, X.shape[1]))
        np.add.at(g_scores, inv, scores)
        G = g_scores.shape[0]
        n, p = X.shape
        adj = G / (G - 1) * (n - 1) / (n - p)            # usual small-sample correction (as in Stata/statsmodels)
        cov_cluster = adj * info_inv @ (g_scores.T @ g_scores) @ info_inv
    eps = 1e-15
    ll = float(np.sum(y * np.log(mu + eps) + (1 - y) * np.log(1 - mu + eps)))
    return LogitFit(b, info_inv, info_inv @ meat @ info_inv, cov_cluster, ll, it, converged)
