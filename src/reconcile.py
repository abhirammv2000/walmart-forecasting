"""Hierarchical forecast reconciliation.

Forecasts made independently at each level of a hierarchy are almost never
**coherent**: the item forecasts don't add up to the department forecast, which
doesn't add up to the total. That's a problem operationally - finance plans off
the top, replenishment off the bottom, and they disagree. Reconciliation forces
coherence, and the good methods (MinT) *improve accuracy* while doing it by
borrowing strength across levels.

The maths (Hyndman & Athanasopoulos; Wickramasuriya et al. 2019)
--------------------------------------------------------------
Let ``S`` be the (n_nodes x n_bottom) **summing matrix** mapping the bottom series
to every node in the hierarchy (its bottom block is the identity). Base forecasts
``yhat`` (one per node) are reconciled as::

    ytilde = S @ G @ yhat

where ``G`` (n_bottom x n_nodes) maps base forecasts to reconciled bottom-level
forecasts. The methods differ only in ``G``:

* **bottom-up**    G selects the bottom rows (ignore aggregate forecasts).
* **top-down**     disaggregate the top forecast by historical proportions.
* **OLS**          G = (SᵀS)⁻¹Sᵀ  - MinT with an identity error covariance.
* **WLS (struct)** weight each node by how many bottom series it aggregates.
* **MinT (shrink)** weight by the (shrunk) covariance of base forecast errors -
  the minimum-trace optimal reconciliation.

All reconciled forecasts are coherent by construction (they are ``S`` times a
bottom-level vector).
"""
from __future__ import annotations

import numpy as np


def bottom_up_G(n_nodes: int, n_bottom: int) -> np.ndarray:
    """G that just picks the bottom rows (bottom block of S is the identity)."""
    G = np.zeros((n_bottom, n_nodes))
    G[:, n_nodes - n_bottom:] = np.eye(n_bottom)
    return G


def _proj_G(S: np.ndarray, W: np.ndarray | None) -> np.ndarray:
    """G = (Sᵀ W⁻¹ S)⁻¹ Sᵀ W⁻¹  (OLS when W is None/identity)."""
    if W is None:
        StWi = S.T                                  # W = I
    else:
        Wi = np.linalg.inv(W)
        StWi = S.T @ Wi
    return np.linalg.solve(StWi @ S, StWi)          # (n_bottom x n_nodes)


def ols_G(S: np.ndarray) -> np.ndarray:
    return _proj_G(S, None)


def wls_structural_G(S: np.ndarray) -> np.ndarray:
    """WLS weighting each node by the number of bottom series it aggregates."""
    w = S.sum(axis=1)                               # row sums = #bottom per node
    return _proj_G(S, np.diag(w))


def shrunk_covariance(residuals: np.ndarray) -> np.ndarray:
    """Schafer-Strimmer shrinkage of the residual covariance toward its diagonal.

    ``residuals`` is (n_obs x n_nodes). Returns an (n_nodes x n_nodes) estimate
    that is well-conditioned even when n_obs is small - which the raw sample
    covariance is not, and MinT needs to invert.
    """
    n = residuals.shape[0]
    resid = residuals - residuals.mean(axis=0, keepdims=True)
    samp = (resid.T @ resid) / (n - 1)              # sample covariance
    var = np.diag(samp)
    # Shrinkage target is the diagonal of sample variances; we shrink only the
    # off-diagonal (correlation) entries toward zero.
    sd = np.sqrt(var)
    corr = samp / np.outer(sd, sd)
    np.fill_diagonal(corr, 0.0)
    # var of each off-diagonal correlation estimate (Schafer-Strimmer approx)
    r2 = (resid ** 2)
    w = (r2.T @ r2) / (n - 1) - samp ** 2
    off = ~np.eye(len(var), dtype=bool)
    denom = (corr ** 2)[off].sum()
    lam = (w[off].sum() / (denom + 1e-12)) / (n)
    lam = float(np.clip(lam, 0.0, 1.0))
    shrunk = samp.copy()
    shrunk[off] *= (1.0 - lam)
    return shrunk


def mint_shrink_G(S: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """Minimum-trace reconciliation with a shrunk error covariance."""
    return _proj_G(S, shrunk_covariance(residuals))


def top_down_G(S: np.ndarray, proportions: np.ndarray) -> np.ndarray:
    """Disaggregate the top-level (row 0) forecast by fixed ``proportions``.

    ``proportions`` sums to 1 over the bottom series. G picks the top forecast
    and splits it; every other base forecast is ignored.
    """
    n_bottom = S.shape[1]
    G = np.zeros((n_bottom, S.shape[0]))
    G[:, 0] = proportions                           # total is row 0
    return G


def reconcile(base: np.ndarray, S: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Coherent reconciled forecasts ``S @ G @ base`` for every node.

    ``base`` is (n_nodes,) or (n_nodes, H); returns the same shape.
    """
    return S @ (G @ base)


def is_coherent(y: np.ndarray, S: np.ndarray, atol: float = 1e-6) -> bool:
    """True if ``y`` satisfies the hierarchy: every node = sum of its children."""
    bottom = y[S.shape[0] - S.shape[1]:]
    return np.allclose(y, S @ bottom, atol=atol)
