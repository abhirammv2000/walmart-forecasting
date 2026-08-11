"""Tests for hierarchical reconciliation.

Reconciliation is a few lines of linear algebra that are easy to get subtly wrong
(transpose the wrong matrix, weight the wrong axis). The invariants below fully
characterise correct behaviour: outputs must be coherent, and any method that is
a projection must leave already-coherent forecasts untouched.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import reconcile as rec

# A tiny 2-bottom hierarchy: nodes = [total, b1, b2], bottom block = I.
S = np.array([[1.0, 1.0],
              [1.0, 0.0],
              [0.0, 1.0]])


def test_bottom_up_sums_children():
    base = np.array([99.0, 3.0, 5.0])          # top is nonsense; BU ignores it
    y = rec.reconcile(base, S, rec.bottom_up_G(3, 2))
    assert np.allclose(y, [8.0, 3.0, 5.0])     # total becomes 3+5


@pytest.mark.parametrize("G_fn", [
    lambda: rec.bottom_up_G(3, 2),
    lambda: rec.ols_G(S),
    lambda: rec.wls_structural_G(S),
])
def test_output_is_always_coherent(G_fn):
    base = np.array([10.0, 3.0, 3.0])          # incoherent (3+3 != 10)
    y = rec.reconcile(base, S, G_fn())
    assert rec.is_coherent(y, S)


def test_projections_preserve_already_coherent_forecasts():
    """OLS/WLS/MinT are projections onto the coherent subspace: coherent in ->
    unchanged out."""
    bottom = np.array([4.0, 7.0])
    coherent = S @ bottom                       # [11, 4, 7]
    for G in (rec.ols_G(S), rec.wls_structural_G(S), rec.bottom_up_G(3, 2)):
        assert np.allclose(rec.reconcile(coherent, S, G), coherent)


def test_ols_splits_the_discrepancy_evenly():
    """With equal weights, OLS spreads the top/bottom disagreement symmetrically.

    base total=10 but bottoms say 3 and 3; OLS reconciles the two identical
    bottoms to the same value, and the reconciled total is their sum.
    """
    base = np.array([10.0, 3.0, 3.0])
    y = rec.reconcile(base, S, rec.ols_G(S))
    assert y[1] == pytest.approx(y[2])          # symmetry
    assert y[0] == pytest.approx(y[1] + y[2])   # coherence


def test_top_down_disaggregates_by_proportions():
    props = np.array([0.25, 0.75])
    base = np.array([100.0, 999.0, 999.0])      # only the top (100) is used
    y = rec.reconcile(base, S, rec.top_down_G(S, props))
    assert np.allclose(y, [100.0, 25.0, 75.0])


def test_reconcile_handles_multi_horizon():
    base = np.tile(np.array([10.0, 3.0, 3.0])[:, None], (1, 5))   # 5 horizon days
    y = rec.reconcile(base, S, rec.ols_G(S))
    assert y.shape == (3, 5)
    for h in range(5):
        assert rec.is_coherent(y[:, h], S)


def test_shrunk_covariance_is_symmetric_and_diagonally_dominant():
    rng = np.random.default_rng(0)
    resid = rng.standard_normal((40, 3))
    cov = rec.shrunk_covariance(resid)
    assert np.allclose(cov, cov.T)
    # diagonal (variances) untouched by shrinkage
    raw = np.cov(resid, rowvar=False)
    assert np.allclose(np.diag(cov), np.diag(raw))


def test_mint_output_is_coherent():
    rng = np.random.default_rng(1)
    resid = rng.standard_normal((50, 3))
    base = np.array([10.0, 3.0, 3.0])
    y = rec.reconcile(base, S, rec.mint_shrink_G(S, resid))
    assert rec.is_coherent(y, S)
