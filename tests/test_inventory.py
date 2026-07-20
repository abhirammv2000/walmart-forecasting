"""Tests for the inventory decision layer.

These pin the *economics*, not just the plumbing. If the newsvendor logic is
subtly wrong (e.g. the critical ratio inverted), the simulation would still run
happily and produce plausible-looking numbers - so the properties are asserted
directly, including the headline claim that the cost-optimal order is a quantile
rather than the mean.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import inventory as inv


# --------------------------------------------------------------------------- #
# Critical ratio
# --------------------------------------------------------------------------- #
def test_critical_ratio_symmetric_costs_is_median():
    """Cu == Co is the only case where ordering the median is optimal."""
    assert inv.critical_ratio(0.2, 0.2) == pytest.approx(0.5)


def test_critical_ratio_rises_when_stockouts_hurt_more():
    """Expensive stockouts push the optimum into the upper tail."""
    assert inv.critical_ratio(0.30, 0.03) > 0.9
    assert inv.critical_ratio(0.03, 0.30) < 0.1


# --------------------------------------------------------------------------- #
# Order-up-to levels from discrete quantiles
# --------------------------------------------------------------------------- #
@pytest.fixture
def cube():
    """(2 series, 3 days, 5 quantiles) with a known, ascending distribution."""
    quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    base = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    c = np.tile(base, (2, 3, 1))
    return c, quantiles


def test_order_up_to_matches_an_exact_quantile(cube):
    c, q = cube
    got = inv.order_up_to_levels(c, q, 0.75)
    assert np.allclose(got, 8.0)


def test_order_up_to_interpolates_between_quantiles(cube):
    """An arbitrary service level must interpolate the quantile function."""
    c, q = cube
    got = inv.order_up_to_levels(c, q, 0.625)          # midway 0.5 -> 0.75
    assert np.allclose(got, 6.0)                        # midway 4 -> 8


def test_order_up_to_is_monotone_in_service_level(cube):
    """Higher service level can never mean ordering less."""
    c, q = cube
    levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    qty = [inv.order_up_to_levels(c, q, s).mean() for s in levels]
    assert all(b >= a for a, b in zip(qty, qty[1:]))


def test_order_up_to_clips_to_forecast_range(cube):
    """We must not extrapolate beyond the quantiles we actually modelled."""
    c, q = cube
    assert np.allclose(inv.order_up_to_levels(c, q, 0.999), 16.0)
    assert np.allclose(inv.order_up_to_levels(c, q, 0.001), 1.0)


# --------------------------------------------------------------------------- #
# Simulation mechanics
# --------------------------------------------------------------------------- #
def test_perfect_foresight_has_no_lost_sales_and_no_leftovers():
    demand = np.array([[3.0, 5.0, 0.0, 2.0]])
    price = np.ones_like(demand)
    m = inv.simulate(demand.copy(), demand, price, carryover=True)
    assert m["fill_rate"] == pytest.approx(1.0)
    assert m["units_lost"] == pytest.approx(0.0)
    assert m["total_cost"] == pytest.approx(0.0)


def test_ordering_nothing_loses_everything():
    demand = np.array([[3.0, 5.0, 1.0]])
    price = np.ones_like(demand)
    m = inv.simulate(np.zeros_like(demand), demand, price)
    assert m["fill_rate"] == pytest.approx(0.0)
    assert m["units_lost"] == pytest.approx(9.0)
    assert m["holding_cost"] == pytest.approx(0.0)


def test_overstocking_incurs_holding_not_shortage():
    demand = np.array([[1.0, 1.0]])
    price = np.ones_like(demand)
    m = inv.simulate(np.full_like(demand, 10.0), demand, price,
                     underage_frac=0.3, overage_frac=0.03, carryover=True)
    assert m["shortage_cost"] == pytest.approx(0.0)
    assert m["holding_cost"] > 0
    assert m["fill_rate"] == pytest.approx(1.0)


def test_carryover_reduces_ordering_versus_independent_periods():
    """Leftover stock should offset tomorrow's order - that's what carryover means."""
    demand = np.array([[0.0, 0.0, 0.0, 0.0]])
    price = np.ones_like(demand)
    target = np.full_like(demand, 5.0)
    with_co = inv.simulate(target, demand, price, carryover=True)
    without = inv.simulate(target, demand, price, carryover=False)
    assert with_co["units_ordered"] < without["units_ordered"]


def test_fill_rate_increases_with_service_level():
    rng = np.random.default_rng(0)
    quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    demand = rng.poisson(3.0, size=(200, 28)).astype(float)
    # A plausible quantile cube around the true mean.
    base = np.array([0.0, 1.0, 3.0, 5.0, 7.0])
    cube = np.tile(base, (200, 28, 1))
    price = np.ones_like(demand)

    lo = inv.simulate(inv.order_up_to_levels(cube, quantiles, 0.25), demand, price)
    hi = inv.simulate(inv.order_up_to_levels(cube, quantiles, 0.9), demand, price)
    assert hi["fill_rate"] > lo["fill_rate"]


# --------------------------------------------------------------------------- #
# The headline economic claim
# --------------------------------------------------------------------------- #
def test_newsvendor_quantile_beats_ordering_the_mean():
    """With asymmetric costs, the CR-quantile policy must cost less than the mean.

    This is the entire justification for forecasting a distribution: under
    Cu >> Co, ordering the central forecast systematically under-stocks and pays
    for it in lost sales.
    """
    rng = np.random.default_rng(42)
    n, days = 400, 28
    demand = rng.poisson(4.0, size=(n, days)).astype(float)
    price = np.ones((n, days))

    quantiles = np.array([0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995])
    # Correctly-specified Poisson(4) quantiles, broadcast to every series/day.
    cube = np.empty((n, days, len(quantiles)))
    for k, q in enumerate(quantiles):
        cube[:, :, k] = _poisson_ppf(q, 4.0)

    cu, co = 0.30, 0.03
    cr = inv.critical_ratio(cu, co)

    mean_policy = inv.mean_forecast_policy(cube, quantiles)
    nv_policy = inv.order_up_to_levels(cube, quantiles, cr)

    mean_cost = inv.simulate(mean_policy, demand, price, cu, co)["total_cost"]
    nv_cost = inv.simulate(nv_policy, demand, price, cu, co)["total_cost"]

    assert nv_cost < mean_cost, (
        f"newsvendor policy ({nv_cost:.1f}) should beat mean policy ({mean_cost:.1f})")


def _poisson_ppf(q: float, lam: float) -> float:
    """Tiny Poisson quantile function (avoids a scipy dependency)."""
    from math import exp
    cum, p, k = 0.0, exp(-lam), 0
    while cum + p < q and k < 1000:
        cum += p
        k += 1
        p *= lam / k
    return float(k)
