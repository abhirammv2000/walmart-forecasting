"""Tests for the cold-start analog forecaster.

The leave-one-out group mean is the piece most likely to be wrong (including the
item's own history would leak the very thing we're pretending we don't have), so
it's pinned directly.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import coldstart as cs


def test_loo_group_mean_excludes_self():
    # group A = rows 0,1,2 ; group B = rows 3,4
    values = np.array([[1.0], [3.0], [5.0], [10.0], [20.0]])
    groups = np.array(["A", "A", "A", "B", "B"])
    loo = cs.group_mean_excluding_self(values, groups)
    assert loo[0, 0] == pytest.approx((3 + 5) / 2)     # excludes the 1
    assert loo[2, 0] == pytest.approx((1 + 3) / 2)     # excludes the 5
    assert loo[3, 0] == pytest.approx(20.0)            # only the other B row
    assert loo[4, 0] == pytest.approx(10.0)


def test_singleton_group_has_no_analogs():
    values = np.array([[2.0], [9.0]])
    groups = np.array(["A", "B"])
    loo = cs.group_mean_excluding_self(values, groups)
    assert np.isnan(loo).all()                         # each group size 1


def test_price_factor_direction_and_bounds():
    # Cheaper than analogs (item 5 vs analog 10) -> multiplier > 1.
    assert cs.price_factor(np.array([5.0]), np.array([10.0]))[0] > 1.0
    # Pricier than analogs -> multiplier < 1.
    assert cs.price_factor(np.array([20.0]), np.array([10.0]))[0] < 1.0
    # Bounded.
    assert cs.price_factor(np.array([0.01]), np.array([100.0]))[0] == 2.0
    assert cs.price_factor(np.array([100.0]), np.array([0.01]))[0] == 0.5


def test_analog_forecast_uses_group_profile_over_the_horizon():
    # Two groups, distinct weekday profiles; horizon repeats weekdays 0,1.
    profiles = np.array([
        [2.0, 4.0, 0, 0, 0, 0, 0],   # A
        [2.0, 4.0, 0, 0, 0, 0, 0],   # A (identical, so LOO of each = the other)
        [9.0, 9.0, 0, 0, 0, 0, 0],   # B
        [9.0, 9.0, 0, 0, 0, 0, 0],   # B
    ])
    groups = np.array(["A", "A", "B", "B"])
    fc = cs.analog_forecast(profiles, groups, np.array([0, 1, 0]))
    assert fc.shape == (4, 3)
    assert np.allclose(fc[0], [2.0, 4.0, 2.0])         # inherits A's profile
    assert np.allclose(fc[2], [9.0, 9.0, 9.0])         # inherits B's profile


def test_forecast_is_nonnegative():
    rng = np.random.default_rng(0)
    profiles = rng.standard_normal((50, 7))            # can be negative
    groups = rng.integers(0, 5, 50)
    fc = cs.analog_forecast(profiles, groups, np.arange(7))
    assert (fc[~np.isnan(fc)] >= 0).all()
