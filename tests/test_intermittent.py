"""Tests for the intermittent-demand methods.

Croston/TSB are easy to implement subtly wrong (off-by-one on intervals, updating
the wrong quantity), so the properties are pinned on cases with known answers.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import intermittent as it


# --------------------------------------------------------------------------- #
# Croston / SBA
# --------------------------------------------------------------------------- #
def test_all_zero_history_forecasts_zero():
    assert it.croston(np.zeros(50)) == 0.0
    assert it.tsb(np.zeros(50)) == 0.0


def test_constant_demand_every_period_recovers_the_level():
    """Demand of 5 every period: rate = size 5 / interval 1 = 5."""
    assert it.croston(np.full(30, 5.0)) == pytest.approx(5.0, abs=1e-9)


def test_every_other_period_halves_the_rate():
    """Demand of 4 every 2nd period -> rate 2 (size 4 / interval 2).

    Demand starts at index 1 so *every* interval (including the first) is exactly
    2, and the interval SES has a constant input -> the rate is exactly 2.
    """
    y = np.zeros(40)
    y[1::2] = 4.0
    assert it.croston(y) == pytest.approx(2.0, rel=1e-9)


def test_sba_is_below_croston_by_the_debias_factor():
    """SBA = Croston x (1 - alpha/2), so it is strictly smaller for alpha>0."""
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.3) * rng.integers(1, 6, 200)
    a = 0.2
    assert it.croston(y, alpha=a, variant="sba") == pytest.approx(
        it.croston(y, alpha=a) * (1 - a / 2), rel=1e-9)


def test_croston_forecast_is_nonnegative_and_finite():
    rng = np.random.default_rng(1)
    for _ in range(20):
        y = (rng.random(300) < 0.1) * rng.integers(1, 10, 300)
        r = it.croston(y)
        assert np.isfinite(r) and r >= 0


# --------------------------------------------------------------------------- #
# TSB - the obsolescence property
# --------------------------------------------------------------------------- #
def test_tsb_decays_toward_zero_after_demand_stops():
    """A SKU that sold, then went dead, should get a lower TSB rate than one
    still selling - because TSB decays the demand probability every period."""
    still_selling = np.tile([0, 0, 3, 0], 30).astype(float)
    went_dead = still_selling.copy()
    went_dead[len(went_dead) // 2:] = 0.0            # stops halfway
    assert it.tsb(went_dead) < it.tsb(still_selling)


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def test_classification_quadrants():
    # Smooth: frequent, low variability.
    assert it.classify(np.full(100, 5.0)) == "smooth"
    # Intermittent: infrequent, low size variability.
    y = np.zeros(100); y[::5] = 3.0
    assert it.classify(y) == "intermittent"
    # Lumpy: infrequent AND highly variable sizes (CV^2 >= 0.49).
    rng = np.random.default_rng(2)
    y = np.zeros(400)
    hits = rng.choice(400, 40, replace=False)
    y[hits] = rng.exponential(20.0, 40) + 0.5        # heavy-tailed -> CV^2 ~ 1
    assert it.classify(y) == "lumpy"


def test_adi_and_cv2_values():
    y = np.zeros(100); y[::4] = 2.0                  # 25 demands in 100 periods
    adi, cv2 = it.demand_stats(y)
    assert adi == pytest.approx(4.0)                 # 100 / 25
    assert cv2 == pytest.approx(0.0)                 # all sizes equal
