"""Tests for the safety-stock / reorder-point formulas."""
from __future__ import annotations

import numpy as np
import pytest

from src import safety_stock as ss


def test_service_factor_known_values():
    assert ss.service_factor(0.5) == pytest.approx(0.0, abs=1e-9)
    assert ss.service_factor(0.8413) == pytest.approx(1.0, abs=1e-3)
    assert ss.service_factor(0.9772) == pytest.approx(2.0, abs=1e-3)


def test_safety_stock_scales_with_sqrt_protection_interval():
    """Protection interval 4 vs 1 -> safety stock doubles (sqrt(4)/sqrt(1))."""
    wide = ss.safety_stock(2.0, lead_time=3, review_period=1, service_level=0.95)
    narrow = ss.safety_stock(2.0, lead_time=0, review_period=1, service_level=0.95)
    assert wide == pytest.approx(2 * narrow)


def test_safety_stock_increases_with_service_level():
    lo = ss.safety_stock(1.0, 2, service_level=0.80)
    hi = ss.safety_stock(1.0, 2, service_level=0.99)
    assert hi > lo > 0


def test_median_service_level_needs_no_safety_stock():
    assert ss.safety_stock(3.0, 5, service_level=0.5) == pytest.approx(0.0, abs=1e-9)


def test_order_up_to_is_mean_demand_plus_safety_stock():
    mu, sigma, L, R, sl = 4.0, 2.0, 2, 1, 0.95
    expected = mu * (L + R) + ss.safety_stock(sigma, L, R, sl)
    assert ss.order_up_to_normal(mu, sigma, L, R, sl) == pytest.approx(expected)


def test_reorder_point_components():
    mu, sigma, L, sl = 5.0, 1.0, 4, 0.90
    z = ss.service_factor(sl)
    assert ss.reorder_point(mu, sigma, L, sl) == pytest.approx(
        mu * L + z * sigma * np.sqrt(L))


def test_vectorised_over_series():
    mu = np.array([1.0, 10.0])
    sigma = np.array([1.0, 5.0])
    out = ss.order_up_to_normal(mu, sigma, lead_time=1, review_period=1,
                                service_level=0.95)
    assert out.shape == (2,) and (out > 0).all()
