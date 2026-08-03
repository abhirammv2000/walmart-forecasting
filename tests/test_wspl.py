"""Correctness tests for the WSPL (M5 Uncertainty) metric.

Pinball loss is easy to get subtly backwards (penalising the wrong tail), so the
asymmetry is pinned directly, then the full evaluator is checked against
properties that must hold for any correct implementation.
"""
from __future__ import annotations

import numpy as np
import pytest

from src import config, data
from src.wspl import QUANTILES, WSPLEvaluator, pinball_loss

LAST_TRAIN_DAY = 1913
VALID_COLS = [f"d_{d}" for d in range(1914, 1942)]
GCOLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


# --------------------------------------------------------------------------- #
# Pure-function tests (fast, no data)
# --------------------------------------------------------------------------- #
def test_pinball_is_zero_for_perfect_forecast():
    y = np.array([0.0, 3.0, 10.0])
    for u in QUANTILES:
        assert np.allclose(pinball_loss(y, y, u), 0.0)


def test_pinball_median_is_symmetric():
    """At u=0.5 over- and under-forecasting by the same amount cost the same."""
    y = np.array([10.0])
    assert pinball_loss(y, np.array([8.0]), 0.5) == pytest.approx(
        pinball_loss(y, np.array([12.0]), 0.5))


def test_pinball_high_quantile_punishes_under_forecasting():
    """At u=0.995, missing demand on the high side must hurt far more.

    This is the asymmetry that makes high quantiles useful for service levels.
    """
    y = np.array([10.0])
    under = pinball_loss(y, np.array([9.0]), 0.995).item()   # forecast too low
    over = pinball_loss(y, np.array([11.0]), 0.995).item()   # forecast too high
    assert under > over
    assert under / over == pytest.approx(0.995 / 0.005, rel=1e-9)


def test_pinball_low_quantile_punishes_over_forecasting():
    y = np.array([10.0])
    under = pinball_loss(y, np.array([9.0]), 0.005)
    over = pinball_loss(y, np.array([11.0]), 0.005)
    assert over > under


# --------------------------------------------------------------------------- #
# Full evaluator (uses real data)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def evaluator_and_truth():
    eval_wide = data.load_sales_wide(evaluation=True)
    calendar = data.load_calendar()
    prices = data.load_prices()
    train_cols = config.ID_COLS + [f"d_{d}" for d in range(1, LAST_TRAIN_DAY + 1)]
    ev = WSPLEvaluator(eval_wide[train_cols],
                       eval_wide[config.ID_COLS + VALID_COLS][VALID_COLS],
                       calendar, prices)
    return ev, eval_wide[config.ID_COLS + VALID_COLS], eval_wide


def _const_pred(truth, values):
    p = truth[GCOLS].copy()
    for c in VALID_COLS:
        p[c] = values
    return p


def test_perfect_forecast_at_every_quantile_scores_zero(evaluator_and_truth):
    """If every quantile equals the actual, pinball loss is 0 everywhere."""
    ev, truth, _ = evaluator_and_truth
    perfect = truth[GCOLS + VALID_COLS]
    score = ev.score({u: perfect for u in QUANTILES})
    assert score == pytest.approx(0.0, abs=1e-9)


def test_score_is_finite_and_has_12_levels(evaluator_and_truth):
    ev, truth, eval_wide = evaluator_and_truth
    mean28 = eval_wide[[f"d_{d}" for d in range(1886, 1914)]].mean(axis=1).to_numpy()
    flat = {u: _const_pred(truth, mean28) for u in QUANTILES}
    score, levels = ev.score(flat, return_levels=True)
    assert np.isfinite(score) and score > 0
    assert len(levels) == 12 and np.all(np.isfinite(levels))


def test_spread_quantiles_beat_a_degenerate_point_forecast(evaluator_and_truth):
    """A sensible spread must score better than pretending there's no uncertainty.

    Collapsing all nine quantiles onto the mean throws away the distribution;
    a monotone spread around it should score strictly lower (better) WSPL.
    """
    ev, truth, eval_wide = evaluator_and_truth
    mean28 = eval_wide[[f"d_{d}" for d in range(1886, 1914)]].mean(axis=1).to_numpy()

    degenerate = {u: _const_pred(truth, mean28) for u in QUANTILES}
    # Crude but monotone spread: scale the mean by a quantile-dependent factor.
    spread = {u: _const_pred(truth, mean28 * (0.4 + 1.6 * u)) for u in QUANTILES}

    assert ev.score(spread) < ev.score(degenerate)


def test_missing_quantile_raises(evaluator_and_truth):
    ev, truth, eval_wide = evaluator_and_truth
    mean28 = eval_wide[[f"d_{d}" for d in range(1886, 1914)]].mean(axis=1).to_numpy()
    partial = {u: _const_pred(truth, mean28) for u in QUANTILES[:-1]}
    with pytest.raises(ValueError, match="missing quantile"):
        ev.score(partial)
