"""Correctness tests for the WRMSSE metric.

The metric is the foundation of every number in this project, so it gets pinned
against values we can reason about independently:

* a perfect forecast must score exactly 0
* published/naive reference points must reproduce (last-day ~1.46, 28-day-mean
  ~1.08) - these match well-known M5 benchmark behaviour
* the score must never be inf/NaN (regression test for the zero-scale bug)

These use the real data, so they take a couple of minutes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, data
from src.wrmsse import WRMSSEEvaluator

LAST_TRAIN_DAY = 1913
VALID_COLS = [f"d_{d}" for d in range(1914, 1942)]


@pytest.fixture(scope="module")
def evaluator_and_truth():
    eval_wide = data.load_sales_wide(evaluation=True)
    calendar = data.load_calendar()
    prices = data.load_prices()
    train_cols = config.ID_COLS + [f"d_{d}" for d in range(1, LAST_TRAIN_DAY + 1)]
    train_wide = eval_wide[train_cols]
    valid_truth = eval_wide[config.ID_COLS + VALID_COLS]
    ev = WRMSSEEvaluator(train_wide, valid_truth[VALID_COLS], calendar, prices)
    return ev, valid_truth, eval_wide


def test_perfect_forecast_scores_zero(evaluator_and_truth):
    """Predicting the actuals exactly must give WRMSSE == 0."""
    ev, valid_truth, _ = evaluator_and_truth
    score = ev.score(valid_truth[["item_id", "dept_id", "cat_id",
                                  "store_id", "state_id"] + VALID_COLS])
    assert score == pytest.approx(0.0, abs=1e-9)


def test_naive_reference_values(evaluator_and_truth):
    """Naive forecasts reproduce known M5 benchmark magnitudes."""
    ev, valid_truth, eval_wide = evaluator_and_truth
    gcols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

    # (a) repeat the last training day
    last_day = eval_wide[f"d_{LAST_TRAIN_DAY}"].to_numpy()
    p1 = valid_truth[gcols].copy()
    for c in VALID_COLS:
        p1[c] = last_day
    s_last = ev.score(p1)

    # (b) repeat the mean of the last 28 training days (seasonal-naive-ish)
    mean28 = eval_wide[[f"d_{d}" for d in range(1886, 1914)]].mean(axis=1).to_numpy()
    p2 = valid_truth[gcols].copy()
    for c in VALID_COLS:
        p2[c] = mean28
    s_mean = ev.score(p2)

    assert 1.35 < s_last < 1.60, f"last-day naive out of range: {s_last}"
    assert 1.00 < s_mean < 1.20, f"28-day-mean naive out of range: {s_mean}"
    # Averaging must beat blindly repeating one noisy day.
    assert s_mean < s_last


def test_score_is_finite_and_levels_sane(evaluator_and_truth):
    """Regression test: zero-scale series must not inject inf/NaN."""
    ev, valid_truth, eval_wide = evaluator_and_truth
    gcols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    mean28 = eval_wide[[f"d_{d}" for d in range(1886, 1914)]].mean(axis=1).to_numpy()
    preds = valid_truth[gcols].copy()
    for c in VALID_COLS:
        preds[c] = mean28

    score, levels = ev.score(preds, return_levels=True)
    assert np.isfinite(score), "WRMSSE must be finite"
    assert np.all(np.isfinite(levels)), f"non-finite level scores: {levels}"
    assert len(levels) == 12, "M5 defines 12 aggregation levels"

    # Degenerate (constant) series exist but must be a tiny minority.
    excluded = ev.excluded_series()
    bottom_level_excluded = excluded[12]
    assert bottom_level_excluded < 0.05 * 30490, (
        f"too many series dropped for undefined scale: {bottom_level_excluded}")
