"""Leakage tests - the most important correctness guarantee in the project.

Our whole design rests on one claim:

    *Every sales-derived feature for a day `d` depends only on actual sales from
    at least 28 days earlier.*

Two things follow from it, and both are load-bearing:

1. **No leakage.** A forecast for d_1914..d_1941 never peeks at data that
   wouldn't exist at forecast time.
2. **The build-once-slice-many cache is valid** (see ``src/dataset.py``): we
   engineer features once over the full timeline and slice per CV fold, which is
   only legitimate if a feature's value doesn't depend on where the fold's
   training cut-off falls.

The decisive test: build features twice - once with the full actuals, once with
every actual after the cut-off erased - and assert the prediction-window
features are *identical*. If any feature peeked at the future, the two must
differ.

This must be re-run if short/recursive lags (< 28) are ever introduced; it is
precisely the test that will fail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, data, features

STORE = "CA_1"
CUTOFF = 1913                       # pretend "today" is d_1913
PRED_DAYS = range(1914, 1942)       # the 28-day forecast window


def _build(sales_wide: pd.DataFrame, calendar, prices, encoders) -> pd.DataFrame:
    long_df = data.melt_store(sales_wide, STORE, add_horizon_days=0)
    return features.build_features(long_df, calendar, prices, encoders)


@pytest.fixture(scope="module")
def paired_feature_frames():
    calendar = data.load_calendar()
    prices = data.load_prices()
    sales_wide = data.load_sales_wide(evaluation=True)      # actuals d_1..d_1941
    encoders = features.build_label_encoders(sales_wide, calendar)

    full = _build(sales_wide, calendar, prices, encoders)

    # Same data, but with everything after the cut-off erased - i.e. exactly what
    # we would know when standing at d_1913 making a real forecast.
    masked_wide = sales_wide.copy()
    for d in PRED_DAYS:
        masked_wide[f"d_{d}"] = np.nan
    masked = _build(masked_wide, calendar, prices, encoders)
    return full, masked


def test_prediction_window_features_ignore_the_future(paired_feature_frames):
    """Features on the forecast rows must not change when the future is erased."""
    full, masked = paired_feature_frames

    f = full[full["d_int"].isin(PRED_DAYS)].sort_values(["id", "d_int"]).reset_index(drop=True)
    m = masked[masked["d_int"].isin(PRED_DAYS)].sort_values(["id", "d_int"]).reset_index(drop=True)

    assert len(f) == len(m) and len(f) > 0, "row sets must align and be non-empty"

    # 'sales' is the target (it is deliberately erased in `masked`), so exclude it.
    feat_cols = [c for c in features.feature_columns(f) if c != "sales"]
    assert any(c.startswith("sales_lag_") for c in feat_cols), "sanity: lags present"

    mismatched = []
    for col in feat_cols:
        a, b = f[col], m[col]
        if pd.api.types.is_numeric_dtype(a):
            same = np.allclose(a.to_numpy(dtype="float64"),
                               b.to_numpy(dtype="float64"),
                               equal_nan=True)
        else:
            same = a.equals(b)
        if not same:
            mismatched.append(col)

    assert not mismatched, (
        "LEAKAGE: these forecast-window features changed when future actuals "
        f"were erased, so they depend on the future: {mismatched}")


def test_all_sales_lags_are_at_least_the_horizon():
    """Structural guard: a lag < 28 would break non-recursion and the cache."""
    assert min(features.LAG_DAYS) >= config.HORIZON
    assert features.ROLL_BASE_LAG >= config.HORIZON
