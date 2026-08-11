"""New-product / cold-start forecasting by attribute analogs.

A brand-new SKU has **no sales history**, so every lag/rolling feature the main
model relies on is undefined. You forecast it the way a merchant does: by looking
at **analogs** - existing products with the same attributes (department, store,
price tier) - and assuming the newcomer behaves like its peers. This is the skill
that separates people who understand the *business* from people who only know
time series.

Method
------
1. Summarise every existing series by its **day-of-week demand profile** (mean
   units sold on each weekday over the training period) - this carries the weekly
   seasonality a new item will inherit.
2. For a new (item, store), the forecast profile is the **mean profile of its
   analog group** (same department x store), computed **excluding the item
   itself** so no history of the "new" product leaks in.
3. Optionally scale by a **price factor**: cheaper-than-peers items sell more.

The functions here are the reusable, testable core; ``src.run_coldstart`` runs the
hold-out experiment.
"""
from __future__ import annotations

import numpy as np


def group_mean_excluding_self(values: np.ndarray,
                              group_ids: np.ndarray) -> np.ndarray:
    """Leave-one-out group mean: each row -> mean of *other* rows in its group.

    ``values`` is (n, k); ``group_ids`` is (n,). Vectorised. Rows whose group has
    only one member get NaN (no analogs).
    """
    _, inv = np.unique(group_ids, return_inverse=True)
    n_groups = inv.max() + 1
    gsum = np.zeros((n_groups, values.shape[1]), dtype=float)
    np.add.at(gsum, inv, values)
    gcount = np.bincount(inv, minlength=n_groups)
    per_row_sum = gsum[inv]
    per_row_count = gcount[inv][:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        loo = (per_row_sum - values) / (per_row_count - 1)
    loo[(per_row_count == 1)[:, 0]] = np.nan
    return loo


def price_factor(item_price: np.ndarray, analog_price: np.ndarray,
                 beta: float = 0.5, clip: tuple[float, float] = (0.5, 2.0)
                 ) -> np.ndarray:
    """Demand multiplier from relative price: (analog_price / item_price)^beta.

    A soft, bounded elasticity proxy - an item priced below its peers is expected
    to sell somewhat more, and vice versa. Not a causal estimate; a sensible prior
    for a product with no data of its own.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(item_price > 0, analog_price / item_price, 1.0)
    return np.clip(ratio ** beta, *clip)


def analog_forecast(weekday_profiles: np.ndarray, group_ids: np.ndarray,
                    horizon_weekdays: np.ndarray,
                    scale: np.ndarray | None = None) -> np.ndarray:
    """Cold-start forecast for every series over a horizon.

    Parameters
    ----------
    weekday_profiles:
        (n_series, 7) mean demand by weekday over training.
    group_ids:
        (n_series,) analog-group id (e.g. department x store).
    horizon_weekdays:
        (H,) weekday index (0-6) for each forecast day.
    scale:
        Optional (n_series,) per-series multiplier (e.g. a price factor).

    Returns (n_series, H) forecasts.
    """
    analog = group_mean_excluding_self(weekday_profiles, group_ids)  # (n, 7)
    fc = analog[:, horizon_weekdays]                                 # (n, H)
    if scale is not None:
        fc = fc * scale[:, None]
    return np.clip(fc, 0, None)
