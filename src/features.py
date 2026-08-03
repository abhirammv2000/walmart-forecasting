"""Feature engineering for the baseline model.

This is the *single source of truth* for features: the same function is used to
build the training table and the prediction table, so the two can never drift
apart (a bug that plagued the original train/predict split).

Design choices for the baseline
-------------------------------
* **Non-recursive lags.** Every lag is >= 28 days. To forecast any day in the
  28-day horizon (d_1914..d_1941), a lag of 28 reaches back to a day that is
  already known (d_1886..d_1913). That means a *single* model can predict the
  whole horizon in one shot with no recursion - simple and leak-free.
* **Global categorical encoding.** Category codes are fit once over the whole
  dataset and applied per store, so ``item_id`` (etc.) means the same thing in
  every store's rows. Without this a global model would be nonsense.
* **Per-store construction.** Features are built one store at a time to keep
  peak memory bounded; the caller concatenates the (windowed) results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Columns treated as categorical by LightGBM (encoded to int16 codes).
ID_CAT_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
EVENT_CAT_COLS = ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]
CATEGORICAL_FEATURES = ID_CAT_COLS + EVENT_CAT_COLS

# Lag / rolling configuration (all lags >= HORIZON keep the model non-recursive).
LAG_DAYS = [28, 29, 30, 31, 32, 33, 34, 35]
ROLL_BASE_LAG = 28
ROLL_WINDOWS = [7, 14, 28]

# Optional feature groups, toggled per experiment (see docs/05-features.md).
# Sales-derived groups stay non-recursive (built on the 28-day lag). Price and
# calendar groups may use current/future values because M5 gives future prices
# and the calendar in advance - they are known at forecast time, not leakage.
ALL_GROUPS = ("roll_ext", "price_ext", "cal_ext")

NON_FEATURE_COLS = {"id", "sales", "d_int"}


def build_label_encoders(sales_wide: pd.DataFrame,
                         calendar: pd.DataFrame) -> dict[str, dict]:
    """Build value -> integer-code maps for every categorical column.

    Fit globally so codes are consistent across stores. Missing event values
    (NaN) are folded into the ``"__none__"`` bucket and get a stable code.
    """
    encoders: dict[str, dict] = {}
    for col in ID_CAT_COLS:
        vals = sorted(sales_wide[col].unique())
        encoders[col] = {v: i for i, v in enumerate(vals)}
    for col in EVENT_CAT_COLS:
        vals = sorted(calendar[col].fillna("__none__").unique())
        encoders[col] = {v: i for i, v in enumerate(vals)}
    return encoders


def _apply_encoders(df: pd.DataFrame, encoders: dict[str, dict]) -> pd.DataFrame:
    for col, mapping in encoders.items():
        if col in EVENT_CAT_COLS:
            df[col] = df[col].fillna("__none__")
        df[col] = df[col].map(mapping).astype(np.int16)
    return df


def build_features(long_df: pd.DataFrame,
                   calendar: pd.DataFrame,
                   prices: pd.DataFrame,
                   encoders: dict[str, dict],
                   groups: frozenset[str] = frozenset()) -> pd.DataFrame:
    """Attach calendar, price, lag, rolling and date features to one store.

    ``long_df`` is the melted (series, day) table for a single store, including
    any empty future days to be predicted. ``groups`` selects optional feature
    groups (subset of ``ALL_GROUPS``) on top of the always-on base features.
    """
    cal_cols = ["d", "wm_yr_wk", "date", "event_name_1", "event_type_1",
                "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI"]
    df = long_df.merge(calendar[cal_cols], on="d", how="left")
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    # Release filtering: drop rows before an item's first sale in this store.
    # A product has no ``sell_price`` until it is stocked, so pre-release rows are
    # structural zeros (the item didn't exist yet), not real demand. Keeping them
    # only teaches the model to predict "not sold" and wastes ~10% of rows. We
    # keep everything from the first priced day onward (``cummax`` of "has price"),
    # a fixed per-series property, so the CV folds stay leak-free.
    df = df.sort_values(["id", "d_int"])
    released = df.groupby("id")["sell_price"].transform(lambda s: s.notna().cummax())
    df = df[released.astype(bool)].copy()

    # Encode categoricals with the global maps.
    df = _apply_encoders(df, encoders)

    # Calendar / date features.
    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.day.astype(np.int8)
    df["week"] = df["date"].dt.isocalendar().week.astype(np.int8)
    df["month"] = df["date"].dt.month.astype(np.int8)
    df["year"] = (df["date"].dt.year - 2011).astype(np.int8)  # 0..5, compact
    df["dayofweek"] = df["date"].dt.dayofweek.astype(np.int8)

    # Lag features (per series). Within one store, ``id`` identifies a series.
    df = df.sort_values(["id", "d_int"])
    grp = df.groupby("id")["sales"]
    for lag in LAG_DAYS:
        df[f"sales_lag_{lag}"] = grp.shift(lag).astype(np.float32)

    # Rolling mean / std built on the 28-day lag (so they stay non-recursive).
    base = df.groupby("id")[f"sales_lag_{ROLL_BASE_LAG}"]
    for w in ROLL_WINDOWS:
        df[f"rmean_l{ROLL_BASE_LAG}_w{w}"] = (
            base.transform(lambda s: s.rolling(w).mean()).astype(np.float32))
        df[f"rstd_l{ROLL_BASE_LAG}_w{w}"] = (
            base.transform(lambda s: s.rolling(w).std()).astype(np.float32))

    # Price features (base). Prices are known in advance in M5, so price features
    # are not bound by the >=28-day rule.
    item_mean_price = df.groupby(["store_id", "item_id"])["sell_price"].transform("mean")
    df["price_momentum"] = (df["sell_price"] / item_mean_price).astype(np.float32)

    # ------------------------------------------------------------------ #
    # Optional feature groups (Milestone B). See docs/05-features.md.
    # ------------------------------------------------------------------ #
    if "roll_ext" in groups:
        # Longer trend/volatility windows and the recent range, all built on the
        # 28-day lag so they remain non-recursive (like the base rolling stats).
        base_lag = df.groupby("id")[f"sales_lag_{ROLL_BASE_LAG}"]
        for w in (60, 180):
            df[f"rmean_l{ROLL_BASE_LAG}_w{w}"] = (
                base_lag.transform(lambda s: s.rolling(w).mean()).astype(np.float32))
        df[f"rstd_l{ROLL_BASE_LAG}_w60"] = (
            base_lag.transform(lambda s: s.rolling(60).std()).astype(np.float32))
        df[f"rmax_l{ROLL_BASE_LAG}_w28"] = (
            base_lag.transform(lambda s: s.rolling(28).max()).astype(np.float32))
        df[f"rmin_l{ROLL_BASE_LAG}_w28"] = (
            base_lag.transform(lambda s: s.rolling(28).min()).astype(np.float32))

    if "price_ext" in groups:
        # Promotion / price-position signals. Grouped per series (store fixed).
        g = df.groupby("id")["sell_price"]
        pmax = g.transform("max")
        df["price_max"] = pmax.astype(np.float32)
        df["price_min"] = g.transform("min").astype(np.float32)
        df["price_norm"] = (df["sell_price"] / pmax).astype(np.float32)      # 0..1 position
        df["price_nunique"] = g.transform("nunique").astype(np.float32)      # # distinct prices ~ promo activity
        # Week-over-week change: price is constant within a Walmart week, so a
        # 7-day shift compares against the previous week's price.
        df["price_change_w"] = (df["sell_price"] / g.shift(7)).astype(np.float32)

    if "cal_ext" in groups:
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(np.int8)
        df["week_of_month"] = ((df["day"] - 1) // 7 + 1).astype(np.int8)

    df.drop(columns=["date", "wm_yr_wk", "d"], inplace=True)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the model input columns: everything except identifiers/target.

    Robust to optional feature groups - any engineered column present in ``df``
    (base or from an enabled group) is used, while ``id``/``sales``/``d_int``
    are excluded.
    """
    return [c for c in df.columns if c not in NON_FEATURE_COLS]
