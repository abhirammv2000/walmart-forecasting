"""Data loading helpers for the M5 dataset.

The raw competition data comes in three files:

* ``sales_train_*.csv`` - one row per (item, store) series in *wide* format,
  with one column per day (``d_1`` .. ``d_1913`` or ``d_1941``).
* ``calendar.csv``      - one row per day, mapping ``d_N`` to a real date plus
  events and SNAP (food-stamp) flags.
* ``sell_prices.csv``   - weekly selling price per (store, item).

The modelling code works on a *long* (tidy) table with one row per
(series, day). This module loads the raw files and reshapes the sales table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def reduce_mem_usage(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Downcast numeric columns to the smallest dtype that holds their range.

    The M5 long table is tens of millions of rows, so shrinking int64/float64
    to int8/float32 is the difference between fitting in RAM and not.
    """
    start = df.memory_usage(deep=True).sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_integer_dtype(col_type):
            c_min, c_max = df[col].min(), df[col].max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        elif pd.api.types.is_float_dtype(col_type):
            # float32 is plenty of precision for sales counts / prices and
            # avoids the surprises that come with float16 rounding.
            df[col] = df[col].astype(np.float32)
    if verbose:
        end = df.memory_usage(deep=True).sum() / 1024 ** 2
        print(f"  reduce_mem_usage: {start:.0f} MB -> {end:.0f} MB "
              f"({100 * (start - end) / start:.0f}% smaller)")
    return df


def load_calendar() -> pd.DataFrame:
    """Load calendar and add an integer day index ``d_int`` (the N in d_N)."""
    cal = pd.read_csv(config.CALENDAR_CSV)
    cal["d_int"] = cal["d"].str[2:].astype(np.int16)
    return cal


def load_prices() -> pd.DataFrame:
    """Load the weekly (store, item) selling prices."""
    return pd.read_csv(config.PRICES_CSV)


def load_sales_wide(evaluation: bool = True) -> pd.DataFrame:
    """Load the wide sales table.

    Parameters
    ----------
    evaluation:
        If True load ``sales_train_evaluation.csv`` (d_1..d_1941); this gives us
        the validation-window ground truth (d_1914..d_1941) for local scoring.
        If False load ``sales_train_validation.csv`` (d_1..d_1913).
    """
    path = (config.SALES_EVALUATION_CSV if evaluation
            else config.SALES_VALIDATION_CSV)
    return pd.read_csv(path)


def melt_store(sales_wide: pd.DataFrame, store_id: str,
               add_horizon_days: int = 0) -> pd.DataFrame:
    """Melt one store's wide rows into long (series, day) format.

    Parameters
    ----------
    sales_wide:
        The full wide sales table.
    store_id:
        Which store to extract (keeps peak memory to one store at a time).
    add_horizon_days:
        Number of empty future days to append (with sales = NaN) so we can
        attach features and predict them. Days are numbered continuing from the
        last day present in ``sales_wide``.
    """
    store_wide = sales_wide[sales_wide["store_id"] == store_id].copy()

    day_cols = [c for c in store_wide.columns if c.startswith("d_")]
    last_day = max(int(c[2:]) for c in day_cols)

    # Append empty future columns to be filled by prediction.
    for h in range(1, add_horizon_days + 1):
        store_wide[f"d_{last_day + h}"] = np.nan

    long_df = store_wide.melt(
        id_vars=config.ID_COLS, var_name="d", value_name="sales",
    )
    long_df["d_int"] = long_df["d"].str[2:].astype(np.int16)
    long_df["sales"] = long_df["sales"].astype(np.float32)
    return long_df
