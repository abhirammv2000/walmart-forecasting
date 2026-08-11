"""Cold-start experiment: forecast every item-store as if it were brand new.

We hide each series' own history and forecast it purely from **attribute analogs**
(same department x store, day-of-week profile, optional price adjustment), then
score against the real held-out demand - and against the full history-based
LightGBM model, to quantify the *cost of having no history*.

Run::

    python -m src.run_coldstart
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config, coldstart, data

LAST_TRAIN_DAY = config.LAST_TRAIN_DAY_VALIDATION
VALID_DAYS = list(range(1914, 1942))
VALID_COLS = [f"d_{d}" for d in VALID_DAYS]


def main() -> None:
    t0 = time.time()
    sales = data.load_sales_wide(evaluation=True)
    calendar = data.load_calendar()
    prices = data.load_prices()

    train_cols = [f"d_{d}" for d in range(1, LAST_TRAIN_DAY + 1)]
    train_mat = sales[train_cols].to_numpy(dtype=np.float64)
    actual = sales[VALID_COLS].to_numpy(dtype=np.float64)
    n = len(sales)
    print(f"{n:,} item-store series.")

    # Weekday index for train days and horizon days.
    day_wd = calendar.set_index("d_int")["wday"].astype(int)          # 1..7
    train_days = np.arange(1, LAST_TRAIN_DAY + 1)
    train_wd = day_wd.loc[train_days].to_numpy() - 1                  # 0..6
    horizon_wd = day_wd.loc[VALID_DAYS].to_numpy() - 1

    # Per-series weekday demand profile over training (mean units by weekday).
    print("Building weekday profiles...")
    profiles = np.zeros((n, 7))
    for w in range(7):
        cols = train_wd == w
        profiles[:, w] = train_mat[:, cols].mean(axis=1)

    # Analog group = department x store; price for the adjustment.
    group = (sales["dept_id"].astype(str) + "|" + sales["store_id"].astype(str)).to_numpy()
    last_wk = calendar.loc[calendar["d_int"] == LAST_TRAIN_DAY, "wm_yr_wk"].iloc[0]
    pr = prices[prices["wm_yr_wk"] == last_wk].set_index(["store_id", "item_id"])["sell_price"]
    item_price = sales.set_index(["store_id", "item_id"]).index.map(pr).to_numpy(dtype=float)
    item_price = np.where(np.isnan(item_price), np.nanmedian(item_price), item_price)
    analog_price = coldstart.group_mean_excluding_self(item_price[:, None], group)[:, 0]
    analog_price = np.where(np.isnan(analog_price), item_price, analog_price)
    pfactor = coldstart.price_factor(item_price, analog_price)

    # --- forecasts ------------------------------------------------------------
    fc_analog = coldstart.analog_forecast(profiles, group, horizon_wd)
    fc_analog_price = coldstart.analog_forecast(profiles, group, horizon_wd, scale=pfactor)
    # Naive baseline available to a new item: the global weekday profile.
    global_prof = np.nanmean(profiles, axis=0)
    fc_global = np.repeat(global_prof[horizon_wd][None, :], n, axis=0)
    # Ceiling: the full model that DOES use history.
    pf = pd.read_csv(config.PREDICTION_DIR / "baseline_validation.csv")
    pf["base_id"] = pf["id"].str.rsplit("_", n=1).str[0]
    pf = pf.set_index("base_id")
    base_id = sales["id"].str.rsplit("_", n=1).str[0]
    fc_lgbm = pf.loc[base_id, VALID_COLS].to_numpy(dtype=np.float64)

    # --- score (MASE) ---------------------------------------------------------
    scale = np.array([np.mean(np.abs(np.diff(train_mat[i]))) for i in range(n)])
    scale[scale == 0] = np.nan

    def mase(fc: np.ndarray) -> float:
        mae = np.abs(actual - fc).mean(axis=1)
        return float(np.nanmean(mae / scale))

    methods = {
        "global weekday profile": fc_global,
        "analog (dept x store)": fc_analog,
        "analog + price factor": fc_analog_price,
        "LightGBM (uses history)": fc_lgbm,
    }
    rows = [{"method": k, "MASE": round(mase(v), 3)} for k, v in methods.items()]
    table = pd.DataFrame(rows).set_index("method")
    cold = table.loc["analog + price factor", "MASE"]
    warm = table.loc["LightGBM (uses history)", "MASE"]

    print("\nCold-start accuracy (no own history) vs the history model:")
    print(table.to_string())
    print(f"\n  cost of being new: cold-start MASE {cold} vs {warm} with history "
          f"(+{(cold-warm)/warm*100:.0f}%); the analog forecast recovers "
          f"{(1 - (cold-warm)/(table.loc['global weekday profile','MASE']-warm))*100:.0f}% "
          f"of the gap between a global prior and the full model.")

    out = config.OUTPUT_DIR / "coldstart_benchmark.csv"
    table.to_csv(out)
    print(f"\nsaved -> {out}\nDone in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
