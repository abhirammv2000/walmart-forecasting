"""Project 2 driver: turn quantile forecasts into stocking decisions and cost them.

Loads the saved nine-quantile forecasts for the held-out window, applies two
competing policies against the **actual** realised demand, and reports the
business metrics plus the service/cost trade-off curve.

The comparison that matters::

    policy A: order the central (median) forecast   <- "we have a point forecast"
    policy B: order Q* = F^-1(Cu/(Cu+Co))           <- newsvendor, uses the tail

Run::

    python -m src.run_inventory --quantiles outputs/predictions/quantiles_*.parquet
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd

from . import config, data, inventory
from .wspl import QUANTILES

VALID_DAYS = list(range(1914, 1942))
DAY_COLS = [f"d_{d}" for d in VALID_DAYS]


def load_quantile_cube(path: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Load saved forecasts into a ``(n_series, n_days, n_quantiles)`` cube.

    Returns the cube plus the series key frame (row order matches axis 0).
    """
    long = pd.read_parquet(path)
    quantiles = sorted(long["quantile"].unique())
    if len(quantiles) != len(QUANTILES):
        raise ValueError(f"expected {len(QUANTILES)} quantiles, found {len(quantiles)}")

    first = long[long["quantile"] == quantiles[0]].sort_values("id").reset_index(drop=True)
    keys = first[["id", "item_id", "store_id"]].copy()

    cube = np.empty((len(keys), len(DAY_COLS), len(quantiles)), dtype=np.float64)
    for k, q in enumerate(quantiles):
        block = (long[long["quantile"] == q].sort_values("id").reset_index(drop=True))
        if not block["id"].equals(keys["id"]):
            raise ValueError("series order differs between quantiles")
        cube[:, :, k] = block[DAY_COLS].to_numpy(dtype=np.float64)

    # Guarantee monotone quantiles (predict already sorts; be defensive).
    cube = np.sort(cube, axis=2)
    return cube, keys


def load_actuals(keys: pd.DataFrame) -> np.ndarray:
    """Realised demand for the held-out window, aligned to ``keys`` row order."""
    ew = data.load_sales_wide(evaluation=True)
    ew = ew.set_index("id").loc[keys["id"]]
    return ew[DAY_COLS].to_numpy(dtype=np.float64)


def load_prices(keys: pd.DataFrame) -> np.ndarray:
    """Unit sell price per (series, day), aligned to ``keys`` row order.

    Prices are weekly (keyed by ``wm_yr_wk``), so we map each forecast day to its
    Walmart week and join. M5 publishes prices for the forecast period, so this
    is legitimately known at decision time.
    """
    calendar = data.load_calendar()
    prices = data.load_prices()
    day_to_week = (calendar[calendar["d"].isin(DAY_COLS)]
                   .set_index("d")["wm_yr_wk"].to_dict())

    base = keys[["item_id", "store_id"]].copy()
    base["_row"] = np.arange(len(base))
    frames = []
    for j, dcol in enumerate(DAY_COLS):
        f = base.copy()
        f["wm_yr_wk"] = day_to_week[dcol]
        f["_col"] = j
        frames.append(f)
    long = pd.concat(frames, ignore_index=True)
    long = long.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")

    out = np.full((len(keys), len(DAY_COLS)), np.nan)
    out[long["_row"].to_numpy(), long["_col"].to_numpy()] = long["sell_price"].to_numpy()

    # Fall back to the series' own median price, then the global median, so a
    # missing week never silently zeroes an item's economics.
    row_med = np.nanmedian(out, axis=1)
    row_med = np.where(np.isnan(row_med), np.nanmedian(out), row_med)
    idx = np.where(np.isnan(out))
    out[idx] = row_med[idx[0]]
    return out


def main():
    ap = argparse.ArgumentParser(description="M5 inventory policy simulation")
    ap.add_argument("--quantiles", required=True,
                    help="Path (or glob) to the saved quantile parquet.")
    ap.add_argument("--underage-frac", type=float, default=inventory.DEFAULT_UNDERAGE_FRAC)
    ap.add_argument("--overage-frac", type=float, default=inventory.DEFAULT_OVERAGE_FRAC)
    ap.add_argument("--no-carryover", action="store_true",
                    help="Treat each day as an independent newsvendor period.")
    args = ap.parse_args()

    matches = sorted(glob.glob(args.quantiles))
    if not matches:
        raise SystemExit(f"no quantile file matched: {args.quantiles}")
    path = matches[-1]
    print(f"Loading quantile forecasts: {path}")

    cube, keys = load_quantile_cube(path)
    demand = load_actuals(keys)
    price = load_prices(keys)
    qs = np.array(sorted(QUANTILES))
    carry = not args.no_carryover
    cu, co = args.underage_frac, args.overage_frac
    cr = inventory.critical_ratio(cu, co)

    print(f"  series={cube.shape[0]:,}  days={cube.shape[1]}  quantiles={cube.shape[2]}")
    print(f"  Cu={cu:.3f}*price  Co={co:.3f}*price  ->  critical ratio = {cr:.4f}")

    # --- the headline comparison ------------------------------------------- #
    policies = {
        "mean_forecast (order the median)": inventory.mean_forecast_policy(cube, qs),
        f"newsvendor (order Q* at CR={cr:.3f})": inventory.order_up_to_levels(cube, qs, cr),
    }
    rows = []
    for name, q in policies.items():
        m = inventory.simulate(q, demand, price, cu, co, carryover=carry)
        m["policy"] = name
        rows.append(m)
    comp = pd.DataFrame(rows).set_index("policy")

    print("\n=== Policy comparison (held-out window d1914-1941) ===")
    print(comp[["fill_rate", "stockout_rate", "holding_cost",
                "shortage_cost", "total_cost"]].to_string(
        float_format=lambda v: f"{v:,.2f}"))

    a, b = comp["total_cost"].iloc[0], comp["total_cost"].iloc[1]
    print(f"\n  newsvendor vs mean-forecast: total cost {a:,.0f} -> {b:,.0f} "
          f"({100*(a-b)/a:.1f}% lower)")
    print(f"  fill rate: {comp['fill_rate'].iloc[0]:.3%} -> "
          f"{comp['fill_rate'].iloc[1]:.3%}")

    comp_path = config.OUTPUT_DIR / "inventory_policy_comparison.csv"
    comp.to_csv(comp_path)

    # --- trade-off curve ---------------------------------------------------- #
    print("\n=== Service-level / cost trade-off ===")
    curve = inventory.service_level_curve(cube, qs, demand, price,
                                          underage_frac=cu, overage_frac=co,
                                          carryover=carry)
    print(curve[["fill_rate", "holding_cost", "shortage_cost", "total_cost"]]
          .to_string(float_format=lambda v: f"{v:,.2f}"))

    best = curve["total_cost"].idxmin()
    print(f"\n  empirical cost-minimising service level = {best:.3f}")
    print(f"  theoretical critical ratio               = {cr:.3f}")

    curve_path = config.OUTPUT_DIR / "inventory_service_curve.csv"
    curve.to_csv(curve_path)
    print(f"\n  saved -> {comp_path}\n  saved -> {curve_path}")


if __name__ == "__main__":
    main()
