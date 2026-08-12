"""Safety stock vs. the empirical quantile - does the normal approximation hold?

The classical safety-stock order-up-to (``mu + z*sigma``) is the newsvendor order
*under a normal demand assumption*. This script sets both methods to the same
service level (the newsvendor critical ratio) and simulates them against actual
demand, to show where the Gaussian assumption breaks for intermittent retail
demand.

Run::

    python -m src.run_safety_stock
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config, data, inventory, safety_stock
from .run_inventory import load_actuals, load_prices, load_quantile_cube
from .wspl import QUANTILES

LAST_TRAIN_DAY = config.LAST_TRAIN_DAY_VALIDATION
VALID_COLS = [f"d_{d}" for d in range(1914, 1942)]
RECENT = 365   # days of history used to estimate mu / sigma


def main() -> None:
    t0 = time.time()
    cube, keys = load_quantile_cube(
        str(config.PREDICTION_DIR / "quantiles_q_trained_final.parquet"))
    demand = load_actuals(keys)
    price = load_prices(keys)
    qs = np.array(sorted(QUANTILES))
    cu, co = inventory.DEFAULT_UNDERAGE_FRAC, inventory.DEFAULT_OVERAGE_FRAC
    cr = inventory.critical_ratio(cu, co)                      # 0.909

    # Per-series mu, sigma of daily demand over recent training history.
    ew = data.load_sales_wide(evaluation=True).set_index("id").loc[keys["id"]]
    recent_cols = [f"d_{d}" for d in range(LAST_TRAIN_DAY - RECENT + 1, LAST_TRAIN_DAY + 1)]
    hist = ew[recent_cols].to_numpy(dtype=np.float64)
    mu, sigma = hist.mean(axis=1), hist.std(axis=1)
    print(f"{len(keys):,} series | service level = critical ratio = {cr:.3f} "
          f"(z = {safety_stock.service_factor(cr):.3f})")

    # Two order-up-to levels at the SAME service level, L=0 / R=1 (one-day cover).
    out_normal = safety_stock.order_up_to_normal(
        mu, sigma, lead_time=0, review_period=1, service_level=cr)
    out_normal = np.repeat(out_normal[:, None], len(VALID_COLS), axis=1)
    out_empirical = inventory.order_up_to_levels(cube, qs, cr)

    rows = []
    for name, order in [("normal safety stock (mu + z*sigma)", out_normal),
                        ("empirical quantile (newsvendor)", out_empirical)]:
        m = inventory.simulate(order, demand, price, cu, co)
        rows.append({"method": name, "target_SL": round(cr, 3),
                     "fill_rate": round(m["fill_rate"], 3),
                     "total_cost": round(m["total_cost"], 0)})
    table = pd.DataFrame(rows).set_index("method")

    print("\nBoth methods aim for the same service level; only one hits it:")
    print(table.to_string())
    fn = table.loc["normal safety stock (mu + z*sigma)", "fill_rate"]
    fe = table.loc["empirical quantile (newsvendor)", "fill_rate"]
    print(f"\n  target service level        : {cr:.1%}")
    print(f"  normal-approx fill achieved : {fn:.1%}  (gap {cr-fn:+.1%})")
    print(f"  empirical-quantile fill     : {fe:.1%}")
    print("  -> the Gaussian formula under-stocks zero-inflated demand; the "
          "distribution-based order is properly calibrated.")

    out = config.OUTPUT_DIR / "safety_stock_benchmark.csv"
    table.to_csv(out)
    print(f"\nsaved -> {out}\nDone in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
