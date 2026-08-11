"""Benchmark classical intermittent methods against the ML model on M5.

Produces the honest, nuanced answer to "do you know Croston/TSB, and do you know
when they win?": we score Croston / SBA / TSB and a seasonal-naive benchmark on
the held-out window, compare to the LightGBM point forecast, and break the result
down by Syntetos-Boylan demand class.

Run::

    python -m src.run_intermittent
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config, data, intermittent
from .wrmsse import WRMSSEEvaluator

LAST_TRAIN_DAY = config.LAST_TRAIN_DAY_VALIDATION      # 1913
VALID_DAYS = list(range(1914, 1942))
VALID_COLS = [f"d_{d}" for d in VALID_DAYS]
GROUP_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]


def _flat_forecast(train_mat: np.ndarray, fn) -> np.ndarray:
    """Apply a per-series scalar-rate method to every row; tile over 28 days."""
    rates = np.array([fn(train_mat[i]) for i in range(train_mat.shape[0])])
    return np.repeat(rates[:, None], len(VALID_COLS), axis=1)


def main() -> None:
    t0 = time.time()
    ew = data.load_sales_wide(evaluation=True)
    calendar, prices = data.load_calendar(), data.load_prices()
    train_cols = [f"d_{d}" for d in range(1, LAST_TRAIN_DAY + 1)]
    train_mat = ew[train_cols].to_numpy(dtype=np.float64)
    print(f"Loaded {ew.shape[0]:,} series x {len(train_cols)} train days.")

    # --- forecasts ------------------------------------------------------------
    print("Fitting classical methods per series...")
    methods = {
        "croston": lambda y: intermittent.croston(y, alpha=0.1),
        "sba":     lambda y: intermittent.croston(y, alpha=0.1, variant="sba"),
        "tsb":     lambda y: intermittent.tsb(y, alpha=0.1, beta=0.1),
    }
    preds = {name: _flat_forecast(train_mat, fn) for name, fn in methods.items()}
    # Seasonal-naive benchmark: mean of the last 28 training days.
    preds["seasonal_naive"] = np.repeat(
        train_mat[:, -28:].mean(axis=1)[:, None], len(VALID_COLS), axis=1)
    # LightGBM point forecast (aligned by base id: _validation vs _evaluation).
    pf = pd.read_csv(config.PREDICTION_DIR / "baseline_validation.csv")
    pf["base_id"] = pf["id"].str.rsplit("_", n=1).str[0]
    pf = pf.set_index("base_id")
    base = ew["id"].str.rsplit("_", n=1).str[0]
    preds["lightgbm"] = pf.loc[base, VALID_COLS].to_numpy(dtype=np.float64)

    # --- score with WRMSSE ----------------------------------------------------
    print("Scoring WRMSSE...")
    evaluator = WRMSSEEvaluator(ew[config.ID_COLS + train_cols],
                                ew[config.ID_COLS + VALID_COLS][VALID_COLS],
                                calendar, prices)
    overall = {}
    for name, mat in preds.items():
        frame = ew[GROUP_COLS].copy()
        frame[VALID_COLS] = mat
        overall[name] = evaluator.score(frame)

    # --- demand-class breakdown (bottom-level RMSSE, weighted) ----------------
    classes = np.array([intermittent.classify(train_mat[i])
                        for i in range(train_mat.shape[0])])
    truth = ew[VALID_COLS].to_numpy(dtype=np.float64)
    # Per-series RMSSE denominator (mean squared 1-step diff over active history).
    scale = np.array([
        np.mean(np.diff(train_mat[i][np.argmax(train_mat[i] != 0):]) ** 2)
        if (train_mat[i] != 0).any() else np.nan
        for i in range(train_mat.shape[0])])

    def series_rmsse(mat: np.ndarray) -> np.ndarray:
        mse = ((truth - mat) ** 2).mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.sqrt(mse / scale)

    rows = []
    for name, mat in preds.items():
        r = series_rmsse(mat)
        row = {"method": name, "WRMSSE": round(overall[name], 4)}
        for cls in ["smooth", "intermittent", "erratic", "lumpy"]:
            m = (classes == cls) & np.isfinite(r)
            row[cls] = round(float(np.nanmean(r[m])), 3) if m.any() else np.nan
        rows.append(row)
    table = pd.DataFrame(rows).set_index("method")

    dist = pd.Series(classes).value_counts()
    print("\nDemand-class mix (of 30,490 series):")
    for cls in ["smooth", "intermittent", "erratic", "lumpy"]:
        print(f"  {cls:12s} {dist.get(cls, 0):6,}  ({dist.get(cls, 0)/len(classes)*100:.0f}%)")

    print("\nWRMSSE (overall) and mean bottom-level RMSSE by demand class:")
    print(table.to_string(float_format=lambda v: f"{v:.3f}"))

    out = config.OUTPUT_DIR / "intermittent_benchmark.csv"
    table.to_csv(out)
    print(f"\nsaved -> {out}\nDone in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
