"""Demonstrate hierarchical reconciliation on the M5 product hierarchy.

Hierarchy (national, summed over stores): Total -> Category (3) -> Department (7)
-> Item (3,049). We fit an *independent* exponential-smoothing forecast at every
node (which makes the base forecasts incoherent), then reconcile with bottom-up,
top-down, OLS, WLS and MinT, and measure accuracy per level with MASE.

Run::

    python -m src.run_reconcile
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from . import config, reconcile

LAST_TRAIN_DAY = config.LAST_TRAIN_DAY_VALIDATION
VALID_DAYS = list(range(1914, 1942))
ALPHAS = np.round(np.linspace(0.05, 0.95, 19), 3)


def build_hierarchy():
    """Return (node_series, S, level_of_node, item_order).

    ``node_series`` is (n_nodes, n_days) national daily demand for every node,
    ordered [total, categories, departments, items]; ``S`` is the summing matrix.
    """
    sales = pd.read_csv(config.SALES_EVALUATION_CSV)
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    # National item demand (sum across stores).
    items = sales.groupby("item_id")[day_cols].sum()
    item_ids = items.index.to_numpy()
    item_mat = items.to_numpy(dtype=np.float64)                      # (3049, D)

    meta = sales[["item_id", "cat_id", "dept_id"]].drop_duplicates("item_id")
    meta = meta.set_index("item_id").loc[item_ids]
    cats = sorted(meta["cat_id"].unique())
    depts = sorted(meta["dept_id"].unique())

    total = item_mat.sum(axis=0, keepdims=True)
    cat_mat = np.vstack([item_mat[(meta["cat_id"] == c).to_numpy()].sum(axis=0)
                         for c in cats])
    dept_mat = np.vstack([item_mat[(meta["dept_id"] == d).to_numpy()].sum(axis=0)
                          for d in depts])
    node_series = np.vstack([total, cat_mat, dept_mat, item_mat])

    # Summing matrix S: rows = nodes, cols = items (bottom). Bottom block = I.
    n_items = len(item_ids)
    rows = [np.ones(n_items)]                                        # total
    for c in cats:
        rows.append((meta["cat_id"] == c).to_numpy(dtype=float))
    for d in depts:
        rows.append((meta["dept_id"] == d).to_numpy(dtype=float))
    S = np.vstack(rows + [np.eye(n_items)])

    level = (["Total"] + ["Category"] * len(cats) + ["Department"] * len(depts)
             + ["Item"] * n_items)
    return node_series, S, np.array(level)


def ses_base_forecasts(train: np.ndarray):
    """Per-node SES with grid-searched alpha. Returns (forecast, residuals).

    Vectorised over nodes: the recursion runs on the whole node vector at once for
    each candidate alpha, and we keep the best alpha per node. ``forecast`` is the
    final level (a flat h-step forecast); ``residuals`` are the in-sample one-step
    errors at each node's best alpha (needed by MinT).
    """
    n_nodes, T = train.shape
    best_sse = np.full(n_nodes, np.inf)
    best_level = train[:, 0].copy()
    best_resid = np.zeros((n_nodes, T - 1))
    for a in ALPHAS:
        level = train[:, 0].copy()
        sse = np.zeros(n_nodes)
        resid = np.zeros((n_nodes, T - 1))
        for t in range(1, T):
            e = train[:, t] - level
            resid[:, t - 1] = e
            sse += e ** 2
            level += a * e
        better = sse < best_sse
        best_sse[better] = sse[better]
        best_level[better] = level[better]
        best_resid[better] = resid[better]
    return best_level, best_resid


def main() -> None:
    t0 = time.time()
    node_series, S, level = build_hierarchy()
    n_nodes, n_items = S.shape
    print(f"Hierarchy: {n_nodes:,} nodes ({n_items:,} bottom items) "
          f"across {len(np.unique(level))} levels.")

    train = node_series[:, :LAST_TRAIN_DAY]
    test = node_series[:, LAST_TRAIN_DAY:LAST_TRAIN_DAY + len(VALID_DAYS)]

    print("Fitting per-node SES base forecasts...")
    base_level, resid = ses_base_forecasts(train)

    # Incoherence of the base forecast (top vs. sum of items).
    top, item_sum = base_level[0], base_level[n_nodes - n_items:].sum()
    print(f"  base-forecast incoherence: total={top:,.0f} vs sum(items)="
          f"{item_sum:,.0f}  gap={abs(top-item_sum)/top*100:.1f}%")

    # Reconcile (base forecast is flat, so reconcile the level vector).
    Gs = {
        "base (unreconciled)": None,
        "bottom_up": reconcile.bottom_up_G(n_nodes, n_items),
        "ols": reconcile.ols_G(S),
        "wls_struct": reconcile.wls_structural_G(S),
        "mint_shrink": reconcile.mint_shrink_G(S, resid.T),
    }

    # MASE scale per node: mean abs 1-step diff over the (active) training series.
    scale = np.array([np.mean(np.abs(np.diff(train[i]))) for i in range(n_nodes)])
    scale[scale == 0] = np.nan

    def mase_by_level(fc_level: np.ndarray) -> dict:
        mae = np.abs(test - fc_level[:, None]).mean(axis=1)
        m = mae / scale
        return {lv: round(float(np.nanmean(m[level == lv])), 3)
                for lv in ["Total", "Category", "Department", "Item"]}

    rows = []
    for name, G in Gs.items():
        fc = base_level if G is None else reconcile.reconcile(base_level, S, G)
        coherent = reconcile.is_coherent(fc, S)
        row = {"method": name, "coherent": coherent, **mase_by_level(fc)}
        row["overall"] = round(float(np.nanmean(
            [row["Total"], row["Category"], row["Department"], row["Item"]])), 3)
        rows.append(row)
    table = pd.DataFrame(rows).set_index("method")

    print("\nMASE by level (lower is better); 'coherent' = forecasts add up:")
    print(table.to_string())
    out = config.OUTPUT_DIR / "reconciliation_benchmark.csv"
    table.to_csv(out)
    print(f"\nsaved -> {out}\nDone in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
