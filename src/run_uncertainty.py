"""Run a quantile-forecasting experiment and score it.

Trains the nine quantile models on a fold's training window, forecasts the
28-day horizon, and scores **bottom-level (item x store) weighted SPL** - the
level the inventory policy in Project 2 actually consumes.

Why bottom level only (for now)
-------------------------------
Official WSPL averages all 12 hierarchy levels, but quantiles are **not
additive**: summing bottom-level quantiles does not give a valid quantile of a
store or state total. Scoring upper levels would therefore require models fit at
those levels. We score level 12 honestly rather than report a number built on an
invalid aggregation. See ``docs/07-uncertainty.md``.

Run::

    python -m src.run_uncertainty --name q_ts700 --train-start-day 700 --day-floor 300
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime

import numpy as np
import pandas as pd

from . import config, data, dataset, features, quantile
from .wspl import QUANTILES, WSPLEvaluator

UNCERTAINTY_CSV = config.OUTPUT_DIR / "uncertainty_experiments.csv"
UNCERTAINTY_MD = config.ROOT / "docs" / "uncertainty-experiments.md"

# Same fold definitions as the point track (src/cv.py).
DEV_FOLDS = {"cv1": 1829, "cv2": 1857, "cv3": 1885}
FINAL_FOLD = {"final": 1913}
BOTTOM_LEVEL = (12,)


def run_fold(feats, last_train_day, train_start_day, n_estimators,
             eval_wide, decode, calendar, prices,
             backend="lightgbm", device="cuda"):
    """Train the quantile model(s), forecast, and score bottom-level weighted SPL."""
    print(f"  training {len(QUANTILES)} quantiles via {backend}...", flush=True)
    model = quantile.train_quantile_models(
        feats, last_train_day, train_start_day, n_estimators=n_estimators,
        backend=backend, device=device)

    preds = quantile.predict_quantiles(model, feats, last_train_day, decode)
    cross = quantile.crossing_rate(model, feats, last_train_day)

    valid_days = range(last_train_day + 1, last_train_day + 1 + config.HORIZON)
    valid_cols = [f"d_{d}" for d in valid_days]
    train_cols = config.ID_COLS + [f"d_{d}" for d in range(1, last_train_day + 1)]
    gcols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

    ev = WSPLEvaluator(eval_wide[train_cols],
                       eval_wide[config.ID_COLS + valid_cols][valid_cols],
                       calendar, prices)
    spl = ev.score({u: preds[u][gcols + valid_cols] for u in QUANTILES},
                   levels=BOTTOM_LEVEL)
    return spl, cross, model.best_iters, preds


def main():
    ap = argparse.ArgumentParser(description="M5 uncertainty (quantile) experiment")
    ap.add_argument("--name", required=True)
    ap.add_argument("--desc", default="")
    ap.add_argument("--train-start-day", type=int, default=700)
    ap.add_argument("--day-floor", type=int, default=dataset.DEFAULT_DAY_FLOOR)
    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--fold", default="cv3",
                    choices=list(DEV_FOLDS) + list(FINAL_FOLD),
                    help="Which fold to train/score on.")
    ap.add_argument("--backend", default="lightgbm", choices=["lightgbm", "xgboost"],
                    help="lightgbm = 9 CPU fits; xgboost = 1 multi-quantile GPU fit.")
    ap.add_argument("--device", default="cuda", help="xgboost device: cuda or cpu.")
    ap.add_argument("--features", default="")
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--save-preds", action="store_true",
                    help="Persist the quantile forecasts (Project 2 input).")
    args = ap.parse_args()

    groups = frozenset(g.strip() for g in args.features.split(",") if g.strip())
    folds = {**DEV_FOLDS, **FINAL_FOLD}
    last_train_day = folds[args.fold]

    t0 = time.time()
    feats = dataset.load_feature_cache(groups=groups, day_floor=args.day_floor,
                                       rebuild=args.rebuild_features)
    print(f"Loaded {len(feats):,} feature rows.", flush=True)

    eval_wide = data.load_sales_wide(evaluation=True)
    decode = eval_wide[config.ID_COLS]
    calendar, prices = data.load_calendar(), data.load_prices()

    spl, cross, iters, preds = run_fold(
        feats, last_train_day, args.train_start_day, args.n_estimators,
        eval_wide, decode, calendar, prices,
        backend=args.backend, device=args.device)

    print(f"\n  >>> bottom-level weighted SPL = {spl:.4f}")
    print(f"      quantile-crossing rate (pre-sort) = {cross:.3%}")

    if args.save_preds:
        out = config.PREDICTION_DIR / f"quantiles_{args.name}_{args.fold}.parquet"
        long = pd.concat(
            [preds[u].assign(quantile=u) for u in QUANTILES], ignore_index=True)
        long.to_parquet(out, index=False)
        print(f"      quantile forecasts -> {out}")

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "name": args.name,
        "fold": args.fold,
        "backend": args.backend,
        "train_start_day": args.train_start_day,
        "n_estimators": args.n_estimators,
        "bottom_spl": round(spl, 4),
        "crossing_rate": round(cross, 5),
        "best_iters": "/".join(str(iters[u]) for u in QUANTILES),
        "runtime_s": int(time.time() - t0),
        "desc": args.desc,
    }
    _log(row)


def _log(row: dict) -> None:
    cols = list(row.keys())
    new = not UNCERTAINTY_CSV.exists()
    with open(UNCERTAINTY_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow(row)
    if not UNCERTAINTY_MD.exists():
        UNCERTAINTY_MD.write_text(
            "# Uncertainty Experiment Log\n\nBottom-level (item x store) weighted "
            "scaled pinball loss over the nine M5 quantiles. Lower is better.\n\n"
            "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|\n",
            encoding="utf-8")
    with open(UNCERTAINTY_MD, "a", encoding="utf-8") as f:
        f.write("| " + " | ".join(str(row[c]) for c in cols) + " |\n")


if __name__ == "__main__":
    main()
