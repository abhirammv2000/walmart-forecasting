"""Cross-validation harness and experiment log.

This is the measurement backbone for iterative improvement. It evaluates a model
configuration on several rolling 28-day windows and records the result so every
change can be judged honestly.

Folds (all scored with WRMSSE against actuals in sales_train_evaluation.csv):

* **Dev folds** - what we steer on (the mean is the headline CV number):
    - ``cv1``: train <= d_1857, forecast d_1858..d_1885
    - ``cv2``: train <= d_1885, forecast d_1886..d_1913
* **Final test** - held out, evaluated *once* at the very end:
    - ``final``: train <= d_1913, forecast d_1914..d_1941

We cannot score d_1942..d_1969 locally (labels not in the repo), so the most
recent labelled window (d_1914..d_1941) serves as the untouched final test.

Run::

    python -m src.cv --name baseline --desc "current baseline re-scored on CV"
"""
from __future__ import annotations

import argparse
import csv
import gc
import time
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config, data, dataset, features
from .wrmsse import WRMSSEEvaluator

# last_train_day for each fold; forecast window is the following 28 days.
# Three rolling dev folds (chronological) make the CV mean steadier so we can
# trust smaller improvements; the final fold stays held out.
DEV_FOLDS = {"cv1": 1829, "cv2": 1857, "cv3": 1885}   # predict 1830-57 / 58-85 / 86-1913
FINAL_FOLD = {"final": 1913}                            # predict 1914-1941 (held out)

EXPERIMENTS_CSV = config.OUTPUT_DIR / "experiments.csv"
EXPERIMENTS_MD = config.ROOT / "docs" / "experiments.md"


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _truth_frames(eval_wide: pd.DataFrame, last_train_day: int):
    """Return (train_wide, valid_truth_28cols) for a fold from the actuals."""
    valid_days = range(last_train_day + 1, last_train_day + 1 + config.HORIZON)
    train_cols = config.ID_COLS + [f"d_{d}" for d in range(1, last_train_day + 1)]
    valid_cols = [f"d_{d}" for d in valid_days]
    return eval_wide[train_cols], eval_wide[config.ID_COLS + valid_cols]


def _assemble_pred_wide(predict_rows: pd.DataFrame, preds: np.ndarray,
                        decode: pd.DataFrame) -> pd.DataFrame:
    """Wide prediction frame (ID_COLS + d_* cols) from per-row predictions.

    The cache is built from sales_train_evaluation.csv, so its ``id`` matches the
    ground-truth file exactly - a direct decode on ``id`` is correct here.
    """
    long = pd.DataFrame({"id": predict_rows["id"].to_numpy(),
                         "d_int": predict_rows["d_int"].to_numpy(),
                         "pred": np.clip(preds, 0, None)})
    wide = long.pivot(index="id", columns="d_int", values="pred")
    wide.columns = [f"d_{int(c)}" for c in wide.columns]
    wide = wide.reset_index().merge(decode, on="id", how="left")
    return wide


# --------------------------------------------------------------------------- #
# Train + score one fold
# --------------------------------------------------------------------------- #
def run_fold(feats: pd.DataFrame, last_train_day: int, train_start_day: int,
             params: dict, n_estimators: int, eval_wide: pd.DataFrame,
             decode: pd.DataFrame, calendar: pd.DataFrame, prices: pd.DataFrame):
    """Train on the fold's window, forecast its 28 days, return WRMSSE + levels."""
    feat_cols = features.feature_columns(feats)
    cat_cols = [c for c in features.CATEGORICAL_FEATURES if c in feat_cols]

    valid_start = last_train_day - config.HORIZON + 1   # last 28 train days = early-stop
    tr = feats[(feats["d_int"] >= train_start_day) & (feats["d_int"] < valid_start)]
    va = feats[(feats["d_int"] >= valid_start) & (feats["d_int"] <= last_train_day)]

    dtrain = lgb.Dataset(tr[feat_cols], label=tr["sales"],
                         categorical_feature=cat_cols, free_raw_data=True)
    dvalid = lgb.Dataset(va[feat_cols], label=va["sales"],
                         categorical_feature=cat_cols, free_raw_data=True)
    model = lgb.train({**params, "n_estimators": n_estimators}, dtrain,
                      valid_sets=[dvalid], valid_names=["valid"],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    del dtrain, dvalid, tr, va
    gc.collect()

    pred_days = list(range(last_train_day + 1, last_train_day + 1 + config.HORIZON))
    prows = feats[feats["d_int"].isin(pred_days)]
    preds = model.predict(prows[feat_cols], num_iteration=model.best_iteration)
    pred_wide = _assemble_pred_wide(prows, preds, decode)

    train_wide, valid_truth = _truth_frames(eval_wide, last_train_day)
    valid_cols = [f"d_{d}" for d in pred_days]
    group_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    evaluator = WRMSSEEvaluator(train_wide, valid_truth[valid_cols], calendar, prices)
    wrmsse, levels = evaluator.score(pred_wide[group_cols + valid_cols], return_levels=True)
    return wrmsse, levels, int(model.best_iteration)


# --------------------------------------------------------------------------- #
# Experiment runner + logging
# --------------------------------------------------------------------------- #
def run_experiment(name: str, desc: str, params: dict, train_start_day: int,
                   n_estimators: int, feats: pd.DataFrame,
                   run_final: bool = False) -> dict:
    """Evaluate a config across dev folds (+ optional final test) and log it."""
    eval_wide = data.load_sales_wide(evaluation=True)
    decode = eval_wide[config.ID_COLS]
    calendar = data.load_calendar()
    prices = data.load_prices()

    t0 = time.time()
    results, iters = {}, {}
    folds = {**DEV_FOLDS, **(FINAL_FOLD if run_final else {})}
    for fold, ltd in folds.items():
        ft0 = time.time()
        wrmsse, levels, best_it = run_fold(feats, ltd, train_start_day, params,
                                           n_estimators, eval_wide, decode,
                                           calendar, prices)
        results[fold] = wrmsse
        iters[fold] = best_it
        print(f"  [{fold}] WRMSSE={wrmsse:.4f}  best_it={best_it}  "
              f"({time.time() - ft0:.0f}s)")

    cv_mean = float(np.mean([results[f] for f in DEV_FOLDS]))
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "name": name,
        "train_start_day": train_start_day,
        "n_estimators": n_estimators,
        **{f: round(results[f], 4) for f in DEV_FOLDS},
        "cv_mean": round(cv_mean, 4),
        "final_test": round(results["final"], 4) if "final" in results else "",
        "best_iters": "/".join(str(iters[f]) for f in folds),
        "runtime_s": int(time.time() - t0),
        "desc": desc,
    }
    _log_experiment(row)
    print(f"\n  >>> CV mean WRMSSE = {cv_mean:.4f}"
          + (f" | final test = {results['final']:.4f}" if "final" in results else ""))
    return row


def _log_experiment(row: dict) -> None:
    """Append the experiment to the CSV ledger and the Markdown table.

    Schema-driven: the columns follow ``row``'s keys, so adding/removing folds
    just works.
    """
    cols = list(row.keys())
    new = not EXPERIMENTS_CSV.exists()
    with open(EXPERIMENTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow(row)

    if not EXPERIMENTS_MD.exists():
        intro = ("# Experiment Log\n\nWRMSSE on rolling dev folds "
                 "(cv1: d1830-1857, cv2: d1858-1885, cv3: d1886-1913) and the "
                 "held-out final test (d1914-1941). Lower is better. We steer on "
                 "**cv_mean**; the final test is touched only at milestones.\n\n")
        header = "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|\n"
        EXPERIMENTS_MD.write_text(intro + header, encoding="utf-8")
    with open(EXPERIMENTS_MD, "a", encoding="utf-8") as f:
        f.write("| " + " | ".join(str(row[c]) for c in cols) + " |\n")


def main():
    ap = argparse.ArgumentParser(description="M5 CV harness / experiment runner")
    ap.add_argument("--name", required=True, help="Short experiment id.")
    ap.add_argument("--desc", default="", help="What changed in this experiment.")
    ap.add_argument("--train-start-day", type=int, default=1300)
    ap.add_argument("--n-estimators", type=int, default=1500)
    ap.add_argument("--day-floor", type=int, default=dataset.DEFAULT_DAY_FLOOR,
                    help="Earliest day kept in the feature cache. Must be <= the "
                         "smallest train-start-day you intend to use. Only takes "
                         "effect when the cache is (re)built.")
    ap.add_argument("--features", default="",
                    help="Comma-separated optional feature groups on top of base "
                         f"(any of: {','.join(features.ALL_GROUPS)}).")
    ap.add_argument("--final-test", action="store_true",
                    help="Also score the held-out final window (use sparingly).")
    ap.add_argument("--rebuild-features", action="store_true")
    args = ap.parse_args()

    if args.train_start_day < args.day_floor:
        ap.error(f"train-start-day ({args.train_start_day}) < day-floor "
                 f"({args.day_floor}); rebuild the cache with a lower --day-floor.")

    groups = frozenset(g.strip() for g in args.features.split(",") if g.strip())
    unknown = groups - set(features.ALL_GROUPS)
    if unknown:
        ap.error(f"unknown feature group(s): {sorted(unknown)}; "
                 f"choose from {features.ALL_GROUPS}")

    from .baseline import LGB_PARAMS
    feats = dataset.load_feature_cache(groups=groups, day_floor=args.day_floor,
                                       rebuild=args.rebuild_features)
    print(f"Loaded {len(feats):,} feature rows.")
    run_experiment(args.name, args.desc, LGB_PARAMS, args.train_start_day,
                   args.n_estimators, feats, run_final=args.final_test)


if __name__ == "__main__":
    main()
