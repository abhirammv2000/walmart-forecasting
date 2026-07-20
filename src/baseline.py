"""Baseline M5 model: a single global LightGBM with non-recursive features.

.. note::
   **This is the original standalone train->predict->submit script.** Day-to-day
   experimentation now runs through :mod:`src.cv` (rolling-origin CV + experiment
   log) on top of the :mod:`src.dataset` feature cache, which is the path all
   reported numbers come from. This module is kept because it is the only place
   that emits a Kaggle-format submission, and it shares :mod:`src.features` so
   the feature recipe cannot drift. It does **not** read the feature cache and
   does not know about CV folds - prefer ``python -m src.cv`` for evaluation.

Two modes
---------
* ``validation``  - train on d_1..d_1913, forecast d_1914..d_1941, and score
  locally with WRMSSE against the ground truth in sales_train_evaluation.csv.
  This is how we measure progress.
* ``evaluation``  - train on d_1..d_1941, forecast d_1942..d_1969, and write a
  Kaggle submission (cannot be scored locally; the labels were never released).

Run::

    python -m src.baseline --mode validation --train-start-day 1300

Feature engineering is done one store at a time to keep peak memory bounded;
only the rows inside the training window (plus the 28 horizon rows) are kept.
"""
from __future__ import annotations

import argparse
import gc
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config, data, features
from .wrmsse import WRMSSEEvaluator

LGB_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "tweedie",          # intermittent, non-negative count demand
    "tweedie_variance_power": 1.1,
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 255,
    "min_data_in_leaf": 255,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.1,
    "force_row_wise": True,
    # Reproducibility: without this, LightGBM's multithreaded histogram building
    # is non-deterministic, and we measured ~+/-0.01 WRMSSE run-to-run on the
    # same config - larger than several feature effects we were trying to judge.
    # Requires force_row_wise/force_col_wise. Costs a little speed, buys
    # trustworthy A/B comparisons.
    "deterministic": True,
    "seed": config.SEED,
    "verbose": -1,
    "n_jobs": -1,
}


def build_dataset(mode: str, train_start_day: int):
    """Build the long feature table (training rows + horizon rows) for a mode."""
    if mode == "validation":
        sales_wide = data.load_sales_wide(evaluation=False)   # ends d_1913
        last_train_day = config.LAST_TRAIN_DAY_VALIDATION
    else:
        sales_wide = data.load_sales_wide(evaluation=True)    # ends d_1941
        last_train_day = config.LAST_TRAIN_DAY_EVALUATION
    horizon_days = list(range(last_train_day + 1, last_train_day + 1 + config.HORIZON))

    calendar = data.load_calendar()
    prices = data.load_prices()
    encoders = features.build_label_encoders(sales_wide, calendar)

    keep_from = train_start_day - max(features.LAG_DAYS) - max(features.ROLL_WINDOWS)
    train_parts, horizon_parts = [], []

    for store in config.STORES:
        t0 = time.time()
        long_df = data.melt_store(sales_wide, store, add_horizon_days=config.HORIZON)
        feat = features.build_features(long_df, calendar, prices, encoders)
        feat = data.reduce_mem_usage(feat)

        # Training rows: inside the window and with a known target.
        train_mask = (feat["d_int"] >= train_start_day) & (feat["d_int"] <= last_train_day)
        train_parts.append(feat[train_mask].copy())
        horizon_parts.append(feat[feat["d_int"].isin(horizon_days)].copy())

        del long_df, feat
        gc.collect()
        print(f"  [{store}] features built in {time.time() - t0:.0f}s")

    train_df = pd.concat(train_parts, ignore_index=True)
    horizon_df = pd.concat(horizon_parts, ignore_index=True)
    del train_parts, horizon_parts, sales_wide
    gc.collect()
    return train_df, horizon_df, last_train_day, calendar, prices


def train_model(train_df: pd.DataFrame, feat_cols: list[str], cat_cols: list[str],
                last_train_day: int, n_estimators: int):
    """Train one global LightGBM; early-stop on the last 28 days of the window."""
    valid_start = last_train_day - config.HORIZON + 1
    tr = train_df[train_df["d_int"] < valid_start]
    va = train_df[train_df["d_int"] >= valid_start]
    print(f"  train rows: {len(tr):,}   early-stop rows: {len(va):,}")

    dtrain = lgb.Dataset(tr[feat_cols], label=tr["sales"],
                         categorical_feature=cat_cols, free_raw_data=True)
    dvalid = lgb.Dataset(va[feat_cols], label=va["sales"],
                         categorical_feature=cat_cols, free_raw_data=True)
    del tr, va
    gc.collect()

    params = {**LGB_PARAMS, "n_estimators": n_estimators}
    model = lgb.train(
        params, dtrain, valid_sets=[dtrain, dvalid], valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    return model


def predict_horizon(model, horizon_df, feat_cols, last_train_day):
    """Predict the 28 horizon days; return a wide frame: ID_COLS + d_* columns."""
    preds = np.clip(
        model.predict(horizon_df[feat_cols], num_iteration=model.best_iteration),
        0, None)                                     # demand can't be negative

    # Pivot on the unique string ``id`` (one row per series, one column per day).
    long = pd.DataFrame({"id": horizon_df["id"].to_numpy(),
                         "d_int": horizon_df["d_int"].to_numpy(),
                         "pred": preds})
    wide = long.pivot(index="id", columns="d_int", values="pred")
    wide.columns = [f"d_{int(c)}" for c in wide.columns]
    wide = wide.reset_index()

    # Re-attach the decoded (string) ID columns. The series ``id`` differs only
    # by a ``_validation`` / ``_evaluation`` suffix between the two raw files, so
    # we join on the suffix-stripped base id (item_store), which is shared.
    wide["base_id"] = wide["id"].str.rsplit("_", n=1).str[0]
    decode = pd.read_csv(config.SALES_EVALUATION_CSV, usecols=config.ID_COLS)
    decode["base_id"] = decode["id"].str.rsplit("_", n=1).str[0]
    decode = decode.drop(columns="id").drop_duplicates("base_id")
    wide = wide.merge(decode, on="base_id", how="left").drop(columns="base_id")
    return wide[config.ID_COLS + [c for c in wide.columns if c.startswith("d_")]]


def score_validation(pred_wide, last_train_day, calendar, prices):
    """Compute WRMSSE for a validation-mode prediction frame.

    WRMSSE aggregates by the category columns (item/dept/cat/store/state), never
    by the raw ``id``, so we score on those - which sidesteps the
    ``_validation`` vs ``_evaluation`` id-suffix mismatch entirely.
    """
    eval_wide = data.load_sales_wide(evaluation=True)
    train_cols = config.ID_COLS + [f"d_{d}" for d in range(1, last_train_day + 1)]
    valid_cols = [f"d_{d}" for d in config.VALIDATION_DAYS]
    train_wide = eval_wide[train_cols]
    valid_truth = eval_wide[config.ID_COLS + valid_cols]

    group_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    pred = pred_wide[group_cols + valid_cols].copy()

    evaluator = WRMSSEEvaluator(train_wide, valid_truth[valid_cols], calendar, prices)
    return evaluator.score(pred, return_levels=True)


def write_submission(pred_wide, mode, tag):
    """Write predictions in Kaggle F1..F28 format."""
    fcols = [f"F{i}" for i in range(1, config.HORIZON + 1)]
    sub = pred_wide.copy()
    dcols = [c for c in sub.columns if c.startswith("d_")]
    sub = sub.rename(columns=dict(zip(sorted(dcols, key=lambda c: int(c[2:])), fcols)))
    suffix = "validation" if mode == "validation" else "evaluation"
    sub = sub[["id"] + fcols]
    path = config.SUBMISSION_DIR / f"submission_{tag}_{suffix}.csv"
    sub.to_csv(path, index=False)
    return path


def main():
    ap = argparse.ArgumentParser(description="M5 baseline (global LightGBM)")
    ap.add_argument("--mode", choices=["validation", "evaluation"], default="validation")
    ap.add_argument("--train-start-day", type=int, default=1300,
                    help="First day used for training (lower = more data, more RAM).")
    ap.add_argument("--n-estimators", type=int, default=1500)
    ap.add_argument("--tag", type=str, default="baseline")
    args = ap.parse_args()

    t_start = time.time()
    print(f"== M5 baseline | mode={args.mode} | train_start_day={args.train_start_day} ==")

    print("Building features (per store)...")
    train_df, horizon_df, last_train_day, calendar, prices = build_dataset(
        args.mode, args.train_start_day)
    feat_cols = features.feature_columns(train_df)
    cat_cols = [c for c in features.CATEGORICAL_FEATURES if c in feat_cols]
    print(f"  {len(feat_cols)} features, {len(train_df):,} training rows")

    print("Training global LightGBM...")
    model = train_model(train_df, feat_cols, cat_cols, last_train_day, args.n_estimators)
    model_path = config.MODEL_DIR / f"{args.tag}_{args.mode}.txt"
    model.save_model(str(model_path), num_iteration=model.best_iteration)
    print(f"  best_iteration={model.best_iteration}, saved -> {model_path}")
    del train_df
    gc.collect()

    print("Predicting horizon...")
    pred_wide = predict_horizon(model, horizon_df, feat_cols, last_train_day)
    pred_path = config.PREDICTION_DIR / f"{args.tag}_{args.mode}.csv"
    pred_wide.to_csv(pred_path, index=False)

    sub_path = write_submission(pred_wide, args.mode, args.tag)
    print(f"  predictions -> {pred_path}")
    print(f"  submission  -> {sub_path}")

    if args.mode == "validation":
        print("Scoring WRMSSE (vs d_1914..d_1941 ground truth)...")
        wrmsse, levels = score_validation(pred_wide, last_train_day, calendar, prices)
        print("\n  Per-level WRMSSE:")
        for i, s in enumerate(levels, start=1):
            print(f"    Level {i:2d}: {s:.4f}")
        print(f"\n  >>> OVERALL WRMSSE = {wrmsse:.4f}\n")

    print(f"Done in {time.time() - t_start:.0f}s.")


if __name__ == "__main__":
    main()
