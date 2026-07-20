"""Nine-quantile demand forecasting (M5 Uncertainty track).

Produces a **predictive distribution** per (item, store, day) rather than a
single number. This is what the inventory policy in Project 2 consumes: a
newsvendor order quantity *is* a quantile of demand, so the distribution is the
deliverable, not a nice-to-have.

Two backends
------------
``lightgbm`` (CPU)
    Nine independent fits, ``objective="quantile", alpha=u``. Matches the engine
    used for the point model, so the two tracks share tuning intuition. Slow:
    nine passes over ~33M rows.

``xgboost`` (GPU)
    **One** fit. XGBoost's ``reg:quantileerror`` accepts ``quantile_alpha`` as a
    *list* and trains a multi-output model, so all nine quantiles come from a
    single booster - collapsing 9 sequential trainings into 1, on the GPU.
    This is the reason the GPU is worth it here; it is a structural win, not
    just faster arithmetic.

Both return a :class:`QuantileModel`, so the rest of the pipeline is
backend-agnostic.

Quantile crossing
-----------------
Nothing forces ``q_0.005 <= ... <= q_0.995`` (independent fits in the LightGBM
case; no monotonicity constraint in either). A crossed set is not a valid
distribution and would break the newsvendor lookup, so predictions are sorted
per (series, day) - the standard, loss-preserving fix - and the pre-sort
crossing rate is reported as a diagnostic.

Aggregation
-----------
Quantiles are **not additive**: these bottom-level quantiles must not be summed
to score upper hierarchy levels. See ``docs/07-uncertainty.md``.
"""
from __future__ import annotations

import gc
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from . import config, features
from .baseline import LGB_PARAMS
from .wspl import QUANTILES


class QuantileModel:
    """Backend-agnostic handle over fitted quantile models."""

    def __init__(self, backend: str, obj, quantiles: tuple[float, ...],
                 feat_cols: list[str], cat_cols: list[str],
                 best_iters: dict[float, int],
                 cat_dtypes: dict[str, pd.CategoricalDtype] | None = None):
        self.backend = backend
        self._obj = obj                 # dict[u]->Booster (lgb) or Booster (xgb)
        self.quantiles = tuple(quantiles)
        self.feat_cols = feat_cols
        self.cat_cols = cat_cols
        self.best_iters = best_iters
        self.cat_dtypes = cat_dtypes or {}

    def predict_matrix(self, x: pd.DataFrame) -> np.ndarray:
        """Predict all quantiles -> array (n_rows, n_quantiles), clipped+sorted."""
        if self.backend == "lightgbm":
            raw = np.column_stack([
                self._obj[u].predict(x, num_iteration=self._obj[u].best_iteration)
                for u in self.quantiles])
        else:
            raw = _xgb_predict(self._obj, x, self.cat_dtypes)
        raw = np.clip(raw, 0, None)          # demand can't be negative
        return np.sort(raw, axis=1)          # fix quantile crossing


# --------------------------------------------------------------------------- #
# LightGBM backend (CPU)
# --------------------------------------------------------------------------- #
def _train_lightgbm(x_tr, y_tr, x_va, y_va, cat_cols, quantiles,
                    n_estimators, verbose) -> tuple[dict, dict]:
    # Bin the data ONCE and reuse for all nine fits: LightGBM's Dataset is just
    # binned features + labels (the objective is a training parameter), so
    # rebuilding per quantile would re-bin ~33M rows nine times for nothing.
    dtrain = lgb.Dataset(x_tr, label=y_tr, categorical_feature=cat_cols,
                         free_raw_data=False)
    dvalid = lgb.Dataset(x_va, label=y_va, categorical_feature=cat_cols,
                         free_raw_data=False)

    models, iters = {}, {}
    for u in quantiles:
        t0 = time.time()
        params = {k: v for k, v in LGB_PARAMS.items() if k != "tweedie_variance_power"}
        params.update({"objective": "quantile", "alpha": u, "metric": "quantile",
                       "n_estimators": n_estimators})
        models[u] = lgb.train(params, dtrain, valid_sets=[dvalid],
                              valid_names=["valid"],
                              callbacks=[lgb.early_stopping(50, verbose=False)])
        iters[u] = int(models[u].best_iteration)
        if verbose:
            print(f"    q={u:<5} best_it={iters[u]:<5} ({time.time()-t0:.0f}s)",
                  flush=True)
    del dtrain, dvalid
    gc.collect()
    return models, iters


# --------------------------------------------------------------------------- #
# XGBoost backend (GPU, multi-quantile in one fit)
# --------------------------------------------------------------------------- #
def build_cat_dtypes(feats: pd.DataFrame,
                     cat_cols: list[str]) -> dict[str, pd.CategoricalDtype]:
    """Fix the category set for each categorical column, once, from all rows.

    XGBoost requires train/valid/predict to share *identical* category sets - a
    plain ``astype("category")`` per slice derives categories from whatever
    happens to be in that slice, so a value present only in validation raises
    "Found a category not in the training set". Deriving the dtype once from the
    full frame makes every slice consistent by construction.
    """
    return {c: pd.CategoricalDtype(categories=np.sort(feats[c].unique()))
            for c in cat_cols}


def _as_categorical(df: pd.DataFrame,
                    cat_dtypes: dict[str, pd.CategoricalDtype]) -> pd.DataFrame:
    """Apply the shared categorical dtypes (see :func:`build_cat_dtypes`)."""
    out = df.copy()
    for c, dtype in cat_dtypes.items():
        if c in out.columns:
            out[c] = out[c].astype(dtype)
    return out


def _xgb_predict(booster, x: pd.DataFrame,
                 cat_dtypes: dict[str, pd.CategoricalDtype]) -> np.ndarray:
    import xgboost as xgb
    dm = xgb.DMatrix(_as_categorical(x, cat_dtypes), enable_categorical=True)
    pred = booster.predict(dm, iteration_range=(0, booster.best_iteration + 1))
    return pred.reshape(len(x), -1)


def _train_xgboost(x_tr, y_tr, x_va, y_va, cat_dtypes, quantiles,
                   n_estimators, verbose, device: str = "cuda") -> tuple[object, dict]:
    import xgboost as xgb

    t0 = time.time()
    # QuantileDMatrix builds the histogram index directly, which is markedly more
    # memory-frugal than DMatrix - important on a 16 GB T4 with tens of millions
    # of rows.
    dtrain = xgb.QuantileDMatrix(_as_categorical(x_tr, cat_dtypes), label=y_tr,
                                 enable_categorical=True)
    dvalid = xgb.QuantileDMatrix(_as_categorical(x_va, cat_dtypes), label=y_va,
                                 enable_categorical=True, ref=dtrain)

    params = {
        "objective": "reg:quantileerror",
        "quantile_alpha": np.array(quantiles),   # <- all nine in ONE model
        "tree_method": "hist",
        "device": device,
        # `max_leaves` is only honoured under the lossguide grow policy. Without
        # this, XGBoost defaults to depthwise and `max_depth=0` means UNLIMITED
        # depth - it grows enormous trees and training crawls. lossguide +
        # max_leaves is the direct analogue of LightGBM's leaf-wise growth.
        "grow_policy": "lossguide",
        "learning_rate": LGB_PARAMS["learning_rate"],
        "max_leaves": LGB_PARAMS["num_leaves"],
        "min_child_weight": LGB_PARAMS["min_data_in_leaf"],
        "colsample_bytree": LGB_PARAMS["feature_fraction"],
        "subsample": LGB_PARAMS["bagging_fraction"],
        "reg_alpha": LGB_PARAMS["lambda_l1"],
        "reg_lambda": LGB_PARAMS["lambda_l2"],
        "max_depth": 0,                          # depth-wise off; grow by leaves
        "seed": config.SEED,
    }
    evals_result: dict = {}
    booster = xgb.train(params, dtrain, num_boost_round=n_estimators,
                        evals=[(dvalid, "valid")], early_stopping_rounds=50,
                        evals_result=evals_result, verbose_eval=100 if verbose else False)
    if verbose:
        print(f"    xgb multi-quantile best_it={booster.best_iteration} "
              f"({time.time()-t0:.0f}s)", flush=True)
    iters = {u: int(booster.best_iteration) for u in quantiles}
    del dtrain, dvalid
    gc.collect()
    return booster, iters


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def train_quantile_models(feats: pd.DataFrame, last_train_day: int,
                          train_start_day: int, n_estimators: int = 800,
                          quantiles: tuple[float, ...] = QUANTILES,
                          backend: str = "lightgbm", device: str = "cuda",
                          verbose: bool = True) -> QuantileModel:
    """Fit quantile models on the fold's training window."""
    feat_cols = features.feature_columns(feats)
    cat_cols = [c for c in features.CATEGORICAL_FEATURES if c in feat_cols]

    valid_start = last_train_day - config.HORIZON + 1     # last 28 days = early stop
    tr = feats[(feats["d_int"] >= train_start_day) & (feats["d_int"] < valid_start)]
    va = feats[(feats["d_int"] >= valid_start) & (feats["d_int"] <= last_train_day)]
    x_tr, y_tr = tr[feat_cols], tr["sales"]
    x_va, y_va = va[feat_cols], va["sales"]
    del tr, va
    gc.collect()
    if verbose:
        print(f"    train rows={len(x_tr):,}  early-stop rows={len(x_va):,}",
              flush=True)

    cat_dtypes: dict = {}
    if backend == "lightgbm":
        obj, iters = _train_lightgbm(x_tr, y_tr, x_va, y_va, cat_cols,
                                     quantiles, n_estimators, verbose)
    elif backend == "xgboost":
        # Category sets must be identical across train/valid/predict, so derive
        # them once from every row we will ever score.
        cat_dtypes = build_cat_dtypes(feats, cat_cols)
        obj, iters = _train_xgboost(x_tr, y_tr, x_va, y_va, cat_dtypes,
                                    quantiles, n_estimators, verbose, device)
    else:
        raise ValueError(f"unknown backend: {backend}")

    del x_tr, y_tr, x_va, y_va
    gc.collect()
    return QuantileModel(backend, obj, quantiles, feat_cols, cat_cols, iters,
                         cat_dtypes)


def predict_quantiles(model: QuantileModel, feats: pd.DataFrame,
                      last_train_day: int, decode: pd.DataFrame,
                      ) -> dict[float, pd.DataFrame]:
    """Forecast the 28-day horizon at every quantile.

    Returns ``{quantile: wide frame}`` (ID columns + 28 ``d_*`` columns).
    """
    pred_days = list(range(last_train_day + 1, last_train_day + 1 + config.HORIZON))
    prows = feats[feats["d_int"].isin(pred_days)]
    mat = model.predict_matrix(prows[model.feat_cols])

    ids = prows["id"].to_numpy()
    days = prows["d_int"].to_numpy()

    out: dict[float, pd.DataFrame] = {}
    for j, u in enumerate(model.quantiles):
        long = pd.DataFrame({"id": ids, "d_int": days, "pred": mat[:, j]})
        wide = long.pivot(index="id", columns="d_int", values="pred")
        wide.columns = [f"d_{int(c)}" for c in wide.columns]
        out[u] = wide.reset_index().merge(decode, on="id", how="left")
    return out


def crossing_rate(model: QuantileModel, feats: pd.DataFrame,
                  last_train_day: int) -> float:
    """Fraction of (series, day) cells where the *raw* quantiles crossed.

    Diagnostic only - :meth:`QuantileModel.predict_matrix` already sorts. A high
    rate means the fits disagree badly and the distribution is unreliable.
    """
    pred_days = list(range(last_train_day + 1, last_train_day + 1 + config.HORIZON))
    prows = feats[feats["d_int"].isin(pred_days)]
    x = prows[model.feat_cols]
    if model.backend == "lightgbm":
        raw = np.column_stack([
            model._obj[u].predict(x, num_iteration=model._obj[u].best_iteration)
            for u in model.quantiles])
    else:
        raw = _xgb_predict(model._obj, x, model.cat_dtypes)
    return float((np.diff(raw, axis=1) < 0).any(axis=1).mean())
