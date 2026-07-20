"""Build-once feature cache.

Because every feature is non-recursive (all lags >= 28; see
[`src/features.py`](features.py)), the feature values for a given (series, day)
depend only on actual sales from >= 28 days earlier. That means we can build the
full feature matrix **once** over the entire timeline and then *slice* it for any
cross-validation fold - the training and prediction rows of every fold within the
observed range get identical, leak-free features either way.

This is purely an efficiency choice: it turns "rebuild 18M rows of features per
fold" into "build once, slice many". **It must be revisited if we ever add short
or recursive lags** (lag < 28), which would make a day's features depend on the
fold's cut-off.

The cache is a single Parquet file under ``outputs/cache/``. Delete it (or pass
``--rebuild-features``) to regenerate after changing the feature recipe.
"""
from __future__ import annotations

import gc
import time
from pathlib import Path

import pandas as pd

from . import config, data, features

CACHE_DIR = config.OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Keep only rows from this day onward in the cache. Lower = more training
# history available to folds, but more rows / RAM. Features for these rows still
# use earlier history (computed over the full series before slicing).
DEFAULT_DAY_FLOOR = 1200


def cache_path_for(groups: frozenset[str], day_floor: int) -> Path:
    """Distinct cache file per (feature-group set, day floor) so configs never
    collide and identical configs are reused."""
    tag = "base" if not groups else "base+" + "+".join(sorted(groups))
    return CACHE_DIR / f"features_{tag}_f{day_floor}.parquet"


def build_feature_cache(groups: frozenset[str] = frozenset(),
                        day_floor: int = DEFAULT_DAY_FLOOR,
                        cache_path: Path | None = None,
                        verbose: bool = True) -> Path:
    """Engineer features for all stores over the full timeline; write Parquet.

    Uses ``sales_train_evaluation.csv`` (actuals through d_1941). Rows with
    ``d_int >= day_floor`` are kept. Safe to slice for any fold ending <= d_1941.
    """
    cache_path = cache_path or cache_path_for(groups, day_floor)
    sales_wide = data.load_sales_wide(evaluation=True)        # actuals d_1..d_1941
    calendar = data.load_calendar()
    prices = data.load_prices()
    encoders = features.build_label_encoders(sales_wide, calendar)

    parts = []
    for store in config.STORES:
        t0 = time.time()
        long_df = data.melt_store(sales_wide, store, add_horizon_days=0)
        feat = features.build_features(long_df, calendar, prices, encoders, groups=groups)
        feat = feat[feat["d_int"] >= day_floor].copy()
        feat = data.reduce_mem_usage(feat)
        parts.append(feat)
        del long_df, feat
        gc.collect()
        if verbose:
            print(f"  [{store}] features built in {time.time() - t0:.0f}s")

    full = pd.concat(parts, ignore_index=True)
    del parts, sales_wide
    gc.collect()
    # ``id`` is the only string column and there are only 30,490 unique values;
    # storing it as a category keeps the in-memory cache ~10x smaller.
    full["id"] = full["id"].astype("category")
    full.to_parquet(cache_path, index=False)
    if verbose:
        print(f"  cache written: {cache_path.name}  ({len(full):,} rows, "
              f"day_floor={day_floor}, groups={sorted(groups) or ['base']})")
    return cache_path


def load_feature_cache(groups: frozenset[str] = frozenset(),
                       day_floor: int = DEFAULT_DAY_FLOOR,
                       rebuild: bool = False) -> pd.DataFrame:
    """Load the feature cache for a (groups, day_floor) config; build if needed."""
    cache_path = cache_path_for(groups, day_floor)
    if rebuild or not cache_path.exists():
        print(f"Building feature cache {cache_path.name} (once per recipe)...")
        build_feature_cache(groups=groups, day_floor=day_floor, cache_path=cache_path)
    return pd.read_parquet(cache_path)
