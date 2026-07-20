"""Shared M5 hierarchy machinery for both competition metrics.

M5 scores two tracks, and they share almost all of their structure:

* **Accuracy** -> WRMSSE (squared error, scaled by squared first differences)
* **Uncertainty** -> WSPL (pinball loss, scaled by *absolute* first differences)

Both aggregate the same 30,490 bottom series into the same **12 levels**
(42,840 series total), and both weight each series by the **dollars it sold in
the last 28 training days**, normalised to sum to 1 within each level.

Only two things differ: how the training-history scale is reduced (squared vs
absolute), and the loss applied to the forecast. Everything else lives here so
the two metrics cannot drift apart - the weights and level definitions are
computed in exactly one place.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The 12 aggregation levels, expressed as the group-by key for each.
GROUP_IDS = (
    "all_id",                         # 1  - grand total (1 series)
    "state_id",                       # 2  - 3
    "store_id",                       # 3  - 10
    "cat_id",                         # 4  - 3
    "dept_id",                        # 5  - 7
    ["state_id", "cat_id"],           # 6  - 9
    ["state_id", "dept_id"],          # 7  - 21
    ["store_id", "cat_id"],           # 8  - 30
    ["store_id", "dept_id"],          # 9  - 70
    "item_id",                        # 10 - 3,049
    ["item_id", "state_id"],          # 11 - 9,147
    ["item_id", "store_id"],          # 12 - 30,490 (bottom)
)

N_LEVELS = len(GROUP_IDS)


class HierarchicalEvaluator:
    """Base: builds per-level scales, weights and aggregated ground truth.

    Subclasses supply :meth:`_scale` (how to reduce the training first
    differences) and a ``score`` method implementing their loss.

    Parameters
    ----------
    train_df:
        Wide sales for the training period: ID columns + ``d_1`` .. ``d_N``.
    valid_df:
        Wide ground-truth sales for the 28-day horizon (ID columns optional).
    calendar, prices:
        Raw tables, used to convert units into dollar weights.
    """

    def __init__(self, train_df: pd.DataFrame, valid_df: pd.DataFrame,
                 calendar: pd.DataFrame, prices: pd.DataFrame):
        train_df = train_df.copy()
        train_df["all_id"] = "all"

        self.train_target_cols = [c for c in train_df.columns if c.startswith("d_")]
        self.valid_target_cols = [c for c in valid_df.columns if c.startswith("d_")]
        self.id_cols = [c for c in train_df.columns if not c.startswith("d_")]
        self.weight_cols = self.train_target_cols[-28:]   # last 28 training days

        if not all(c in valid_df.columns for c in self.id_cols):
            valid_df = pd.concat([train_df[self.id_cols].reset_index(drop=True),
                                  valid_df.reset_index(drop=True)], axis=1)

        self.train_df = train_df
        self.valid_df = valid_df
        self.calendar = calendar
        self.prices = prices

        weight_df = self._build_weight_df()

        for i, gid in enumerate(GROUP_IDS, start=1):
            train_y = train_df.groupby(gid)[self.train_target_cols].sum()

            scales = []
            for _, row in train_y.iterrows():
                vals = row.values
                active = vals[np.argmax(vals != 0):]      # trim pre-launch zeros
                diffs = np.diff(active)
                s = self._scale(diffs) if len(diffs) else np.nan
                # Zero scale = the series never moved over its active history.
                # The metric is undefined there; mark NaN and exclude it rather
                # than letting a divide-by-zero inject inf into the score.
                scales.append(s if s and s > 0 else np.nan)
            setattr(self, f"lv{i}_scale", np.array(scales, dtype=np.float64))

            setattr(self, f"lv{i}_valid",
                    valid_df.groupby(gid)[self.valid_target_cols].sum())

            lv_weight = weight_df.groupby(gid)[self.weight_cols].sum().sum(axis=1)
            setattr(self, f"lv{i}_weight", lv_weight / lv_weight.sum())

    # -- to be provided by subclasses ------------------------------------- #
    @staticmethod
    def _scale(diffs: np.ndarray) -> float:
        """Reduce training first-differences to the scaling denominator."""
        raise NotImplementedError

    # -- shared helpers ---------------------------------------------------- #
    def _build_weight_df(self) -> pd.DataFrame:
        """Dollar sales (units x price) per series over the last 28 train days."""
        day_to_week = self.calendar.set_index("d")["wm_yr_wk"].to_dict()
        wdf = (self.train_df[["item_id", "store_id"] + self.weight_cols]
               .set_index(["item_id", "store_id"]).stack().reset_index())
        wdf.columns = ["item_id", "store_id", "d", "units"]
        wdf["wm_yr_wk"] = wdf["d"].map(day_to_week)
        wdf = wdf.merge(self.prices, how="left", on=["item_id", "store_id", "wm_yr_wk"])
        wdf["dollars"] = wdf["units"] * wdf["sell_price"]
        wdf = wdf.set_index(["item_id", "store_id", "d"])["dollars"].unstack("d")
        wdf = wdf.loc[list(zip(self.train_df.item_id, self.train_df.store_id))]
        wdf = wdf.reset_index(drop=True)
        return pd.concat([self.train_df[self.id_cols].reset_index(drop=True), wdf],
                         axis=1)

    def _prep(self, preds: pd.DataFrame) -> pd.DataFrame:
        """Attach the total-level key and align day-column names to the truth."""
        preds = preds.copy()
        preds["all_id"] = "all"
        pred_day_cols = [c for c in preds.columns if c.startswith("d_")]
        return preds.rename(columns=dict(zip(pred_day_cols, self.valid_target_cols)))

    def _aggregate(self, preds: pd.DataFrame, level: int) -> pd.DataFrame:
        """Sum predictions to a level; rows align with that level's truth index."""
        return preds.groupby(GROUP_IDS[level - 1])[self.valid_target_cols].sum()

    def excluded_series(self) -> dict[int, int]:
        """Per level, how many series have an undefined (degenerate) scale.

        Reported for transparency: these carry negligible dollar weight, but the
        count should stay small.
        """
        return {i: int(np.isnan(getattr(self, f"lv{i}_scale")).sum())
                for i in range(1, N_LEVELS + 1)}
