"""WRMSSE - the official M5 *Accuracy* metric.

The competition scores point forecasts with the **Weighted Root Mean Squared
Scaled Error**, averaged over **12 levels of aggregation** (grand total down to
the 30,490 individual item-store series).

For a single series *i*::

                  sqrt( mean_t   (y_t - yhat_t)^2  over the 28 horizon days )
    RMSSE_i =     ------------------------------------------------------------
                  sqrt( mean_t   (y_t - y_{t-1})^2 over the training history )

* **Numerator**: RMS forecast error over the horizon.
* **Denominator**: RMS error of a *naive one-step* forecast on the training
  history, trimmed to start at the series' first non-zero sale so the long
  "not yet launched" prefix doesn't deflate the scale.
* "Scaled" -> dividing by the naive error makes series comparable regardless of
  volume. **RMSSE < 1 means you beat the naive forecast.**

Each series is weighted by its dollar sales over the last 28 training days
(normalised to sum to 1 per level), then::

    WRMSSE = (1/12) * sum_over_levels sum_over_series  w_i * RMSSE_i

The 12-level aggregation and the dollar weights live in :mod:`src.hierarchy`,
shared with the uncertainty metric (:mod:`src.wspl`) so the two cannot drift.

Validated in ``tests/test_wrmsse.py``: a perfect forecast scores 0, and naive
baselines reproduce known M5 magnitudes (~1.46 last-day, ~1.08 28-day-mean).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .hierarchy import N_LEVELS, HierarchicalEvaluator


class WRMSSEEvaluator(HierarchicalEvaluator):
    """Scores point forecasts with WRMSSE. See module docstring."""

    @staticmethod
    def _scale(diffs: np.ndarray) -> float:
        """Mean *squared* first difference (the RMSSE denominator, pre-sqrt)."""
        return float((diffs ** 2).mean())

    def _rmsse(self, preds_grouped: pd.DataFrame, level: int) -> pd.Series:
        valid_y = getattr(self, f"lv{level}_valid")
        mse = ((valid_y.values - preds_grouped.values) ** 2).mean(axis=1)
        scale = getattr(self, f"lv{level}_scale")
        # `scale` is NaN for degenerate series, so the result is NaN and the
        # weighted sum below skips it. No inf can leak into the score.
        with np.errstate(divide="ignore", invalid="ignore"):
            rmsse = np.sqrt(mse / scale)
        return pd.Series(rmsse, index=valid_y.index)

    def score(self, preds: pd.DataFrame, return_levels: bool = False):
        """Score a wide prediction frame (group columns + 28 ``d_*`` columns).

        Returns the overall WRMSSE, or ``(wrmsse, per_level_array)`` when
        ``return_levels`` is True.
        """
        preds = self._prep(preds)

        level_scores = []
        for level in range(1, N_LEVELS + 1):
            grouped = self._aggregate(preds, level)
            rmsse = self._rmsse(grouped, level)
            weight = getattr(self, f"lv{level}_weight")
            level_scores.append(float((weight * rmsse).sum()))

        wrmsse = float(np.mean(level_scores))
        if return_levels:
            return wrmsse, np.array(level_scores)
        return wrmsse
