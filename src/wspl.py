"""WSPL - the official M5 *Uncertainty* metric.

Where WRMSSE grades a single number per day, the uncertainty track grades a
**predictive distribution**, represented by nine quantiles. This is the metric
that matters for downstream decisions: an inventory policy needs to know how bad
the tail is, not just the mean (see ``docs/08-inventory.md``).

The nine M5 quantiles are deliberately asymmetric-tail-heavy::

    0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995

**Pinball loss** for a quantile *u*, actual *y*, forecast *q*::

    L = u * (y - q)        if y >= q      (penalty for under-forecasting)
        (1 - u) * (q - y)  if y <  q      (penalty for over-forecasting)

At u = 0.5 this is symmetric (median). At u = 0.995 under-forecasting is
penalised ~200x more than over-forecasting, which is exactly the asymmetry a
service-level constraint encodes.

**Scaled** pinball loss divides by the naive one-step error on the training
history - but note the scaling uses **mean absolute** first differences, not
squared as in RMSSE::

                mean_t  pinball_u(y_t, q_t)  over the 28 horizon days
    SPL_i,u =   ---------------------------------------------------------
                mean_t  |y_t - y_{t-1}|      over the training history

Then, using the same dollar weights and 12 levels as WRMSSE::

    WSPL = (1/12) * sum_over_levels sum_over_series  w_i * mean_over_u( SPL_i,u )

Aggregation caveat
------------------
Quantiles are **not additive**: the 0.99 quantile of a store total is not the
sum of the 0.99 quantiles of its items. So predictions must be supplied per
level, or generated at each level - never summed up from the bottom and passed
off as upper-level quantiles. This evaluator sums whatever you give it, so it is
the caller's job to provide coherent per-level forecasts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .hierarchy import N_LEVELS, HierarchicalEvaluator

# The nine quantiles M5 scores, in ascending order.
QUANTILES: tuple[float, ...] = (0.005, 0.025, 0.165, 0.250, 0.500,
                                0.750, 0.835, 0.975, 0.995)


def pinball_loss(y: np.ndarray, q: np.ndarray, u: float) -> np.ndarray:
    """Element-wise pinball loss for quantile ``u``.

    ``u*(y-q)`` when the actual exceeds the forecast, ``(1-u)*(q-y)`` otherwise.
    Written as a single expression: ``(u - 1{y<q}) * (y - q)``.
    """
    diff = y - q
    return np.where(diff >= 0, u * diff, (u - 1.0) * diff)


class WSPLEvaluator(HierarchicalEvaluator):
    """Scores nine-quantile forecasts with WSPL. See module docstring."""

    @staticmethod
    def _scale(diffs: np.ndarray) -> float:
        """Mean *absolute* first difference (MAE-style, unlike RMSSE)."""
        return float(np.abs(diffs).mean())

    def _spl(self, preds_grouped: pd.DataFrame, level: int, u: float) -> pd.Series:
        """Scaled pinball loss per series at one level and one quantile."""
        valid_y = getattr(self, f"lv{level}_valid")
        loss = pinball_loss(valid_y.values, preds_grouped.values, u).mean(axis=1)
        scale = getattr(self, f"lv{level}_scale")
        with np.errstate(divide="ignore", invalid="ignore"):
            spl = loss / scale
        return pd.Series(spl, index=valid_y.index)

    def score(self, preds_by_quantile: dict[float, pd.DataFrame],
              return_levels: bool = False,
              levels: tuple[int, ...] | None = None):
        """Score a dict of ``{quantile: wide prediction frame}``.

        Each frame needs the group columns (item_id/dept_id/cat_id/store_id/
        state_id) plus the 28 ``d_*`` columns. All nine quantiles are required.

        Parameters
        ----------
        levels:
            Which hierarchy levels to score. ``None`` scores all 12 (the official
            WSPL). Pass e.g. ``(12,)`` when you only have *bottom-level*
            quantiles - summing them to upper levels would be statistically
            invalid, because quantiles are not additive.

        Returns overall WSPL (mean over the scored levels), or
        ``(wspl, per_level_array)``.
        """
        missing = set(QUANTILES) - set(preds_by_quantile)
        if missing:
            raise ValueError(f"missing quantile forecasts for: {sorted(missing)}")

        prepped = {u: self._prep(preds_by_quantile[u]) for u in QUANTILES}
        levels = levels or tuple(range(1, N_LEVELS + 1))

        level_scores = []
        for level in levels:
            # Mean SPL across the nine quantiles, per series.
            spl_sum = None
            for u in QUANTILES:
                grouped = self._aggregate(prepped[u], level)
                spl_u = self._spl(grouped, level, u)
                spl_sum = spl_u if spl_sum is None else spl_sum + spl_u
            spl = spl_sum / len(QUANTILES)

            weight = getattr(self, f"lv{level}_weight")
            level_scores.append(float((weight * spl).sum()))

        wspl = float(np.mean(level_scores))
        if return_levels:
            return wspl, np.array(level_scores)
        return wspl
