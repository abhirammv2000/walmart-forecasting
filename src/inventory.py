"""Project 2 - turning a demand *distribution* into a stocking decision.

This is the layer that makes the forecast worth anything. A forecast is not a
decision; an order quantity is. The bridge is the **newsvendor model**.

The newsvendor result
---------------------
For a single stocking period with

* ``Cu`` = cost of being one unit **short** (lost margin on a missed sale), and
* ``Co`` = cost of one unit **left over** (holding, markdown, spoilage),

expected cost is minimised by ordering the quantity that satisfies demand with
probability equal to the **critical ratio**::

    CR  = Cu / (Cu + Co)
    Q*  = F^-1(CR)          <- a QUANTILE of the demand distribution

Two consequences drive this whole project:

1. **The optimal order is not the mean forecast.** Whenever Cu != Co the optimum
   sits in a tail. In retail Cu > Co (losing the sale usually costs more than
   holding the unit), so CR > 0.5 and you deliberately stock *above* the point
   forecast. Ordering the mean is only optimal in the knife-edge case Cu == Co.
2. **You cannot compute Q\\* from a point forecast at all** - you need
   ``F^-1``, i.e. the quantile forecasts from :mod:`src.quantile`. This is why
   the uncertainty track exists.

What this module provides
-------------------------
* :func:`order_up_to_levels` - read Q* off the nine quantiles (interpolating).
* :func:`simulate` - run a periodic-review inventory policy against the *actual*
  realised demand, with inventory carrying over day to day, and measure what a
  supply-chain team actually reports: fill rate, stockout rate, holding cost,
  shortage cost, total cost.
* :func:`service_level_curve` - sweep the target service level to trace the
  cost/service trade-off, so we can check the empirical cost minimum lands where
  newsvendor theory says it should.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Cost assumptions, expressed as fractions of unit sell price so they scale with
# item value. These are *assumptions* - stated explicitly, and swept in the
# trade-off analysis rather than treated as ground truth.
DEFAULT_UNDERAGE_FRAC = 0.30   # lost gross margin when we miss a sale
DEFAULT_OVERAGE_FRAC = 0.03    # per-day holding + markdown/spoilage risk


def critical_ratio(underage_frac: float = DEFAULT_UNDERAGE_FRAC,
                   overage_frac: float = DEFAULT_OVERAGE_FRAC) -> float:
    """``Cu / (Cu + Co)`` - the demand probability the optimal order covers.

    Because both costs scale with the unit price, price cancels and the critical
    ratio is the *same* for every item; only the resulting quantity differs.
    """
    return underage_frac / (underage_frac + overage_frac)


def order_up_to_levels(quantile_cube: np.ndarray, quantiles: np.ndarray,
                       service_level: float) -> np.ndarray:
    """Q* at an arbitrary service level, from the nine discrete quantiles.

    Parameters
    ----------
    quantile_cube:
        Array ``(n_series, n_days, n_quantiles)``, ascending in the last axis.
    quantiles:
        The nine probability levels matching that last axis.
    service_level:
        Target probability of covering demand (the critical ratio).

    We only forecast nine points of ``F``, so an arbitrary CR requires
    interpolating the quantile function between them. The target is clipped to
    the forecast range - we cannot honestly extrapolate beyond the 0.5th/99.5th
    percentile we actually modelled.
    """
    s = float(np.clip(service_level, quantiles[0], quantiles[-1]))
    # np.interp is 1-D, so interpolate along the quantile axis via searchsorted.
    hi = int(np.searchsorted(quantiles, s, side="left"))
    if hi == 0:
        return quantile_cube[:, :, 0].copy()
    lo = hi - 1
    span = quantiles[hi] - quantiles[lo]
    w = 0.0 if span == 0 else (s - quantiles[lo]) / span
    return (1.0 - w) * quantile_cube[:, :, lo] + w * quantile_cube[:, :, hi]


def sample_demand_paths(quantile_cube: np.ndarray, quantiles: np.ndarray,
                        n_samples: int = 100, seed: int = 0) -> np.ndarray:
    """Draw demand samples per (series, day) by inverting the quantile function.

    We only have nine points of ``F``, so we treat the quantile function as
    piecewise linear between them and push uniform draws through it. Returns
    ``(n_series, n_days, n_samples)`` as float32 to keep memory sane.
    """
    rng = np.random.default_rng(seed)
    n_series, n_days, n_q = quantile_cube.shape
    out = np.empty((n_series, n_days, n_samples), dtype=np.float32)

    # Draw INDEPENDENTLY per (series, day, sample). Sharing one uniform across
    # days would make every day land at the same percentile - perfectly
    # correlated paths whose sum just reproduces naive quantile summation, which
    # is exactly the error this function exists to avoid.
    # Generated a day at a time to keep peak memory bounded.
    for t in range(n_days):
        u = rng.random((n_series, n_samples))
        u = np.clip(u, quantiles[0], quantiles[-1])   # don't extrapolate
        hi = np.clip(np.searchsorted(quantiles, u, side="left"), 1, n_q - 1)
        lo = hi - 1
        q_lo, q_hi = quantiles[lo], quantiles[hi]
        span = q_hi - q_lo
        w = np.where(span > 0, (u - q_lo) / np.where(span > 0, span, 1.0), 0.0)
        cube_t = quantile_cube[:, t, :]
        v_lo = np.take_along_axis(cube_t, lo, axis=1)
        v_hi = np.take_along_axis(cube_t, hi, axis=1)
        out[:, t, :] = ((1.0 - w) * v_lo + w * v_hi).astype(np.float32)
    return out


def protection_interval_levels(quantile_cube: np.ndarray, quantiles: np.ndarray,
                               service_level: float, lead_time: int = 0,
                               review_period: int = 1, n_samples: int = 100,
                               seed: int = 0) -> np.ndarray:
    """Order-up-to level covering demand over the protection interval.

    With a lead time ``L`` and review period ``R``, an order placed now is the
    last chance to influence stock until the *next* order arrives - so the
    order-up-to level must cover demand over ``W = L + R`` days, not one day.

    **Why this needs sampling.** Quantiles are not additive: the 90th percentile
    of three-day demand is *not* the sum of three daily 90th percentiles (that
    would assume all three extremes land together, badly overstating the
    spread). So we draw sample paths from the daily distributions, sum them
    across the protection interval, and read the quantile off the *summed*
    distribution - which is coherent.

    Assumes demand is independent across days. That ignores autocorrelation and
    so understates the spread somewhat; modelling multi-day demand directly
    would remove the assumption.
    """
    n_series, n_days, _ = quantile_cube.shape
    window = max(1, lead_time + review_period)
    paths = sample_demand_paths(quantile_cube, quantiles, n_samples, seed)

    levels = np.empty((n_series, n_days), dtype=np.float64)
    for t in range(n_days):
        end = min(t + window, n_days)              # clip at the horizon edge
        total = paths[:, t:end, :].sum(axis=1)     # (n_series, n_samples)
        levels[:, t] = np.quantile(total, service_level, axis=1)
    return levels


def simulate(order_up_to: np.ndarray, demand: np.ndarray, price: np.ndarray,
             underage_frac: float = DEFAULT_UNDERAGE_FRAC,
             overage_frac: float = DEFAULT_OVERAGE_FRAC,
             carryover: bool = True, lead_time: int = 0) -> dict[str, float]:
    """Run the stocking policy against realised demand and cost the outcome.

    Periodic review with an order-up-to level: each day we top inventory up to
    ``order_up_to`` (never negative ordering - we cannot return stock), sell what
    we can, lose the rest, and carry leftovers into tomorrow.

    Parameters
    ----------
    order_up_to, demand:
        ``(n_series, n_days)`` arrays. ``demand`` is the *actual* realised sales.
    price:
        ``(n_series, n_days)`` unit sell price, used to value the costs.
    carryover:
        If True, unsold stock persists to the next day (realistic). If False,
        each day is an independent single-period newsvendor.

    Returns a dict of business metrics - the numbers an ops team is judged on.
    """
    n_series, n_days = demand.shape
    cu = underage_frac * price
    co = overage_frac * price

    on_hand = np.zeros(n_series, dtype=np.float64)
    # Orders placed at t arrive at t + lead_time; until then they sit "on order"
    # and must still be counted, or we would re-order the same units every day.
    pipeline = np.zeros((n_series, n_days + lead_time + 1), dtype=np.float64)

    tot_sold = tot_demand = tot_lost = 0.0
    tot_holding = tot_shortage = 0.0
    tot_units_ordered = 0.0
    stockout_days = 0

    for t in range(n_days):
        on_hand += pipeline[:, t]                    # receive today's arrivals
        pipeline[:, t] = 0.0
        on_order = pipeline[:, t:].sum(axis=1)       # still in transit

        target = order_up_to[:, t]
        if carryover:
            # Order against the inventory *position* (on hand + on order).
            order = np.maximum(0.0, target - on_hand - on_order)
        else:
            order = target

        if lead_time == 0:
            on_hand += order                          # arrives immediately
        else:
            pipeline[:, t + lead_time] += order

        d = demand[:, t]
        sold = np.minimum(d, on_hand)                 # can only sell what's here
        lost = d - sold
        on_hand -= sold

        tot_units_ordered += float(order.sum())
        tot_sold += float(sold.sum())
        tot_demand += float(d.sum())
        tot_lost += float(lost.sum())
        tot_shortage += float((cu[:, t] * lost).sum())
        tot_holding += float((co[:, t] * on_hand).sum())   # end-of-day stock
        stockout_days += int((lost > 0).sum())

        if not carryover:
            on_hand = np.zeros_like(on_hand)

    return {
        "fill_rate": tot_sold / tot_demand if tot_demand else np.nan,
        "stockout_rate": stockout_days / (n_series * n_days),
        "holding_cost": tot_holding,
        "shortage_cost": tot_shortage,
        "total_cost": tot_holding + tot_shortage,
        "units_ordered": tot_units_ordered,
        "units_lost": tot_lost,
        "units_demanded": tot_demand,
    }


def service_level_curve(quantile_cube: np.ndarray, quantiles: np.ndarray,
                        demand: np.ndarray, price: np.ndarray,
                        service_levels: np.ndarray | None = None,
                        underage_frac: float = DEFAULT_UNDERAGE_FRAC,
                        overage_frac: float = DEFAULT_OVERAGE_FRAC,
                        carryover: bool = True) -> pd.DataFrame:
    """Sweep the target service level and cost each resulting policy.

    Produces the classic service-vs-cost trade-off curve. The interesting check:
    the *empirical* cost minimum should sit near the theoretical critical ratio.
    """
    if service_levels is None:
        service_levels = np.array([0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90,
                                   0.925, 0.95, 0.965, 0.975, 0.985, 0.995])
    rows = []
    for s in service_levels:
        q = order_up_to_levels(quantile_cube, quantiles, s)
        m = simulate(q, demand, price, underage_frac, overage_frac, carryover)
        m["service_level"] = float(s)
        rows.append(m)
    return pd.DataFrame(rows).set_index("service_level")


def median_forecast_policy(quantile_cube: np.ndarray, quantiles: np.ndarray
                           ) -> np.ndarray:
    """Reference benchmark: order the **median** (q=0.5) forecast.

    Read straight off the same quantile model, so the comparison is purely about
    the *decision rule* (which quantile you stock to), not two different models.
    Because the median of a right-skewed, zero-heavy demand distribution sits
    low, this deliberately under-stocks - by construction it covers realised
    demand only ~50% of the time, so its fill rate is ~50%. For a fairer,
    stronger baseline, ``src.run_inventory`` also compares against the point
    (conditional-mean) forecast from the Tweedie model.
    """
    mid = int(np.searchsorted(quantiles, 0.5))
    return quantile_cube[:, :, mid].copy()
