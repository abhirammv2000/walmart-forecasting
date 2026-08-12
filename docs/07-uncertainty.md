# 07 - Uncertainty: Forecasting the Distribution

> A point forecast says "we'll sell about 4." An inventory system needs "there's
> a 5% chance we sell more than 11." Those are different questions, and only the
> second one lets you choose a stock level. This is the deliverable that
> Project 2 (inventory) consumes.

## Why this exists

M5 has two tracks. The Accuracy track grades a single number per day
([WRMSSE](02-metric-wrmsse.md)); the **Uncertainty track grades a predictive
distribution**, represented by nine quantiles, using **weighted scaled pinball
loss**.

The uncertainty track is the one that matters operationally. You cannot derive a
service level, a safety stock, or a newsvendor order quantity from a mean, all
of them are statements about the *tail* of the demand distribution.

## The nine quantiles

```
0.005, 0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975, 0.995
```

Deliberately tail-heavy: four of the nine sit in the outer 17% of the
distribution, because that's where stockout risk lives.

## The metric: WSPL

**Pinball loss** at quantile *u*, actual *y*, forecast *q*:

$$
L_u(y, q) = \begin{cases} u \cdot (y - q) & y \ge q \quad \text{(under-forecast)} \\ (1-u) \cdot (q - y) & y < q \quad \text{(over-forecast)} \end{cases}
$$

The asymmetry *is* the point. At `u = 0.995`, under-forecasting is penalised
**199×** more than over-forecasting, exactly the trade-off a high service level
encodes. At `u = 0.5` it's symmetric and reduces to the median.

**Scaled**, then weighted and averaged over the same 12 hierarchy levels and the
same dollar weights as WRMSSE:

$$
\text{SPL}_{i,u} = \frac{\frac{1}{h}\sum_t L_u(y_t, q_t)}{\frac{1}{n-1}\sum_{t=2}^{n} |y_t - y_{t-1}|}
\qquad
\text{WSPL} = \frac{1}{12}\sum_{\ell}\sum_{i \in \ell} w_i \cdot \overline{\text{SPL}_{i,u}}
$$

**Note the scaling differs from RMSSE**: it uses **mean absolute** first
differences, not squared. Getting this wrong silently rescales every number.

The shared 12-level aggregation and dollar weights live in
[`src/hierarchy.py`](../src/hierarchy.py), used by *both* metrics so they cannot
drift apart.

## Method: direct quantile regression

Nine LightGBM models, one per quantile, `objective="quantile", alpha=u`, trained
on **exactly the same feature matrix** as the point model.

**Why not a point forecast plus an assumed distribution?** Intermittent retail
demand is zero-inflated and heteroscedastic, spread depends on price, weekday,
SNAP, and recent volatility. Quantile regression *learns* that conditional
spread from the features instead of assuming a parametric shape that would be
wrong for a slow-moving SKU and a fast-moving one simultaneously.

A sanity check that the method behaves: the `0.005` and `0.025` models converge
almost immediately and predict ~0 nearly everywhere. That is **correct**, with
68% of bottom-level series-days being zeros, the 0.5th percentile of demand really
is zero for most SKU-days.

### Quantile crossing

The nine models are fit independently, so nothing forces
`q_0.005 ≤ … ≤ q_0.995`. A crossed set isn't a valid distribution and would
break the newsvendor lookup. We **sort the nine predictions per (series, day)**
the standard, loss-preserving fix, and report the pre-sort
`crossing_rate` as a diagnostic.

## An honest limitation: quantiles don't add up

The 99th percentile of a store's total is **not** the sum of the 99th
percentiles of its items, those extremes don't all happen on the same day.
Summing bottom-level quantiles would systematically **overstate** the spread of
aggregates.

So we score **level 12 (item × store) only**, which is:
* statistically valid, and
* exactly the level the inventory policy operates at.

Reporting a full 12-level WSPL from summed bottom quantiles would be a bigger
number *and* a wrong one. Doing it properly needs either models fit at each
level, or Monte-Carlo sample paths aggregated up the hierarchy, recorded as
future work rather than faked. The evaluator supports this via
`score(..., levels=(12,))`.

## Results

Final held-out window d1914–1941, XGBoost multi-quantile:

| metric | value |
|---|---|
| Bottom-level weighted SPL | **0.2618** |
| Quantile-crossing rate (pre-sort) | 16.9% (corrected by sorting) |

**This model is trained on limited data** (`train_start=1750`, 4.1M rows), far
less than the point model's 42M, because multi-quantile training is expensive
(every boosting round grows nine trees). It exists to make the inventory work
real; the SPL number is **not** a competitive result and should not be read as
one.

**A useful negative finding.** We retrained at 600 rounds to see if the first
result (150 rounds, SPL 0.2618) was under-fit. It was not: validation loss
plateaued by ~round 200 and 600 rounds only reached **SPL 0.2606**, a rounding-
error improvement, while the pre-sort crossing rate *worsened* from 16.9% to
55.8%. More rounds just overfit each quantile independently. So **rounds were
never the constraint; training-data volume is.** The clear next win is more data
(lower `train_start`), which needs a longer compute budget. High crossing is a
symptom of the same data starvation; sorting keeps the output a valid
distribution regardless.

## Correctness guarantees

`tests/test_wspl.py` (all passing) pins:
* pinball loss is 0 for a perfect forecast, symmetric at `u=0.5`
* at `u=0.995` under-forecasting costs exactly `0.995/0.005 = 199×`
  over-forecasting (and the reverse at `u=0.005`), the asymmetry can't silently
  invert
* a perfect forecast at every quantile scores WSPL = 0
* a sensible monotone spread beats collapsing all nine quantiles onto the mean
  (i.e. the metric actually rewards modelling uncertainty)
* missing quantiles raise rather than silently scoring a partial distribution
