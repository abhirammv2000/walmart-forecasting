# 02 - The Metric: WRMSSE

M5 Accuracy is scored by the **Weighted Root Mean Squared Scaled Error**. You
cannot improve what you cannot measure, so this is the most important piece of
the project. It lives in [`src/wrmsse.py`](../src/wrmsse.py) and is validated
against known reference values (see bottom).

## The formula, piece by piece

### 1. RMSSE for one series

For a single series with 28-day horizon forecasts $\hat{y}_t$ and actuals $y_t$:

$$
\text{RMSSE} = \sqrt{ \frac{\frac{1}{h}\sum_{t=1}^{h} (y_t - \hat{y}_t)^2}
                          {\frac{1}{n-1}\sum_{t=2}^{n} (y_t - y_{t-1})^2} }
$$

* **Numerator**: mean squared forecast error over the 28 horizon days.
* **Denominator**: mean squared error of a **naive one-step forecast**
  ($\hat{y}_t = y_{t-1}$) over the *training* history. The series is trimmed to
  start at its first non-zero sale, so the long zero "not yet launched" prefix
  doesn't deflate the scale.
* "Scaled" → dividing by the naive error makes series comparable regardless of
  volume. **RMSSE < 1 means you beat the naive forecast.**

### 2. Weighting

Each series gets a weight $w_i$ equal to its **cumulative dollar sales (units ×
sell price) over the last 28 training days**, normalised so weights sum to 1
*within each aggregation level*. High-revenue series matter more.

### 3. Averaging over 12 levels

The same forecast is aggregated to **12 levels** (total, per-state, per-store,
per-category, per-department, several cross products, per-item, and the bottom
item×store level — see [`GROUP_IDS`](../src/wrmsse.py)). WRMSSE is the simple
average of the weighted RMSSE at each level:

$$
\text{WRMSSE} = \frac{1}{12} \sum_{\ell=1}^{12} \sum_{i \in \ell} w_i \cdot \text{RMSSE}_i
$$

Because the levels are averaged equally, **getting the aggregate levels right
matters as much as the 30,490 bottom series** — a model that nails individual
items but drifts on the state/store totals still scores poorly.

## How we use it

`WRMSSEEvaluator` pre-computes, per level: the scale denominator (from training
history), the weights (from last-28-day dollars), and the aggregated ground
truth. Then `.score(pred_wide)` aggregates a prediction frame to each level and
returns the overall WRMSSE (optionally the 12 per-level scores).

We score the **validation window (d_1914–1941)** locally on every run, training
only on d_1..d_1913 so there is no leakage.

## Sanity checks (reference values)

Running trivial forecasts through the evaluator reproduces well-known M5
benchmark numbers, confirming the implementation:

| Forecast | WRMSSE |
|----------|--------|
| Repeat last training day (d_1913) for 28 days | ~1.46 |
| Mean of last 28 training days, repeated | ~1.08 |

The per-level pattern is also correct: bottom levels (item×store ≈ 0.87) score
better than aggregate levels (total/state ≈ 1.1–1.2), exactly as expected for a
flat naive forecast.

## Intuition for targets

| WRMSSE | Meaning |
|--------|---------|
| ~1.0   | About as good as a naive seasonal forecast |
| ~0.55–0.65 | Solid single-model LightGBM baseline |
| ~0.52  | Top public-LB single models |
| ~0.520 | M5 competition winner (heavy ensembling) |
