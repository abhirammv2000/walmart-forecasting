# 03 - Baseline Model

> **Result (multi-fold CV): CV mean WRMSSE = 0.6247** (cv1=0.7028, cv2=0.5465);
> **held-out final test (d1914–1941) = 0.6569**. Single global LightGBM,
> train_start_day=1300. See [docs/04-validation.md](04-validation.md) for the
> fold design and [docs/experiments.md](experiments.md) for the log.
>
> The two dev folds differ by ~0.16 WRMSSE, concrete proof that a single window
> is a noisy signal, and why we steer on the **CV mean**. (The 0.6569 final-test
> number reproduces the earlier single-window run of 0.6567, cross-validating the
> harness.)

This is the foundation the rest of the project builds on. It is deliberately a
*simple, honest, fully-measured* model rather than a leaderboard-chasing
ensemble, the kind of first model you would actually ship in industry: easy to
reason about, cheap to retrain, and instrumented so every later change can be
judged against it.

## The model in one paragraph

A **single global LightGBM** regressor predicts daily unit sales for all 30,490
series at once. It uses a **Tweedie** objective (well suited to intermittent,
non-negative, zero-inflated demand), **non-recursive** lag/rolling features, and
calendar + price signals. One model forecasts the entire 28-day horizon.

## Key design decisions (and why)

### 1 - One global model, not per-series or per-horizon
30,490 individual models would be unmaintainable and would ignore the huge
amount of shared structure (a holiday lifts demand across many items). A single
global model learns cross-series patterns and is one artifact to deploy. We also
*avoid* the original code's **28 separate horizon models**, unnecessary given
the non-recursive trick below.

### 2 - Non-recursive features (every lag ≥ 28)
To forecast day *d* we only use information available at d_1913. Because every
lag is ≥ 28 days, the features for any horizon day (d_1914..d_1941) reach back
only to days that are already known (d_1886..d_1913). So:

* **No recursion**, we don't feed predictions back as inputs, so errors don't
  compound across the horizon.
* **No leakage**, a feature can never see the future.
* **One model, one shot**, the same feature recipe works for all 28 days.

This is the single most important structural choice in the baseline.

### 3 - Tweedie objective
Daily item sales are mostly 0 or small integers with occasional spikes , 
classic intermittent demand. Tweedie regression (`tweedie_variance_power=1.1`)
models this far better than plain RMSE/L2, which would over-smooth toward the
mean. Predictions are clipped at 0 (negative demand is meaningless).

### 4 - Memory-bounded, per-store feature engineering
The melted table is ~59M rows; the dev box has ~5 GB free. So features are built
**one store at a time** (≈3,049 series each) and only the rows inside the
training window are kept. `train_start_day` trades RAM/time for data volume , 
1300 keeps ~18.7M training rows and fits comfortably. Lowering it (more history)
is the first lever to pull for a better score on a bigger machine.

### 5 - Single source of truth for features
[`src/features.py`](../src/features.py) builds features for *both* training and
prediction. The original project had two separate, drifting feature scripts
(and a real bug from it); one function eliminates that whole class of error.

## Features

| Group | Features |
|-------|----------|
| **Identifiers** (categorical) | item_id, dept_id, cat_id, store_id, state_id |
| **Calendar** | year, month, week, day, dayofweek |
| **Events** (categorical) | event_name_1/2, event_type_1/2 |
| **SNAP** | snap_CA, snap_TX, snap_WI |
| **Price** | sell_price, price_momentum (price ÷ item's mean price) |
| **Lags** | sales_lag_28 … sales_lag_35 |
| **Rolling** (on lag-28) | mean & std over 7, 14, 28-day windows |

33 features total. Categoricals are integer-encoded **globally** (a code means
the same thing in every store), which is required for a global model.

## How it's trained and validated

```
train on d_1..d_1885  ──►  early-stop on d_1886..d_1913  ──►  forecast d_1914..d_1941  ──►  WRMSSE
```

We hold out the last 28 days of the training window for LightGBM early stopping,
then score the *real* validation window against ground truth from
`sales_train_evaluation.csv`. No part of the horizon is ever seen during
training.

## Results

Overall **WRMSSE = 0.6567**. Per level:

| Level | What it aggregates | WRMSSE |
|------:|--------------------|-------:|
| 1  | Total (all sales)            | 0.536 |
| 2  | State                        | 0.553 |
| 3  | Store                        | 0.611 |
| 4  | Category                     | 0.542 |
| 5  | Department                   | 0.563 |
| 6  | State × Category             | 0.579 |
| 7  | State × Department           | 0.601 |
| 8  | Store × Category             | 0.646 |
| 9  | Store × Department           | 0.684 |
| 10 | Item                         | 0.863 |
| 11 | Item × State                 | 0.855 |
| 12 | Item × Store (bottom)        | 0.846 |

**Reading the table:** aggregate levels (1–9) score well (~0.54–0.68); the
hardest levels are the sparse bottom ones (10–12, ~0.85), exactly as expected , 
individual item-store demand is noisy and hard to pin down day-to-day. For
context, a naive seasonal forecast scores ~1.08 overall, so the baseline is a
clear, well-rounded improvement.

## Honest limitations / next levers

In rough order of expected payoff:

1. **More training history**, lower `train_start_day` (needs more RAM); top
   solutions use years of data.
2. **Recency-weighted / longer-window features**, 60- and 180-day rolling
   stats, days-since-last-sale, rolling skew.
3. **Better price features**, price change vs last week, price rank within
   department, promo detection.
4. **Per-store or per-category models**, let the global model specialise.
5. **Recursive features + multi-step**, riskier but unlocks short lags (1–27).
6. **Tuning & ensembling**, hyperparameter search, seed/feature bagging,
   blending with a statistical model. (Where M5 winners earned their last
   ~0.13.)

## Reproducing

```bash
# Validation score (what the numbers above come from)
python -m src.baseline --mode validation --train-start-day 1300

# Kaggle submission for the private window d_1942..d_1969
python -m src.baseline --mode evaluation --train-start-day 1300
```

Artifacts land in `outputs/` (`models/`, `predictions/`, `submissions/`).
