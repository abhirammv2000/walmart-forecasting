# 05 - Feature Engineering (Milestone B)

This documents every feature in the model, why it exists, and the correctness
rules that keep it leak-free. Features are grouped so each group can be toggled
and measured independently on CV (`--features roll_ext,price_ext,...`).

## The one rule that governs everything: what is known at forecast time?

To forecast day *d* we may only use information a real forecaster would have on
d_1913 (the last training day). Two kinds of inputs qualify:

1. **Sales-derived features must be ≥ 28 days lagged.** We never know future
   sales, and our single non-recursive model predicts all 28 horizon days at
   once, so any feature built from `sales` is computed on `sales.shift(28)` (or a
   longer lag). A lag of 28 for day d_1941 reaches back only to d_1913 - known.
2. **Prices and the calendar are known in advance.** M5 *gives* you
   `sell_prices.csv` and `calendar.csv` for the forecast weeks. So price and
   calendar features may use the current (even "future") value - that is not
   leakage, it is legitimately available. This is a defining property of the M5
   task and is why price features are so useful.

This rule is also what makes our **build-once-slice-many** cache valid (see
[04-validation.md](04-validation.md)): every feature depends only on
known-at-forecast-time information, so a row's feature values are identical no
matter which fold's cut-off we slice at.

## Base features (always on)

| Feature | Type | Meaning |
|---|---|---|
| item_id, dept_id, cat_id, store_id, state_id | categorical | series identity (globally encoded) |
| event_name_1/2, event_type_1/2 | categorical | calendar events (holidays, sporting, etc.) |
| snap_CA/TX/WI | binary | SNAP food-stamp day per state (lifts FOODS demand) |
| year, month, week, day, dayofweek | date | seasonality |
| sell_price | price | current selling price |
| price_momentum | price | price ÷ item's mean price |
| sales_lag_28 ... sales_lag_35 | sales (lagged) | recent demand, ≥28-day lags |
| rmean/rstd_l28_w{7,14,28} | sales (lagged) | short-term trend & volatility on lag-28 |

Plus **release filtering** (Milestone A): rows before an item's first sale are
dropped - structural zeros, not demand.

## Milestone B feature groups (toggled + measured)

### `roll_ext` - longer & richer rolling stats *(sales-derived, ≥28-lagged)*
Built on `sales_lag_28`, so leak-free like the base rolling stats.

| Feature | Rationale |
|---|---|
| rmean_l28_w60, rmean_l28_w180 | longer trend (season-scale) beyond the 28-day view |
| rstd_l28_w60 | longer-horizon volatility |
| rmax_l28_w28, rmin_l28_w28 | recent demand range (peaks/troughs) |

Grounding: multi-window rolling means (7/30/60/180) are standard in the strong
M5 public kernels (Yakovlev) and top solutions - they let the tree read demand
at several time-scales at once.

### `price_ext` - price / promotion signals *(price-derived, known in advance)*

| Feature | Rationale |
|---|---|
| price_max, price_min | the item's price band |
| price_norm = sell_price ÷ price_max | where today's price sits in that band (0-1); low = on promotion |
| price_nunique | # of distinct prices ever seen ~ how promo-active the item is |
| price_change_w = price ÷ price 7 days ago | week-over-week price move (price is constant within a Walmart week) |

Grounding: price features (normalisation, distinct-price counts, momentum) were
one of the clearest single-model contributors in M5 - demand is highly
price/promotion sensitive, and future prices are known, so the model can
anticipate promo-driven spikes.

### `cal_ext` - calendar refinements *(known in advance)*

| Feature | Rationale |
|---|---|
| is_weekend (dayofweek ≥ 5) | strong weekly demand pattern; weekends differ sharply |
| week_of_month | intra-month structure (e.g., paycheck/SNAP timing) |

## How each group is evaluated

Each group is screened on the CV dev folds at `train_start=700` (faster than the
full `train_start=300`), added on top of the current baseline. Groups that
improve the **CV mean** beyond noise are kept; the final winning combination is
then confirmed with one run at `train_start=300`. Every run is recorded in
[experiments.md](experiments.md).

## Screening results (measured, not assumed)

Reference = base recipe at train_start=700 = **CV mean 0.6466**.

| group added | cv1 | cv2 | cv3 | CV mean | vs base |
|---|---|---|---|---|---|
| *(base)* | 0.728 | 0.665 | 0.546 | **0.6466** |, |
| roll_ext | 0.719 | 0.694 | 0.538 | 0.6502 | +0.0036 (worse) |
| price_ext | 0.728 | 0.673 | 0.546 | 0.6493 | +0.0027 (worse) |
| cal_ext | 0.720 | 0.672 | 0.549 | 0.6472 | +0.0006 (flat) |
| all three | 0.708 | 0.688 | 0.552 | 0.6489 | +0.0023 (worse) |

**Conclusion: none of these groups beat the base recipe.** cal_ext is neutral;
roll_ext and price_ext slightly hurt. Note the *per-fold* pattern: several groups
improved cv1 but hurt cv2, roughly cancelling - a textbook reminder of why we
average over folds rather than trusting one window.

**Why (honest read):** the base already carries the core signal (lags 28-35,
rolling 7/14/28, `sell_price`, `price_momentum`, calendar, SNAP, events). Bolting
on more features **at fixed hyperparameters** mostly lets the model fit more
trees without generalising better (e.g. price_ext pushed best-iter to ~800 with
no CV gain - a sign of overfitting). In the winning M5 solutions these features
helped, but alongside heavier tuning and/or per-store models.

**Decision:** keep the base recipe as the model; leave these groups implemented
but **off by default** (revisit after hyperparameter tuning, which may change the
picture). The next lever is therefore **tuning / model structure**, not more
features.

## Deliberately deferred (with reasons)

* **Recursive lags (lag < 28).** Would let the model use very recent demand, but
  breaks the non-recursive design and the build-once cache, and risks error
  compounding across the horizon. A separate, carefully-built experiment later.
* **Target mean-encodings.** Powerful but leakage-prone; require per-fold,
  past-only computation (which also breaks the cache shortcut). Deferred until we
  can do them safely.
* **days-since-last-sale.** Useful intermittency signal but fiddly to make
  strictly ≥28-lagged; revisit after the cheaper wins above.
