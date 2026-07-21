# 01 - The M5 Dataset

The M5 Forecasting – Accuracy competition asks us to forecast **28 days** of
daily unit sales for Walmart products, then scores the forecast with WRMSSE
(see [02-metric-wrmsse.md](02-metric-wrmsse.md)).

## The series

There are **30,490 bottom-level series**, each a unique (item, store) pair:

| Dimension | Values | Notes |
|-----------|--------|-------|
| Items     | 3,049  | e.g. `HOBBIES_1_001` |
| Stores    | 10     | `CA_1..4`, `TX_1..3`, `WI_1..3` |
| States    | 3      | CA, TX, WI |
| Categories| 3      | HOBBIES, HOUSEHOLD, FOODS |
| Departments | 7    | e.g. `FOODS_3` |

3,049 items × 10 stores = 30,490 series. These roll up into **12 aggregation
levels** (total → state → store → category → department → item → item×store),
42,840 series in all, which is what WRMSSE averages over.

## The files (`data/`)

| File | Shape | What it is |
|------|-------|------------|
| `sales_train_validation.csv` | 30,490 × (6 + 1,913) | Wide. ID columns + one column `d_1..d_1913` per day. Sales up to the public-LB cutoff. |
| `sales_train_evaluation.csv` | 30,490 × (6 + 1,941) | Same, extended to `d_1941`. **Contains the d_1914..d_1941 ground truth** we use for local scoring. |
| `calendar.csv` | 1,969 × 14 | One row per day `d_1..d_1969`: date, `wm_yr_wk` (Walmart week), weekday, events, and SNAP food-stamp flags per state. |
| `sell_prices.csv` | ~6.8M × 4 | Weekly selling price per (store, item, `wm_yr_wk`). Missing before a product is first sold. |
| `sample_submission.csv` | 60,980 × 29 | `id`, `F1..F28`. Two rows per series: `*_validation` (d_1914..1941) and `*_evaluation` (d_1942..1969). |

## The timeline (important)

```
 d_1 ........................ d_1913 | d_1914 ... d_1941 | d_1942 ... d_1969
 |------------ train -----------------|--- validation ---|--- evaluation ---|
                                       (public LB)         (private LB)
```

* **Validation window** d_1914–1941: labels are public (inside
  `sales_train_evaluation.csv`), so **we score these locally** to measure
  progress. This is our development signal.
* **Evaluation window** d_1942–1969: labels were never released; we can only
  produce a submission for them.

## Wide vs long

The sales files are **wide** (one column per day). The model works on a **long**
table (one row per series-day) so we can attach lag/rolling/calendar features.
[`src/data.py`](../src/data.py) does the melt, one store at a time to bound
memory. The full melt is ~30,490 × 1,941 ≈ **59M rows**, which is why memory
discipline (downcasting, per-store processing) matters.

## A note on the raw data quirks

* **Leading zeros are not real demand.** A product shows `0` sales before it was
  introduced to a store. WRMSSE's scale denominator trims these leading zeros;
  our features simply produce `NaN` lags there, which LightGBM handles.
* **Prices are weekly**, keyed by `wm_yr_wk`, not daily. We merge them onto each
  day via the calendar's `wm_yr_wk` mapping.
* **SNAP days** (when food stamps can be spent) materially lift FOODS demand and
  differ by state — hence `snap_CA/TX/WI` are kept as features.
