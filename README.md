# M5 Walmart Forecasting

A clean, local-first reimplementation of the
[M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy)
Kaggle competition: forecast **28 days** of daily unit sales for **30,490**
Walmart item-store series, scored by **WRMSSE** across 12 aggregation levels.

This repo was rebuilt from scratch with three priorities: **everything runs
reproducibly**, **everything is measured** against real held-out ground truth,
and **everything is documented**.

The repo covers **two linked projects**:

1. **Demand forecasting** — point forecasts (WRMSSE) *and* the uncertainty track
   (nine quantiles, weighted scaled pinball loss).
2. **Inventory / replenishment** — turning those quantiles into stocking
   decisions and costing them. *This is the part that makes the forecast useful.*

## Results at a glance

**Project 1 — forecasting** (held-out window d1914–1941)

| | value |
|---|---|
| **Held-out final test WRMSSE** — frozen, scored once | **0.6475** |
| 3-fold rolling-origin CV mean (steering metric) | 0.6401 |
| Seasonal-naive reference | ~1.08 |
| Bottom-level weighted SPL (9 quantiles) | 0.2618 |

CV and held-out agree within 0.007 — the validation scheme was not overfit.
See [docs/06-results.md](docs/06-results.md).

**Project 2 — inventory decision** (same held-out window, 30,490 series)

| policy | fill rate | **total cost** |
|---|---|---|
| order the median forecast | 47.6% | 681,517 |
| **newsvendor `Q* = F⁻¹(CR)`** | **92.5%** | **317,730 (−53%)** |

And the check that ties it together: the **empirically cost-minimising service
level (0.900)** matches the **theoretical critical ratio (0.909)**. Optimising the
forecast and optimising the *decision* give different answers — which is the whole
reason supply chains forecast distributions. See
[docs/08-inventory.md](docs/08-inventory.md).

## Quickstart

```bash
pip install -r requirements.txt          # pandas, numpy, lightgbm, pyarrow
# Put the competition CSVs in data/ (see docs/01-data.md)

# Train + score the baseline on the validation window (d_1914..d_1941):
python -m src.baseline --mode validation --train-start-day 1300

# Produce a Kaggle submission for the evaluation window (d_1942..d_1969):
python -m src.baseline --mode evaluation --train-start-day 1300
```

`validation` mode prints a **WRMSSE** score (it has local ground truth);
`evaluation` mode writes a submission to `outputs/submissions/`.

```bash
# Uncertainty track: fit the nine quantiles and score bottom-level SPL
python -m src.run_uncertainty --name q --fold final --train-start-day 1750     --day-floor 300 --n-estimators 150 --backend xgboost --device cpu --save-preds

# Project 2: turn those quantiles into stocking decisions and cost them
python -m src.run_inventory --quantiles "outputs/predictions/quantiles_*.parquet"
```

## Layout

```
data/                Raw competition CSVs (see docs/01-data.md)
src/
  config.py          Paths, competition constants, day boundaries
  data.py            Loaders + per-store melt + memory downcasting
  features.py        Feature engineering (single source of truth)
  dataset.py         Build-once feature cache (engineer features, slice per fold)
  hierarchy.py       Shared 12-level aggregation + dollar weights
  wrmsse.py          WRMSSE metric (Accuracy track)
  wspl.py            WSPL metric (Uncertainty track, 9 quantiles)
  quantile.py        Quantile forecasting (LightGBM CPU / XGBoost multi-quantile)
  inventory.py       Newsvendor policy + inventory simulation (Project 2)
  cv.py              Cross-validation harness + experiment log
  baseline.py        Global LightGBM: train -> predict -> score -> submit
outputs/
  models/            Saved LightGBM boosters (.txt)
  predictions/       Wide prediction frames
  submissions/       Kaggle-format F1..F28 files
  cache/             Cached engineered features (Parquet)
  experiments.csv    Machine-readable experiment ledger
docs/                Written explanations (data, metric, validation, baseline)
notebooks/reference/ Public reference notebooks (read-only, for ideas)
legacy/              Previous attempt (SageMaker/Docker) - kept, not used
```

## Tests

Correctness is pinned by an automated suite — run it before trusting any number:

```bash
pip install pytest
PYTHONPATH=. python -m pytest tests/ -q
```

| Test | What it guarantees |
|------|--------------------|
| `tests/test_wrmsse.py` | A perfect forecast scores exactly 0; naive baselines reproduce known M5 magnitudes (last-day ≈1.46, 28-day-mean ≈1.08); the score is always finite (regression test for a zero-scale divide-by-zero). |
| `tests/test_leakage.py` | **The critical one.** Rebuilds features with all future actuals erased and asserts the forecast-window features are byte-identical — proving no feature peeks at the future, and that the build-once-slice-many cache is valid. Also guards that every sales lag ≥ 28 days. |
| `tests/test_wspl.py` | Pinball loss is 0 for a perfect forecast, symmetric at u=0.5, and at u=0.995 punishes under-forecasting exactly 199× — the asymmetry can't silently invert. Modelling a spread must beat collapsing to the mean. |
| `tests/test_inventory.py` | The **economics**: Cu==Co ⇒ critical ratio 0.5; order-up-to levels are monotone in service level and never extrapolate; simulation mechanics (perfect foresight ⇒ zero cost, carryover reduces ordering); and the headline claim that the newsvendor quantile policy beats ordering the mean. |

The leakage test is the one to re-run if short/recursive lags (< 28) are ever
introduced — it is precisely the test that will fail.

## Iterating

We improve the model one measured experiment at a time. Evaluate any config on
the cross-validation folds and append the result to the experiment log:

```bash
python -m src.cv --name my_experiment --desc "what changed"
```

See [docs/04-validation.md](docs/04-validation.md) for the fold design (we steer
on the **CV mean** and keep d_1914–1941 as a held-out test) and
[docs/experiments.md](docs/experiments.md) for the running results.

## The approach (baseline)

A single **global LightGBM** with a **Tweedie** objective (right for
intermittent, non-negative demand) and **non-recursive** lag/rolling features
(every lag ≥ 28 days, so one model forecasts the whole 28-day horizon with no
recursion and no leakage). Feature engineering runs **one store at a time** to
keep peak memory bounded. See [docs/03-baseline.md](docs/03-baseline.md) for the
full writeup and the recorded score.

## How we measure

We train only on d_1..d_1913 and score the validation window d_1914..d_1941
locally with WRMSSE (ground truth lives in `sales_train_evaluation.csv`). The
metric and its reference-value sanity checks are in
[docs/02-metric-wrmsse.md](docs/02-metric-wrmsse.md).

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/01-data.md](docs/01-data.md) | The dataset, files, timeline, and quirks |
| [docs/02-metric-wrmsse.md](docs/02-metric-wrmsse.md) | The WRMSSE metric, explained and validated |
| [docs/03-baseline.md](docs/03-baseline.md) | Baseline model, design choices, results |
| [docs/04-validation.md](docs/04-validation.md) | Cross-validation strategy and the experiment log |
| [docs/05-features.md](docs/05-features.md) | Feature groups, rationale, and what did/didn't work |
| [docs/07-uncertainty.md](docs/07-uncertainty.md) | Quantile forecasting, pinball loss, WSPL |
| [docs/08-inventory.md](docs/08-inventory.md) | **Project 2**: newsvendor policy, simulation, cost/service trade-off |
| [docs/06-results.md](docs/06-results.md) | **Headline results**, measurement-noise finding, limitations |
| [docs/experiments.md](docs/experiments.md) | Running results of every experiment |
