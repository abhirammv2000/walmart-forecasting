# 06 - Results (Project 1: point forecast)

## Headline

| | WRMSSE |
|---|---|
| **Held-out final test (d1914–1941)**, never used for any decision | **0.6475** |
| 3-fold rolling-origin CV mean (steering metric) | 0.6401 |
| Seasonal-naive reference (28-day mean) | ~1.08 |
| Last-day-naive reference | ~1.46 |

**Config:** single global LightGBM, Tweedie objective, non-recursive features
(all lags ≥ 28), release filtering, `train_start_day=300` (42.4M training rows).

Reproduce:
```bash
python -m src.cv --name best --desc "best config" \
  --train-start-day 300 --day-floor 300 --final-test
```

## Why the final-test number is the one that counts

d1914–1941 was frozen from the start: no feature, hyperparameter, or history
decision was ever made by looking at it. It was scored exactly once, at the end.

The most reassuring result is that **CV mean (0.6401) and the held-out test
(0.6475) agree to within 0.007.** Our rolling-origin CV was an honest predictor
of unseen performance, i.e. we did not overfit the validation scheme, which is
the classic M5 failure mode.

## How we got here

| Step | CV mean | Note |
|---|---|---|
| Baseline (train_start=1300) | 0.6740 | |
| + more history (700) | 0.6466 | biggest single win |
| + near-full history (300) | 0.6371->0.6401 | current best (see noise below) |
| Feature groups (rolling/price/calendar) | 0.647–0.650 | **no measurable gain, shelved** |

## The measurement-noise finding (important)

Re-running the *identical* config reproduced **different** numbers:

| fold | run 1 | run 2 | Δ |
|---|---|---|---|
| cv1 | 0.6975 | 0.7081 | +0.011 |
| cv2 | 0.6655 | 0.6730 | +0.008 |
| cv3 | 0.5483 | 0.5391 | -0.009 |
| **mean** | **0.6371** | **0.6401** | **+0.003** |

Cause: LightGBM's multithreaded histogram building is **non-deterministic** by
default, and early stopping then lands on a different iteration.

Two consequences we now act on:

1. **Per-fold noise is ~±0.01; the 3-fold mean is ~±0.003.** Averaging folds
   buys real stability, more justification for multi-fold CV.
2. **Several Milestone-B feature deltas (0.001–0.004) were inside the noise
   floor.** We can honestly say those features showed *no measurable gain*, but
   *not* that they were harmful. Claiming otherwise would be reading noise.

**Fix applied:** `deterministic: True` in the LightGBM params (with
`force_row_wise`) so future A/B comparisons are reproducible. For decisions near
the noise floor, prefer multi-seed averaging over a single run.

## Honest limitations

* We are at **0.6475**; strong public M5 single models reach ~0.52. The gap is
  mostly hyperparameter tuning, per-store/per-category models, recursive-lag
  variants, and ensembling, none of which we did.
* Feature engineering beyond the base set produced no measurable gain, which
  suggests the base lags/rolling already capture most of the signal available to
  a single global model at this capacity.
* **This is the point forecast only.** M5's uncertainty track (9 quantiles,
  weighted scaled pinball loss) is a separate deliverable, and it is the
  prerequisite for the inventory work. See `docs/07-uncertainty.md`.

## Competition submission

`outputs/submissions/submission_best_evaluation.csv` is the forecast for the
private window **d1942–1969**, produced by retraining the best point config
(`train_start=300`) on all data through d1941:

```bash
python -m src.baseline --mode evaluation --train-start-day 300 --tag best
```

30,490 rows in Kaggle `F1..F28` format, all `_evaluation` ids, non-negative. This
is the half of the competition that was never scoreable locally (labels never
released). A full Kaggle upload would also include the 30,490 `_validation` rows
(d1914–1941), whose labels are now public and which we score directly instead.

## Correctness guarantees

Not just asserted, tested (`tests/`, all passing):
* perfect forecast ⇒ WRMSSE exactly 0; naive baselines reproduce known magnitudes
* the score is always finite (regression test for a zero-scale divide-by-zero)
* **leakage test:** rebuilding features with all future actuals erased leaves the
  forecast-window features byte-identical ⇒ no feature sees the future, and the
  build-once-slice-many cache is valid
