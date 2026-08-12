# 04 - Validation Strategy

> "You can't improve what you don't measure", and in forecasting you can't even
> *trust* a single measurement. This is how we measure so that improvements are
> real and not noise.

## The trap we're avoiding

In M5, the public leaderboard was a **single 28-day window**. Thousands of
competitors tuned against it and were shocked when the private leaderboard (a
*different* 28-day window) reshuffled everything. The lesson: **one window is not
a reliable signal.** A change that helps one month can hurt the next.

So we evaluate every change on **multiple rolling windows** and steer on the
average, while keeping one window completely untouched as a final reality check.

## Our folds

All windows are scored with [WRMSSE](02-metric-wrmsse.md) against the actuals in
`sales_train_evaluation.csv` (which we have through d_1941).

| Fold | Train days | Forecast window | Role |
|------|-----------|-----------------|------|
| `cv1` | ≤ d_1857 | d_1858 – d_1885 | Dev (steer on it) |
| `cv2` | ≤ d_1885 | d_1886 – d_1913 | Dev (steer on it) |
| **`final`** | ≤ d_1913 | d_1914 – d_1941 | **Held-out test** |

* **CV mean** = mean(cv1, cv2). This is the headline number we optimise.
* **Final test** (d_1914–1941) is touched **rarely**, only to confirm a
  milestone, never to pick between tweaks. Optimising against it would just
  recreate the public-LB overfitting trap on a smaller scale.

### Why these windows?

They are the most recent labelled 28-day windows, adjacent to the test window, so
they sit in a similar part of the calendar/season. (A natural future improvement
is to add windows exactly one year earlier for seasonal alignment.)

### What about d_1942–1969?

That's the competition's private window. **Its labels are not in this repo**, so
we cannot score it locally, hence d_1914–1941 is our final test instead. If the
released M5 evaluation answers are added to `data/`, we can promote d_1942–1969 to
the true final test.

## The build-once-slice-many shortcut (and its caveat)

All current features are **non-recursive** (every lag ≥ 28 days), so a given
(series, day) feature value depends only on actuals ≥ 28 days earlier, never on
where a fold's training cut-off falls. That lets us **engineer features once over
the whole timeline** ([`src/dataset.py`](../src/dataset.py)) and simply *slice*
the rows for each fold. It's a big speed-up and is exactly leak-equivalent to
rebuilding per fold.

**Caveat:** the moment we introduce a lag < 28 (a recursive feature), this
shortcut breaks, those features would depend on the fold cut-off and must be
rebuilt per fold. The code notes this; revisit it when we get to recursive
experiments.

## The experiment log

Every run appends a row to [`docs/experiments.md`](experiments.md) (human-readable)
and `outputs/experiments.csv` (machine-readable): date, name, per-fold WRMSSE, CV
mean, final test (if run), tree count, and a one-line description of what changed.
This is the project's memory, the record of what we tried and what worked.

## How to run

```bash
# Score a config on the dev folds (steering signal):
python -m src.cv --name my_experiment --desc "what changed"

# Also score the held-out final window (use sparingly, at milestones):
python -m src.cv --name my_experiment --desc "..." --final-test

# Force a feature-cache rebuild after changing the feature recipe:
python -m src.cv --name my_experiment --desc "..." --rebuild-features
```
