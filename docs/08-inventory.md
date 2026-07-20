# 08 - Inventory: From Forecast to Decision

> A forecast is not a decision. Nobody ships a WRMSSE. The question a retailer
> actually asks is **"how many units do I put on the shelf?"** — and answering it
> is what makes the forecasting work worth doing.

This is the layer that consumes the [quantile forecasts](07-uncertainty.md) and
turns them into stocking decisions, then measures those decisions in money and
service — not in loss-function units.

## The newsvendor model

One stocking period, uncertain demand, two ways to be wrong:

* **`Cu`** — cost of being one unit **short**: the lost margin on a sale you
  couldn't serve (plus goodwill you can't easily price).
* **`Co`** — cost of one unit **left over**: holding, markdown, spoilage.

Expected cost is minimised by stocking the quantity that covers demand with
probability equal to the **critical ratio**:

$$
CR = \frac{C_u}{C_u + C_o}, \qquad Q^* = F^{-1}(CR)
$$

**`Q*` is a quantile of demand.** Two things follow, and together they are the
entire argument of this project:

1. **The optimal order is not the mean forecast.** Ordering the mean is optimal
   *only* in the knife-edge case `Cu == Co`. In retail `Cu > Co` — losing a sale
   usually costs more than holding a unit — so `CR > 0.5` and you deliberately
   stock **above** the point forecast. A team that "orders the forecast" is
   systematically under-stocking and paying for it in lost sales.
2. **You literally cannot compute `Q*` from a point forecast.** It requires
   `F^{-1}` — the inverse CDF. This is *why* the uncertainty track exists, and
   why "just optimise RMSE" is the beginner move.

## Cost assumptions (stated, not smuggled in)

Both costs are expressed as fractions of unit sell price, so they scale with item
value:

| | default | meaning |
|---|---|---|
| `underage_frac` | 0.30 | lost gross margin per unmet unit |
| `overage_frac` | 0.03 | per-day holding + markdown/spoilage risk |

Because both scale with price, **price cancels in the critical ratio** — `CR` is
the same for every item (here ≈0.909); only the resulting *quantity* differs per
item. These are assumptions, so we don't rest on them: the trade-off curve
**sweeps** the service level rather than trusting one point.

## The simulation

Periodic review with an order-up-to level, run against the **actual realised
demand** of the held-out window (d1914–1941):

```
each day:  order    = max(0, order_up_to - on_hand)   # can't un-order stock
           available= on_hand + order
           sold     = min(demand, available)
           lost     = demand - sold                    # stockout
           on_hand  = available - sold                 # carries to tomorrow
```

Inventory **carries over**, which is what makes it an inventory simulation rather
than 28 independent bets. Costs accrue as `Co × end-of-day stock` and
`Cu × lost units`.

Reported metrics are the ones an ops team is actually judged on:

* **fill rate** — share of demanded units actually served
* **stockout rate** — share of item-days with any unmet demand
* **holding / shortage / total cost**

## Results

Run:
```bash
python -m src.run_inventory --quantiles "outputs/predictions/quantiles_*.parquet"
```

Outputs `outputs/inventory_policy_comparison.csv` and
`outputs/inventory_service_curve.csv`.

### Policy comparison — held-out window d1914–1941, all 30,490 series

| policy | fill rate | stockout rate | holding | shortage | **total cost** |
|---|---|---|---|---|---|
| order the median forecast | 47.6% | 38% | 19,816 | 661,701 | **681,517** |
| **newsvendor `Q* = F⁻¹(0.909)`** | **92.5%** | **7%** | 225,511 | 92,219 | **317,730** |

**The newsvendor policy costs 53.4% less** (681,517 → 317,730) *and* lifts fill
rate from 47.6% to 92.5%. It buys ~11× more holding cost and eliminates ~7× more
shortage cost — a trade that is strongly worth making when `Cu` is 10× `Co`.

**Why the mean policy fails so badly** — and this is the sharpest result here.
With ~60% of bottom-level series being zeros, the **median forecast is 0 for most
SKU-days**. "Order the point forecast" therefore literally means "stock nothing"
for a large share of the catalogue, and you lose the sale every time demand shows
up. Intermittent demand is exactly where point forecasts fail as decisions, and
no amount of extra WRMSSE tuning fixes it — you need the distribution.

### The trade-off curve

| service level | fill rate | holding | shortage | total cost |
|---|---|---|---|---|
| 0.50 | 0.48 | 19,816 | 661,701 | 681,517 |
| 0.70 | 0.69 | 64,290 | 388,279 | 452,569 |
| 0.80 | 0.79 | 102,375 | 261,530 | 363,905 |
| 0.85 | 0.85 | 141,325 | 183,255 | 324,580 |
| **0.90** | **0.92** | 212,143 | 102,433 | **314,576** ← minimum |
| 0.95 | 0.95 | 287,131 | 59,499 | 346,630 |
| 0.975 | 0.96 | 325,668 | 47,136 | 372,804 |
| 0.995 | 0.99 | 537,298 | 17,680 | 554,978 |

Textbook U-shape: holding rises and shortage falls with service level, and total
cost bottoms out in between. Chasing 99.5% service nearly **doubles** cost versus
the optimum — "maximise fill rate" is not the goal; *minimising total cost* is.

### The validation that matters

| | |
|---|---|
| Empirical cost-minimising service level | **0.900** |
| Theoretical critical ratio `Cu/(Cu+Co)` | **0.909** |

These agree to within one grid step. The optimum predicted by newsvendor theory
from the *cost structure alone* is where the simulation against **real, held-out
demand** actually bottoms out. That is a genuine end-to-end check: it says the
forecast distribution is calibrated well enough that the theory transfers to real
data, rather than the policy just happening to work.

## Honest limitations

* **Lead time is zero** and review is daily. Real replenishment has a lead time
  `L`, which means you need the distribution of demand over `L + R` days — and
  because quantiles don't add, that is *not* the sum of daily quantiles. Doing it
  properly needs either multi-horizon quantile models or simulated sample paths.
* **No capacity, batching, or MOQ constraints** — no case packs, shelf limits, or
  supplier minimums.
* **Costs are assumed**, not observed. Real `Cu`/`Co` vary by category (fresh
  produce spoils; canned goods don't). The sweep is the mitigation.
* **Independent series.** We stock each item-store separately, ignoring
  substitution (a stockout pushes demand to a substitute) and shared shelf space.

None of these change the core result — that the cost-optimal order is a quantile,
not the mean — but they are what you'd tackle before putting this near a real
replenishment system.

## Correctness guarantees

`tests/test_inventory.py` (12 tests, all passing) pins the economics, not just
the plumbing:

* `Cu == Co` ⇒ critical ratio exactly 0.5 (median); asymmetric costs push it into
  the correct tail
* order-up-to levels interpolate the quantile function correctly, are **monotone
  in service level**, and never extrapolate beyond the modelled quantiles
* simulation mechanics: perfect foresight ⇒ zero cost and 100% fill; ordering
  nothing ⇒ zero fill and no holding cost; over-ordering ⇒ holding but no
  shortage; carryover genuinely reduces ordering
* **the headline claim**: with `Cu >> Co`, the newsvendor quantile policy costs
  strictly less than ordering the mean
