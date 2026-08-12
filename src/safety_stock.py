"""Classical safety-stock / reorder-point inventory formulas.

The textbook continuous- and periodic-review models. They matter here for two
reasons: they're the language planners actually use, and they make the link
between the **newsvendor** decision and the everyday "safety stock" number
explicit.

The formulas (normal-demand approximation)
------------------------------------------
With daily demand mean ``mu`` and standard deviation ``sigma``, lead time ``L``
and review period ``R``:

* **service factor**   ``z = Phi^-1(service_level)``  (inverse normal CDF)
* **safety stock**     ``SS = z * sigma * sqrt(L + R)``
* **reorder point**    ``ROP = mu * L + z * sigma * sqrt(L)``  (continuous review)
* **order-up-to**      ``S = mu * (L + R) + z * sigma * sqrt(L + R)``  (periodic review)

The `sqrt(L+R)` is the **protection interval**: you must cover demand variability
over the whole window until the next order arrives, and variance adds over
independent days.

The connection to the newsvendor
--------------------------------
Order-up-to ``S = mu + z*sigma`` is nothing but the **demand quantile at
probability ``Phi(z)``** *assuming demand is normal*. Set the service level equal
to the newsvendor critical ratio ``CR = Cu/(Cu+Co)`` and this **is** the
newsvendor order - just written as "mean + buffer" instead of ``F^-1(CR)``.

So safety-stock IS newsvendor under a Gaussian demand model. Its weakness is
exactly that assumption: retail demand is zero-inflated and right-skewed, the
normal has thin symmetric tails (and can even imply negative demand), so the
formula **under-stocks** intermittent items. That is the quantitative case for
forecasting the distribution directly (``src.quantile``) instead.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def service_factor(service_level: float | np.ndarray) -> np.ndarray:
    """``z = Phi^-1(service_level)`` - the safety-stock multiplier."""
    return norm.ppf(service_level)


def safety_stock(sigma_daily, lead_time: float, review_period: float = 1.0,
                 service_level: float = 0.95) -> np.ndarray:
    """``z * sigma * sqrt(L + R)`` - buffer for the protection interval."""
    z = service_factor(service_level)
    return z * np.asarray(sigma_daily) * np.sqrt(lead_time + review_period)


def reorder_point(mu_daily, sigma_daily, lead_time: float,
                  service_level: float = 0.95) -> np.ndarray:
    """Continuous-review ROP: expected lead-time demand + safety stock."""
    z = service_factor(service_level)
    mu, sigma = np.asarray(mu_daily), np.asarray(sigma_daily)
    return mu * lead_time + z * sigma * np.sqrt(lead_time)


def order_up_to_normal(mu_daily, sigma_daily, lead_time: float,
                       review_period: float = 1.0,
                       service_level: float = 0.95) -> np.ndarray:
    """Periodic-review base-stock level under the normal approximation."""
    z = service_factor(service_level)
    mu, sigma = np.asarray(mu_daily), np.asarray(sigma_daily)
    w = lead_time + review_period
    return np.clip(mu * w + z * sigma * np.sqrt(w), 0, None)
