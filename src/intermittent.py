"""Classical intermittent-demand methods: Croston, SBA, TSB + demand classification.

Most M5 bottom-level series are **intermittent** - lots of zeros, sporadic
spikes. The standard tools for this are not ML; they are the Croston family,
which forecast by separating *how much* sells from *how often*. They matter for
two reasons in a retail interview:

1. They are the right, robust default for genuinely intermittent / spare-parts
   demand (no seasonality, no covariates, little history).
2. Knowing them - and knowing *when a feature-rich gradient booster beats them*
   (when there IS seasonality and covariates, as in M5) - is exactly the
   judgement interviewers probe.

Methods
-------
* **Croston (1972)** - SES on the non-zero demand sizes and on the intervals
  between them, separately; forecast = size / interval.
* **SBA (Syntetos-Boylan Approximation)** - Croston is known to be biased; SBA
  multiplies by ``(1 - alpha/2)`` to debias. The recommended default in practice.
* **TSB (Teunter-Syntetos-Babai, 2011)** - updates the demand *probability*
  every period (not the interval), so it can decay a SKU toward zero when it
  stops selling - the right choice for obsolescence / end-of-life.

Demand classification (Syntetos-Boylan-Croston)
-----------------------------------------------
``ADI`` = average inter-demand interval, ``CV2`` = squared coefficient of
variation of non-zero sizes. The (ADI, CV2) plane splits demand into
*smooth / intermittent / erratic / lumpy*, which tells you which method to reach
for. Cutoffs: ADI = 1.32, CV2 = 0.49.
"""
from __future__ import annotations

import numpy as np

ADI_CUT = 1.32
CV2_CUT = 0.49


def _ses_forecast_rate(sizes: np.ndarray, intervals: np.ndarray,
                       alpha: float) -> tuple[float, float]:
    """Exponentially smooth demand sizes and intervals; return (z, p)."""
    z, p = float(sizes[0]), float(intervals[0])
    for i in range(1, len(sizes)):
        z += alpha * (sizes[i] - z)
        p += alpha * (intervals[i] - p)
    return z, p


def croston(y: np.ndarray, alpha: float = 0.1, variant: str = "croston") -> float:
    """Per-period demand-rate forecast from Croston / SBA.

    Parameters
    ----------
    y:
        1-D history of demand (zeros allowed).
    alpha:
        Smoothing constant for both the size and interval SES.
    variant:
        ``"croston"`` (classic) or ``"sba"`` (bias-corrected, recommended).

    Returns a single constant rate - the classical methods are flat forecasts,
    with no day-of-week structure (which is exactly why they lose to a feature
    model on seasonal M5 data).
    """
    y = np.asarray(y, dtype=float)
    nz = np.flatnonzero(y > 0)
    if nz.size == 0:
        return 0.0
    sizes = y[nz]
    # Interval to the first demand is (index + 1); then gaps between demands.
    intervals = np.diff(np.concatenate(([-1], nz))).astype(float)
    z, p = _ses_forecast_rate(sizes, intervals, alpha)
    rate = z / p if p > 0 else 0.0
    if variant == "sba":
        rate *= (1.0 - alpha / 2.0)      # Syntetos-Boylan debiasing
    return rate


def tsb(y: np.ndarray, alpha: float = 0.1, beta: float = 0.1) -> float:
    """Per-period demand-rate forecast from TSB.

    Updates the demand *probability* every period (smoothing ``beta``) and the
    demand *size* only when demand occurs (smoothing ``alpha``); forecast is
    ``probability * size``. Unlike Croston it decays toward zero for a SKU that
    stops selling, so it is the right tool for obsolescence.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    nz = np.flatnonzero(y > 0)
    if nz.size == 0:
        return 0.0
    first = int(nz[0])
    z = float(y[first])
    prob = 1.0 / (first + 1)             # initial demand probability
    for t in range(first + 1, n):
        occurred = 1.0 if y[t] > 0 else 0.0
        prob += beta * (occurred - prob)
        if y[t] > 0:
            z += alpha * (y[t] - z)
    return prob * z


def demand_stats(y: np.ndarray) -> tuple[float, float]:
    """Return (ADI, CV2) - average demand interval and squared CV of sizes."""
    y = np.asarray(y, dtype=float)
    nz = y[y > 0]
    if nz.size == 0:
        return np.inf, 0.0
    adi = y.size / nz.size
    mean = nz.mean()
    cv2 = (nz.std() / mean) ** 2 if mean > 0 else 0.0
    return float(adi), float(cv2)


def classify(y: np.ndarray) -> str:
    """Syntetos-Boylan-Croston class: smooth / intermittent / erratic / lumpy."""
    adi, cv2 = demand_stats(y)
    if adi < ADI_CUT:
        return "erratic" if cv2 >= CV2_CUT else "smooth"
    return "lumpy" if cv2 >= CV2_CUT else "intermittent"


def recommended_method(y: np.ndarray) -> str:
    """The textbook method for a series' class (what a planner would pick)."""
    return {"smooth": "SES/ETS", "intermittent": "Croston",
            "erratic": "SBA", "lumpy": "SBA/TSB"}[classify(y)]
