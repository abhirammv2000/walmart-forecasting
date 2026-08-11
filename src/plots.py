"""Generate the project's figures from saved result artifacts.

Every chart here is built from a file already produced by the pipeline (the
inventory CSVs, the saved quantile forecasts, the point-forecast CSV, the trained
booster) - so the figures are reproducible, never hand-drawn, and regenerate with:

    python -m src.plots

Design notes
------------
* Colours use the **Okabe-Ito** palette, which is colour-vision-deficiency safe;
  every categorical pair here was checked for CVD separation rather than eyeballed.
* One measure per axis, recessive grid, thin marks, direct value labels. Titles
  state the *takeaway*, not just the axes.

Figures are written to ``docs/img/`` and embedded in the README and docs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib as mpl
mpl.use("Agg")                      # headless: render to file, never a window
import matplotlib.pyplot as plt

from . import config, data

IMG_DIR = config.ROOT / "docs" / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Okabe-Ito, assigned by role and used in this fixed order (never cycled).
INK = "#1a1a1a"
MUTED = "#6b6b6b"
BLUE = "#0072B2"        # our approach / forecast
VERMILLION = "#D55E00"  # baseline / naive
SKY = "#56B4E9"         # cost component A (holding)
ORANGE = "#E69F00"      # cost component B (shortage)
GREEN = "#009E73"       # actuals / chosen optimum


def _style() -> None:
    """Apply a clean, consistent house style to every figure."""
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "figure.facecolor": "white",
        "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.8,
        "axes.axisbelow": True,
    })


def _despine(ax, keep=("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)
    ax.grid(axis="x", visible=False)


def _save(fig, name: str) -> Path:
    fig.tight_layout()
    out = IMG_DIR / name
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(config.ROOT)}")
    return out


# --------------------------------------------------------------------------- #
# 1. Cost vs service level - the "there is an optimum" chart
# --------------------------------------------------------------------------- #
def cost_service_curve(curve_csv: Path = config.OUTPUT_DIR / "inventory_service_curve.csv"):
    df = pd.read_csv(curve_csv).sort_values("service_level")
    x, y = df["service_level"].to_numpy(), df["total_cost"].to_numpy()
    i_min = int(np.argmin(y))
    cr = 0.909   # theoretical critical ratio Cu/(Cu+Co) for 0.30/0.03

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(x, y, color=BLUE, lw=2, zorder=3)
    ax.scatter([x[i_min]], [y[i_min]], color=GREEN, s=90, zorder=5,
               label=f"empirical optimum (SL={x[i_min]:.2f})")
    ax.axvline(cr, color=VERMILLION, ls="--", lw=1.5,
               label=f"newsvendor theory (CR={cr:.3f})")
    ax.annotate("cost-minimising\nservice level", xy=(x[i_min], y[i_min]),
                xytext=(x[i_min] - 0.28, y[i_min] + 0.16 * (y.max() - y.min())),
                fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED))
    ax.set_xlabel("target service level  (order-up-to quantile)")
    ax.set_ylabel("total cost  (holding + shortage)")
    ax.set_title("Theory meets practice: cost is minimised at the critical ratio")
    ax.legend(frameon=False, fontsize=9, loc="upper center")
    _despine(ax)
    return _save(fig, "cost_service_curve.png")


# --------------------------------------------------------------------------- #
# 2. Newsvendor vs ordering the mean - the headline business result
# --------------------------------------------------------------------------- #
def policy_comparison(cmp_csv: Path = config.OUTPUT_DIR / "inventory_policy_comparison.csv"):
    df = pd.read_csv(cmp_csv)

    def _kind(p: str) -> str:
        p = p.lower()
        return ("newsvendor" if "newsvendor" in p
                else "point" if ("point" in p or "mean" in p) else "median")
    df["kind"] = df["policy"].map(_kind)
    order = {"point": 0, "median": 1, "newsvendor": 2}
    df = df.sort_values("kind", key=lambda s: s.map(order)).reset_index(drop=True)
    label = {"point": "Point forecast\n(order the mean)",
             "median": "Median forecast\n(order P50)",
             "newsvendor": "Newsvendor\n(order a quantile)"}
    df["label"] = df["kind"].map(label)
    # Baselines neutral, our approach highlighted.
    bar_color = {"point": VERMILLION, "median": "#9a9a9a", "newsvendor": BLUE}
    colors = [bar_color[k] for k in df["kind"]]

    # Headline numbers: point forecast (the fair baseline) vs newsvendor.
    base = df[df["kind"] == "point"].iloc[0] if (df["kind"] == "point").any() \
        else df[df["kind"] == "median"].iloc[0]
    nv = df[df["kind"] == "newsvendor"].iloc[0]
    cost_drop = (1 - nv["total_cost"] / base["total_cost"]) * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.7))
    x = np.arange(len(df))

    # Left: total cost, split into holding + shortage.
    hold, short = df["holding_cost"].to_numpy(), df["shortage_cost"].to_numpy()
    ax1.bar(x, hold, color=SKY, label="holding", width=0.62)
    ax1.bar(x, short, bottom=hold, color=ORANGE, label="shortage", width=0.62)
    for xi, tot in zip(x, df["total_cost"]):
        ax1.text(xi, tot, f"${tot/1e3:,.0f}K", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color=INK)
    ax1.set_xticks(x); ax1.set_xticklabels(df["label"], fontsize=8.5)
    ax1.set_ylabel("total cost ($)")
    ax1.set_title(f"{cost_drop:.0f}% lower cost vs the point forecast")
    ax1.legend(frameon=False, fontsize=9, loc="upper right")
    ax1.set_ylim(0, df["total_cost"].max() * 1.18)
    _despine(ax1)

    # Right: fill rate.
    fr = df["fill_rate"].to_numpy()
    ax2.bar(x, fr * 100, color=colors, width=0.62)
    for xi, f in zip(x, fr):
        ax2.text(xi, f * 100, f"{f*100:.0f}%", ha="center", va="bottom",
                 fontsize=10, fontweight="bold", color=INK)
    ax2.set_xticks(x); ax2.set_xticklabels(df["label"], fontsize=8.5)
    ax2.set_ylabel("fill rate (%)")
    ax2.set_title(f"{base['fill_rate']*100:.0f}% → {nv['fill_rate']*100:.0f}% demand met")
    ax2.set_ylim(0, 108)
    _despine(ax2)

    fig.suptitle("Ordering a demand quantile, not the point forecast",
                 fontsize=14, fontweight="bold", y=1.02)
    return _save(fig, "policy_comparison.png")


# --------------------------------------------------------------------------- #
# 3. Forecast vs actuals at the aggregate level
# --------------------------------------------------------------------------- #
def forecast_vs_actual(pred_csv: Path = config.PREDICTION_DIR / "baseline_validation.csv"):
    valid_cols = [f"d_{d}" for d in range(1914, 1942)]
    pred = pd.read_csv(pred_csv)
    # baseline_validation.csv is stored as ID_COLS + d_1914..d_1941.
    pred_days = [c for c in pred.columns if c.startswith("d_")]
    fc_total = pred[pred_days].sum(axis=0).to_numpy()

    eval_wide = data.load_sales_wide(evaluation=True)
    ac_total = eval_wide[valid_cols].sum(axis=0).to_numpy()

    days = np.arange(1, 29)
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    ax.plot(days, ac_total, color=GREEN, lw=2, marker="o", ms=4, label="actual")
    ax.plot(days, fc_total, color=BLUE, lw=2, marker="o", ms=4, label="forecast")
    mape = np.mean(np.abs(fc_total - ac_total) / ac_total) * 100
    ax.set_xlabel("forecast horizon (day)")
    ax.set_ylabel("total units sold (all 30,490 series)")
    ax.set_title(f"Aggregate forecast tracks actual demand  (MAPE {mape:.1f}%)")
    ax.legend(frameon=False, fontsize=10)
    ax.set_xlim(0.5, 28.5)
    _despine(ax)
    return _save(fig, "forecast_vs_actual.png")


# --------------------------------------------------------------------------- #
# 4. Quantile fan - the distribution, not a point
# --------------------------------------------------------------------------- #
def quantile_fan(quant_parquet: Path = config.PREDICTION_DIR / "quantiles_q_trained_final.parquet"):
    q = pd.read_parquet(quant_parquet)
    valid_cols = [f"d_{d}" for d in range(1914, 1942)]

    # Pick a *representative* high-volume series (90th pct of horizon demand) -
    # legible, but not the single most volatile outlier, whose extremes would
    # misleadingly suggest the intervals never cover.
    eval_wide = data.load_sales_wide(evaluation=True)
    totals = eval_wide.set_index("id")[valid_cols].sum(axis=1)
    target = totals.quantile(0.90)
    sample_id = (totals - target).abs().idxmin()
    actual = eval_wide.set_index("id").loc[sample_id, valid_cols].to_numpy(dtype=float)

    qs = sorted(q["quantile"].unique())
    band = {u: q[q["quantile"] == u].set_index("id").loc[sample_id, valid_cols]
            .to_numpy(dtype=float) for u in qs}
    days = np.arange(1, 29)

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    # Nested prediction-interval bands, lighter as they widen.
    pairs = [(0.005, 0.995, 0.12), (0.165, 0.835, 0.22), (0.25, 0.75, 0.32)]
    for lo, hi, a in pairs:
        ax.fill_between(days, band[lo], band[hi], color=BLUE, alpha=a, lw=0,
                        label=f"{int((hi-lo)*100)}% interval")
    ax.plot(days, band[0.5], color=BLUE, lw=2, label="median forecast")
    ax.plot(days, actual, color=GREEN, lw=0, marker="o", ms=5, label="actual")
    ax.set_xlabel("forecast horizon (day)")
    ax.set_ylabel("units sold")
    ax.set_title("A distribution, not a point: 9-quantile forecast vs actual")
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper left")
    ax.set_xlim(0.5, 28.5)
    _despine(ax)
    return _save(fig, "quantile_fan.png")


# --------------------------------------------------------------------------- #
# 5. Feature importance
# --------------------------------------------------------------------------- #
def feature_importance(model_txt: Path = config.MODEL_DIR / "baseline_validation.txt",
                       top_n: int = 15):
    import lightgbm as lgb
    booster = lgb.Booster(model_file=str(model_txt))
    imp = pd.DataFrame({
        "feature": booster.feature_name(),
        "gain": booster.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=True).tail(top_n)
    imp["gain"] /= imp["gain"].sum()   # share of total gain among shown

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.barh(imp["feature"], imp["gain"] * 100, color=BLUE, height=0.72)
    for y_, v in enumerate(imp["gain"] * 100):
        ax.text(v, y_, f" {v:.0f}%", va="center", fontsize=9, color=MUTED)
    ax.set_xlabel("share of model gain (%)")
    ax.set_title(f"What the model relies on (top {top_n} features)")
    _despine(ax)
    ax.grid(axis="y", visible=False)
    return _save(fig, "feature_importance.png")


# --------------------------------------------------------------------------- #
# 6. Demand intermittency - why this problem is hard
# --------------------------------------------------------------------------- #
def demand_intermittency():
    """Distribution of the zero-sales-day fraction across the 30,490 series."""
    sales = pd.read_csv(config.SALES_EVALUATION_CSV)
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    vals = sales[day_cols].to_numpy()
    zero_frac = (vals == 0).mean(axis=1)          # per series
    overall_zero = float((vals == 0).mean())

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.hist(zero_frac * 100, bins=40, color=BLUE, edgecolor="white", linewidth=0.4)
    ax.axvline(overall_zero * 100, color=VERMILLION, ls="--", lw=1.5,
               label=f"overall {overall_zero*100:.0f}% of series-days are zero")
    ax.set_xlabel("% of days a series records zero sales")
    ax.set_ylabel("number of series")
    ax.set_title("Intermittent demand: most items don't sell most days")
    ax.legend(frameon=False, fontsize=9)
    _despine(ax)
    return _save(fig, "demand_intermittency.png")


# --------------------------------------------------------------------------- #
# 7. Classical intermittent methods vs the ML model
# --------------------------------------------------------------------------- #
def intermittent_benchmark(csv: Path = config.OUTPUT_DIR / "intermittent_benchmark.csv"):
    df = pd.read_csv(csv).set_index("method")
    order = ["seasonal_naive", "croston", "sba", "tsb", "lightgbm"]
    df = df.reindex([m for m in order if m in df.index])
    names = {"seasonal_naive": "Seasonal\nnaive", "croston": "Croston",
             "sba": "SBA", "tsb": "TSB", "lightgbm": "LightGBM"}
    labels = [names.get(m, m) for m in df.index]
    colors = [BLUE if m == "lightgbm" else "#9a9a9a" for m in df.index]

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    x = np.arange(len(df))
    ax.bar(x, df["WRMSSE"], color=colors, width=0.62)
    for xi, v in zip(x, df["WRMSSE"]):
        ax.text(xi, v, f"{v:.2f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("WRMSSE  (lower is better)")
    ax.set_title("Classical intermittent methods vs. the feature-based model")
    ax.set_ylim(0, df["WRMSSE"].max() * 1.15)
    _despine(ax)
    ax.grid(axis="x", visible=False)
    return _save(fig, "intermittent_benchmark.png")


def main() -> None:
    _style()
    print("Generating figures -> docs/img/")
    demand_intermittency()
    forecast_vs_actual()
    quantile_fan()
    feature_importance()
    cost_service_curve()
    policy_comparison()
    intermittent_benchmark()
    print("Done.")


if __name__ == "__main__":
    main()
