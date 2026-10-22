"""Figures for notebooks, the report and the slides.

One visual system everywhere: each model keeps a fixed colour (colour follows
the entity, never its rank), 2px lines, hairline recessive grids, a legend for
every multi-series chart, and text in ink colours rather than series colours.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import qcol

INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
GRID, SURFACE = "#e4e3df", "#ffffff"
BASELINE_GRAY = "#9a9994"

# fixed categorical order (validated palette): blue, orange, aqua, violet
MODEL_COLORS = {
    "kalman": "#2a78d6",
    "lightgbm": "#eb6834",
    "tsb": "#1baf7a",
    "lightgbm_lag28": "#6250d6",
    "seasonal_naive": BASELINE_GRAY,          # the reference baseline recedes
}
MODEL_LABELS = {
    "seasonal_naive": "Seasonal naive",
    "tsb": "TSB (Croston-type)",
    "kalman": "Kalman filter (local)",
    "lightgbm": "LightGBM (global)",
    "lightgbm_lag28": "LightGBM lag-28 only",
}
MODEL_ORDER = ["seasonal_naive", "tsb", "kalman", "lightgbm", "lightgbm_lag28"]
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110, "figure.constrained_layout.use": True, "savefig.dpi": 300, "savefig.bbox": "tight",
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": INK_3, "ytick.color": INK_3, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "lines.linewidth": 2.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 8, "legend.labelcolor": INK_2,
    })


def legend_below(fig, ax, ncols: int = 4) -> None:
    """One figure-level legend under the plots, so it never covers data."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=ncols)


def save(fig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # no creation timestamp, so re-running a script only changes a PDF when the figure changed
    fig.savefig(path, metadata={"CreationDate": None} if path.suffix == ".pdf" else None)
    if path.suffix == ".pdf":                                  # PNG twin for slides
        fig.savefig(path.with_suffix(".png"))


def _models(df: pd.DataFrame) -> list[str]:
    present = set(df["model"])
    return [m for m in MODEL_ORDER if m in present]


def metric_by_horizon(table: pd.DataFrame, metric: str, ax=None, models=None):
    """Line per model of ``metric`` against forecast horizon h = 1..28."""
    ax = ax or plt.subplots(figsize=(4.2, 2.8))[1]
    for m in models or _models(table):
        d = table[table["model"] == m].sort_values("h")
        ax.plot(d["h"], d[metric], color=MODEL_COLORS[m], label=MODEL_LABELS[m],
                lw=1.6 if m == "seasonal_naive" else 2.0)
    ax.set_xlabel("Forecast horizon (days ahead)")
    ax.set_ylabel(metric)
    ax.set_xticks([1, 7, 14, 21, 28])
    return ax


def quantile_calibration(evals: pd.DataFrame, levels) -> pd.DataFrame:
    """Calibration bounds of each predictive quantile, per model.

    Sales are counts: the a-quantile of a discrete distribution is 0 whenever
    P(Y = 0) >= a, so the share of actuals below it cannot equal a exactly.
    A quantile is calibrated when  P(y < q) <= a <= P(y <= q).
    """
    rows = []
    for m, d in evals.groupby("model"):
        y = d["y"].to_numpy()
        for a in levels:
            q = d[qcol(a)].to_numpy()
            rows.append({"model": m, "level": a, "below": np.mean(y < q), "at_or_below": np.mean(y <= q)})
    out = pd.DataFrame(rows)
    out["calibrated"] = (out["below"] <= out["level"]) & (out["level"] <= out["at_or_below"])
    return out


def reliability(evals: pd.DataFrame, levels, ax=None, models=None):
    """Reliability diagram for count data: each quantile is drawn as the interval
    [P(y < q), P(y <= q)]; it is calibrated when the interval crosses the diagonal."""
    ax = ax or plt.subplots(figsize=(3.2, 3.0))[1]
    ax.plot([0, 1], [0, 1], color=INK_3, lw=1, label="Perfect calibration")
    models = models or _models(evals)
    cal = quantile_calibration(evals[evals["model"].isin(models)], levels)
    for j, m in enumerate(models):
        d = cal[cal["model"] == m]
        x = d["level"].to_numpy() + (j - (len(models) - 1) / 2) * 0.014
        ax.vlines(x, d["below"], d["at_or_below"], color=MODEL_COLORS[m], lw=2.2, label=MODEL_LABELS[m])
        ax.scatter(x, (d["below"] + d["at_or_below"]) / 2, color=MODEL_COLORS[m], s=12, zorder=3,
                   edgecolor=SURFACE, linewidth=0.8)    # keeps zero-length intervals visible
    ax.set_xlabel("Nominal quantile level")
    ax.set_ylabel("Share of actuals below / at quantile")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    return ax


def grouped_bars(table: pd.DataFrame, by: str, metric: str, order=None, ax=None, models=None):
    """Grouped columns: one group per segment, one column per model (<= 24px-like slim bars)."""
    models = models or _models(table)
    groups = order or list(table[by].drop_duplicates())
    ax = ax or plt.subplots(figsize=(5.0, 2.8))[1]
    width = 0.8 / len(models)
    x = np.arange(len(groups))
    for j, m in enumerate(models):
        vals = [table.loc[(table["model"] == m) & (table[by] == g), metric].squeeze() for g in groups]
        vals = [v if np.isscalar(v) else np.nan for v in vals]
        ax.bar(x + (j - (len(models) - 1) / 2) * width, vals, width * 0.86, color=MODEL_COLORS[m],
               label=MODEL_LABELS[m], edgecolor=SURFACE, linewidth=0)
    ax.set_xticks(x, groups)
    ax.set_ylabel(metric)
    ax.grid(axis="x", visible=False)
    return ax


def fan_chart(history: np.ndarray, actual: np.ndarray, fc: pd.DataFrame, model: str,
              start_day: int, ax=None, stockout_hist: np.ndarray | None = None, title=None):
    """History, actuals and the 50% / 90% predictive bands of one model for one series."""
    ax = ax or plt.subplots(figsize=(5.2, 2.6))[1]
    color = MODEL_COLORS[model]
    hx = np.arange(start_day - len(history), start_day)
    fx = start_day + fc["h"].to_numpy() - 1
    ax.plot(hx, history, color=INK_3, lw=1.2, label="History")
    if stockout_hist is not None and stockout_hist.any():
        ax.scatter(hx[stockout_hist], history[stockout_hist], s=10, color=INK, marker="x", lw=0.8,
                   label="Flagged stockout", zorder=4)
    ax.fill_between(fx, fc[qcol(0.05)], fc[qcol(0.95)], color=color, alpha=0.15, lw=0, label="90% interval")
    ax.fill_between(fx, fc[qcol(0.25)], fc[qcol(0.75)], color=color, alpha=0.30, lw=0, label="50% interval")
    ax.plot(fx, fc[qcol(0.5)], color=color, lw=2, label=f"{MODEL_LABELS[model]} median")
    ax.plot(fx, actual, color=INK, lw=1.2, marker="o", ms=2.5, label="Actual")
    ax.axvline(start_day - 0.5, color=INK_3, lw=0.8)
    ax.set_xlabel("Day index")
    ax.set_ylabel("Units sold")
    if title:
        ax.set_title(title)
    return ax
