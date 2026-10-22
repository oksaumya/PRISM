"""Forecast accuracy metrics.

Each point metric is evaluated on the functional it elicits (Gneiting, 2011):
the median minimises absolute error, the mean minimises squared error. So

* MAE, WAPE   use the predictive median  (q = 0.5)
* RMSE, RMSSE use the predictive mean    (``point``)

Probabilistic quality is scored with the pinball (quantile) loss, a strictly
consistent scoring rule for quantiles whose average over levels approximates
the CRPS (Gneiting & Raftery, 2007), plus empirical interval coverage.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def qcol(level: float) -> str:
    return f"q_{level:.2f}"


def pinball_loss(y: np.ndarray, q: np.ndarray, level: float) -> np.ndarray:
    """Pinball loss  rho_a(u) = max(a u, (a - 1) u),  u = y - q."""
    u = y - q
    return np.maximum(level * u, (level - 1) * u)


def add_losses(df: pd.DataFrame, levels) -> pd.DataFrame:
    """Row-level loss columns, so any grouping can be aggregated cheaply."""
    df = df.copy()
    y = df["y"].to_numpy()
    med = df[qcol(0.5)].to_numpy()
    df["abs_err"] = np.abs(y - med)
    df["abs_err_mean"] = np.abs(y - df["point"].to_numpy())
    df["sq_err"] = (y - df["point"].to_numpy()) ** 2
    df["pinball"] = np.mean([pinball_loss(y, df[qcol(a)].to_numpy(), a) for a in levels], axis=0)
    for lo, hi, tag in [(0.10, 0.90, "80"), (0.05, 0.95, "90")]:
        if qcol(lo) in df and qcol(hi) in df:
            l, u = df[qcol(lo)].to_numpy(), df[qcol(hi)].to_numpy()
            df[f"cover{tag}"] = (y >= l) & (y <= u)
            df[f"width{tag}"] = u - l
    return df


def _scaled(df: pd.DataFrame, col: str, scale: str, root: bool) -> float:
    """Mean over (series, origin) of mean(col)/scale (M5-style RMSSE / scaled pinball)."""
    g = df.groupby(["series", "origin"], sort=False)
    per = g[col].mean() / g[scale].first()
    per = per.replace([np.inf, -np.inf], np.nan)
    return float(np.sqrt(per).mean() if root else per.mean())


def summarize(df: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Aggregate row-level losses (from :func:`add_losses`) into a metric table."""
    by = by or []

    def agg(g: pd.DataFrame) -> dict:
        out = {
            "n": len(g),
            "MAE": g["abs_err"].mean(),
            "RMSE": np.sqrt(g["sq_err"].mean()),
            "WAPE": g["abs_err"].sum() / max(g["y"].sum(), 1e-9),
            "Pinball": g["pinball"].mean(),
            "RMSSE": _scaled(g, "sq_err", "scale_sq", root=True),
            "SPL": _scaled(g, "pinball", "scale_abs", root=False),
        }
        for tag in ["80", "90"]:
            if f"cover{tag}" in g:
                out[f"Cov{tag}"] = g[f"cover{tag}"].mean()
                out[f"Width{tag}"] = g[f"width{tag}"].mean()
        return out

    # An explicit loop rather than groupby.apply: apply drops the grouping columns from
    # each group, and the scaled metrics need "series" and "origin" even when grouping by them.
    keys = ["model", *by]
    rows = [{**dict(zip(keys, key)), **agg(g)} for key, g in df.groupby(keys, sort=True, observed=True)]
    return pd.DataFrame(rows)


def naive_scales(y_hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """In-sample one-step naive scales per series (M5 / Hyndman & Koehler, 2006).

    scale_sq  = mean (y_t - y_{t-1})^2   for RMSSE
    scale_abs = mean |y_t - y_{t-1}|     for scaled pinball loss
    """
    d = np.diff(y_hist, axis=1)
    with warnings.catch_warnings():                      # series with < 2 observed days -> NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        sq = np.nanmean(d ** 2, axis=1)
        ab = np.nanmean(np.abs(d), axis=1)
    sq[sq == 0] = np.nan
    ab[ab == 0] = np.nan
    return sq, ab
