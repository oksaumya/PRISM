"""Significance tests for comparing forecasting models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def newey_west_var(d: np.ndarray, lag: int) -> float:
    """HAC (Bartlett kernel) long-run variance of a series."""
    d = d - d.mean()
    T = d.size
    v = d @ d / T
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * (d[k:] @ d[:-k]) / T
    return v


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lag: int = 7) -> tuple[float, float]:
    """Diebold-Mariano (1995) test of equal predictive accuracy.

    d_t = L_a,t - L_b,t. H0: E[d_t] = 0. A negative statistic means model A
    has lower loss. HAC variance accounts for autocorrelation in d_t.
    Returns (statistic, two-sided p-value).
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    T = d.size
    if T <= lag + 1:
        raise ValueError(f"need more than lag + 1 = {lag + 1} loss values, got {T}")
    var = newey_west_var(d, lag)
    if var <= 0:                      # identical (or constant-difference) losses: no variance to test
        return (0.0, 1.0) if d.mean() == 0 else (float(np.sign(d.mean()) * np.inf), 0.0)
    stat = d.mean() / np.sqrt(var / T)
    return float(stat), float(2 * stats.t.sf(abs(stat), df=T - 1))


def panel_dm(df: pd.DataFrame, model_a: str, model_b: str, loss: str = "pinball", lag: int = 7) -> dict:
    """DM test on the daily cross-sectional mean loss (one value per target day)."""
    daily = df.groupby(["model", "day"])[loss].mean().unstack("model")
    stat, p = diebold_mariano(daily[model_a].to_numpy(), daily[model_b].to_numpy(), lag)
    return {"model_a": model_a, "model_b": model_b, "loss": loss, "days": len(daily),
            "mean_diff": float((daily[model_a] - daily[model_b]).mean()), "DM": stat, "p_value": p}


def series_wilcoxon(df: pd.DataFrame, model_a: str, model_b: str, loss: str = "pinball") -> dict:
    """Wilcoxon signed-rank test on per-series mean loss differences."""
    per = df.groupby(["model", "series"])[loss].mean().unstack("model")
    diff = (per[model_a] - per[model_b]).dropna()
    if (diff == 0).all():             # the two models are indistinguishable on every series
        w, p = 0.0, 1.0
    else:
        res = stats.wilcoxon(diff)
        w, p = float(res.statistic), float(res.pvalue)
    return {"model_a": model_a, "model_b": model_b, "loss": loss, "series": len(diff),
            "share_a_better": float((diff < 0).mean()), "W": w, "p_value": p}
