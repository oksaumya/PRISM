"""Seasonal naive baseline with empirical predictive quantiles."""

from __future__ import annotations

import numpy as np

from ..data import Panel
from .base import Forecast, Forecaster, empirical_quantiles, observed_history


def last_same_weekday(y: np.ndarray, season: int) -> np.ndarray:
    """Most recent valid value for each of the last ``season`` positions.

    If the value one season back is missing (unlaunched or stockout) we step
    back another season, so a stockout never becomes a "zero" forecast.
    Returns (N, season), aligned so that column j is the phase of day T-season+j.
    """
    n, T = y.shape
    out = np.full((n, season), np.nan, dtype=np.float64)
    for j in range(season):
        col = T - season + j
        idx = np.arange(col, -1, -season)
        vals = y[:, idx]                                    # newest first
        has = ~np.isnan(vals)
        first = np.argmax(has, axis=1)
        out[:, j] = np.where(has.any(axis=1), vals[np.arange(n), first], np.nan)
    return out


class SeasonalNaive(Forecaster):
    """y_hat(T+h) = last in-stock value on the same weekday.

    Uncertainty: empirical quantiles of the k-season-ahead naive errors
    y_t - y_{t-k*s} over the last ``residual_window`` days, where
    k = ceil(h / s) matches how far ahead the forecast reaches.
    """

    name = "seasonal_naive"

    def fit_predict(self, panel: Panel, horizon: int, levels: np.ndarray) -> Forecast:
        s, window = self.params.get("season", 7), self.params.get("residual_window", 364)
        y = observed_history(panel)
        base = last_same_weekday(y, s)
        # series without any valid same-weekday value fall back to their history mean
        fallback = np.nanmean(y[:, -window:], axis=1, keepdims=True)
        fallback = np.where(np.isnan(fallback), 0.0, fallback)
        base = np.where(np.isnan(base), fallback, base)

        h = np.arange(1, horizon + 1)
        point = base[:, (h - 1) % s]
        quantiles = np.empty((y.shape[0], horizon, len(levels)))
        recent = y[:, -window:]
        for k in np.unique(np.ceil(h / s).astype(int)):
            lag = k * s
            resid = recent[:, lag:] - recent[:, :-lag]
            qk = empirical_quantiles(resid, levels)
            cols = np.flatnonzero(np.ceil(h / s).astype(int) == k)
            quantiles[:, cols, :] = point[:, cols, None] + qk[:, None, :]
        return Forecast(point=point, quantiles=quantiles, levels=levels)
