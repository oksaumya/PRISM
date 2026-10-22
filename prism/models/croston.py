"""TSB (Teunter, Syntetos & Babai, 2011): Croston-type method for intermittent demand.

Croston (1972) smooths demand *sizes* and *intervals* separately. TSB instead
smooths the demand *probability* every period, so the forecast decays when an
item stops selling (important for obsolescence and for stockout-like gaps):

    if y_t > 0:  z_t = z_{t-1} + alpha (y_t - z_{t-1})
    p_t = p_{t-1} + beta (1[y_t > 0] - p_{t-1})
    y_hat = p_t * z_t       (flat over the horizon)
"""

from __future__ import annotations

import numpy as np

from ..data import Panel
from .base import Forecast, Forecaster, empirical_quantiles, observed_history


class TSB(Forecaster):
    name = "tsb"

    def fit_predict(self, panel: Panel, horizon: int, levels: np.ndarray) -> Forecast:
        alpha, beta = self.params.get("alpha", 0.1), self.params.get("beta", 0.1)
        window = self.params.get("residual_window", 364)
        y = observed_history(panel)
        n, T = y.shape
        z = np.full(n, np.nan)
        p = np.full(n, np.nan)
        fitted = np.full((n, T), np.nan)                     # one-step-ahead forecasts
        for t in range(T):
            fitted[:, t] = p * z
            yt = y[:, t]
            obs = ~np.isnan(yt)
            pos = obs & (yt > 0)
            # initialise on the first observation, then smooth
            init = obs & np.isnan(p)
            upd = obs & ~init
            p = np.where(init, pos.astype(float), p)
            p = np.where(upd, p + beta * (pos.astype(float) - p), p)
            z = np.where(pos, np.where(np.isnan(z), yt, z + alpha * (yt - z)), z)
        z = np.where(np.isnan(z), 0.0, z)
        p = np.where(np.isnan(p), 0.0, p)
        point = np.repeat((p * z)[:, None], horizon, axis=1)
        resid = (y - fitted)[:, -window:]
        q = empirical_quantiles(resid, levels)
        quantiles = point[:, :, None] + q[:, None, :]
        return Forecast(point=point, quantiles=quantiles, levels=levels)
