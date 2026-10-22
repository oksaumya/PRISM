"""Common model interface.

Every model, local or global, ML or (in Phase 2) deep learning, is fit on the
information set at one origin and returns a point forecast plus quantiles for
the next ``horizon`` days of *every* series. The backtest harness never needs
to know what is inside.
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data import Panel


@dataclass
class Forecast:
    point: np.ndarray        # (N, H)     mean forecast
    quantiles: np.ndarray    # (N, H, Q)  predictive quantiles
    levels: np.ndarray       # (Q,)       quantile levels
    cross_size: np.ndarray = field(init=False)   # (N, H) worst quantile crossing in units (0 = none)

    def __post_init__(self):
        # record quantile crossing before fixing it: independent quantile models (LightGBM) can
        # predict e.g. q_0.90 < q_0.75, which is impossible for a real distribution. We keep the
        # size (how far a higher quantile falls below a lower one), since tiny crossings of a
        # few thousandths of a unit near zero are numerical noise, not a modelling problem.
        self.cross_size = np.clip(-np.diff(self.quantiles, axis=-1), 0, None).max(axis=-1)
        # enforce valid predictive distributions: non-negative, non-crossing
        self.point = np.clip(self.point, 0, None)
        self.quantiles = np.sort(np.clip(self.quantiles, 0, None), axis=-1)


class Forecaster(ABC):
    name: str = "base"

    def __init__(self, **params):
        self.params = params
        # optional small tables a model reports about its own fit (e.g. feature importance);
        # the backtest saves them next to the forecast cache
        self.diagnostics: dict[str, pd.DataFrame] = {}

    @abstractmethod
    def fit_predict(self, panel: Panel, horizon: int, levels: np.ndarray) -> Forecast:
        """Fit on ``panel`` (already truncated at the origin) and forecast ``horizon`` days."""


def observed_history(panel: Panel, mask_stockouts: bool = True) -> np.ndarray:
    """Sales history with unobserved and (optionally) stockout days set to NaN."""
    y = panel.y.copy()
    if mask_stockouts:
        y[panel.stockout] = np.nan
    return y


def empirical_quantiles(resid: np.ndarray, levels: np.ndarray, min_count: int = 14) -> np.ndarray:
    """Per-series quantiles of residuals (NaN-aware). (N, T) -> (N, Q).

    Series with fewer than ``min_count`` residuals borrow the pooled
    residual distribution of all series.
    """
    with warnings.catch_warnings():                      # all-NaN rows (not launched) are replaced below
        warnings.simplefilter("ignore", RuntimeWarning)
        q = np.nanquantile(resid, levels, axis=1).T
    pooled = np.nanquantile(resid, levels)
    few = np.sum(~np.isnan(resid), axis=1) < min_count
    q[few] = pooled
    return q
