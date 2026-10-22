"""Synthetic panel so tests run in seconds and need no Kaggle download."""

import numpy as np
import pandas as pd
import pytest

from prism.data import Panel
from prism.stockout import flag_stockouts

N_SERIES, N_DAYS, HORIZON = 12, 420, 28


def make_calendar(n_days: int) -> pd.DataFrame:
    dates = pd.date_range("2015-01-03", periods=n_days, freq="D")   # a Saturday, like M5 day 1
    rng = np.random.default_rng(1)
    events = np.where(rng.random(n_days) < 0.03, "Holiday", None)
    return pd.DataFrame({
        "date": dates,
        "wm_yr_wk": np.arange(n_days) // 7,
        "wday": (np.arange(n_days) % 7) + 1,
        "month": dates.month,
        "event_name_1": events,
        "event_type_1": np.where(events == None, None, "National"),  # noqa: E711
        "snap_CA": (dates.day <= 10).astype(int),
        "snap_TX": (dates.day <= 10).astype(int),
        "snap_WI": (dates.day <= 10).astype(int),
    })


@pytest.fixture(scope="session")
def panel() -> Panel:
    rng = np.random.default_rng(0)
    n_cal = N_DAYS + HORIZON
    weekly = np.array([1.3, 1.2, 0.9, 0.8, 0.8, 1.0, 1.1])
    rate = rng.uniform(0.3, 8, N_SERIES)[:, None] * weekly[np.arange(N_DAYS) % 7][None, :]
    y = rng.poisson(rate).astype(np.float32)
    y[0, 300:330] = 0                                # an obvious stockout on a busy series
    launch = np.zeros(N_SERIES, dtype=int)
    launch[-1] = N_DAYS - 40                         # a cold-start series
    y_obs = y.copy()
    y_obs[np.arange(N_DAYS)[None, :] < launch[:, None]] = np.nan
    meta = pd.DataFrame({
        "id": [f"S{i}" for i in range(N_SERIES)],
        "item_id": [f"ITEM_{i % 6}" for i in range(N_SERIES)],
        "dept_id": [f"D{i % 3}" for i in range(N_SERIES)],
        "cat_id": [f"C{i % 2}" for i in range(N_SERIES)],
        "store_id": ["CA_1" if i < 6 else "TX_1" for i in range(N_SERIES)],
        "state_id": ["CA" if i < 6 else "TX" for i in range(N_SERIES)],
        "demand_class": ["smooth"] * N_SERIES,
        "sim_cold_start": [False] * (N_SERIES - 1) + [True],
    })
    params = dict(p_threshold=1e-3, window=56, min_obs=14)
    pad = lambda a: np.hstack([a, np.full((N_SERIES, HORIZON), np.nan, dtype=np.float32)])
    return Panel(
        y=y_obs, y_raw=y, stockout=flag_stockouts(y_obs, **params),
        price=np.full((N_SERIES, n_cal), 2.5, dtype=np.float32),
        item_other=pad(y * 0.9), dept_other=pad(y * 1.1), launch=launch,
        meta=meta, calendar=make_calendar(n_cal), stockout_params=params,
    )


@pytest.fixture(scope="session")
def cfg() -> dict:
    return {
        "seed": 0,
        "quantiles": [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
        "backtest": {"horizon": HORIZON, "n_origins": 2, "step": HORIZON, "last_day": N_DAYS},
        "cold_start": {"max_history_days": 90, "min_history_days": 7},
        "models": {
            "seasonal_naive": {"season": 7, "residual_window": 364},
            "tsb": {"alpha": 0.1, "beta": 0.1, "residual_window": 364},
            "kalman": {"fit_window": 365, "q_level_grid": [-4, 0, 5], "q_season_grid": [-6, -2, 3],
                       "refine": False, "min_obs_for_mle": 56,
                       "default_q_level": 0.01, "default_q_season": 1e-4},
        },
    }
