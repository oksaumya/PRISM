"""Leakage-safe feature engineering for the global gradient-boosting model.

Horizon-safe *direct* design: every sales-derived feature for target day t is
built only from days <= t - min_lag. With min_lag >= h, the same model can
forecast day T + h from information available at the origin T, with no
recursive feeding of its own predictions and no peeking. Calendar and price
features are known in advance (planned prices and holidays), as in M5.

All features are computed on wide (N, D) matrices and flattened to one row per
(series, day).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .data import Panel
from .stockout import zero_streak

CATEGORICAL = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "event_name_1", "event_type_1"]


def shift(A: np.ndarray, k: int) -> np.ndarray:
    """Shift right along time by k days (value at t is A[t - k])."""
    out = np.full(A.shape, np.nan, dtype=np.float32)
    out[:, k:] = A[:, : A.shape[1] - k]
    return out


def rolling(A: np.ndarray, window: int, how: str = "mean") -> np.ndarray:
    """NaN-aware trailing rolling statistic along time."""
    r = pd.DataFrame(A.T).rolling(window, min_periods=1)
    return getattr(r, how)().to_numpy(dtype=np.float32).T


def extend(A: np.ndarray, length: int) -> np.ndarray:
    """Pad a (N, D) matrix with NaN up to ``length`` days (the unknown future)."""
    if A.shape[1] >= length:
        return A[:, :length].astype(np.float32)
    pad = np.full((A.shape[0], length - A.shape[1]), np.nan, dtype=np.float32)
    return np.hstack([A.astype(np.float32), pad])


def calendar_features(calendar: pd.DataFrame) -> pd.DataFrame:
    cal = calendar.copy()
    out = pd.DataFrame(index=cal.index)
    out["wday"] = cal["wday"]
    out["month"] = cal["month"]
    out["mday"] = cal["date"].dt.day
    out["week_of_year"] = cal["date"].dt.isocalendar().week.astype(int).to_numpy()
    for col in ["event_name_1", "event_type_1"]:
        out[col] = pd.factorize(cal[col])[0] + 1           # 0 = no event
    has_event = cal["event_name_1"].notna().to_numpy()
    idx = np.arange(len(cal))
    ev = np.flatnonzero(has_event)
    nxt = np.searchsorted(ev, idx)                          # events are known ahead of time
    out["days_to_event"] = np.where(nxt < len(ev), ev[np.minimum(nxt, len(ev) - 1)] - idx, 99)
    prv = np.searchsorted(ev, idx, side="right") - 1
    out["days_since_event"] = np.where(prv >= 0, idx - ev[np.maximum(prv, 0)], 99)
    return out


def sales_features(y: np.ndarray, min_lag: int) -> dict[str, np.ndarray]:
    """Lag / rolling features of a sales matrix, all shifted by at least ``min_lag``."""
    L = min_lag
    f: dict[str, np.ndarray] = {}
    for k in [L, L + 1, L + 2, L + 7, L + 14, L + 21, L + 28, 364]:
        f[f"lag_{k}"] = shift(y, k)
    base = shift(y, L)
    for w in [7, 28, 56, 112, 364]:
        f[f"rmean_{w}"] = rolling(base, w, "mean")
    f["rstd_28"] = rolling(base, 28, "std")
    f["rmax_28"] = rolling(base, 28, "max")
    f["rzero_28"] = rolling((base == 0).astype(np.float32) + np.where(np.isnan(base), np.nan, 0), 28, "mean")
    with warnings.catch_warnings():                          # all-NaN slices -> NaN is intended
        warnings.simplefilter("ignore", RuntimeWarning)
        f["dow_mean_4"] = np.nanmean(np.stack([shift(y, L + 7 * i) for i in range(4)]), axis=0)
    f["momentum_7_28"] = f["rmean_7"] / (f["rmean_28"] + 1e-3)
    f["expanding_mean"] = pd.DataFrame(base.T).expanding(min_periods=1).mean().to_numpy(np.float32).T
    f["zero_streak"] = shift(zero_streak(y), L)
    return f


def build_features(panel: Panel, min_lag: int, horizon: int,
                   t_start: int, t_end: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Feature rows for days ``t_start .. t_end-1`` of every series.

    ``panel`` must already be truncated at the origin; days after the cutoff
    are NaN in every sales-derived matrix, so a feature can only "see" the
    future if it were shifted by less than the horizon, which ``min_lag``
    prevents. Returns (X, series_index, day_index).
    """
    cutoff = panel.cutoff if panel.cutoff is not None else panel.n_days
    D = cutoff + horizon
    N = panel.n_series
    y = extend(panel.y, D)

    feats = sales_features(y, min_lag)
    item_o = shift(extend(panel.item_other, D), min_lag)
    dept_o = shift(extend(panel.dept_other, D), min_lag)
    feats["item_other_7"] = rolling(item_o, 7)
    feats["item_other_28"] = rolling(item_o, 28)
    feats["dept_other_28"] = rolling(dept_o, 28)
    feats["own_vs_item_other"] = feats["rmean_28"] / (feats["item_other_28"] + 1e-3)

    price = extend(panel.price, D)                          # planned prices: known ahead
    feats["price"] = price
    feats["price_norm"] = price / (rolling(price, 364, "mean") + 1e-6)
    feats["price_max_ratio"] = price / (rolling(price, 364, "max") + 1e-6)
    feats["price_change_7"] = price / (shift(price, 7) + 1e-6) - 1

    days = np.arange(D)
    feats["item_age"] = (days[None, :] - panel.launch[:, None]).astype(np.float32)

    cal = calendar_features(panel.calendar).iloc[:D]
    snap = panel.calendar[[f"snap_{s}" for s in ["CA", "TX", "WI"]]].to_numpy()[:D]
    state_col = panel.meta["state_id"].map({"CA": 0, "TX": 1, "WI": 2}).to_numpy()
    feats["snap"] = snap[:, state_col].T.astype(np.float32)

    # flatten the requested window
    sl = slice(t_start, t_end)
    width = t_end - t_start
    cols = {name: A[:, sl].reshape(-1) for name, A in feats.items()}
    for c in cal.columns:
        cols[c] = np.tile(cal[c].to_numpy()[sl], N)
    for c in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
        cols[c] = np.repeat(pd.Categorical(panel.meta[c]).codes, width)
    X = pd.DataFrame(cols)
    series_idx = np.repeat(np.arange(N), width)
    day_idx = np.tile(np.arange(t_start, t_end), N)
    return X, series_idx, day_idx


def target_window(panel: Panel, series_idx: np.ndarray, day_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Targets and a mask of usable training rows (observed and not a likely stockout)."""
    y = panel.y[series_idx, day_idx]
    usable = ~np.isnan(y) & ~panel.stockout[series_idx, day_idx]
    return y, usable
