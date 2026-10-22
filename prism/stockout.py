"""Stockout *marking* (never imputation).

M5 has no inventory data, so a stockout must be inferred from the sales
signal. We flag a run of ``L`` consecutive zero-sales days as a likely
stockout when it is implausible under the item's own recent behaviour:

    pi0 = (#zero days + 1) / (#days + 2)      # Laplace-smoothed P(zero day),
                                               # trailing in-stock window
    P(run of L zeros) ~= pi0 ** L             # independence approximation
    flag the run  <=>  pi0 ** L < p_threshold

The trailing window only contains days *before* the run that were not
themselves flagged, so the rule never looks ahead of the run start. Flags are
used to (1) drop suspicious zeros from model training, (2) treat them as
missing observations in the Kalman filter and (3) report "stockout-masked"
metrics next to raw ones.
"""

from __future__ import annotations

import numpy as np


def _zero_runs(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start (inclusive) and end (exclusive) indices of runs where y == 0."""
    is_zero = np.concatenate([[False], y == 0, [False]])
    edges = np.flatnonzero(np.diff(is_zero.astype(np.int8)))
    return edges[0::2], edges[1::2]


def _run_is_stockout(hist: np.ndarray, run_len: int, p_threshold: float, min_obs: int) -> bool:
    if hist.size < min_obs:
        return False
    pi0 = (np.count_nonzero(hist == 0) + 1.0) / (hist.size + 2.0)
    return run_len * np.log(pi0) < np.log(p_threshold)


def flag_series(y: np.ndarray, p_threshold: float, window: int, min_obs: int) -> np.ndarray:
    """Flag likely-stockout days of one series. NaN marks unobserved days."""
    flags = np.zeros(y.shape, dtype=bool)
    starts, ends = _zero_runs(y)
    for s, e in zip(starts, ends):
        lo = max(0, s - 4 * window)              # look back far enough to find `window` usable days
        past = y[lo:s]
        usable = past[~np.isnan(past) & ~flags[lo:s]][-window:]
        if _run_is_stockout(usable, e - s, p_threshold, min_obs):
            flags[s:e] = True
    return flags


def flag_stockouts(Y: np.ndarray, p_threshold: float = 1e-3, window: int = 56,
                   min_obs: int = 14) -> np.ndarray:
    """Flag likely-stockout days for a (n_series, n_days) sales matrix (ex-post, full data)."""
    return np.vstack([flag_series(row, p_threshold, window, min_obs) for row in Y])


def flags_at_cutoff(Y: np.ndarray, full_flags: np.ndarray, cutoff: int,
                    p_threshold: float = 1e-3, window: int = 56, min_obs: int = 14) -> np.ndarray:
    """Flags as they would have been computed with data up to ``cutoff`` only.

    Runs that end before the cutoff are judged identically with or without the
    future (the rule is backward looking), so only a zero-run still in
    progress at the cutoff has to be re-judged with its truncated length.
    """
    flags = full_flags[:, :cutoff].copy()
    for i in np.flatnonzero(Y[:, cutoff - 1] == 0):
        y = Y[i, :cutoff]
        nz = np.flatnonzero(y != 0)                # NaN != 0 is True: unobserved days end runs
        s = nz[-1] + 1 if nz.size else 0
        lo = max(0, s - 4 * window)
        past = y[lo:s]
        usable = past[~np.isnan(past) & ~flags[i, lo:s]][-window:]
        flags[i, s:cutoff] = _run_is_stockout(usable, cutoff - s, p_threshold, min_obs)
    return flags


def zero_streak(Y: np.ndarray) -> np.ndarray:
    """Length of the zero-sales streak ending at each day (causal feature)."""
    out = np.zeros(Y.shape, dtype=np.float32)
    run = np.zeros(Y.shape[0], dtype=np.float32)
    for t in range(Y.shape[1]):
        run = np.where(Y[:, t] == 0, run + 1, 0)
        out[:, t] = run
    out[np.isnan(Y)] = np.nan
    return out
