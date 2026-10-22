"""Demand classification and stratified series sampling."""

from __future__ import annotations

import numpy as np
import pandas as pd

DEMAND_CLASSES = ["smooth", "erratic", "intermittent", "lumpy"]


def demand_class(y: np.ndarray, adi_cutoff: float = 1.32, cv2_cutoff: float = 0.49) -> tuple[str, float, float]:
    """Syntetos-Boylan (2005) class of one series from its observed history.

    ADI  = average inter-demand interval  = #days / #non-zero days
    CV^2 = squared coefficient of variation of the non-zero demand sizes
    """
    obs = y[~np.isnan(y)]
    nz = obs[obs > 0]
    if nz.size < 2:
        return "lumpy", np.inf, np.nan
    adi = obs.size / nz.size
    cv2 = (nz.std() / nz.mean()) ** 2
    smooth_timing = adi < adi_cutoff
    stable_size = cv2 < cv2_cutoff
    if smooth_timing:
        return ("smooth" if stable_size else "erratic"), adi, cv2
    return ("intermittent" if stable_size else "lumpy"), adi, cv2


def balanced_allocation(sizes: pd.Series, total: int) -> pd.Series:
    """Spread ``total`` draws as evenly as possible over strata of the given sizes.

    Strata smaller than their fair share are taken whole; the remainder is
    re-split over the larger strata (water-filling).
    """
    alloc = pd.Series(0, index=sizes.index)
    remaining, open_ = total, list(sizes.index)
    while remaining > 0 and open_:
        share = remaining // len(open_)
        if share == 0:                              # hand out the last few one by one
            for k in sorted(open_, key=lambda k: -(sizes[k] - alloc[k]))[:remaining]:
                alloc[k] += 1
            break
        for k in list(open_):
            take = min(share, sizes[k] - alloc[k])
            alloc[k] += take
            remaining -= take
            if alloc[k] >= sizes[k]:
                open_.remove(k)
    return alloc


def stratified_sample(meta: pd.DataFrame, n_per_store: int, strata: list[str], seed: int) -> pd.Index:
    """Sample ``n_per_store`` series per store, balanced over the given strata."""
    rng = np.random.default_rng(seed)
    chosen = []
    for _, store_df in meta.groupby("store_id", sort=True):
        groups = store_df.groupby(strata).groups
        sizes = pd.Series({k: len(v) for k, v in groups.items()})
        alloc = balanced_allocation(sizes, min(n_per_store, len(store_df)))
        for k, n in alloc.items():
            if n > 0:
                chosen.extend(rng.choice(np.asarray(groups[k]), size=n, replace=False))
    return pd.Index(sorted(chosen))
