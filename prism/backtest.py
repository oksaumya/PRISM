"""Rolling-origin backtest (Tashman, 2000).

At each origin the panel is truncated (no data after the cutoff exists for the
model), every model is refit from scratch and forecasts the same 28 days for
the same series. All models are therefore scored on identical
(series, origin, horizon) cells.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import forecast_origins
from .data import Panel
from .metrics import add_losses, naive_scales, qcol
from .models import Forecast, get_model

log = logging.getLogger(__name__)


def forecast_frame(fc: Forecast, model: str, cutoff: int) -> pd.DataFrame:
    N, H = fc.point.shape
    df = pd.DataFrame({
        "model": model,
        "origin": cutoff,
        "series": np.repeat(np.arange(N), H),
        "h": np.tile(np.arange(1, H + 1), N),
        "point": fc.point.reshape(-1).astype(np.float32),
    })
    df["day"] = df["origin"] + df["h"] - 1
    for j, a in enumerate(fc.levels):
        df[qcol(a)] = fc.quantiles[:, :, j].reshape(-1).astype(np.float32)
    return df


def run_backtest(panel: Panel, cfg: dict, models: list[str], cache_dir: Path | None = None) -> pd.DataFrame:
    """Forecasts of every model at every origin as one long frame (cached per model/origin)."""
    H = cfg["backtest"]["horizon"]
    levels = np.asarray(cfg["quantiles"], dtype=float)
    frames, timings = [], []
    for cutoff in forecast_origins(cfg):
        info = panel.truncate(cutoff)
        for name in models:
            path = cache_dir / f"{name}_{cutoff}.parquet" if cache_dir else None
            if path is not None and path.exists():
                cached = pd.read_parquet(path)
                frames.append(cached)
                log.info("origin %d  %-16s cached", cutoff, name)
                continue
            model = get_model(name, seed=cfg["seed"], **cfg["models"].get(name, {}))
            t = time.perf_counter()
            fc = model.fit_predict(info, H, levels)
            secs = time.perf_counter() - t
            out = [forecast_frame(fc, name, cutoff)]
            out += [forecast_frame(f, extra, cutoff) for extra, f in getattr(model, "extras", {}).items()]
            out = pd.concat(out, ignore_index=True)
            out["fit_seconds"] = secs
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                out.to_parquet(path)
            frames.append(out)
            log.info("origin %d  %-16s %.1fs", cutoff, name, secs)
    return pd.concat(frames, ignore_index=True)


def horizon_bucket(h: pd.Series) -> pd.Categorical:
    return pd.cut(h, bins=[0, 7, 14, 28], labels=["h1-7", "h8-14", "h15-28"])


def evaluation_frame(forecasts: pd.DataFrame, panel: Panel, cfg: dict) -> pd.DataFrame:
    """Join forecasts with actuals, segments and scales; keep scorable cells only."""
    levels = cfg["quantiles"]
    df = forecasts.copy()
    s, d = df["series"].to_numpy(), df["day"].to_numpy()
    df["y"] = panel.y[s, d]
    df["stockout"] = panel.stockout[s, d]                    # ex-post flag of the target day
    df["history_days"] = np.clip(df["origin"].to_numpy() - panel.launch[s], 0, None)

    ccfg = cfg["cold_start"]
    df["cold_start"] = df["history_days"] < ccfg["max_history_days"]
    df = df[(df["history_days"] >= ccfg["min_history_days"]) & df["y"].notna()].copy()

    # per (series, origin) naive scales from the history available at the origin
    scales = []
    for cutoff in df["origin"].unique():
        sq, ab = naive_scales(panel.y[:, :cutoff])
        scales.append(pd.DataFrame({"series": np.arange(panel.n_series), "origin": cutoff,
                                    "scale_sq": sq, "scale_abs": ab}))
    df = df.merge(pd.concat(scales), on=["series", "origin"], how="left")

    meta_cols = ["store_id", "cat_id", "dept_id", "demand_class", "sim_cold_start"]
    df = df.join(panel.meta[meta_cols], on="series")
    df["horizon_bucket"] = horizon_bucket(df["h"])
    return add_losses(df, levels)
