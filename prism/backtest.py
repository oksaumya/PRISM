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

from .config import fingerprint, forecast_origins, panel_fingerprint
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
        "cross_size": fc.cross_size.reshape(-1).astype(np.float32),
    })
    df["day"] = df["origin"] + df["h"] - 1
    for j, a in enumerate(fc.levels):
        df[qcol(a)] = fc.quantiles[:, :, j].reshape(-1).astype(np.float32)
    return df


def check_forecast(fc: Forecast, info: Panel, cutoff: int, horizon: int, levels: np.ndarray, model: str) -> None:
    """The harness's own checks, so a broken model fails loudly instead of skewing the tables.

    1. no leak: the model only received data up to the origin
    2. shape: a forecast for every series, every horizon and every quantile level
    3. valid numbers: finite, non-negative and non-crossing quantiles
    """
    if info.y.shape[1] != cutoff or info.cutoff != cutoff:
        raise ValueError(f"{model}: the panel given to the model is not truncated at the origin {cutoff}")
    N, Q = info.n_series, len(levels)
    if fc.point.shape != (N, horizon) or fc.quantiles.shape != (N, horizon, Q):
        raise ValueError(f"{model}: forecast shapes {fc.point.shape}, {fc.quantiles.shape}; "
                         f"expected ({N}, {horizon}) and ({N}, {horizon}, {Q})")
    bad = ~np.isfinite(fc.point).all(axis=1) | ~np.isfinite(fc.quantiles).all(axis=(1, 2))
    if bad.any():
        raise ValueError(f"{model}: NaN or infinite forecasts for {bad.sum()} series at origin {cutoff}")
    if (fc.quantiles < 0).any() or (np.diff(fc.quantiles, axis=-1) < 0).any():
        raise ValueError(f"{model}: negative or crossing quantiles at origin {cutoff}")


# Bump when the content of cached files changes (new columns, new diagnostics), so old
# caches are not reused. 2: model diagnostics; 3: "cross_size" column.
CACHE_VERSION = 3


def cache_key(cfg: dict, model: str) -> str:
    """Everything a cached forecast depends on: the cache format, the panel, the model's
    settings, the horizon, the quantile levels and the seed."""
    return fingerprint(CACHE_VERSION, panel_fingerprint(cfg), model, cfg["models"].get(model, {}),
                       cfg["backtest"]["horizon"], cfg["quantiles"], cfg["seed"])


def run_backtest(panel: Panel, cfg: dict, models: list[str], cache_dir: Path | None = None) -> pd.DataFrame:
    """Forecasts of every model at every origin as one long frame (cached per model/origin)."""
    H = cfg["backtest"]["horizon"]
    levels = np.asarray(cfg["quantiles"], dtype=float)
    frames = []
    for cutoff in forecast_origins(cfg):
        info = panel.truncate(cutoff)
        for name in models:
            path = cache_dir / f"{name}_{cutoff}_{cache_key(cfg, name)}.parquet" if cache_dir else None
            if path is not None and path.exists():
                cached = pd.read_parquet(path)
                frames.append(cached)
                log.info("origin %d  %-16s cached", cutoff, name)
                continue
            model = get_model(name, seed=cfg["seed"], **cfg["models"].get(name, {}))
            t = time.perf_counter()
            fc = model.fit_predict(info, H, levels)
            secs = time.perf_counter() - t
            extras = getattr(model, "extras", {})
            for label, f in [(name, fc), *extras.items()]:
                check_forecast(f, info, cutoff, H, levels, label)
            out = [forecast_frame(fc, name, cutoff)]
            out += [forecast_frame(f, extra, cutoff) for extra, f in extras.items()]
            out = pd.concat(out, ignore_index=True)
            out["fit_seconds"] = secs
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                out.to_parquet(path)
                for table, diag in model.diagnostics.items():       # e.g. feature importance
                    diag.to_parquet(diagnostics_path(path, table))
            frames.append(out)
            log.info("origin %d  %-16s %.1fs", cutoff, name, secs)
    return pd.concat(frames, ignore_index=True)


def diagnostics_path(forecast_path: Path, table: str) -> Path:
    return forecast_path.with_name(f"{forecast_path.stem}__{table}.parquet")


def load_diagnostics(cfg: dict, cache_dir: Path, model: str, table: str) -> pd.DataFrame:
    """A model's diagnostics table for every origin, read from the cache of the current config."""
    frames = []
    for cutoff in forecast_origins(cfg):
        path = diagnostics_path(cache_dir / f"{model}_{cutoff}_{cache_key(cfg, model)}.parquet", table)
        if path.exists():
            frames.append(pd.read_parquet(path).assign(origin=cutoff))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
