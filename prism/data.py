"""Data loading and the :class:`Panel` container used by every model.

Raw M5 data is wide: one row per series, one column per day. We keep that
layout as NumPy matrices, which makes lags, rolling windows and a batched
Kalman filter simple and fast:

* ``y``      (N, D)   unit sales, NaN where the series is not yet launched
* ``price``  (N, Dc)  sell price (known ahead of time), Dc = calendar length
* ``stockout`` (N, D) likely-stockout flags (see :mod:`prism.stockout`)

``Panel.truncate(cutoff)`` returns the information set at a forecast origin:
sales after the cutoff are physically removed, so a model cannot peek.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import forecast_origins, panel_fingerprint, resolve
from .sampling import demand_class, stratified_sample
from .stockout import flag_stockouts, flags_at_cutoff

log = logging.getLogger(__name__)


@dataclass
class Panel:
    y: np.ndarray                 # (N, D) observed sales (NaN = not launched / unobserved)
    y_raw: np.ndarray             # (N, D) raw M5 sales (only used for leakage-safe aggregates)
    stockout: np.ndarray          # (N, D) likely-stockout flags
    price: np.ndarray             # (N, Dc) sell price, known ahead
    item_other: np.ndarray        # (N, Dc) mean sales of the same item in the *other* 9 stores
    dept_other: np.ndarray        # (N, Dc) mean sales per item of the same dept & store, own item excluded
    launch: np.ndarray            # (N,) first observed day index
    meta: pd.DataFrame            # N rows: id, item_id, dept_id, cat_id, store_id, state_id, ...
    calendar: pd.DataFrame        # Dc rows
    cutoff: int | None = None     # number of observed days (None = full data)
    stockout_params: dict | None = None

    @property
    def n_series(self) -> int:
        return self.y.shape[0]

    @property
    def n_days(self) -> int:
        return self.y.shape[1]

    def history_days(self, cutoff: int) -> np.ndarray:
        """Days of history each series has at a cutoff."""
        return np.clip(cutoff - self.launch, 0, None)

    def truncate(self, cutoff: int) -> "Panel":
        """Information set at a forecast origin: sales up to ``cutoff`` only.

        Exogenous inputs that a retailer knows in advance (calendar, planned
        prices) stay complete. Cross-store aggregates are cut too, because
        they are sales data.
        """
        p = self.stockout_params or {}
        stock = flags_at_cutoff(self.y, self.stockout, cutoff, **p)
        cut = lambda a: a[:, :cutoff].copy()
        return replace(
            self, y=cut(self.y), y_raw=cut(self.y_raw), stockout=stock,
            item_other=cut(self.item_other), dept_other=cut(self.dept_other), cutoff=cutoff,
        )


# --------------------------------------------------------------------------- raw loading

ID_COLS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def load_raw(raw_dir: Path) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Return (meta for all 30,490 series, sales matrix, calendar, sell_prices)."""
    sales = pd.read_csv(raw_dir / "sales_train_evaluation.csv")
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    Y = sales[day_cols].to_numpy(dtype=np.float32)
    meta = sales[ID_COLS].copy()
    calendar = pd.read_csv(raw_dir / "calendar.csv", parse_dates=["date"])
    prices = pd.read_csv(raw_dir / "sell_prices.csv")
    return meta, Y, calendar, prices


def price_matrix(meta: pd.DataFrame, prices: pd.DataFrame, calendar: pd.DataFrame) -> np.ndarray:
    """Expand weekly prices to a (N, n_calendar_days) daily matrix (NaN = not on sale)."""
    weeks = np.sort(calendar["wm_yr_wk"].unique())
    p = prices.merge(meta[["store_id", "item_id"]].reset_index(), on=["store_id", "item_id"])
    wide = p.pivot_table(index="index", columns="wm_yr_wk", values="sell_price")
    wide = wide.reindex(index=range(len(meta)), columns=weeks)
    week_pos = np.searchsorted(weeks, calendar["wm_yr_wk"].to_numpy())
    return wide.to_numpy(dtype=np.float32)[:, week_pos]


def leakage_safe_aggregates(meta_all: pd.DataFrame, Y_all: np.ndarray, rows: np.ndarray,
                            n_cal: int) -> tuple[np.ndarray, np.ndarray]:
    """Cross-sectional context features computed from the *full* M5 data.

    * item_other: mean daily sales of the same item in the other 9 stores
    * dept_other: mean daily sales per item of the same dept in the same store,
      excluding the item itself

    The series' own sales are always subtracted, so hidden history of a
    simulated cold-start series cannot leak through an aggregate.
    """
    own = Y_all[rows]
    item_key = meta_all["item_id"].to_numpy()
    ds_key = (meta_all["dept_id"] + "|" + meta_all["store_id"]).to_numpy()

    def group_sum(keys):
        codes, uniq = pd.factorize(keys)
        sums = np.zeros((len(uniq), Y_all.shape[1]), dtype=np.float64)
        np.add.at(sums, codes, Y_all)
        counts = np.bincount(codes)
        return sums[codes[rows]], counts[codes[rows]]

    item_sum, item_n = group_sum(item_key)
    ds_sum, ds_n = group_sum(ds_key)
    item_other = (item_sum - own) / np.maximum(item_n - 1, 1)[:, None]
    dept_other = (ds_sum - own) / np.maximum(ds_n - 1, 1)[:, None]

    pad = lambda a: np.hstack([a, np.full((a.shape[0], n_cal - a.shape[1]), np.nan)]).astype(np.float32)
    return pad(item_other), pad(dept_other)


# --------------------------------------------------------------------------- preparation

def first_sale(Y: np.ndarray) -> np.ndarray:
    """Index of the first non-zero sale (M5 convention for when a series is live)."""
    nz = Y > 0
    return np.where(nz.any(axis=1), nz.argmax(axis=1), Y.shape[1])


def prepare_panel(cfg: dict) -> Panel:
    """Build the experiment panel from raw M5 files (sampling, flags, cold start)."""
    dcfg, scfg = cfg["data"], cfg["stockout"]
    rng = np.random.default_rng(cfg["seed"])
    meta_all, Y_all, calendar, prices = load_raw(resolve(dcfg["raw_dir"]))
    first_origin = forecast_origins(cfg)[0]

    # 1. candidate series and their demand class, judged on pre-backtest history only
    cand = meta_all[meta_all["store_id"].isin(dcfg["stores"])].copy()
    launch_all = first_sale(Y_all)
    classes = []
    for i in cand.index:
        hist = Y_all[i, launch_all[i]:first_origin]
        classes.append(demand_class(hist, **cfg["demand_class"]) if hist.size else ("new", np.nan, np.nan))
    cand[["demand_class", "adi", "cv2"]] = pd.DataFrame(classes, index=cand.index)

    # 2. stratified sample: balanced over category x demand class within each store
    rows = stratified_sample(cand, dcfg["items_per_store"], ["cat_id", "demand_class"], cfg["seed"]).to_numpy()
    meta = cand.loc[rows].reset_index(drop=True)
    meta["m5_row"] = rows
    Y_raw = Y_all[rows]
    launch = launch_all[rows].copy()

    # 3. simulated cold start: some long-established series "launch" shortly before the first origin
    ccfg = cfg["cold_start"]
    eligible = np.flatnonzero(launch < first_origin - 400)
    n_sim = int(round(ccfg["simulate_fraction"] * len(meta)))
    sim = rng.choice(eligible, size=min(n_sim, eligible.size), replace=False)
    hist_len = rng.choice(ccfg["simulate_history_days"], size=sim.size)
    meta["sim_cold_start"] = False
    meta.loc[sim, "sim_cold_start"] = True
    meta["true_launch"] = launch.copy()
    launch[sim] = first_origin - hist_len

    # 4. observed sales: NaN before launch
    Y = Y_raw.copy()
    Y[np.arange(Y.shape[1])[None, :] < launch[:, None]] = np.nan
    meta["launch"] = launch

    # 5. ex-post stockout flags on the observed data
    stock_params = dict(p_threshold=scfg["p_threshold"], window=scfg["window"], min_obs=scfg["min_obs"])
    stockout = flag_stockouts(Y, **stock_params)

    # 6. exogenous inputs
    price = price_matrix(meta, prices, calendar)
    item_other, dept_other = leakage_safe_aggregates(meta_all, Y_all, rows, len(calendar))

    return Panel(y=Y, y_raw=Y_raw, stockout=stockout, price=price, item_other=item_other,
                 dept_other=dept_other, launch=launch, meta=meta, calendar=calendar,
                 stockout_params=stock_params)


# --------------------------------------------------------------------------- caching

ARRAYS = ["y", "y_raw", "stockout", "price", "item_other", "dept_other", "launch"]


def save_panel(panel: Panel, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "panel.npz", **{k: getattr(panel, k) for k in ARRAYS})
    panel.meta.to_parquet(out_dir / "meta.parquet")
    panel.calendar.to_parquet(out_dir / "calendar.parquet")
    (out_dir / "stockout_params.json").write_text(json.dumps(panel.stockout_params))


def load_panel(cfg: dict) -> Panel:
    """Load the cached panel, building it from raw data on first use.

    The cache is rebuilt whenever a setting that shapes the panel (sample size,
    seed, stockout rule, cold start, origins) differs from the one it was built with.
    """
    out_dir = resolve(cfg["data"]["processed_dir"])
    stamp = out_dir / "fingerprint.txt"
    fp = panel_fingerprint(cfg)
    if not (out_dir / "panel.npz").exists() or not stamp.exists() or stamp.read_text() != fp:
        log.info("building panel from raw data (config fingerprint %s)", fp)
        save_panel(prepare_panel(cfg), out_dir)
        stamp.write_text(fp)
    arrays = np.load(out_dir / "panel.npz")
    return Panel(
        **{k: arrays[k] for k in ARRAYS},
        meta=pd.read_parquet(out_dir / "meta.parquet"),
        calendar=pd.read_parquet(out_dir / "calendar.parquet"),
        stockout_params=json.loads((out_dir / "stockout_params.json").read_text()),
    )
