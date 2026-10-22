import numpy as np

from prism.backtest import evaluation_frame, run_backtest
from prism.config import forecast_origins
from prism.metrics import summarize

MODELS = ["seasonal_naive", "tsb", "kalman"]


def test_origins_end_at_last_day(cfg):
    origins = forecast_origins(cfg)
    assert origins[-1] + cfg["backtest"]["horizon"] == cfg["backtest"]["last_day"]
    assert np.all(np.diff(origins) == cfg["backtest"]["step"])


def test_all_models_scored_on_identical_cells(panel, cfg):
    fc = run_backtest(panel, cfg, MODELS)
    ev = evaluation_frame(fc, panel, cfg)
    cells = {m: set(map(tuple, g[["series", "origin", "h"]].to_numpy())) for m, g in ev.groupby("model")}
    assert len({frozenset(c) for c in cells.values()}) == 1
    q = ev[[c for c in ev.columns if c.startswith("q_")]].to_numpy()
    assert np.isfinite(q).all() and (np.diff(q, axis=1) >= 0).all()    # valid, non-crossing quantiles
    table = summarize(ev)
    assert set(table["model"]) == set(MODELS)
    assert (table["Cov90"] > 0.5).all()


def test_cold_start_series_gets_scored(panel, cfg):
    ev = evaluation_frame(run_backtest(panel, cfg, ["seasonal_naive"]), panel, cfg)
    assert ev.loc[ev["series"] == panel.n_series - 1, "cold_start"].all()
