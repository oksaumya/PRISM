"""Model diagnostics: quantile crossing, LightGBM feature importance, Kalman fallbacks."""

import copy

import numpy as np

from prism.backtest import load_diagnostics, run_backtest
from prism.models import Forecast, get_model

LGBM = {
    "train_window": 200,
    "variants": {"lag28": {"min_lag": 28, "horizons": [8, 28]}, "lag7": {"min_lag": 7, "horizons": [1, 7]}},
    "point_params": {"objective": "tweedie", "num_leaves": 7, "min_data_in_leaf": 20, "verbose": -1},
    "point_rounds": 30,
    "early_stopping_valid_days": 28,
    "quantile_params": {"num_leaves": 7, "min_data_in_leaf": 20, "verbose": -1},
    "quantile_rounds": 20,
}


def test_forecast_records_crossing_before_sorting():
    q = np.array([[[1.0, 3.0, 2.0], [0.0, 1.0, 2.0]]])        # first cell crosses by 1 unit, second not
    fc = Forecast(point=np.ones((1, 2)), quantiles=q, levels=np.array([0.1, 0.5, 0.9]))
    np.testing.assert_allclose(fc.cross_size, [[1.0, 0.0]])
    assert (np.diff(fc.quantiles, axis=-1) >= 0).all()           # and the stored quantiles are sorted


def test_lightgbm_forecasts_and_feature_importance(panel, cfg):
    model = get_model("lightgbm", seed=0, **LGBM)
    fc = model.fit_predict(panel.truncate(392), 28, np.asarray(cfg["quantiles"]))
    assert fc.point.shape == (panel.n_series, 28) and np.isfinite(fc.quantiles).all()
    assert "lightgbm_lag28" in model.extras
    imp = model.diagnostics["feature_importance"]
    assert set(imp["variant"]) == {"lag7", "lag28"}
    for _, g in imp.groupby("variant"):
        assert abs(g["gain_share"].sum() - 1) < 1e-9


def test_kalman_reports_fallbacks_and_cache_keeps_them(panel, cfg, tmp_path):
    run_backtest(panel, cfg, ["kalman"], cache_dir=tmp_path)
    fb = load_diagnostics(cfg, tmp_path, "kalman", "fallbacks")
    assert len(fb) == panel.n_series * cfg["backtest"]["n_origins"]
    # the cold-start series (40 days of history at most) is too short for the MLE
    assert fb.loc[fb["series"] == panel.n_series - 1, "default_ratios"].all()
    assert not fb.loc[fb["series"] == 0, "default_ratios"].any()
    # a different config does not see these diagnostics
    cfg2 = copy.deepcopy(cfg)
    cfg2["models"]["kalman"]["refine"] = True
    assert load_diagnostics(cfg2, tmp_path, "kalman", "fallbacks").empty
