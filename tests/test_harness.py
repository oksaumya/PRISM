"""The backtest harness checks itself: a zero model, broken forecasts, bad configs, edge cases."""

import numpy as np
import pytest

from prism.backtest import check_forecast, evaluation_frame, run_backtest
from prism.models import Forecast, get_model
from prism.stats_tests import diebold_mariano, series_wilcoxon


def test_zero_model_runs_end_to_end_with_known_scores(panel, cfg):
    ev = evaluation_frame(run_backtest(panel, cfg, ["zero"]), panel, cfg)
    from prism.metrics import summarize
    table = summarize(ev)
    assert table.loc[0, "WAPE"] == pytest.approx(1.0)                  # sum|y - 0| / sum y
    assert table.loc[0, "MAE"] == pytest.approx(ev["y"].mean())
    assert table.loc[0, "Cov90"] == pytest.approx((ev["y"] == 0).mean())   # [0, 0] only covers zeros


def _fc(n, h, q, fill=1.0):
    return Forecast(point=np.full((n, h), fill), quantiles=np.full((n, h, q), fill), levels=np.linspace(0.1, 0.9, q))


def test_harness_rejects_broken_forecasts(panel, cfg):
    info, levels = panel.truncate(392), np.asarray(cfg["quantiles"])
    N, Q = panel.n_series, len(levels)
    check_forecast(_fc(N, 28, Q), info, 392, 28, levels, "ok")             # a valid forecast passes
    with pytest.raises(ValueError, match="shapes"):
        check_forecast(_fc(N - 1, 28, Q), info, 392, 28, levels, "short")
    with pytest.raises(ValueError, match="NaN"):
        check_forecast(_fc(N, 28, Q, fill=np.nan), info, 392, 28, levels, "nan")
    with pytest.raises(ValueError, match="truncated"):
        check_forecast(_fc(N, 28, Q), panel.truncate(380), 392, 28, levels, "leak")


def test_unknown_model_name_is_a_clear_error():
    with pytest.raises(ValueError, match="available"):
        get_model("not_a_model")


def test_lightgbm_rejects_leaky_or_incomplete_variants():
    leaky = get_model("lightgbm", variants={"lag7": {"min_lag": 7, "horizons": [1, 28]}})
    with pytest.raises(ValueError, match="leak"):
        leaky.check_variants(28)
    gap = get_model("lightgbm", variants={"lag7": {"min_lag": 7, "horizons": [1, 7]}})
    with pytest.raises(ValueError, match="cover"):
        gap.check_variants(28)


def test_dm_test_edge_cases():
    rng = np.random.default_rng(0)
    a = rng.normal(1.0, 0.1, 200)
    assert diebold_mariano(a, a) == (0.0, 1.0)                           # identical losses
    stat, p = diebold_mariano(a - 0.2, a + rng.normal(0, 0.05, 200))     # A clearly better
    assert stat < 0 and p < 1e-6
    with pytest.raises(ValueError):
        diebold_mariano(a[:5], a[:5], lag=7)


def test_wilcoxon_identical_models():
    import pandas as pd
    df = pd.DataFrame({"model": ["a"] * 3 + ["b"] * 3, "series": [0, 1, 2] * 2, "pinball": [1.0, 2.0, 3.0] * 2})
    assert series_wilcoxon(df, "a", "b")["p_value"] == 1.0
