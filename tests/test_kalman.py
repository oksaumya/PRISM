"""The from-scratch Kalman filter must agree with statsmodels' state-space implementation."""

import numpy as np
import pytest
import statsmodels.api as sm

from prism.models.kalman import forecast_from_state, fit_ratios, kalman_filter, loglik, profile_loglik

SIG_EPS, SIG_LEVEL, SIG_SEASON = 0.09, 0.01, 0.001


@pytest.fixture
def series():
    rng = np.random.default_rng(0)
    n = 300
    pattern = np.tile([0.5, -0.2, 0.1, 0.0, 0.3, -0.4, -0.3], n)[:n]
    return np.cumsum(rng.normal(0, 0.1, n)) + pattern + rng.normal(0, 0.3, n)


def _statsmodels(z):
    mod = sm.tsa.UnobservedComponents(z, level="llevel", seasonal=7)
    mod.initialize_approximate_diffuse(1e6)
    return mod.filter([SIG_EPS, SIG_LEVEL, SIG_SEASON])


def _ours(z):
    return kalman_filter(z[None], np.array([SIG_LEVEL, SIG_SEASON]), SIG_EPS, 1e6, burn=7, return_path=True)


@pytest.mark.parametrize("missing", [False, True])
def test_matches_statsmodels(series, missing):
    z = series.copy()
    if missing:
        z[100:120] = np.nan                     # e.g. a stockout: predict-only steps
    ref, ours = _statsmodels(z), _ours(z)
    assert loglik(ours)[0] == pytest.approx(ref.llf, abs=1e-6)
    np.testing.assert_allclose(ours["pred_mean"][0, 10:], ref.forecasts[0, 10:], atol=1e-6)
    np.testing.assert_allclose(ours["pred_var"][0, 10:], ref.forecasts_error_cov[0, 0, 10:], atol=1e-6)


def test_multistep_forecast_matches_statsmodels(series):
    ref, ours = _statsmodels(series), _ours(series)
    fc = ref.get_forecast(14)
    q = np.array([[SIG_LEVEL, SIG_SEASON]])
    # unscaled system: pass s2 = 1 and subtract the unit observation term, add sigma_eps^2
    m, v = forecast_from_state(ours["a"], ours["P"], q, np.array([1.0]), 14)
    np.testing.assert_allclose(m[0], fc.predicted_mean, atol=1e-6)
    np.testing.assert_allclose(v[0] - 1 + SIG_EPS, fc.var_pred_mean, atol=1e-6)


def test_profile_likelihood_recovers_ratios(series):
    """Grid MLE of signal-to-noise ratios should be near the truth on simulated data."""
    rng = np.random.default_rng(3)
    n, q_true = 2000, np.array([0.05, 0.002])
    level = np.cumsum(rng.normal(0, np.sqrt(q_true[0]), n))
    season = np.zeros(n + 7)
    season[:6] = rng.normal(0, 0.5, 6)
    for t in range(6, n + 6):
        season[t] = -season[t - 6:t].sum() + rng.normal(0, np.sqrt(q_true[1]))
    z = level + season[6:n + 6] + rng.normal(0, 1.0, n)
    cfg = {"q_level_grid": [-4, 0, 17], "q_season_grid": [-5, -1, 17], "refine": True}
    q_hat, _ = fit_ratios(z[None], cfg)
    assert abs(np.log10(q_hat[0, 0]) - np.log10(q_true[0])) < 0.5
    _, s2 = profile_loglik(kalman_filter(z[None], q_hat[0]))
    assert s2[0] == pytest.approx(1.0, rel=0.15)
