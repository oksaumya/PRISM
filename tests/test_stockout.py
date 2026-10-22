import numpy as np

from prism.stockout import flag_series, flag_stockouts, flags_at_cutoff, zero_streak

P = dict(p_threshold=1e-3, window=56, min_obs=14)


def test_long_gap_in_busy_series_is_flagged():
    rng = np.random.default_rng(0)
    y = rng.poisson(6, 200).astype(float)
    y[120:130] = 0
    assert flag_series(y, **P)[120:130].all()


def test_gap_in_intermittent_series_is_not_flagged():
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.2).astype(float)     # sells on ~20% of days
    y[120:130] = 0                                # 10 zeros are unremarkable here
    assert not flag_series(y, **P)[120:130].any()


def test_cutoff_flags_equal_flags_recomputed_from_truncated_data():
    """Flags at an origin may only use data up to that origin."""
    rng = np.random.default_rng(1)
    Y = rng.poisson(rng.uniform(0.1, 5, (40, 1))[:, [0] * 300]).astype(float)
    Y[::3, 200:260] = 0                            # runs crossing the cutoff
    Y[1, :50] = np.nan                             # not-yet-launched days
    full = flag_stockouts(Y, **P)
    for cutoff in [150, 230, 299]:
        np.testing.assert_array_equal(flags_at_cutoff(Y, full, cutoff, **P), flag_stockouts(Y[:, :cutoff], **P))


def test_zero_streak():
    y = np.array([[1.0, 0, 0, 2, 0, np.nan]])
    np.testing.assert_array_equal(zero_streak(y)[0, :5], [0, 1, 2, 0, 1])
    assert np.isnan(zero_streak(y)[0, 5])


def test_data_quality_on_synthetic_panel(panel):
    from prism.data import data_quality
    dq = data_quality(panel).set_index("metric")["value"]
    assert 0 < dq["share of observed days flagged as likely stockout"] < 0.5
    assert dq["share of series-days excluded from training and masked scoring"] >= \
        dq["share of series-days not observed (pre-launch or hidden cold-start history)"]
