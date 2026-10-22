import numpy as np
import pandas as pd
import pytest

from prism.metrics import add_losses, naive_scales, pinball_loss, summarize


def test_pinball_hand_computed():
    y = np.array([10.0, 10.0])
    q = np.array([8.0, 13.0])
    # under-forecast by 2 at level 0.9 -> 0.9*2 ; over-forecast by 3 -> 0.1*3
    assert np.allclose(pinball_loss(y, q, 0.9), [1.8, 0.3])
    # at the median the pinball loss is half the absolute error
    assert np.allclose(pinball_loss(y, q, 0.5), 0.5 * np.abs(y - q))


def test_pinball_is_minimised_by_the_true_quantile():
    rng = np.random.default_rng(0)
    y = rng.gamma(2.0, 3.0, 20000)
    grid = np.linspace(0, 20, 401)
    losses = [pinball_loss(y, g, 0.8).mean() for g in grid]
    assert abs(grid[int(np.argmin(losses))] - np.quantile(y, 0.8)) < 0.15


def _frame():
    levels = [0.1, 0.5, 0.9]
    df = pd.DataFrame({
        "model": "m", "series": [0, 0, 1, 1], "origin": 100, "day": [100, 101, 100, 101],
        "y": [2.0, 4.0, 0.0, 6.0], "point": [3.0, 3.0, 1.0, 5.0],
        "q_0.10": [1.0, 1.0, 0.0, 2.0], "q_0.50": [2.0, 3.0, 1.0, 5.0], "q_0.90": [3.0, 5.0, 2.0, 5.5],
        "scale_sq": [1.0, 1.0, 4.0, 4.0], "scale_abs": [1.0, 1.0, 2.0, 2.0],
    })
    return add_losses(df, levels)


def test_point_metrics():
    m = summarize(_frame()).iloc[0]
    abs_err = np.array([0, 1, 1, 1])            # vs the median
    sq_err = np.array([1, 1, 1, 1])             # vs the mean
    assert m["MAE"] == pytest.approx(abs_err.mean())
    assert m["RMSE"] == pytest.approx(np.sqrt(sq_err.mean()))
    assert m["WAPE"] == pytest.approx(abs_err.sum() / 12.0)
    # RMSSE: series 0 -> sqrt(1/1), series 1 -> sqrt(1/4)
    assert m["RMSSE"] == pytest.approx((1.0 + 0.5) / 2)
    assert m["Cov80"] == pytest.approx(0.75)    # y=6 is above q_0.90=5.5


def test_naive_scales_ignore_unlaunched_days():
    y = np.array([[np.nan, np.nan, 1.0, 3.0, 2.0]])
    sq, ab = naive_scales(y)
    assert sq[0] == pytest.approx((4 + 1) / 2)
    assert ab[0] == pytest.approx((2 + 1) / 2)
