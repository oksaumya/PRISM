"""Every sales-derived feature for target day t may only use data up to t - min_lag."""

import numpy as np
import pytest

from prism.features import build_features


@pytest.mark.parametrize("min_lag", [7, 28])
def test_features_ignore_data_newer_than_min_lag(panel, min_lag):
    t_star = 380
    # A: data only up to t* - min_lag.  B: data up to t* - 1.  Row t* must be identical.
    a = panel.truncate(t_star - min_lag + 1)
    b = panel.truncate(t_star)
    Xa, _, _ = build_features(a, min_lag, horizon=min_lag, t_start=t_star, t_end=t_star + 1)
    Xb, _, _ = build_features(b, min_lag, horizon=1, t_start=t_star, t_end=t_star + 1)
    np.testing.assert_array_equal(Xa.to_numpy(), Xb.to_numpy())


def test_truncate_removes_the_future(panel):
    p = panel.truncate(300)
    assert p.y.shape[1] == 300 and p.item_other.shape[1] == 300
    assert p.price.shape[1] == panel.price.shape[1]    # planned prices are known ahead
