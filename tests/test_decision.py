import numpy as np
import pandas as pd
import pytest

from prism.decision import critical_ratio, decision_table, quantile_at, simulate

LEVELS = [0.1, 0.5, 0.9]


def _frame(model="m"):
    return pd.DataFrame({"model": model, "y": [4.0, 0.0], "point": [2.2, 1.0],
                         "q_0.10": [1.0, 0.0], "q_0.50": [2.0, 1.0], "q_0.90": [5.0, 3.0]})


def test_critical_ratio():
    assert critical_ratio(9, 1) == pytest.approx(0.9)
    assert critical_ratio(1, 1) == pytest.approx(0.5)


def test_quantile_interpolation():
    q = quantile_at(_frame(), LEVELS, 0.7)                 # halfway between q50 and q90
    np.testing.assert_allclose(q, [3.5, 2.0])
    with pytest.raises(ValueError):
        quantile_at(_frame(), LEVELS, 0.95)                # outside the forecast levels


def test_costs_hand_computed():
    # point policy orders ceil(2.2)=3 and ceil(1.0)=1 -> short 1 (cost 9), left over 1 (cost 1)
    point = simulate(_frame(), LEVELS, "point", 9.0, 1.0)
    np.testing.assert_array_equal(point["order"], [3, 1])
    assert point["cost"].sum() == pytest.approx(9 + 1)
    # quantile policy at tau=0.9 orders 5 and 3 -> left over 1 and 3, nothing short
    quant = simulate(_frame(), LEVELS, "quantile", 9.0, 1.0)
    np.testing.assert_array_equal(quant["order"], [5, 3])
    assert quant["cost"].sum() == pytest.approx(1 + 3)
    table = decision_table(pd.concat([_frame("seasonal_naive"), _frame("kalman")]), LEVELS)
    assert set(table["policy"]) == {"point", "quantile"}
    assert (table["fill_rate"] <= 1).all()


def test_ordering_the_critical_quantile_minimises_cost():
    """Newsvendor theory: the tau* quantile of the demand distribution is the cheapest order."""
    rng = np.random.default_rng(0)
    y = rng.poisson(5.0, 50_000)
    cu, co = 9.0, 1.0
    costs = {Q: (cu * np.maximum(y - Q, 0) + co * np.maximum(Q - y, 0)).mean() for Q in range(16)}
    assert min(costs, key=costs.get) == np.quantile(y, critical_ratio(cu, co), method="inverted_cdf")
