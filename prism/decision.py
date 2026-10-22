"""From quantile forecasts to inventory decisions (the newsvendor model).

Every day, for every series, the store orders Q units and then demand y arrives.
Each unsold unit costs c_o (overage: holding, waste) and each unmet unit of
demand costs c_u (underage: the lost sale):

    cost(Q, y) = c_u * max(y - Q, 0) + c_o * max(Q - y, 0)

The order that minimises the expected cost is the demand quantile at the
*critical ratio* (Arrow, Harris & Marschak, 1951):

    tau* = c_u / (c_u + c_o)          e.g. c_u = 9, c_o = 1  ->  tau* = 0.9

This cost is exactly (c_u + c_o) times the pinball loss at tau*, so a model with
better-calibrated quantiles makes cheaper decisions. A point forecast can only
say "order the mean", which is the right answer only when c_u = c_o.

We compare two ordering policies for every model:

* ``point``    : order the point forecast (what a point-forecast system would do)
* ``quantile`` : order the predictive quantile at tau*

Simplifications, stated in the report: one period per day, no lead time, no
stock carried over between days, unmet demand is lost. Orders are whole units
(rounded up). Only stockout-masked days are used, because on a flagged day the
true demand was not observed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import qcol


def critical_ratio(cost_under: float, cost_over: float) -> float:
    return cost_under / (cost_under + cost_over)


def quantile_at(df: pd.DataFrame, levels, tau: float) -> np.ndarray:
    """Predictive quantile at any level ``tau``, interpolated linearly between the forecast levels."""
    levels = np.asarray(levels, dtype=float)
    if not levels[0] <= tau <= levels[-1]:
        raise ValueError(f"tau={tau:.3f} is outside the forecast levels [{levels[0]}, {levels[-1]}]")
    Q = df[[qcol(a) for a in levels]].to_numpy(dtype=float)
    j = int(np.clip(np.searchsorted(levels, tau), 1, len(levels) - 1))
    w = (tau - levels[j - 1]) / (levels[j] - levels[j - 1])
    return Q[:, j - 1] + w * (Q[:, j] - Q[:, j - 1])


def order_quantity(df: pd.DataFrame, levels, policy: str, cost_under: float, cost_over: float) -> np.ndarray:
    if policy == "point":
        target = df["point"].to_numpy(dtype=float)
    elif policy == "quantile":
        target = quantile_at(df, levels, critical_ratio(cost_under, cost_over))
    else:
        raise ValueError(f"unknown policy {policy!r}")
    return np.ceil(np.clip(target, 0, None))


def simulate(df: pd.DataFrame, levels, policy: str, cost_under: float, cost_over: float) -> pd.DataFrame:
    """Row-level outcome of one ordering policy: order, units short, units left over, cost."""
    y = df["y"].to_numpy(dtype=float)
    order = order_quantity(df, levels, policy, cost_under, cost_over)
    short = np.maximum(y - order, 0)
    excess = np.maximum(order - y, 0)
    return pd.DataFrame({"model": df["model"].to_numpy(), "y": y, "order": order, "short": short,
                         "excess": excess, "cost": cost_under * short + cost_over * excess})


def decision_table(df: pd.DataFrame, levels, cost_under: float = 9.0, cost_over: float = 1.0) -> pd.DataFrame:
    """Total cost, units short, units left over and fill rate per model and policy."""
    tables = []
    for policy in ["point", "quantile"]:
        sim = simulate(df, levels, policy, cost_under, cost_over)
        t = sim.groupby("model").agg(total_cost=("cost", "sum"), short_units=("short", "sum"),
                                     excess_units=("excess", "sum"), demand=("y", "sum")).reset_index()
        t["fill_rate"] = 1 - t["short_units"] / t["demand"]
        t["policy"] = policy
        tables.append(t)
    out = pd.concat(tables, ignore_index=True)
    out["tau"] = np.where(out["policy"] == "point", np.nan, critical_ratio(cost_under, cost_over))
    out["cost_vs_naive_point"] = out["total_cost"] / out.loc[
        (out["model"] == "seasonal_naive") & (out["policy"] == "point"), "total_cost"].iloc[0] - 1
    return out[["model", "policy", "tau", "total_cost", "cost_vs_naive_point", "short_units",
                "excess_units", "fill_rate", "demand"]]


def cost_ratio_sweep(df: pd.DataFrame, levels, ratios, cost_over: float = 1.0) -> pd.DataFrame:
    """Total cost of both policies as the underage/overage cost ratio grows."""
    rows = []
    for r in ratios:
        for policy in ["point", "quantile"]:
            sim = simulate(df, levels, policy, r * cost_over, cost_over)
            for model, cost in sim.groupby("model")["cost"].sum().items():
                rows.append({"ratio": r, "tau": critical_ratio(r, 1.0), "policy": policy,
                             "model": model, "total_cost": cost})
    return pd.DataFrame(rows)
