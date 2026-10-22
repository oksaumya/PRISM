"""Forecasting models. All share :class:`prism.models.base.Forecaster`."""

from importlib import import_module

from .base import Forecast, Forecaster

# name -> "module:Class"; imported lazily so light models do not pull in LightGBM
REGISTRY = {
    "zero": "naive:ZeroForecaster",           # harness check, not a real model
    "seasonal_naive": "naive:SeasonalNaive",
    "tsb": "croston:TSB",
    "kalman": "kalman:KalmanForecaster",
    "lightgbm": "gbm:LightGBMForecaster",
}


def get_model(name: str, **params) -> Forecaster:
    if name not in REGISTRY:
        raise ValueError(f"unknown model {name!r}; available: {', '.join(REGISTRY)}")
    module, cls = REGISTRY[name].split(":")
    return getattr(import_module(f"{__name__}.{module}"), cls)(**params)


__all__ = ["Forecast", "Forecaster", "REGISTRY", "get_model"]
