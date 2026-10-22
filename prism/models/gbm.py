"""Global LightGBM model on lag features (the tabular "AML" baseline).

One model is trained on all series at once, so it can share seasonal, event
and price effects across items, which a local filter cannot.

* Point forecast: Tweedie objective (a compound Poisson-Gamma likelihood with
  a point mass at zero, suited to intermittent counts). The number of boosting
  rounds is chosen by early stopping on the last 28 days *of the training
  window*, then the model is refit on the whole window.
* Quantiles: one model per level with the pinball (quantile) loss
  (Koenker & Bassett, 1978); crossings are removed by sorting.
* Horizon-safe direct variants: a ``min_lag=7`` model serves h = 1..7 and a
  ``min_lag=28`` model serves h = 8..28. The pure lag-28 model is also kept as
  an ablation (``lightgbm_lag28``).
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..data import Panel
from ..features import CATEGORICAL, build_features, target_window
from .base import Forecast, Forecaster


class LightGBMForecaster(Forecaster):
    name = "lightgbm"

    def _train(self, X, y, params, rounds):
        ds = lgb.Dataset(X, y, categorical_feature=[c for c in CATEGORICAL if c in X], free_raw_data=False)
        return lgb.train(params, ds, num_boost_round=rounds)

    def _best_rounds(self, X, y, days, cutoff, params, rounds):
        """Early stopping on an inner validation window inside the training data."""
        valid_days = self.params.get("early_stopping_valid_days", 28)
        if not valid_days:
            return rounds
        tr, va = days < cutoff - valid_days, days >= cutoff - valid_days
        cats = [c for c in CATEGORICAL if c in X]
        dtr = lgb.Dataset(X[tr], y[tr], categorical_feature=cats)
        dva = lgb.Dataset(X[va], y[va], categorical_feature=cats, reference=dtr)
        booster = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                            callbacks=[lgb.early_stopping(50, verbose=False)])
        return max(booster.best_iteration, 50)

    def fit_variant(self, panel: Panel, min_lag: int, horizon: int, levels: np.ndarray):
        cutoff = panel.cutoff
        seed = self.params.get("seed", 0)
        t0 = max(0, cutoff - self.params.get("train_window", 730))
        X, s_idx, d_idx = build_features(panel, min_lag, horizon, t0, cutoff)
        y, usable = target_window(panel, s_idx, d_idx)
        X, y, d_idx = X[usable], y[usable], d_idx[usable]

        point_params = {**self.params["point_params"], "seed": seed}
        rounds = self._best_rounds(X, y, d_idx, cutoff, point_params, self.params["point_rounds"])
        point_model = self._train(X, y, point_params, rounds)

        Xf, _, _ = build_features(panel, min_lag, horizon, cutoff, cutoff + horizon)
        N = panel.n_series
        point = point_model.predict(Xf).reshape(N, horizon)
        quantiles = np.empty((N, horizon, len(levels)))
        for j, a in enumerate(levels):
            qp = {**self.params["quantile_params"], "objective": "quantile", "alpha": float(a), "seed": seed}
            qm = self._train(X, y, qp, self.params["quantile_rounds"])
            quantiles[:, :, j] = qm.predict(Xf).reshape(N, horizon)

        self.models_.setdefault(min_lag, {})["point"] = point_model
        self.rounds_[min_lag] = rounds
        return point, quantiles

    def feature_importance(self) -> pd.DataFrame:
        """Total gain of each feature in the point (Tweedie) model of every variant.

        Gain = total reduction of the training loss from splits on that feature. It is a
        sanity check as much as an explanation: a feature that should not be informative
        (or that could leak) showing up on top would be a red flag.
        """
        frames = []
        for min_lag, models in self.models_.items():
            booster = models["point"]
            gain = booster.feature_importance(importance_type="gain")
            frames.append(pd.DataFrame({"variant": f"lag{min_lag}", "feature": booster.feature_name(),
                                        "gain": gain, "gain_share": gain / gain.sum()}))
        return pd.concat(frames, ignore_index=True)

    def check_variants(self, horizon: int) -> None:
        """Before any training: every variant only serves horizons h <= min_lag (otherwise its
        features would need data from after the origin), and together they cover 1..horizon once."""
        served = []
        for name, var in self.params["variants"].items():
            lo, hi = var["horizons"]
            if not 1 <= lo <= hi <= var["min_lag"]:
                raise ValueError(f"variant {name}: horizons [{lo}, {hi}] must satisfy 1 <= lo <= hi <= "
                                 f"min_lag={var['min_lag']}, or its lag features would leak")
            served += range(lo, hi + 1)
        if sorted(served) != list(range(1, horizon + 1)):
            raise ValueError(f"LightGBM variants must cover horizons 1..{horizon} exactly once, got {sorted(served)}")

    def fit_predict(self, panel: Panel, horizon: int, levels: np.ndarray) -> Forecast:
        self.check_variants(horizon)
        self.models_, self.rounds_, self.extras = {}, {}, {}
        N = panel.n_series
        point = np.full((N, horizon), np.nan)
        quantiles = np.full((N, horizon, len(levels)), np.nan)
        for name, var in self.params["variants"].items():
            p, q = self.fit_variant(panel, var["min_lag"], horizon, levels)
            lo, hi = var["horizons"]
            cols = slice(lo - 1, hi)
            point[:, cols], quantiles[:, cols] = p[:, cols], q[:, cols]
            if var["min_lag"] >= horizon:          # this variant alone covers every horizon
                self.extras[f"lightgbm_{name}"] = Forecast(point=p, quantiles=q, levels=levels)
        self.diagnostics = {"feature_importance": self.feature_importance()}
        return Forecast(point=point, quantiles=quantiles, levels=levels)
