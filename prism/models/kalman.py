"""Per-SKU Kalman filter on a structural time-series model, written from scratch.

Model (Harvey, 1989), fit to z_t = log(1 + y_t):

    observation   z_t       = mu_t + gamma_t + eps_t            eps_t   ~ N(0, s2_eps)
    level         mu_{t+1}  = mu_t + eta_t                      eta_t   ~ N(0, s2_eta)
    season (s=7)  gamma_{t+1} = -(gamma_t + ... + gamma_{t-5}) + omega_t,  omega_t ~ N(0, s2_omega)

In state-space form  alpha_t = [mu_t, gamma_t, gamma_{t-1}, ..., gamma_{t-5}]  (7 states):

    z_t = Z alpha_t + eps_t,        alpha_{t+1} = T alpha_t + R eta*_t

Kalman (1960) recursions, with a_t = E[alpha_t | z_1..z_{t-1}] and P_t its covariance:

    v_t = z_t - Z a_t                 (innovation)
    F_t = Z P_t Z' + H                (innovation variance)
    K_t = P_t Z' / F_t                (Kalman gain)
    a_t|t = a_t + K_t v_t,            P_t|t = P_t - K_t F_t K_t'
    a_{t+1} = T a_t|t,                P_{t+1} = T P_t|t T' + Q

A missing observation (not launched, or a flagged stockout) simply skips the
update step, which is the exact treatment of missing data in a linear Gaussian
model; this is how stockouts are *marked* rather than imputed.

Parameters are estimated by maximum likelihood with the observation variance
concentrated out (Durbin & Koopman, 2012, sec. 2.10.2): writing Q = s2_eps * Q*,
the filter is run once with H = 1 and signal-to-noise ratios q = (q_eta, q_omega);
s2_eps then has the closed form  s2_hat = (1/n) sum v_t^2 / F*_t,  and the profile
log-likelihood is

    l_c(q) = -n/2 [log(2 pi) + 1 + log s2_hat] - 1/2 sum log F*_t.

l_c is maximised over a log-spaced grid of q (then a finer local grid). The
filter is vectorised across series, so every grid point costs a single pass over
all SKUs. The initial state is approximately diffuse (P_1 = kappa I) and the
first 7 observed points are excluded from the likelihood (Harvey, 1989).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import norm

from ..data import Panel
from .base import Forecast, Forecaster, observed_history

SEASON = 7


def system_matrices(season: int = SEASON) -> tuple[np.ndarray, np.ndarray]:
    """Transition matrix T and observation vector Z for level + dummy seasonal."""
    k = season                                   # 1 level + (season - 1) seasonal states
    T = np.zeros((k, k))
    T[0, 0] = 1.0                                # random-walk level
    T[1, 1:] = -1.0                              # seasonal effects sum to zero over a week
    T[2:, 1:-1] = np.eye(season - 2)             # shift older seasonal states down
    Z = np.zeros(k)
    Z[:2] = 1.0
    return T, Z


def kalman_filter(z: np.ndarray, q: np.ndarray, h: np.ndarray | float = 1.0,
                  kappa: float = 1e6, burn: int = SEASON, return_path: bool = False) -> dict:
    """Run the Kalman filter for N independent series at once.

    Args:
        z: (N, n) observations, NaN = missing.
        q: (2,) or (N, 2) variances of the level and seasonal disturbances.
        h: observation noise variance, scalar or (N,).
        kappa: variance of the approximately diffuse initial state.
        burn: number of initial *observed* points left out of the likelihood.
        return_path: also return one-step predictions and their variances.

    Returns a dict with the likelihood pieces, the one-step-ahead predicted
    state ``a`` / covariance ``P`` for time n+1, and optionally the paths.
    """
    N, n = z.shape
    T, _ = system_matrices()
    k = T.shape[0]
    q = np.broadcast_to(np.asarray(q, dtype=float), (N, 2))
    h = np.broadcast_to(np.asarray(h, dtype=float), (N,))

    a = np.zeros((N, k))
    P = np.broadcast_to(kappa * np.eye(k), (N, k, k)).copy()
    sum_logF, sum_v2F = np.zeros(N), np.zeros(N)
    n_used, n_seen = np.zeros(N), np.zeros(N)
    if return_path:
        pred_mean, pred_var = np.empty((N, n)), np.empty((N, n))
        level, season = np.empty((N, n)), np.empty((N, n))

    for t in range(n):
        zt = z[:, t]
        obs = ~np.isnan(zt)
        PZ = P[:, :, 0] + P[:, :, 1]             # P Z'   (Z = [1, 1, 0, ...])
        F = PZ[:, 0] + PZ[:, 1] + h              # Z P Z' + H
        yhat = a[:, 0] + a[:, 1]
        if return_path:
            pred_mean[:, t], pred_var[:, t] = yhat, F
        v = np.where(obs, zt - yhat, 0.0)
        K = PZ / F[:, None]
        a = a + K * v[:, None]
        P = P - obs[:, None, None] * (K[:, :, None] * PZ[:, None, :])

        n_seen += obs
        use = obs & (n_seen > burn)
        sum_logF += np.where(use, np.log(F), 0.0)
        sum_v2F += np.where(use, v * v / F, 0.0)
        n_used += use
        if return_path:
            level[:, t], season[:, t] = a[:, 0], a[:, 1]

        a = a @ T.T
        P = T @ P @ T.T
        P[:, 0, 0] += q[:, 0]
        P[:, 1, 1] += q[:, 1]
        P = 0.5 * (P + P.transpose(0, 2, 1))     # keep symmetric against round-off

    out = dict(a=a, P=P, sum_logF=sum_logF, sum_v2F=sum_v2F, n_used=n_used)
    if return_path:
        out.update(pred_mean=pred_mean, pred_var=pred_var, level=level, season=season)
    return out


def loglik(res: dict) -> np.ndarray:
    """Exact Gaussian log-likelihood (unconcentrated) from filter output."""
    n = res["n_used"]
    return -0.5 * (n * np.log(2 * np.pi) + res["sum_logF"] + res["sum_v2F"])


def profile_loglik(res: dict) -> tuple[np.ndarray, np.ndarray]:
    """Concentrated log-likelihood and the MLE of s2_eps (filter run with H = 1)."""
    n = np.maximum(res["n_used"], 1)
    s2 = res["sum_v2F"] / n
    ll = -0.5 * (n * (np.log(2 * np.pi) + 1 + np.log(np.maximum(s2, 1e-12))) + res["sum_logF"])
    return ll, s2


def fit_ratios(z: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Grid-search MLE of the signal-to-noise ratios, per series. Returns (q, loglik)."""
    kappa = cfg.get("diffuse_variance", 1e6)
    g1 = np.logspace(*cfg["q_level_grid"])
    g2 = np.logspace(*cfg["q_season_grid"])
    N = z.shape[0]
    best_ll = np.full(N, -np.inf)
    best_q = np.zeros((N, 2))

    def evaluate(qs):
        nonlocal best_ll, best_q
        for qi in qs:
            ll, _ = profile_loglik(kalman_filter(z, qi, 1.0, kappa))
            better = ll > best_ll
            best_ll = np.where(better, ll, best_ll)
            best_q = np.where(better[:, None], np.broadcast_to(qi, (N, 2)), best_q)

    evaluate([np.array([a, b]) for a in g1 for b in g2])
    if cfg.get("refine", True):
        steps = np.array([-0.5, -0.25, 0.25, 0.5])
        center = best_q.copy()
        evaluate([center * 10.0 ** np.array([d1, d2]) for d1 in [0.0, *steps] for d2 in [0.0, *steps]
                  if (d1, d2) != (0.0, 0.0)])
    return best_q, best_ll


def forecast_from_state(a: np.ndarray, P: np.ndarray, q: np.ndarray, s2: np.ndarray,
                        horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """h-step predictive mean and variance of z, h = 1..horizon (scaled system)."""
    T, _ = system_matrices()
    N = a.shape[0]
    mean, var = np.empty((N, horizon)), np.empty((N, horizon))
    for j in range(horizon):
        mean[:, j] = a[:, 0] + a[:, 1]
        var[:, j] = s2 * (P[:, 0, 0] + 2 * P[:, 0, 1] + P[:, 1, 1] + 1.0)
        a = a @ T.T
        P = T @ P @ T.T
        P[:, 0, 0] += q[:, 0]
        P[:, 1, 1] += q[:, 1]
    return mean, var


class KalmanForecaster(Forecaster):
    """Local model: one structural Kalman filter per SKU, no information shared."""

    name = "kalman"

    def fit_predict(self, panel: Panel, horizon: int, levels: np.ndarray) -> Forecast:
        cfg = self.params
        window = cfg.get("fit_window", 730)
        masked = observed_history(panel)[:, -window:]
        # an item whose whole recent history is one flagged gap is effectively delisted:
        # fall back to its raw (zero) sales rather than filtering on nothing
        few = np.sum(~np.isnan(masked), axis=1) < cfg.get("min_obs_in_stock", 14)
        y = np.where(few[:, None], panel.y[:, -window:], masked)
        z = np.log1p(y.astype(np.float64))
        n_obs = np.sum(~np.isnan(z), axis=1)

        q, _ = fit_ratios(z, cfg)
        short = n_obs < cfg.get("min_obs_for_mle", 56)
        q[short] = [cfg["default_q_level"], cfg["default_q_season"]]
        self.q_ = q

        res = kalman_filter(z, q, 1.0, cfg.get("diffuse_variance", 1e6))
        _, s2 = profile_loglik(res)
        # too few post-burn-in points for the MLE of s2: use the local sample variance
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            local_var = np.nan_to_num(np.nanvar(z, axis=1), nan=0.25)
        s2 = np.where(res["n_used"] >= 14, s2, np.maximum(local_var, 0.05))
        self.s2_ = s2

        m, v = forecast_from_state(res["a"], res["P"], q, s2, horizon)
        # A random-walk level makes the predictive variance grow without bound over long
        # horizons and unobserved gaps, and exp(m + v/2) amplifies it. The forecast is never
        # allowed to be less certain than the series' own marginal distribution of z.
        marginal = np.maximum(local_var, 0.01)
        v = np.clip(v, 1e-8, marginal[:, None])
        sd = np.sqrt(v)
        # quantiles survive the monotone back-transform: Q_y(p) = expm1(Q_z(p))
        quantiles = np.expm1(m[:, :, None] + sd[:, :, None] * norm.ppf(levels)[None, None, :])
        point = np.expm1(m + v / 2)                  # mean of the log-normal predictive
        return Forecast(point=point, quantiles=quantiles, levels=levels)
