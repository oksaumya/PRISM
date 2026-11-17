"""Global DeepAR model (Salinas, Flunkert, Gasthaus & Januschowski, 2020).

One autoregressive LSTM is trained on windows drawn from *all* series. At each
step it sees the previous sales (lags 1, 7, 14, 28, divided by a per-window
scale), calendar and price covariates that are known in advance, and learned
embeddings of the static identifiers (item, department, category, store). It
outputs the parameters of a **negative binomial** distribution for that day's
demand, which suits non-negative, often intermittent counts.

How Phase 1's rules carry over:

* **Stockouts are marked, never imputed.** Days flagged as likely stockouts and
  days before launch contribute nothing to the likelihood. As inputs they are
  set to 0 together with an "observed" indicator that is 0, so the network can
  tell "no data" from "zero demand".
* **No leakage.** The model only ever receives ``panel.truncate(origin)``. Inputs
  after the origin are calendar and planned prices, as in LightGBM; future
  sales are never inputs, they are sampled.
* **Early stopping inside the training data.** Epochs are chosen on the
  likelihood of the last ``valid_days`` days before the origin, with training
  windows ending before them; the model is then retrained from scratch on the
  whole window for that many epochs (``refit_full``), as LightGBM does with its
  boosting rounds.
* **Uncertainty from the model, not from residuals.** Forecasts are ancestral
  samples: each sample path feeds its own previous draws back in. The mean of
  the paths is the point forecast and their empirical quantiles are the
  predictive quantiles, so horizon-wise uncertainty grows naturally.

``use_static_ids: false`` drops every ID embedding (the ``deepar_noid``
ablation), which isolates how much the model gains from knowing *which* item it
is forecasting, as opposed to learning shared dynamics from history alone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from ..data import Panel
from ..features import calendar_features
from .base import Forecast, Forecaster

LAGS = (1, 7, 14, 28)
STATIC = ("item_id", "dept_id", "cat_id", "store_id")


# --------------------------------------------------------------------------- likelihood

def nb_log_prob(z: torch.Tensor, mu: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Log pmf of a negative binomial with mean ``mu`` and dispersion ``alpha``.

    Var = mu + alpha * mu^2 (Salinas et al., 2020, eq. 2); alpha -> 0 is Poisson.
    """
    r = 1.0 / alpha
    log_r_mu = torch.log(r + mu)
    return (torch.lgamma(z + r) - torch.lgamma(r) - torch.lgamma(z + 1.0)
            + r * (torch.log(r) - log_r_mu) + z * (torch.log(mu) - log_r_mu))


def nb_sample(mu: torch.Tensor, alpha: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Draw from NB(mu, alpha) as a gamma-Poisson mixture: lambda ~ Gamma(r, mu / r), z ~ Poisson(lambda)."""
    r = 1.0 / alpha
    lam = torch.distributions.Gamma(r, r / mu).sample()     # global RNG: seeded in _sample
    return torch.poisson(lam, generator=generator)


def nb_params(raw: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Network output (.., 2) and window scale v (.., 1) -> (mu, alpha), as in the paper:
    the mean is multiplied by v and the dispersion divided by sqrt(v)."""
    mu = F.softplus(raw[..., 0]) * scale[..., 0] + 1e-6
    alpha = F.softplus(raw[..., 1]) / torch.sqrt(scale[..., 0]) + 1e-4
    return mu, alpha


# --------------------------------------------------------------------------- network

class DeepARNet(nn.Module):
    def __init__(self, n_inputs: int, cardinalities: list[int], emb_dims: list[int],
                 hidden: int, layers: int, dropout: float):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(c, d) for c, d in zip(cardinalities, emb_dims))
        self.rnn = nn.LSTM(n_inputs + sum(emb_dims), hidden, layers, batch_first=True,
                           dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor, static: torch.Tensor, state=None):
        """x: (B, T, F) step inputs, static: (B, n_static) codes -> raw (B, T, 2), state."""
        if len(self.embeddings):
            e = torch.cat([emb(static[:, j]) for j, emb in enumerate(self.embeddings)], dim=-1)
            x = torch.cat([x, e[:, None, :].expand(-1, x.shape[1], -1)], dim=-1)
        out, state = self.rnn(x, state)
        return self.head(out), state


# --------------------------------------------------------------------------- inputs

@dataclass
class Inputs:
    """Everything the network reads, as left-padded (N, PAD + D) arrays.

    Index ``t + pad`` holds day ``t``; the padding lets a window start before day 0
    (or before launch) without special cases: padded days are "not observed".
    """
    z: np.ndarray          # (N, P+D) sales, 0 where not usable
    usable: np.ndarray     # (N, P+D) observed, in stock and before the origin
    series_cov: np.ndarray  # (N, P+D, Fs) snap, price, age, live
    cal: np.ndarray        # (P+D, Fc) calendar covariates (shared)
    static: np.ndarray     # (N, n_static) integer codes
    pad: int
    cutoff: int


def calendar_inputs(calendar: pd.DataFrame) -> np.ndarray:
    """Calendar covariates known in advance: weekday, month, day of month, week of year, events."""
    cal = calendar_features(calendar)
    wday = np.eye(7, dtype=np.float32)[cal["wday"].to_numpy() - 1]
    m = 2 * np.pi * cal["month"].to_numpy() / 12
    w = 2 * np.pi * cal["week_of_year"].to_numpy() / 53
    n_types = int(cal["event_type_1"].max())
    ev_type = np.eye(n_types + 1, dtype=np.float32)[cal["event_type_1"].to_numpy()][:, 1:]
    cols = [wday, np.sin(m)[:, None], np.cos(m)[:, None], np.sin(w)[:, None], np.cos(w)[:, None],
            (cal["mday"].to_numpy()[:, None] - 16) / 15,
            (cal["event_name_1"].to_numpy() > 0)[:, None], ev_type,
            np.minimum(cal["days_to_event"].to_numpy(), 30)[:, None] / 30,
            np.minimum(cal["days_since_event"].to_numpy(), 30)[:, None] / 30]
    return np.hstack(cols).astype(np.float32)


def build_inputs(panel: Panel, horizon: int, pad: int) -> Inputs:
    cutoff = panel.cutoff if panel.cutoff is not None else panel.n_days
    N, D = panel.n_series, cutoff + horizon
    days = np.arange(D)

    y = np.full((N, D), np.nan, dtype=np.float32)
    y[:, :cutoff] = panel.y[:, :cutoff]
    stock = np.zeros((N, D), dtype=bool)
    stock[:, :cutoff] = panel.stockout[:, :cutoff]
    usable = ~np.isnan(y) & ~stock
    z = np.where(usable, y, 0).astype(np.float32)

    state = panel.meta["state_id"].to_numpy()
    snap = np.stack([panel.calendar[f"snap_{s}"].to_numpy()[:D] for s in state]).astype(np.float32)
    price = panel.price[:, :D].astype(np.float64)
    ref = pd.DataFrame(price.T).rolling(364, min_periods=1).mean().to_numpy().T
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_price = np.log(price / ref)
    price_known = np.isfinite(rel_price)
    rel_price = np.where(price_known, np.clip(rel_price, -2, 2), 0)
    age = days[None, :] - panel.launch[:, None]
    live = age >= 0
    log_age = np.where(live, np.log1p(np.maximum(age, 0)) / np.log(1000), 0)
    series_cov = np.stack([snap, rel_price, price_known, log_age, live], axis=-1).astype(np.float32)

    cal = calendar_inputs(panel.calendar.iloc[:D].reset_index(drop=True))
    static = np.stack([pd.Categorical(panel.meta[c]).codes for c in STATIC], axis=1).astype(np.int64)

    lpad = lambda a: np.concatenate([np.zeros((N, pad) + a.shape[2:], a.dtype), a], axis=1)
    return Inputs(z=lpad(z), usable=lpad(usable), series_cov=lpad(series_cov),
                  cal=np.concatenate([np.zeros((pad, cal.shape[1]), np.float32), cal]),
                  static=static, pad=pad, cutoff=cutoff)


def window_batch(inp: Inputs, series: np.ndarray, end: np.ndarray, length: int, context: int):
    """Teacher-forced inputs for windows covering days ``end - length .. end - 1``.

    Returns x (B, L, F), target z (B, L), target usable mask (B, L), scale v (B, 1)
    and static codes. The scale is 1 + the mean of the usable sales in the first
    ``context`` days of the window (Salinas et al., 2020, Sec. 3.3).
    """
    t = end[:, None] - length + np.arange(length)[None, :] + inp.pad          # (B, L) padded index
    s = series[:, None]
    zc, uc = inp.z[s, t[:, :context]], inp.usable[s, t[:, :context]]
    v = 1.0 + zc.sum(1) / np.maximum(uc.sum(1), 1)
    lag_z = np.stack([inp.z[s, t - k] for k in LAGS], axis=-1) / v[:, None, None]
    lag_u = np.stack([inp.usable[s, t - k] for k in LAGS], axis=-1)
    x = np.concatenate([lag_z, lag_u, inp.cal[t], inp.series_cov[s, t],
                        np.broadcast_to(np.log(v)[:, None, None], t.shape + (1,))], axis=-1)
    return (torch.from_numpy(x.astype(np.float32)), torch.from_numpy(inp.z[s, t]),
            torch.from_numpy(inp.usable[s, t]), torch.from_numpy(v.astype(np.float32)[:, None]),
            torch.from_numpy(inp.static[series]))


# --------------------------------------------------------------------------- forecaster

def pick_device(name: str) -> torch.device:
    """``auto`` = Apple-silicon GPU (MPS) or CUDA when present, else CPU."""
    if name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class DeepARForecaster(Forecaster):
    name = "deepar"

    def _p(self, key, default):
        return self.params.get(key, default)

    def _to(self, batch):
        return tuple(b.to(self.device_) for b in batch)

    def _network(self, inp: Inputs, n_inputs: int) -> DeepARNet:
        if self._p("use_static_ids", True):
            dims = self._p("embedding_dims", {})
            card = [int(inp.static[:, j].max()) + 1 for j in range(len(STATIC))]
            emb = [int(dims.get(c, 4)) for c in STATIC]
        else:
            card, emb = [], []
        return DeepARNet(n_inputs, card, emb, self._p("hidden_size", 64),
                         self._p("num_layers", 2), self._p("dropout", 0.1)).to(self.device_)

    def _nll(self, net, batch, loss_from: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        x, z, u, v, st = batch
        raw, _ = net(x, st)
        mu, alpha = nb_params(raw, v[:, None, :].expand(-1, x.shape[1], -1))
        lp = nb_log_prob(z, mu, alpha)[:, loss_from:]
        m = u[:, loss_from:].float()
        return -(lp * m).sum(), m.sum()

    def _train(self, inp: Inputs, train_end: int, epochs: int, valid: bool, rng: np.random.Generator):
        """Train a fresh network on windows ending at or before ``train_end``.

        With ``valid=True`` the validation likelihood of the days ``train_end .. cutoff-1``
        is tracked after every epoch, training stops after ``patience`` epochs without
        improvement and the best weights are restored.
        """
        C, P = self._p("context_length", 112), self._p("prediction_length", 28)
        L = C + P
        launch = self.launch_
        lo = np.maximum(launch + 1, train_end - self._p("train_window", 730) + P)
        trainable = np.flatnonzero(lo <= train_end)
        x0 = window_batch(inp, trainable[:1], np.array([train_end]), L, C)[0]
        net = self._network(inp, x0.shape[-1])
        opt = torch.optim.Adam(net.parameters(), lr=self._p("learning_rate", 1e-3),
                               weight_decay=self._p("weight_decay", 0.0))
        B, nb = self._p("batch_size", 256), self._p("batches_per_epoch", 100)

        if valid:   # one window per series whose prediction range is the validation period
            vs = np.flatnonzero(inp.usable[:, inp.pad + train_end: inp.pad + inp.cutoff].any(1))
            vbatches = [self._to(window_batch(inp, vs[i:i + 1024], np.full(len(vs[i:i + 1024]), inp.cutoff), L, C))
                        for i in range(0, len(vs), 1024)]

        curve, best, best_state, best_epoch, bad = [], np.inf, None, epochs, 0
        for epoch in range(1, epochs + 1):
            net.train()
            tot, cnt = 0.0, 0.0
            for _ in range(nb):
                s = rng.choice(trainable, size=B)
                e = rng.integers(lo[s], train_end + 1)
                loss_sum, n = self._nll(net, self._to(window_batch(inp, s, e, L, C)))
                loss = loss_sum / n.clamp(min=1)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), self._p("grad_clip", 10.0))
                opt.step()
                tot, cnt = tot + loss_sum.item(), cnt + n.item()
            row = {"epoch": epoch, "train_nll": tot / max(cnt, 1)}
            if valid:
                net.eval()
                with torch.no_grad():
                    vl = [self._nll(net, b, loss_from=L - (inp.cutoff - train_end)) for b in vbatches]
                row["valid_nll"] = sum(a.item() for a, _ in vl) / max(sum(n.item() for _, n in vl), 1)
                if row["valid_nll"] < best - 1e-4:
                    best, best_epoch, bad = row["valid_nll"], epoch, 0
                    best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
            curve.append(row)
            if valid and bad >= self._p("patience", 5):
                break
        if best_state is not None:
            net.load_state_dict(best_state)
        return net, best_epoch, pd.DataFrame(curve)

    @torch.no_grad()
    def _sample(self, net: DeepARNet, inp: Inputs, horizon: int, levels: np.ndarray, seed: int):
        """Ancestral sampling of ``num_samples`` paths for every series.

        The network runs on ``device_``; the NB draws are made on the CPU with a seeded
        generator (MPS has no Poisson sampler), so sampled paths are reproducible.
        """
        net.eval()
        dev = self.device_
        C, S = self._p("context_length", 112), self._p("num_samples", 200)
        N, c, pad = inp.static.shape[0], inp.cutoff, inp.pad
        gen = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)                        # the gamma draw uses the global generator
        mean = np.empty((N, horizon))
        quant = np.empty((N, horizon, len(levels)))
        chunk = max(1, self._p("predict_chunk", 64))
        for i0 in range(0, N, chunk):
            s = np.arange(i0, min(N, i0 + chunk))
            b = len(s)
            # 1. unroll the context with the observed history (inputs for days c-C .. c-1)
            x, _, _, v, st = self._to(window_batch(inp, s, np.full(b, c), C, C))
            _, state = net(x, st)
            # 2. sample: history buffer per path, future draws appended as they are made
            rep = lambda a: a.repeat_interleave(S, dim=0)
            state = tuple(h.repeat_interleave(S, dim=1) for h in state)
            v_s, st_s = rep(v), rep(st)
            hist = slice(pad + c - max(LAGS), pad + c)
            zbuf = rep(torch.from_numpy(inp.z[s, hist]).to(dev))               # (b*S, 28)
            ubuf = rep(torch.from_numpy(inp.usable[s, hist]).float().to(dev))
            draws = torch.empty(b * S, horizon)
            cal = torch.from_numpy(inp.cal[pad + c: pad + c + horizon]).to(dev)
            scov = torch.from_numpy(inp.series_cov[s, pad + c: pad + c + horizon]).to(dev)
            ones = torch.ones(b * S, 1, device=dev)
            logv = torch.log(v_s)
            for h in range(horizon):
                lag_z = torch.stack([zbuf[:, -k] for k in LAGS], dim=-1) / v_s
                lag_u = torch.stack([ubuf[:, -k] for k in LAGS], dim=-1)
                xs = torch.cat([lag_z, lag_u, cal[h].expand(b * S, -1),
                                rep(scov[:, h]), logv], dim=-1)[:, None, :]
                raw, state = net(xs, st_s, state)
                mu, alpha = nb_params(raw[:, 0], v_s)
                zt = nb_sample(mu.cpu(), alpha.cpu(), gen)
                draws[:, h] = zt
                zbuf = torch.cat([zbuf[:, 1:], zt[:, None].to(dev)], dim=1)
                ubuf = torch.cat([ubuf[:, 1:], ones], dim=1)
            paths = draws.view(b, S, horizon).numpy()
            mean[s] = paths.mean(axis=1)
            quant[s] = np.moveaxis(np.quantile(paths, levels, axis=1), 0, -1)
        return mean, quant

    def item_embedding(self, net: DeepARNet, panel: Panel) -> pd.DataFrame:
        if not len(net.embeddings):
            return pd.DataFrame()
        cats = pd.Categorical(panel.meta["item_id"]).categories
        w = net.embeddings[0].weight.detach().cpu().numpy()
        df = pd.DataFrame(w, columns=[f"e{j}" for j in range(w.shape[1])])
        df.insert(0, "item_id", list(cats))
        return df

    def fit_predict(self, panel: Panel, horizon: int, levels: np.ndarray) -> Forecast:
        seed = self.params.get("seed", 0)
        torch.manual_seed(seed)
        torch.set_num_threads(self._p("threads", torch.get_num_threads()))
        rng = np.random.default_rng(seed)
        self.device_ = pick_device(self._p("device", "auto"))
        C, P = self._p("context_length", 112), self._p("prediction_length", 28)
        inp = build_inputs(panel, horizon, pad=C + P + max(LAGS))
        self.launch_ = panel.launch
        cutoff, vdays = inp.cutoff, self._p("valid_days", 28)
        max_epochs = self._p("max_epochs", 40)

        t = time.perf_counter()
        if vdays:
            net, epochs, curve = self._train(inp, cutoff - vdays, max_epochs, True, rng)
            curve["phase"] = "select"
            if self._p("refit_full", True):
                net, _, curve2 = self._train(inp, cutoff, epochs, False, rng)
                curve = pd.concat([curve, curve2.assign(phase="refit")], ignore_index=True)
        else:
            net, epochs, curve = self._train(inp, cutoff, max_epochs, False, rng)
            curve["phase"] = "refit"
        train_s = time.perf_counter() - t
        mean, quant = self._sample(net, inp, horizon, levels, seed)
        curve["best_epochs"] = epochs
        curve["train_seconds"] = train_s
        self.net_ = net
        self.diagnostics = {"training_curve": curve}
        emb = self.item_embedding(net, panel)
        if len(emb):
            self.diagnostics["item_embedding"] = emb
        return Forecast(point=mean, quantiles=quant, levels=levels)
