"""DeepAR: likelihood, sampling, stockout masking, no leakage, and a tiny end-to-end fit.

PyTorch is imported inside the tests (marked ``torch``, run last), never at module level:
see the note in conftest.py.
"""

import copy
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from prism.backtest import check_forecast, run_backtest
from prism.models import get_model

TINY = dict(context_length=56, prediction_length=28, hidden_size=8, num_layers=1, dropout=0.0,
            embedding_dims={"item_id": 3, "dept_id": 2, "cat_id": 2, "store_id": 2},
            batch_size=32, batches_per_epoch=5, max_epochs=3, patience=2, learning_rate=5e-3,
            valid_days=28, refit_full=True, num_samples=50, predict_chunk=6, device="cpu")
CUT = 392


@pytest.fixture
def dar():
    import torch

    from prism.models import deepar
    deepar.torch = torch
    return deepar


def test_backtest_runs_deepar_next_to_lightgbm(panel, cfg, tmp_path):
    """Torch models are fitted in a child process, so one backtest can mix them with LightGBM,
    and a second run reads both from the cache."""
    from test_diagnostics import LGBM
    c = copy.deepcopy(cfg)
    c["backtest"]["n_origins"] = 1
    c["models"].update(lightgbm=LGBM, deepar={**TINY, "max_epochs": 1})
    fc = run_backtest(panel, c, ["lightgbm", "deepar"], cache_dir=tmp_path)
    assert {"lightgbm", "deepar"} <= set(fc["model"])
    again = run_backtest(panel, c, ["lightgbm", "deepar"], cache_dir=tmp_path)
    np.testing.assert_array_equal(fc.sort_values(["model", "series", "h"])["point"].to_numpy(),
                                  again.sort_values(["model", "series", "h"])["point"].to_numpy())


@pytest.mark.skipif(bool(os.environ.get("PRISM_TORCH_CHILD")), reason="already the child run")
def test_torch_tests_pass_in_a_child_process():
    """Runs every test marked ``torch`` in a fresh pytest process (see conftest.py)."""
    env = {**os.environ, "PRISM_TORCH_CHILD": "1"}
    res = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "torch", "-p", "no:cacheprovider", __file__],
                         cwd=Path(__file__).parents[1], env=env, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout[-4000:] + res.stderr[-2000:]


@pytest.mark.torch
def test_nb_log_prob_matches_scipy(dar):
    torch, nb_log_prob = dar.torch, dar.nb_log_prob
    z = torch.tensor([0.0, 1, 3, 10, 40])
    mu, alpha = torch.tensor([0.3, 1.0, 2.5, 8.0, 30.0]), torch.tensor([0.05, 0.5, 1.0, 0.2, 2.0])
    r = 1 / alpha
    ref = stats.nbinom.logpmf(z.numpy(), r.numpy(), (r / (r + mu)).numpy())
    np.testing.assert_allclose(nb_log_prob(z, mu, alpha).numpy(), ref, rtol=1e-5, atol=1e-5)


@pytest.mark.torch
def test_nb_samples_have_the_analytic_mean_variance_and_quantiles(dar):
    torch, nb_sample = dar.torch, dar.nb_sample
    torch.manual_seed(0)
    mu, alpha = 4.0, 0.5
    draws = nb_sample(torch.full((200_000,), mu), torch.full((200_000,), alpha), torch.Generator().manual_seed(0))
    d = draws.numpy()
    assert d.mean() == pytest.approx(mu, rel=0.02)
    assert d.var() == pytest.approx(mu + alpha * mu**2, rel=0.03)
    r = 1 / alpha
    for a in [0.1, 0.5, 0.9]:
        assert np.quantile(d, a) == stats.nbinom.ppf(a, r, r / (r + mu))


@pytest.mark.torch
def test_unusable_days_do_not_enter_the_likelihood(panel, dar):
    """Stockout and pre-launch days carry no weight: changing their sales leaves the loss unchanged."""
    torch, DeepARForecaster, build_inputs, window_batch = dar.torch, dar.DeepARForecaster, dar.build_inputs, dar.window_batch
    m = DeepARForecaster(**TINY)
    m.device_ = torch.device("cpu")
    inp = build_inputs(panel.truncate(CUT), 28, pad=56 + 28 + 28)
    net = m._network(inp, window_batch(inp, np.array([0]), np.array([CUT]), 84, 56)[0].shape[-1])
    s, e = np.array([0, 0, 11]), np.array([330, 360, CUT])       # series 0 has a stockout, 11 launches late
    x, z, u, v, st = window_batch(inp, s, e, 84, 56)
    assert (~u).any()
    base, n = m._nll(net, (x, z, u, v, st))
    z2 = torch.where(u, z, z + 50.0)                             # change only unusable targets
    other, n2 = m._nll(net, (x, z2, u, v, st))
    assert n == n2 and base.item() == pytest.approx(other.item())


@pytest.mark.torch
def test_window_inputs_ignore_data_after_the_origin(panel, dar):
    """A window ending at the origin gets identical inputs from panels cut at the origin or later."""
    torch, build_inputs, window_batch = dar.torch, dar.build_inputs, dar.window_batch
    a = build_inputs(panel.truncate(CUT), 28, pad=112)
    b = build_inputs(panel.truncate(CUT + 28), 28, pad=112)
    s, e = np.arange(panel.n_series), np.full(panel.n_series, CUT)
    for xa, xb in zip(window_batch(a, s, e, 84, 56), window_batch(b, s, e, 84, 56)):
        torch.testing.assert_close(xa, xb)
    assert not a.usable[:, a.pad + CUT:].any()                    # nothing after the origin is observed


@pytest.mark.torch
def test_fit_predict_is_valid_and_reproducible(panel, cfg, dar):
    levels = np.asarray(cfg["quantiles"])
    info = panel.truncate(CUT)
    m1, m2 = get_model("deepar", seed=3, **TINY), get_model("deepar", seed=3, **TINY)
    f1, f2 = m1.fit_predict(info, 28, levels), m2.fit_predict(info, 28, levels)
    check_forecast(f1, info, CUT, 28, levels, "deepar")
    np.testing.assert_array_equal(f1.point, f2.point)
    np.testing.assert_array_equal(f1.quantiles, f2.quantiles)
    curve = m1.diagnostics["training_curve"]
    assert set(curve["phase"]) == {"select", "refit"}
    assert len(m1.diagnostics["item_embedding"]) == panel.meta["item_id"].nunique()


@pytest.mark.torch
def test_noid_ablation_has_no_embeddings(panel, cfg, dar):
    m = get_model("deepar_noid", seed=0, **{**TINY, "use_static_ids": False, "max_epochs": 1})
    m.fit_predict(panel.truncate(CUT), 28, np.asarray(cfg["quantiles"]))
    assert len(m.net_.embeddings) == 0 and "item_embedding" not in m.diagnostics


@pytest.mark.torch
def test_training_reduces_the_loss(panel, dar):
    m = dar.DeepARForecaster(**{**TINY, "hidden_size": 16, "batches_per_epoch": 20, "max_epochs": 6, "valid_days": 0, "learning_rate": 1e-2})
    m.fit_predict(panel.truncate(CUT), 28, np.array([0.1, 0.5, 0.9]))
    nll = m.diagnostics["training_curve"]["train_nll"].to_numpy()
    assert nll[-1] < nll[0]
