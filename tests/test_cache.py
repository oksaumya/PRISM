"""Caches must never return results computed under a different config."""

import copy

import numpy as np

import prism.data as data
from prism.backtest import run_backtest
from prism.config import panel_fingerprint


def test_changed_model_setting_refits(panel, cfg, tmp_path):
    run_backtest(panel, cfg, ["tsb"], cache_dir=tmp_path)
    n_files = len(list(tmp_path.glob("tsb_*.parquet")))
    run_backtest(panel, cfg, ["tsb"], cache_dir=tmp_path)                 # same config: cache hit
    assert len(list(tmp_path.glob("tsb_*.parquet"))) == n_files

    cfg2 = copy.deepcopy(cfg)
    cfg2["models"]["tsb"]["alpha"] = 0.3
    fc = run_backtest(panel, cfg2, ["tsb"], cache_dir=tmp_path)           # new setting: refit
    assert len(list(tmp_path.glob("tsb_*.parquet"))) == 2 * n_files
    fresh = run_backtest(panel, cfg2, ["tsb"])                            # equals an uncached run
    np.testing.assert_allclose(fc["point"], fresh["point"])


def test_panel_rebuilt_when_panel_settings_change(panel, tmp_path, monkeypatch):
    builds = []
    monkeypatch.setattr(data, "prepare_panel", lambda c: builds.append(1) or panel)
    cfg = {"seed": 0, "data": {"processed_dir": str(tmp_path), "items_per_store": 500},
           "stockout": {"p_threshold": 1e-3}}
    data.load_panel(cfg)
    data.load_panel(cfg)
    assert len(builds) == 1                                               # second call is cached

    cfg["data"]["items_per_store"] = 300
    loaded = data.load_panel(cfg)
    assert len(builds) == 2                                               # stale panel is not reused
    assert (tmp_path / "fingerprint.txt").read_text() == panel_fingerprint(cfg)
    np.testing.assert_array_equal(loaded.y, panel.y)


def test_sku_list_round_trip_and_mismatch(panel, tmp_path):
    import pandas as pd
    import pytest
    panel.meta["adi"], panel.meta["cv2"] = 1.0, 0.3
    panel.meta["true_launch"] = panel.meta["launch"] = panel.launch
    path = tmp_path / "sku_list.csv"
    data.write_sku_list(panel, path)
    data.check_sku_list(panel, path)                                      # identical: passes
    ref = pd.read_csv(path)
    ref.loc[0, "id"] = "SOMETHING_ELSE"
    ref.to_csv(path, index=False)
    with pytest.raises(ValueError, match="differ"):
        data.check_sku_list(panel, path)
