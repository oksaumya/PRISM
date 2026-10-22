"""Configuration loading.

The YAML file is the single source of truth for an experiment. We keep it as a
plain nested dict so every value is easy to inspect in notebooks.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "phase1.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG, smoke: bool = False) -> dict:
    """Load a YAML config. With ``smoke=True`` the ``smoke`` overrides are applied."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if smoke:
        cfg = copy.deepcopy(cfg)
        cfg["data"]["items_per_store"] = cfg["smoke"]["items_per_store"]
        cfg["backtest"]["n_origins"] = cfg["smoke"]["n_origins"]
        cfg["data"]["processed_dir"] = str(Path(cfg["data"]["processed_dir"]) / "smoke")
    return cfg


def resolve(path: str | Path) -> Path:
    """Resolve a config path relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def forecast_origins(cfg: dict) -> list[int]:
    """Rolling forecast origins as *cutoffs*: number of observed days.

    A cutoff ``c`` means days ``0 .. c-1`` (d_1 .. d_c) are observed and the
    forecast covers day indices ``c .. c+H-1``. The last origin forecasts the
    final ``H`` days of the data, i.e. the official M5 evaluation window.
    """
    bt = cfg["backtest"]
    last, h, step, n = bt["last_day"], bt["horizon"], bt["step"], bt["n_origins"]
    return [last - h - step * (n - 1 - k) for k in range(n)]
