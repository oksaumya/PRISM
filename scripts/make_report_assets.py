"""Build report tables, figures and headline numbers from results/evaluation.parquet.

    uv run python scripts/make_report_assets.py
"""

import json

import pandas as pd

from prism.backtest import load_diagnostics
from prism.config import load_config, resolve
from prism.data import load_panel
from prism.report_export import build_assets


def main() -> None:
    cfg = load_config()
    panel = load_panel(cfg)
    ev = pd.read_parquet(resolve("results/evaluation.parquet"))
    # small per-model tables saved next to the forecast cache by `make backtest`
    cache = resolve("results/cache")
    diagnostics = {"feature_importance": load_diagnostics(cfg, cache, "lightgbm", "feature_importance"),
                   "fallbacks": load_diagnostics(cfg, cache, "kalman", "fallbacks")}
    head = build_assets(ev, panel, cfg["quantiles"], resolve("report"), resolve("results"),
                        decision_cfg=cfg.get("decision"), diagnostics=diagnostics,
                        min_history=cfg["cold_start"]["min_history_days"],
                        cold_start_days=cfg["cold_start"]["max_history_days"])
    print(json.dumps({k: head[k] for k in ["n_series", "n_cells", "origins", "wape_gain_lgbm_vs_naive",
                                           "spl_gain_lgbm_vs_kalman", "fit_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
