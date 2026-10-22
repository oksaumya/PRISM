"""Build report tables, figures and headline numbers from results/evaluation.parquet.

    uv run python scripts/make_report_assets.py
"""

import json

import pandas as pd

from prism.config import load_config, resolve
from prism.data import load_panel
from prism.report_export import build_assets


def main() -> None:
    cfg = load_config()
    panel = load_panel(cfg)
    ev = pd.read_parquet(resolve("results/evaluation.parquet"))
    head = build_assets(ev, panel, cfg["quantiles"], resolve("report"), resolve("results"))
    print(json.dumps({k: head[k] for k in ["n_series", "n_cells", "origins", "wape_gain_lgbm_vs_naive",
                                           "spl_gain_lgbm_vs_kalman", "fit_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
