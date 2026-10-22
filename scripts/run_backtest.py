"""Run the Phase 1 rolling-origin backtest and write forecasts + metrics.

    uv run python scripts/run_backtest.py            # full experiment (1,500 series, 5 origins)
    uv run python scripts/run_backtest.py --smoke    # 60 series, 1 origin: fast end-to-end check
"""

from __future__ import annotations

import argparse
import logging
import time

from prism.backtest import evaluation_frame, run_backtest
from prism.config import load_config, resolve
from prism.data import check_sku_list, load_panel, write_sku_list
from prism.metrics import summarize

MODELS = ["seasonal_naive", "tsb", "kalman", "lightgbm"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/phase1.yaml")
    ap.add_argument("--smoke", action="store_true", help="tiny run for quick checks")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--no-cache", action="store_true", help="refit even if cached forecasts exist")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    cfg = load_config(resolve(args.config), smoke=args.smoke)
    out_dir = resolve("results/smoke" if args.smoke else "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    t = time.perf_counter()
    panel = load_panel(cfg)
    logging.info("panel: %d series x %d days", panel.n_series, panel.n_days)
    sku_list = out_dir / "sku_list.csv"               # the committed sample: every run must match it
    if sku_list.exists():
        check_sku_list(panel, sku_list)
        logging.info("sample matches %s", sku_list.name)
    else:
        write_sku_list(panel, sku_list)
        logging.info("wrote %s", sku_list.name)
    forecasts = run_backtest(panel, cfg, args.models, cache_dir=None if args.no_cache else out_dir / "cache")
    evals = evaluation_frame(forecasts, panel, cfg)
    evals.to_parquet(out_dir / "evaluation.parquet")

    overall = summarize(evals)
    masked = summarize(evals[~evals["stockout"]])
    overall.to_csv(out_dir / "metrics_overall.csv", index=False)
    masked.to_csv(out_dir / "metrics_masked.csv", index=False)
    cols = ["model", "MAE", "RMSE", "WAPE", "Pinball", "RMSSE", "SPL", "Cov80", "Cov90"]
    print("\nAll target days\n", overall[cols].round(4).to_string(index=False))
    print("\nStockout-masked target days\n", masked[cols].round(4).to_string(index=False))
    logging.info("done in %.1f min", (time.perf_counter() - t) / 60)


if __name__ == "__main__":
    main()
