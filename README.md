# PRISM
**P**robabilistic **R**etail **I**nventory forecasting with **S**tate-space & **M**achine learning

*Inventory demand forecasting via Kalman filtering and global models, on M5 · Phase 1 (Machine Learning)*

PRISM forecasts 28 days of daily unit demand for 1,500 Walmart item-store series, **with predictive quantiles**. It compares a **local** state-space model (a Kalman filter per SKU, written from scratch) with a **global** gradient-boosted model (one LightGBM model across all SKUs) and two classical baselines. Every model is evaluated on an identical rolling-origin backtest that treats **stockouts** and **cold-start items** fairly.

## Results at a glance
Rolling-origin backtest: 1,500 series × 5 origins × 28 days, stockout-masked target days (full tables in `results/` and `report/tables/`).

| Model | WAPE ↓ | RMSSE ↓ | SPL ↓ | 80% cov. | 90% cov. |
|---|---|---|---|---|---|
| Seasonal naive | 0.757 | 0.978 | 0.358 | 0.857 | 0.924 |
| TSB | 0.600 | 0.753 | 0.281 | 0.790 | 0.887 |
| Kalman filter (local) | 0.582 | 0.744 | 0.280 | 0.818 | **0.903** |
| **LightGBM (global)** | **0.569** | **0.732** | **0.264** | **0.798** | 0.915 |

* The global model wins on every accuracy metric (Diebold–Mariano p < 10⁻²⁰; lower pinball loss on 79 % of series).
* The gap **grows with sparsity**: SPL is 3 % lower on smooth series, 9 % on intermittent ones and **19 % for new items with 7–13 days of history**, where the local filter's 90 % interval covers only 71 % of outcomes.
* The Kalman filter is the best local method and the best-calibrated once an item has history. The two are complementary, which motivates the Phase 2 hybrid.

## Why this project
| Question | How PRISM answers it |
|---|---|
| Can a local filter that adapts fast compete with a global model that shares patterns? | Kalman filter per SKU vs global LightGBM, with the same features of the data and the same scoring cells |
| A zero sale may be a stockout, not zero demand | stockouts are **marked, never imputed**: the filter skips the update, the global model drops the rows, and metrics are reported raw and masked |
| New items have no history | a simulated cold-start cohort (7–56 days of history) gives a learning curve per model |
| Point forecasts are not enough for inventory | 7 quantiles per forecast; pinball loss, scaled pinball, 80/90 % coverage and a reliability diagram |
| Is a difference real? | Diebold–Mariano and Wilcoxon tests |

## Repository layout
```
prism/                  library code (importable package)
  config.py             YAML config + forecast origins
  data.py               M5 loading, Panel container, sampling, cold-start simulation, caching
  sampling.py           Syntetos–Boylan demand classes, balanced stratified sampling
  stockout.py           likely-stockout marking (backward-looking zero-run test)
  features.py           leakage-safe lag / rolling / calendar / price / cross-store features
  models/
    base.py             Forecaster interface + Forecast container (Phase 2 models plug in here)
    naive.py            seasonal naive with empirical quantiles
    croston.py          TSB intermittent-demand method
    kalman.py           structural Kalman filter, profile-likelihood MLE, vectorised over SKUs
    gbm.py              global LightGBM: Tweedie mean + quantile models, horizon-safe direct
  backtest.py           rolling-origin harness + evaluation frame
  metrics.py            MAE, RMSE, WAPE, RMSSE, pinball, SPL, coverage
  stats_tests.py        Diebold–Mariano (HAC) and Wilcoxon tests
  plots.py              one consistent figure style
  report_export.py      LaTeX tables, figures, headline numbers
configs/phase1.yaml     every experiment setting
scripts/                download_data.sh · run_backtest.py · make_report_assets.py
notebooks/              01 EDA · 02 stockouts & cold start · 03 Kalman walkthrough · 04 results
tests/                  18 unit tests: metrics, Kalman vs statsmodels, leakage, stockouts, backtest
report/                 IEEE-format LaTeX report (main.tex, refs.bib, figures/, tables/)
slides/                 presentation
docs/viva_notes.md      theory notes, likely viva questions, module ownership
```

## Quick start
```bash
make setup        # uv sync  (Python 3.12, all dependencies pinned in uv.lock)
make data         # Kaggle download into data/raw (needs a Kaggle API token)
make test         # unit tests on synthetic data, ~20 s
make smoke        # 60 series, 1 origin, ~1 min
make backtest     # full experiment: 1,500 series x 5 origins, ~20 min on a laptop
make assets       # report tables + figures from results/evaluation.parquet
make notebooks    # execute all notebooks
```
The report compiles on Overleaf: upload the `report/` folder and compile `main.tex` (pdfLaTeX + BibTeX).

## Method in brief
**Data:** M5 Forecasting Accuracy, stores CA_1 / TX_1 / WI_1, 500 series each, sampled *balanced* over category × Syntetos–Boylan class using only pre-backtest history. A series is live from its first sale. 10 % of established series are turned into simulated new items.

**Stockouts:** a zero-run of length *L* is flagged when π̂₀ᴸ < 10⁻³, where π̂₀ is the item's Laplace-smoothed zero-day rate over the previous 56 in-stock days. The test is backward-looking, so it is valid at any forecast origin.

**Models**
| Model | Type | Uncertainty |
|---|---|---|
| Seasonal naive (s = 7) | local baseline | empirical *k*-week-ahead errors |
| TSB (Teunter et al., 2011) | local, intermittent demand | empirical one-step errors |
| Kalman filter | local structural model (level + weekly season) on log1p, profile-likelihood MLE per SKU, stockouts as missing data | Gaussian on log scale, exact quantile back-transform |
| LightGBM | global, Tweedie mean + 7 quantile models, horizon-safe direct (lag-7 for h ≤ 7, lag-28 for h ≤ 28) | direct quantile regression |

**Evaluation:** 5 rolling origins × 28 days ending at d_1941. All models are refit at each origin on a panel truncated at the origin and scored on identical cells. MAE/WAPE use the median, RMSE/RMSSE the mean, plus pinball, scaled pinball (SPL) and interval coverage. Results are split by horizon, demand class and cold start, on all target days and on stockout-masked days.

## Phase 2 preview
A DeepAR-style global RNN with static item embeddings and dynamic calendar/price covariates implements the same `Forecaster.fit_predict` interface, so it drops into this harness unchanged. The planned **hybrid** feeds Kalman-filtered level, season and state uncertainty to the network as covariates, with ablations.

## Team
| Member | Primary module |
|---|---|
| Member 1 | data pipeline, sampling, EDA |
| Member 2 | stockout marking, cold-start design |
| Member 3 | Kalman filter and local baselines |
| Member 4 | features and global LightGBM |
| Member 5 | backtest, metrics, significance tests, report |
