# PRISM
**P**robabilistic **R**etail **I**nventory forecasting with **S**tate-space & **M**achine learning

*Inventory demand forecasting with Kalman filtering and global models on M5 · Phase 1 (Machine Learning course project)*

PRISM forecasts the next **28 days of daily unit demand** for **1,500 Walmart item-store series** from the [M5 Forecasting – Accuracy](https://www.kaggle.com/competitions/m5-forecasting-accuracy) data. Every forecast comes with **7 predictive quantiles**, not just a point estimate.

It compares:

* a **local** model: a structural Kalman filter fitted separately to every SKU, written from scratch and checked against `statsmodels`;
* a **global** model: one LightGBM model trained on all SKUs at once;
* two classical baselines: **seasonal naive** and **TSB** (Croston-type, for intermittent demand).

All models run through the same **rolling-origin backtest** and are scored on identical cells. The backtest treats **stockouts** (zero sales caused by an empty shelf) and **cold-start items** (little history) explicitly. A **newsvendor decision layer** turns the quantiles into order quantities, which shows what better uncertainty is worth in inventory cost.

**Deliverables:** [IEEE-format report](report/main.pdf) · [presentation (PDF)](presentation/) · [editable slides (PPTX)](slides/PRISM_Phase1.pptx) · [viva notes](docs/viva_notes.md)

---

## Results at a glance
1,500 series × 5 origins × 28 days, scored on the **188,263 stockout-masked cells**. Days flagged as likely stockouts are left out, because true demand was not observed on them. Unmasked results over all 209,860 cells give the same ranking (`results/metrics_overall.csv`).

| Model | WAPE ↓ | RMSSE ↓ | SPL ↓ | 80 % cov. | 90 % cov. |
|---|---|---|---|---|---|
| Seasonal naive | 0.757 | 0.978 | 0.358 | 0.857 | 0.924 |
| TSB | 0.600 | 0.753 | 0.281 | 0.790 | 0.887 |
| Kalman filter (local) | 0.582 | 0.744 | 0.280 | 0.818 | **0.903** |
| **LightGBM (global)** | **0.569** | **0.732** | **0.264** | **0.798** | 0.915 |

<sub>Bold = best; for coverage, closest to the nominal 80 % / 90 %. SPL = scaled pinball loss. Source: `results/metrics_masked.csv`.</sub>

**Key findings**

* **The global model wins on every accuracy metric** (MAE, RMSE, WAPE, RMSSE, pinball, SPL), both masked and unmasked. Against the Kalman filter the Diebold–Mariano p-value on daily pinball loss is about 5 × 10⁻²¹, and LightGBM has the lower pinball loss on **79 % of series**.
* **The gap grows as the data gets sparser.** LightGBM's SPL is 3 % lower than the filter's on smooth series, 9 % lower on intermittent series and **19 % lower on new items with 7–13 days of history**. On those new items the filter's 90 % interval covers only **71 %** of outcomes.
* **The Kalman filter is the strongest local model on WAPE and RMSSE.** Its SPL is essentially tied with TSB (0.2801 vs 0.2809), and its 90 % interval is the closest to nominal overall. Both learned models cut WAPE by 23–25 % relative to seasonal naive.
* **The ranking is stable.** LightGBM has the lowest SPL at **all 5 origins** (`results/metrics_by_origin.csv`).
* **Quantiles pay off in inventory decisions.** When a lost sale costs 9× an unsold unit, ordering the 0.9 quantile instead of the point forecast cuts simulated cost by **21–49 %** for every model. LightGBM's quantiles give the lowest cost, 7.5 % below the filter's.

| Newsvendor cost (c_u = 9, c_o = 1) | Order the point forecast | Order the 0.9 quantile | Fill rate (quantile) |
|---|---|---|---|
| Seasonal naive | 2,017k | 1,037k | 0.925 |
| TSB | 1,378k | 964k | 0.939 |
| Kalman filter | 1,275k | 1,007k | 0.946 |
| **LightGBM** | 1,344k | **931k** | 0.937 |

<sub>Source: `results/decision_costs.csv`; cost-ratio sweep in `results/decision_cost_sweep.csv`.</sub>

## Research questions
| Question | How PRISM answers it |
|---|---|
| Can a local filter that adapts fast compete with a global model that shares patterns across items? | Kalman filter per SKU vs one LightGBM for all SKUs, refit at the same origins and scored on the same cells |
| A zero sale may be a stockout, not zero demand | Stockouts are **marked, never imputed**: the filter skips its update step, the other models ignore those days in training, and metrics are reported both raw and masked |
| New items have little history | A simulated cold-start cohort gives a learning curve from 7 to 168 days of history |
| Point forecasts are not enough for inventory | 7 quantiles per forecast; pinball loss, scaled pinball loss, 80 / 90 % coverage and a reliability diagram |
| Is a difference real or noise? | Diebold–Mariano (HAC) and Wilcoxon signed-rank tests, plus results per origin |
| What is a better forecast worth? | A newsvendor layer orders the critical-ratio quantile and counts the cost |

## Method

```
M5 raw CSVs ──► sample 1,500 series ──► mark stockouts ──► simulate cold start ──► Panel
                (balanced: category ×    (backward-looking   (10 % of series
                 demand class, per store) zero-run test)      "launch" late)
                                                                                     │
          ┌──────────────────────── for each of 5 origins ◄───────────────────────────┘
          ▼
   truncate panel at origin ──► fit every model ──► harness checks ──► forecasts (point + 7 quantiles)
                                                                              │
   metrics · significance tests · newsvendor costs · report tables & figures ◄┘
```

### Data
* **Source:** M5 Forecasting – Accuracy, which has 30,490 item-store series over 1,941 days (2011-01-29 to 2016-05-22), plus a calendar of events and SNAP days and weekly sell prices.
* **Sample:** stores **CA_1, TX_1 and WI_1**, 500 series each. Sampling is *balanced* over category × Syntetos–Boylan demand class (smooth / erratic / intermittent / lumpy). Classes are judged on **pre-backtest history only**, to avoid selection leakage. The 1,500 chosen series are committed in `results/sku_list.csv`.
* **Launch rule:** a series is live from its first sale, and earlier days are missing rather than zero. The price data confirms the rule: 99.3 % of pre-launch days have no listed price, and no day after launch does (`results/data_quality.csv`).

### Stockouts: marked, never imputed
M5 has no inventory data, so stockouts must be inferred from the sales signal. A run of *L* zero-sales days is flagged when π̂₀ᴸ < 10⁻³. Here π̂₀ is the item's Laplace-smoothed share of zero-sales days over its previous 56 in-stock days, and at least 14 such days are required. The test only looks backwards, so it is valid at any forecast origin. A run still in progress at an origin is re-judged on its truncated length. 13.7 % of observed days are flagged.

### Cold start
10 % of long-established series have their early history hidden, so they "launch" 7, 14, 28 or 56 days before the first origin. Over the 5 origins this gives a learning curve from 7 to 168 days of history. Cells with fewer than 7 days of history are not scored, and "cold start" means fewer than 90 days of history. Cross-store features subtract the series' own sales, so the hidden history cannot leak back in.

### Models
| Model | Type | Point forecast | Uncertainty |
|---|---|---|---|
| Seasonal naive (s = 7) | local baseline | last in-stock value on the same weekday | empirical *k*-week-ahead errors |
| TSB (Teunter, Syntetos & Babai, 2011) | local, intermittent demand | smoothed demand probability × smoothed size | empirical one-step errors |
| Kalman filter | local structural model: random-walk level + 7-day seasonal on log1p(y) | log-normal mean | Gaussian on the log scale, exact quantile back-transform |
| LightGBM | global; Tweedie mean model + one quantile model per level | direct, horizon-safe: a lag-7 model serves h = 1–7, a lag-28 model serves h = 8–28 | direct quantile regression |

* **Kalman filter** (`prism/models/kalman.py`): the noise variance is concentrated out of the likelihood. The two signal-to-noise ratios are then found by a profile-likelihood grid search with local refinement, separately for every SKU. The filter is vectorised across SKUs, so one grid point costs one pass over all series. Stockout and pre-launch days are treated as missing observations, which is the exact treatment in a linear Gaussian model. The unit tests check it against `statsmodels.UnobservedComponents` to within 10⁻⁶.
* **LightGBM** (`prism/models/gbm.py`, `prism/features.py`): features are lags, rolling statistics, same-weekday means, zero streaks, cross-store sales of the same item, department sales, prices, calendar events, SNAP days, item age and categorical IDs. Every sales-derived feature for day *t* uses only days ≤ *t* − min_lag, so no recursion is needed and nothing leaks. Early stopping uses the last 28 days *inside* the training window. A lag-28-only model is kept as an ablation: the two-model setup is slightly but significantly better (SPL 0.264 vs 0.265, DM p = 0.003).

### Evaluation
* **Rolling origin:** 5 origins, 28 days apart. The last one forecasts the official M5 test window (d_1914–d_1941). At each origin every model is refit from scratch on a panel that physically contains no later data.
* **Metrics:** each point metric is scored on the statistic it rewards. MAE and WAPE use the predictive **median**; RMSE and RMSSE use the **mean**. Probabilistic metrics are the pinball loss averaged over the 7 levels (0.05 … 0.95), the scaled pinball loss (SPL, M5 style), and the coverage and width of the 80 % / 90 % intervals.
* **Breakdowns:** by horizon bucket, demand class, cold-start segment, history length and origin, each on all days and on stockout-masked days.
* **Decision layer** (`prism/decision.py`): each day, each series orders either its point forecast or its quantile at the critical ratio τ* = c_u / (c_u + c_o), and the units short and left over are costed. It is deliberately simple: one period per day, no lead time, no stock carried over, lost sales.

## Getting started

### Prerequisites
* [**uv**](https://docs.astral.sh/uv/). It installs Python 3.12 and the locked dependencies from `uv.lock`.
* **Kaggle CLI and API token**, to download the data. Install the CLI with `uv tool install kaggle` (or `pip install kaggle`). Put the token in `~/.kaggle/kaggle.json` and accept the [competition rules](https://www.kaggle.com/competitions/m5-forecasting-accuracy/rules). If the official download is refused, the script falls back to a public mirror of the same files.
* **macOS only:** LightGBM needs the OpenMP runtime: `brew install libomp`.

### Run it
```bash
git clone https://github.com/oksaumya/PRISM.git && cd PRISM
make setup        # uv sync: create .venv with pinned dependencies
make test         # 36 unit tests on synthetic data, no download needed (under a minute)
make data         # download M5 into data/raw and check its shape (30,490 series × 1,941 days)
make smoke        # 60 series, 1 origin: fast end-to-end check, writes to results/smoke/
make backtest     # full experiment: 1,500 series × 5 origins, about 20 min on a laptop
make assets       # report tables, figures and results/*.csv from results/evaluation.parquet
make notebooks    # execute all four notebooks in place
```

| Target | Needs |
|---|---|
| `test` | nothing beyond `make setup` |
| `smoke`, `backtest` | raw data (`make data`) |
| `assets`, `notebooks` | raw data **and** `results/evaluation.parquet`, which `make backtest` writes (git-ignored) |

`make all` runs data → test → backtest → assets → notebooks. The committed `results/`, `report/` and slides already hold the outputs of a full run, so you can read every result without re-running anything.

The backtest script takes a few options:
```bash
uv run python scripts/run_backtest.py --models kalman lightgbm   # run a subset of models
uv run python scripts/run_backtest.py --no-cache                 # refit, ignoring cached forecasts
uv run python scripts/run_backtest.py --config my_config.yaml    # another experiment config
```

On the full run, one origin takes on average about 1 minute for the Kalman filter (grid search over all 1,500 SKUs) and about 2.5 minutes for LightGBM (2 variants × 8 boosters). The baselines take under a second (`results/fit_seconds.csv`).

### Build the report
`report/main.pdf` is committed. To rebuild it, upload `report/` to Overleaf and compile `main.tex` with pdfLaTeX + BibTeX. Locally, [Tectonic](https://tectonic-typesetting.github.io) works too: `brew install tectonic`, then `cd report && tectonic main.tex`.

## Configuration
Every experiment setting lives in [`configs/phase1.yaml`](configs/phase1.yaml): seed, stores and sample size, demand-class cut-offs, stockout threshold and window, cold-start design, origins and horizon, quantile levels, newsvendor costs, and all model hyperparameters. The `smoke` section holds the overrides used by `--smoke`.

## Reproducibility and safeguards
* **Fingerprinted caches.** The processed panel (`data/processed/`) and every cached forecast (`results/cache/`) are tagged with a SHA-256 fingerprint of the settings they depend on. A changed setting therefore never silently reuses stale results. `make clean-cache` forces all models to refit.
* **Committed sample.** Every full run checks that it rebuilt exactly the series in `results/sku_list.csv` and fails loudly if not.
* **Harness checks.** Before scoring, every forecast is checked for several things: the model received data only up to the origin, there is a value for every series, horizon and quantile, and all values are finite and non-negative. LightGBM also refuses any variant that would serve a horizon longer than its minimum lag.
* **Quantile crossing is measured, then fixed.** LightGBM's separate quantile models can cross. The raw crossings are logged (`results/quantile_crossing.csv`; only 0.06 % of cells cross by more than one unit) and then removed by sorting, which can only lower the pinball loss.

## Tests
`make test` runs 36 tests on synthetic data in a few dozen seconds:

* **Kalman filter:** matches `statsmodels` (likelihood, predictions and variances, including with missing data); the grid MLE recovers known parameters.
* **No leakage:** feature rows for a day are identical whether the panel is cut just after the latest day the features may use, or at that day itself.
* **Stockouts:** flags at an origin equal flags recomputed from truncated data; gaps are flagged in busy series but not in intermittent ones.
* **Metrics:** hand-computed values; the pinball loss is minimised by the true quantile.
* **Decision layer:** ordering the critical quantile minimises the newsvendor cost.
* **Backtest and harness:** all models are scored on identical cells; a zero forecaster gives WAPE = 1 exactly; broken or leaky forecasts are rejected.
* **Caches:** a changed setting invalidates cached panels and forecasts.

## Repository layout
```
prism/                    library code (importable package)
  config.py               YAML config, forecast origins, config fingerprints
  data.py                 M5 loading, Panel container, sampling, cold-start simulation, caching, data quality
  sampling.py             Syntetos–Boylan demand classes, balanced stratified sampling
  stockout.py             likely-stockout marking (backward-looking zero-run test)
  features.py             leakage-safe lag / rolling / calendar / price / cross-store features
  models/
    base.py               Forecaster interface + Forecast container
    naive.py              seasonal naive (and the zero forecaster used to check the harness)
    croston.py            TSB intermittent-demand method
    kalman.py             structural Kalman filter, profile-likelihood MLE, vectorised over SKUs
    gbm.py                global LightGBM: Tweedie mean + quantile models, horizon-safe direct
  backtest.py             rolling-origin harness, forecast checks, evaluation frame
  metrics.py              MAE, RMSE, WAPE, RMSSE, pinball, SPL, coverage
  stats_tests.py          Diebold–Mariano (HAC) and Wilcoxon tests
  decision.py             newsvendor layer: quantiles → order quantities → cost
  plots.py                shared figure style
  report_export.py        LaTeX tables, figures, results CSVs and summary.json
configs/phase1.yaml       every experiment setting
scripts/                  download_data.sh · run_backtest.py · make_report_assets.py
notebooks/                01 EDA · 02 stockouts & cold start · 03 Kalman walkthrough · 04 results
tests/                    36 unit tests (synthetic data)
results/                  metric tables (CSV), summary.json, sku_list.csv (the committed sample)
report/                   IEEE-format LaTeX report: main.tex, main.pdf, refs.bib, figures/, tables/
presentation/             presentation as PDF
slides/                   PRISM_Phase1.pptx: 18 slides with speaker notes and editable charts
docs/viva_notes.md        theory notes, likely viva questions, module ownership
```

## Adding a model
Every model implements one method, so a new one drops into the harness unchanged:

```python
from prism.models.base import Forecast, Forecaster

class MyModel(Forecaster):
    name = "my_model"

    def fit_predict(self, panel, horizon, levels) -> Forecast:
        # panel is already truncated at the origin; return (N, H) point and (N, H, Q) quantiles
        ...
```

Register it in `REGISTRY` in [`prism/models/__init__.py`](prism/models/__init__.py), add its parameters under `models:` in the config, and run `scripts/run_backtest.py --models my_model ...`.

## Limitations
* Stockouts are *inferred* from sales. The rule assumes independent zero days and cannot tell a stockout from a delisting. A sensitivity analysis over the threshold is in the report, and unmasked metrics are always reported.
* The sample covers 3 of the 10 M5 stores, with 5 origins. The ranking is consistent across origins, but other stores are untested.
* The newsvendor layer compares forecasts, not inventory policies: there are no lead times, no carried-over stock and no order constraints.

## Phase 2 preview
A DeepAR-style global RNN, with static item embeddings and dynamic calendar and price covariates, will implement the same `Forecaster.fit_predict` interface and be scored in this harness. The planned **hybrid** feeds the Kalman filter's level, season and state uncertainty to the network as covariates, with ablations with and without them.

## Team
Newton School of Technology, Rishihood University. Ownership details are in [docs/viva_notes.md](docs/viva_notes.md).

| Member | Enrollment No. | Primary module |
|---|---|---|
| Saumya Kumar | 2401010432 | data pipeline, sampling, EDA |
| Krishiv | 2401010234 | stockout marking, cold-start design |
| Shivansh Upadhayay | 2401020109 | Kalman filter and local baselines |
| Harsha Gonela | 2401010181 | features and global LightGBM |
| Harshvardhan Gupta | 2401010185 | backtest, metrics, significance tests, report |

## Key references
* Makridakis, Spiliotis & Assimakopoulos (2022). *M5 accuracy competition: Results, findings, and conclusions.* International Journal of Forecasting.
* Harvey (1989). *Forecasting, Structural Time Series Models and the Kalman Filter.* · Durbin & Koopman (2012). *Time Series Analysis by State Space Methods.*
* Teunter, Syntetos & Babai (2011). *Intermittent demand: Linking forecasting to inventory obsolescence.* · Syntetos & Boylan (2005).
* Montero-Manso & Hyndman (2021). *Principles and algorithms for forecasting groups of time series: Locality and globality.*
* Diebold & Mariano (1995) · Gneiting (2011) · Tashman (2000).

The full bibliography is in [`report/refs.bib`](report/refs.bib).
