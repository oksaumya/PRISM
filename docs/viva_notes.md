# PRISM: Viva Preparation Notes (Phase 1)

The viva is **individual**. Every member should be able to explain the whole pipeline at a high level and their own module in depth. Each section below gives the key idea, the derivation you may be asked to reproduce, and likely questions with short answers.

---

## 0. Suggested ownership (5 members)

| Member | Module (primary owner) | Code | Notebook / report section |
|---|---|---|---|
| M1 | Data pipeline, sampling and EDA | `prism/data.py`, `prism/sampling.py` | `01_eda.ipynb`; §III Data |
| M2 | Stockout marking and cold-start design | `prism/stockout.py`, cold-start part of `data.py` | `02_stockouts_coldstart.ipynb`; §III-C/D |
| M3 | Local model: Kalman filter (+ seasonal naive, TSB baselines) | `prism/models/kalman.py`, `naive.py`, `croston.py` | `03_kalman_walkthrough.ipynb`; §IV-A/B |
| M4 | Global model: features and LightGBM | `prism/features.py`, `prism/models/gbm.py` | §IV-C; leakage test |
| M5 | Evaluation: backtest, metrics, significance tests, reporting | `prism/backtest.py`, `metrics.py`, `stats_tests.py`, `report_export.py` | `04_results.ipynb`; §V Results |

Everyone: literature review (§II), the local-vs-global story, and the Phase 2 plan.

---

## 1. The one-minute pitch

> We forecast 28 days of daily demand for 1,500 Walmart item-store series from M5, with **uncertainty intervals**. We compare a **local** model (a Kalman filter per SKU, written from scratch) with a **global** model (one LightGBM model shared across all SKUs) and two classical baselines, on a rolling-origin backtest. We handle **stockouts** by marking them, not imputing them, and we measure **cold start** explicitly. The local filter adapts quickly and gives well-calibrated intervals, but it cannot share information across items. The global model can. Where and by how much that matters is what Phase 1 measures, and it motivates the global DeepAR model in Phase 2.

---

## 2. Data (M1)

* **M5:** 3,049 products × 10 stores = 30,490 series, 1,941 days (2011-01-29 to 2016-05-22). Calendar has events and SNAP days; prices are weekly.
* **Sample:** stores CA_1, TX_1, WI_1 × 500 series = 1,500. The sample is **balanced over category × Syntetos–Boylan class** with water-filling allocation (small strata taken whole, the rest split evenly).
* **Why classify on pre-backtest data only?** Choosing the sample with test-period information would bias the evaluation (selection leakage).
* **Launch:** a series is "live" from its first non-zero sale (the M5 scaling convention). Days before launch are NaN, not zero.

**Q: Why not use all 30,490 series?** Compute (an 8 GB laptop) and clarity. A balanced 1,500 still covers every regime, and the config scales up with one line.
**Q: What is ADI / CV²?** ADI = days ÷ non-zero days (how irregular the *timing* is). CV² = (sd/mean)² of the non-zero sizes (how irregular the *quantity* is). The cut-offs 1.32 / 0.49 give smooth, erratic, intermittent and lumpy.
**Q: What did the EDA change in the models?** The weekly pattern led to the seasonal state and day-of-week features. Intermittency led to the log1p scale, the Tweedie loss and TSB. Closures/stockouts led to marking. SNAP and events became known-ahead features.

---

## 3. Stockouts and cold start (M2)

**Rule.** For a zero-run of length L starting at s: π̂₀ = (#zeros + 1)/(W + 2) over the last W = 56 observed, unflagged days before s. Flag the run if π̂₀ᴸ < p\* = 10⁻³.

* The rule is **backward looking**, so at an origin it uses past data only. A run in progress at the origin is re-judged on its truncated length (`flags_at_cutoff`; a unit test checks it equals recomputing from truncated data).
* The flag threshold adapts to each item: a busy item (π₀ = 0.05) is flagged after 3 zeros; an intermittent item (π₀ = 0.8) only after 31.
* **Mark, don't impute:** the Kalman filter skips the update step; LightGBM drops the rows from its loss; naive/TSB skip those values; and the evaluation reports both all days and masked days.

**Q: Why not impute?** Imputed values are invented data. They shrink uncertainty artificially and contaminate both training and evaluation. Missing-data handling is *exact* in a state-space model.
**Q: Weaknesses?** It assumes independent zero days (zeros cluster, so the rule is somewhat liberal), and it confuses delisting with stockout. We show a sensitivity analysis over p\* and always report unmasked metrics too.
**Cold-start simulation:** 10 % of established series have their history hidden so they launch 7/14/28/56 days before the first origin. Across 5 origins this gives a learning curve from 7 to 168 days of history.
**Q: Could the hidden history leak?** Not through the features. The cross-store aggregates subtract the series' own raw sales, so they contain only other series' data.

---

## 4. Kalman filter (M3)

**Model** on z = log(1 + y): level μ (random walk) + 7-day dummy seasonal γ + noise.

**State-space form:** α = (μ, γₜ, …, γₜ₋₅), Z = [1, 1, 0, 0, 0, 0, 0], T has the random-walk row, the "seasonals sum to zero" row (−1s) and shift rows.

**Recursions** (be able to write these on the board):
```
v_t = z_t − Z a_t                      innovation
F_t = Z P_t Zᵀ + σ²_ε                  innovation variance
K_t = P_t Zᵀ / F_t                     Kalman gain
a_{t|t} = a_t + K_t v_t ;  P_{t|t} = P_t − K_t F_t K_tᵀ          update
a_{t+1} = T a_{t|t}      ;  P_{t+1} = T P_{t|t} Tᵀ + Q             predict
```
**Missing z_t:** skip the update. This is exact, because the conditional distribution is unchanged when nothing is observed.

**Likelihood:** prediction-error decomposition ℓ = −½ Σ (log 2π + log F_t + v_t²/F_t).
**Concentration:** with H = 1 and ratios q = (q_η, q_ω), σ̂²_ε = (1/n) Σ v_t²/F*_t, which gives the profile likelihood ℓ_c(q) = −n/2 (log 2π + 1 + log σ̂²) − ½ Σ log F*_t. That leaves only a 2-D search per SKU.
**Optimisation:** a log grid (13 × 7) and then a local refinement grid; the filter is vectorised across all SKUs.
**Initialisation:** approximately diffuse, P₁ = 10⁶ I; the first 7 observations are excluded from the likelihood (one per state).
**Forecast:** iterate the predict step h times. Mean of z = Z a_{T+h}, variance = σ̂²(Z P Zᵀ + 1). Quantiles of y = expm1(quantiles of z), which is exact because quantiles commute with monotone maps. The mean of y is the log-normal mean exp(m + s²/2) − 1.
**Variance cap:** the random-walk variance grows linearly with h and with unobserved gaps; we cap it at the series' marginal variance of z (a forecast cannot be less informative than the unconditional distribution). Without it, a few delisted items produced absurd means.
**Verification:** matches `statsmodels.UnobservedComponents` to ~1e-9 (log-likelihood, predictions, variances, missing data).

**Q: Why is it called "local"?** Each SKU has its own parameters and state; nothing is shared. That is good for adapting, bad for sparse or new items.
**Q: What does a large q_level mean?** The level moves fast relative to the noise, so the filter trusts recent data more (higher gain). It is the state-space version of a high smoothing constant.
**Q: Relation to exponential smoothing?** The local level model's steady-state Kalman filter *is* simple exponential smoothing, with α determined by q.
**Q: Why log1p and not counts?** Variance stabilisation for a Gaussian filter. A Poisson/negative-binomial observation model would need non-linear filtering (extended/particle filters, or DeepAR's likelihood in Phase 2).

**Baselines:** seasonal naive (last same-weekday value that is not a stockout; quantiles from empirical k-week-ahead errors) and TSB (smooths demand probability every period, so forecasts decay for dying items; Teunter et al., 2011).

---

## 5. Global LightGBM (M4)

* **One model for all series:** it shares weekly, event, SNAP and price effects across items (Montero-Manso & Hyndman, 2021: global models can be as complex as the *collection* of series supports).
* **Horizon-safe direct strategy:** every sales feature for target day t uses only days ≤ t − min_lag. Because min_lag ≥ h, the model is valid for every h ≤ min_lag without recursion. We have two variants: lag-7 (serves h = 1–7) and lag-28 (h = 8–28). "LightGBM lag-28 only" is the ablation.
* **Features:** lags (L, L+1, L+2, L+7, …, 364), rolling mean/std/max/zero-share, same-weekday mean, momentum, expanding mean, zero-streak, other-stores item mean, same-dept-store mean, price/relative price/price change, calendar (dow, month, events, days-to-event), SNAP, item age, and categorical IDs.
* **Losses:** Tweedie (ρ = 1.1) for the mean. It is a compound Poisson–Gamma distribution with a point mass at zero, which suits intermittent counts. One quantile (pinball) model per level, with crossings removed by sorting.
* **Early stopping:** on the last 28 days *inside* the training window, then refit on the full window. The test period is never used.
* **Leakage test:** feature rows for day t* are identical whether the panel is cut at t* − L + 1 or at t* (`tests/test_no_leakage.py`).

**Q: Why not recursive forecasting?** Errors compound, and training/inference mismatch (the model is trained on true lags but fed its own predictions). The direct strategy avoids both.
**Q: Isn't using the price on the target day leakage?** No. Prices are planned and known ahead (the M5 setup gives them). Sales-derived inputs are always lagged.
**Q: Why does the global model help cold start?** A new item has no lags, but its category, department and price, and the same item's sales in other stores, are informative. Only a model trained across series can use them.
**Q: What does Tweedie variance power 1.1 mean?** Var(Y) ∝ μ^1.1: close to Poisson (power 1), with mass at zero.

---

## 6. Evaluation (M5)

* **Rolling origin** (Tashman, 2000): 5 origins, 28 days apart, ending at d_1941 (the last origin is the official M5 test window). Every model is refit at every origin on the truncated panel, which physically contains no future data.
* **Same cells for all models** (unit-tested). Cells with < 7 days of history are not scored.
* **Metrics:** MAE and WAPE on the **median**; RMSE and RMSSE on the **mean** (Gneiting, 2011: score each point forecast on the functional it targets). Pinball loss averaged over 7 quantile levels approximates the CRPS. SPL is the pinball loss scaled by each series' in-sample naive MAE (M5 uncertainty style). Coverage and width of the 80 % and 90 % intervals, plus a quantile reliability diagram.
* **WAPE** = Σ|e| / Σy. It is scale-free across the portfolio and dominated by high-volume series. **RMSSE** averages per-series scaled errors, so every series counts equally.
* **Significance:** a Diebold–Mariano test on the daily cross-sectional mean loss (Newey–West HAC, lag 7), plus a Wilcoxon signed-rank test on per-series mean loss differences.

**Q: Why is the pinball loss "proper"?** Its expected value is minimised by the true quantile (`tests/test_metrics.py` checks this numerically). A forecaster cannot gain by misreporting.
**Q: Why can coverage alone be misleading?** An absurdly wide interval has perfect coverage. Pinball loss rewards *sharpness subject to calibration*.
**Q: Why stockout-masked metrics?** On a stockout day the actual is 0 because nothing could be sold, not because nobody wanted it. A model that correctly forecasts demand would be unfairly penalised.
**Q: Why MAPE is not used?** It is undefined or explosive for zero actuals, which are the majority here.

---

## 7. Likely cross-cutting questions

* **Local vs global, in one sentence:** local models fit each series' own dynamics with few parameters; global models fit one function across series and so can transfer patterns, at the cost of needing careful features.
* **What would you do with more time?** Hierarchical reconciliation; conformal calibration of the quantiles; a negative-binomial observation model for the filter; the Phase 2 work below.
* **Phase 2 plan:** a DeepAR-style global RNN (Salinas et al., 2020) with static item embeddings and dynamic calendar/price covariates, trained with a negative-binomial likelihood and scored in the same harness (it only has to implement `Forecaster.fit_predict`). The **hybrid**: feed Kalman-filtered level, season and uncertainty as dynamic covariates to the global network. That gives local adaptivity plus global sharing, and we will run ablations with and without the Kalman features.
