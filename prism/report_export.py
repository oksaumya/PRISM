"""Turn the evaluation frame into report tables (LaTeX + CSV), figures and headline numbers."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import plots
from .data import data_quality
from .decision import cost_ratio_sweep, decision_table
from .metrics import summarize
from .stats_tests import panel_dm, series_wilcoxon

LOWER_BETTER = ["MAE", "RMSE", "WAPE", "Pinball", "RMSSE", "SPL"]
HISTORY_BINS = [7, 14, 28, 56, 90, 180, 10_000]
HISTORY_LABELS = ["7-13", "14-27", "28-55", "56-89", "90-179", "180+"]
DM_LAG = 7                     # Newey-West lag of the Diebold-Mariano tests (one week)


def _fmt(v: float, col: str) -> str:
    if col.startswith("Cov"):
        return f"{v:.3f}"
    if col == "n":
        return f"{int(v):,}"
    return f"{v:.3f}"


def latex_table(df: pd.DataFrame, cols: list[str], caption: str, label: str,
                group_col: str | None = None, nominal: dict | None = None, wide: bool = False) -> str:
    """booktabs table; best model per column (and per group) in bold.

    Coverage columns are "best" when closest to their nominal level.
    """
    nominal = nominal or {"Cov80": 0.8, "Cov90": 0.9}
    env = "table*" if wide else "table"             # table* spans both IEEE columns
    width = r"\textwidth" if wide else r"\columnwidth"
    fit_open, fit_close = r"\resizebox{\ifdim\width>" + width + " " + width + r"\else\width\fi}{!}{%", "}"
    lines = [r"\begin{" + env + "}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\scriptsize", r"\setlength{\tabcolsep}{3.5pt}",
             fit_open, r"\begin{tabular}{l" + ("l" if group_col else "") + "r" * len(cols) + "}", r"\toprule"]
    head = (["Segment"] if group_col else []) + ["Model"] + cols
    lines.append(" & ".join(head) + r" \\")
    lines.append(r"\midrule")
    groups = [(None, df)] if group_col is None else list(df.groupby(group_col, sort=False, observed=True))
    for gi, (g, d) in enumerate(groups):
        best = {}
        for c in cols:
            if c in LOWER_BETTER:
                best[c] = d[c].min()
            elif c in nominal:
                best[c] = d[c].iloc[(d[c] - nominal[c]).abs().argmin()]
        for k, (_, r) in enumerate(d.iterrows()):
            seg = str(g).replace("<", "$<$")
            cells = ([seg if k == 0 else ""] if group_col else []) + [plots.MODEL_LABELS.get(r["model"], r["model"])]
            for c in cols:
                s = _fmt(r[c], c)
                cells.append(r"\textbf{" + s + "}" if c in best and np.isclose(r[c], best[c]) else s)
            lines.append(" & ".join(cells) + r" \\")
        if group_col and gi < len(groups) - 1:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", fit_close, r"\end{" + env + "}"]
    return "\n".join(lines)


def _order(table: pd.DataFrame) -> pd.DataFrame:
    rank = {m: i for i, m in enumerate(plots.MODEL_ORDER)}
    return table.sort_values("model", key=lambda s: s.map(rank), kind="stable")


def decision_assets(masked: pd.DataFrame, levels, dcfg: dict, main_models: list[str],
                    tab_dir: Path, fig_dir: Path, out_results: Path) -> pd.DataFrame:
    """Newsvendor cost table and the cost-ratio sweep figure (see prism/decision.py)."""
    cu, co = dcfg["cost_under"], dcfg["cost_over"]
    d = masked[masked["model"].isin(main_models)]
    table = _order(decision_table(d, levels, cu, co))
    table.to_csv(out_results / "decision_costs.csv", index=False)

    tau = cu / (cu + co)
    rows = [r"\begin{table}[t]", r"\centering",
            r"\caption{Newsvendor simulation on stockout-masked days ($c_u=" + f"{cu:g}" + r"$, $c_o=" + f"{co:g}"
            + r"$, so $\tau^\ast=" + f"{tau:.2f}" + r"$). Each day each series orders the point forecast or the "
            r"$\tau^\ast$ quantile (rounded up). Cost change is relative to seasonal naive ordering its point forecast.}",
            r"\label{tab:decision}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
            r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{llrrrrr}", r"\toprule",
            r"Model & Order & Cost & vs.\ naive & Short & Left over & Fill rate \\", r"\midrule"]
    best = table.loc[table["policy"] == "quantile", "total_cost"].min()
    for policy, label in [("point", "point"), ("quantile", r"$q_{\tau^\ast}$")]:
        for _, r in table[table["policy"] == policy].iterrows():
            cost = f"{r['total_cost'] / 1e3:,.0f}k"
            cost = r"\textbf{" + cost + "}" if r["total_cost"] == best else cost
            rows.append(f"{plots.MODEL_LABELS[r['model']].split(' (')[0]} & {label} & {cost} & "
                        f"{r['cost_vs_naive_point']:+.0%} & {r['short_units'] / 1e3:,.0f}k & "
                        f"{r['excess_units'] / 1e3:,.0f}k & {r['fill_rate']:.3f}".replace("%", r"\%") + r" \\")
        if policy == "point":
            rows.append(r"\midrule")
    rows += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table}"]
    (tab_dir / "decision.tex").write_text("\n".join(rows))

    lo, hi = dcfg["ratio_sweep"]
    sweep = cost_ratio_sweep(d, levels, range(int(lo), int(hi) + 1), co)
    sweep.to_csv(out_results / "decision_cost_sweep.csv", index=False)
    wide = sweep.pivot_table(index=["ratio", "model"], columns="policy", values="total_cost").reset_index()
    naive_q = wide[wide["model"] == "seasonal_naive"].set_index("ratio")["quantile"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for m in [m for m in plots.MODEL_ORDER if m in set(wide["model"])]:
        w = wide[wide["model"] == m].set_index("ratio")
        style = dict(color=plots.MODEL_COLORS[m], label=plots.MODEL_LABELS[m])
        axes[0].plot(w.index, 100 * (1 - w["quantile"] / w["point"]), **style)
        axes[1].plot(w.index, 100 * (w["quantile"] / naive_q - 1), **style)
    axes[0].set_title("Saving from ordering the quantile")
    axes[0].set_ylabel("% cost saved vs. ordering the point")
    axes[1].set_title("Quantile policy vs. seasonal naive")
    axes[1].set_ylabel("% cost change")
    for ax in axes:
        ax.set_xlabel(r"cost ratio $c_u / c_o$")
        ax.axhline(0, color=plots.INK_3, lw=0.8)
    plots.legend_below(fig, axes[0])
    plots.save(fig, fig_dir / "decision_cost_sweep.pdf"); plt.close(fig)
    return table


def diagnostic_assets(ev: pd.DataFrame, panel, diagnostics: dict, fig_dir: Path, out_results: Path,
                      min_history: int, cold_start_days: int) -> dict:
    """Quantile crossing rates, LightGBM feature importance and Kalman fallback counts."""
    head = {}
    if "cross_size" in ev:
        g = ev.groupby("model", sort=False)["cross_size"]
        crossing = pd.DataFrame({"any_crossing": g.apply(lambda s: (s > 0).mean()),
                                 "crossing_over_0.1_units": g.apply(lambda s: (s > 0.1).mean()),
                                 "crossing_over_1_unit": g.apply(lambda s: (s > 1).mean())}).reset_index()
        crossing = _order(crossing)
        crossing.to_csv(out_results / "quantile_crossing.csv", index=False)
        head["crossing"] = crossing.set_index("model").to_dict(orient="index")

    imp = diagnostics.get("feature_importance", pd.DataFrame())
    if not imp.empty:
        imp = (imp.groupby(["variant", "feature"])["gain_share"].mean().rename("gain_share").reset_index()
               .sort_values(["variant", "gain_share"], ascending=[True, False]))
        imp.to_csv(out_results / "feature_importance.csv", index=False)
        variants = sorted(imp["variant"].unique(), key=lambda v: -int(v[3:]))      # lag28 first
        fig, axes = plt.subplots(1, len(variants), figsize=(7.2, 3.6))
        for ax, v in zip(np.atleast_1d(axes), variants):
            top = imp[imp["variant"] == v].head(12).iloc[::-1]
            ax.barh(top["feature"], 100 * top["gain_share"], color=plots.MODEL_COLORS["lightgbm"])
            ax.set_title(f"LightGBM {v.replace('lag', 'lag-')} model")
            ax.set_xlabel("% of total gain (mean over origins)")
            ax.tick_params(axis="y", labelsize=7.5)
            ax.grid(axis="y", visible=False)
        plots.save(fig, fig_dir / "feature_importance.pdf"); plt.close(fig)
        head["top_features"] = {v: imp[imp["variant"] == v].head(5)["feature"].tolist() for v in variants}

    fb = diagnostics.get("fallbacks", pd.DataFrame())
    if not fb.empty:
        # only (series, origin) cells that are scored, i.e. with enough history
        history = fb["origin"].to_numpy() - panel.launch[fb["series"].to_numpy()]
        fb = fb[history >= min_history].assign(history=history[history >= min_history])
        fb = fb.join(panel.meta["demand_class"], on="series")
        flags = ["default_ratios", "raw_history", "sample_variance"]
        fb["segment"] = np.where(fb["history"] < cold_start_days, f"cold start (<{cold_start_days} days)",
                                 "established")
        fb["all"] = "all"
        parts = []
        for group, key in [("demand class", "demand_class"), ("history", "segment"), ("all", "all")]:
            g = fb.groupby(key)
            parts.append(g[flags].mean().assign(cells=g.size(), group=group))
        table = pd.concat(parts).rename_axis("segment").reset_index()[["group", "segment", "cells", *flags]]
        table.to_csv(out_results / "kalman_fallbacks.csv", index=False)
        head["kalman_fallbacks"] = table.set_index("segment")[["cells", *flags]].to_dict(orient="index")
    return head


def build_assets(ev: pd.DataFrame, panel, levels, out_report: Path, out_results: Path,
                 main_models: list[str] | None = None, decision_cfg: dict | None = None,
                 diagnostics: dict | None = None, min_history: int = 7, cold_start_days: int = 90) -> dict:
    """Write every table and figure of the results section; return headline numbers."""
    plots.apply_style()
    fig_dir, tab_dir = out_report / "figures", out_report / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    main_models = main_models or ["seasonal_naive", "tsb", "kalman", "lightgbm"]
    evm = ev[ev["model"].isin(main_models)]
    masked = ev[~ev["stockout"]]
    point_cols = ["MAE", "RMSE", "WAPE", "RMSSE"]
    prob_cols = ["Pinball", "SPL", "Cov80", "Cov90", "Width90"]
    # backtest size, read from the data so captions stay correct for any config
    n_series, n_origins, horizon = ev["series"].nunique(), ev["origin"].nunique(), int(ev["h"].max())

    # ---- tables
    overall = _order(summarize(ev))
    overall_m = _order(summarize(masked))
    overall.to_csv(out_results / "metrics_overall.csv", index=False)
    overall_m.to_csv(out_results / "metrics_masked.csv", index=False)
    both = pd.concat([overall.assign(view="All target days"), overall_m.assign(view="Stockout-masked")])
    (tab_dir / "overall.tex").write_text(latex_table(
        both, point_cols + prob_cols,
        f"Rolling-origin backtest, {n_origins} origins $\\times$ {horizon} days, {n_series:,} series. "
        "MAE/WAPE use the predictive median, RMSE/RMSSE the mean. Best per column in bold (coverage: closest to nominal).",
        "tab:overall", group_col="view", wide=True))

    by_h = _order(summarize(masked, ["horizon_bucket"]))
    by_h.to_csv(out_results / "metrics_by_horizon_bucket.csv", index=False)
    (tab_dir / "by_horizon.tex").write_text(latex_table(
        by_h[by_h["model"].isin(main_models + ["lightgbm_lag28"])], ["WAPE", "RMSSE", "Pinball", "SPL", "Cov90"],
        "Accuracy by horizon bucket (stockout-masked).", "tab:horizon", group_col="horizon_bucket"))

    by_class = _order(summarize(masked[masked["model"].isin(main_models)], ["demand_class"]))
    by_class = by_class[by_class["demand_class"] != "new"]
    by_class.to_csv(out_results / "metrics_by_demand_class.csv", index=False)
    (tab_dir / "by_class.tex").write_text(latex_table(
        by_class, ["WAPE", "RMSSE", "SPL", "Cov90"],
        "Accuracy by Syntetos--Boylan demand class (stockout-masked).", "tab:class", group_col="demand_class"))

    cs = masked[masked["model"].isin(main_models)].assign(
        segment=lambda d: np.where(d["cold_start"], "Cold start (<90 days)", "Established"))
    by_cs = _order(summarize(cs, ["segment"]))
    by_cs.to_csv(out_results / "metrics_cold_start.csv", index=False)
    (tab_dir / "cold_start.tex").write_text(latex_table(
        by_cs, ["n", "WAPE", "RMSSE", "SPL", "Cov90"],
        "Cold-start vs established series (stockout-masked).", "tab:coldstart", group_col="segment"))

    # per-origin results: is the ranking stable across the rolling origins, or driven by one period?
    by_o = _order(summarize(masked[masked["model"].isin(main_models)], ["origin"]))
    by_o.to_csv(out_results / "metrics_by_origin.csv", index=False)
    best_spl = by_o.loc[by_o.groupby("origin")["SPL"].idxmin(), "model"].value_counts()
    rows = [r"\begin{table}[t]", r"\centering",
            r"\caption{Spread across the " + f"{n_origins}" + r" rolling origins (stockout-masked): mean $\pm$ "
            r"standard deviation of the per-origin metric, and the number of origins at which each model has the "
            r"lowest SPL.}",
            r"\label{tab:origin}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{lrrrr}", r"\toprule",
            r"Model & WAPE & SPL & Cov90 & Best SPL \\", r"\midrule"]
    for m, d in by_o.groupby("model", sort=False):
        cells = [f"{d[c].mean():.3f} $\\pm$ {d[c].std():.3f}" for c in ["WAPE", "SPL", "Cov90"]]
        rows.append(f"{plots.MODEL_LABELS[m].split(' (')[0]} & " + " & ".join(cells)
                    + f" & {int(best_spl.get(m, 0))}/{n_origins}" + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (tab_dir / "by_origin.tex").write_text("\n".join(rows))

    # ---- significance
    tests = []
    for a, b in [("lightgbm", "kalman"), ("lightgbm", "tsb"), ("kalman", "tsb"), ("kalman", "seasonal_naive"),
                 ("lightgbm", "lightgbm_lag28")]:
        if {a, b} <= set(ev["model"]):
            for loss in ["pinball", "abs_err"]:
                r = panel_dm(masked, a, b, loss, lag=DM_LAG)
                w = series_wilcoxon(masked, a, b, loss)
                tests.append({**r, "share_series_a_better": w["share_a_better"], "wilcoxon_p": w["p_value"]})
    tests = pd.DataFrame(tests)
    tests.to_csv(out_results / "significance_tests.csv", index=False)
    lab = lambda m: plots.MODEL_LABELS[m].split(" (")[0]
    rows = [r"\begin{table}[t]", r"\centering",
            r"\caption{Diebold--Mariano tests on the daily mean loss (" + f"{masked['day'].nunique()}"
            + r" target days, HAC lag " + f"{DM_LAG}" + r") and the share "
            r"of series where model A has lower mean loss (Wilcoxon signed-rank $p$). Negative DM: A is better.}",
            r"\label{tab:dm}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
            r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{llrrrr}", r"\toprule",
            r"A vs B & Loss & DM & $p$ & A better & Wilcoxon $p$ \\", r"\midrule"]
    for _, r in tests.iterrows():
        rows.append(f"{lab(r['model_a'])} vs {lab(r['model_b'])} & {'pinball' if r['loss']=='pinball' else 'abs. error'} & "
                    f"{r['DM']:.2f} & {r['p_value']:.3g} & {r['share_series_a_better']:.0%} & {r['wilcoxon_p']:.2g}".replace("%", r"\%") + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}", "}", r"\end{table}"]
    (tab_dir / "dm_tests.tex").write_text("\n".join(rows))

    # ---- figures
    per_h = summarize(masked[masked["model"].isin(main_models)], ["h"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    plots.metric_by_horizon(per_h, "WAPE", ax=axes[0])
    plots.metric_by_horizon(per_h, "SPL", ax=axes[1])
    axes[0].set_title("Point accuracy (WAPE)"); axes[1].set_title("Probabilistic accuracy (SPL)")
    plots.legend_below(fig, axes[0])
    plots.save(fig, fig_dir / "metric_by_horizon.pdf"); plt.close(fig)

    plots.quantile_calibration(masked, levels).to_csv(out_results / "quantile_calibration.csv", index=False)
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    plots.reliability(masked[masked["model"].isin(main_models)], levels, ax=ax)
    ax.set_title("Quantile calibration")
    plots.save(fig, fig_dir / "calibration.pdf"); plt.close(fig)

    order = ["smooth", "erratic", "intermittent", "lumpy"]
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    plots.grouped_bars(by_class, "demand_class", "SPL", order=order, ax=ax)
    ax.set_title("Scaled pinball loss by demand class (lower is better)")
    plots.legend_below(fig, ax)
    plots.save(fig, fig_dir / "spl_by_class.pdf"); plt.close(fig)

    sim = masked[masked["sim_cold_start"] & masked["model"].isin(main_models)].copy()
    sim["history"] = pd.cut(sim["history_days"], HISTORY_BINS, right=False, labels=HISTORY_LABELS)
    curve = summarize(sim, ["history"])
    curve.to_csv(out_results / "cold_start_learning_curve.csv", index=False)
    present = [h for h in HISTORY_LABELS if h in set(curve["history"].astype(str))]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for metric, ax in zip(["SPL", "WAPE"], axes):
        for mname in [m for m in plots.MODEL_ORDER if m in set(curve["model"])]:
            d = curve[curve["model"] == mname]
            ax.plot(range(len(d)), d[metric], color=plots.MODEL_COLORS[mname], label=plots.MODEL_LABELS[mname],
                    marker="o", ms=4, mec="white", mew=1)
        ax.set_xticks(range(len(present)), present)
        ax.set_xlabel("Days of history at the origin"); ax.set_ylabel(metric)
    axes[0].set_title("Probabilistic accuracy (SPL)"); axes[1].set_title("Point accuracy (WAPE)")
    plots.legend_below(fig, axes[0])
    plots.save(fig, fig_dir / "cold_start_curve.pdf"); plt.close(fig)

    # fan charts: one smooth and one intermittent series at the last origin, local vs global
    last = ev["origin"].max()
    meta = panel.meta
    picks = []
    for klass in ["smooth", "intermittent"]:
        cand = meta.index[(meta["demand_class"] == klass) & ~meta["sim_cold_start"]]
        vol = np.nanmean(panel.y[cand, last - 56:last], axis=1)
        target = 12 if klass == "smooth" else 1.0
        picks.append(int(cand[np.nanargmin(np.abs(vol - target))]))
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.0), sharey="row", sharex=True)
    for r, s in enumerate(picks):
        for c, mname in enumerate(["kalman", "lightgbm"]):
            d = ev[(ev["model"] == mname) & (ev["origin"] == last) & (ev["series"] == s)].sort_values("h")
            plots.fan_chart(panel.y[s, last - 56:last], d["y"].to_numpy(), d, mname, last, ax=axes[r, c],
                            stockout_hist=panel.stockout[s, last - 56:last],
                            title=f"{meta.loc[s, 'id'].replace('_evaluation', '')} · {mname.replace('lightgbm', 'LightGBM').replace('kalman', 'Kalman')}")
            axes[r, c].title.set_fontsize(8.5)
            if r == 0:
                axes[r, c].set_xlabel("")
            if c == 1:
                axes[r, c].set_ylabel("")
    for c in range(2):                                     # one legend per model column
        h, l = axes[0, c].get_legend_handles_labels()
        axes[0, c].legend(h, l, loc="upper left", fontsize=6.5, ncols=2)
    plots.save(fig, fig_dir / "fan_charts.pdf"); plt.close(fig)

    decisions = None
    if decision_cfg:
        decisions = decision_assets(masked, levels, decision_cfg, main_models, tab_dir, fig_dir, out_results)

    timing = ev.groupby(["model", "origin"])["fit_seconds"].first().groupby("model").mean()
    timing.to_csv(out_results / "fit_seconds.csv")

    # ---- headline numbers for the report text
    get = lambda t, m, c: float(t.loc[t["model"] == m, c].iloc[0])
    head = {
        "n_series": int(panel.n_series), "n_cells": int(ev.groupby("model").size().iloc[0]),
        "origins": sorted(int(o) for o in ev["origin"].unique()),
        "overall": overall.set_index("model").to_dict(orient="index"),
        "masked": overall_m.set_index("model").to_dict(orient="index"),
        "wape_gain_lgbm_vs_naive": 1 - get(overall_m, "lightgbm", "WAPE") / get(overall_m, "seasonal_naive", "WAPE"),
        "spl_gain_lgbm_vs_kalman": 1 - get(overall_m, "lightgbm", "SPL") / get(overall_m, "kalman", "SPL"),
        "fit_seconds": timing.to_dict(),
    }
    if decisions is not None:
        head["decision_costs"] = decisions.to_dict(orient="records")
    head.update(diagnostic_assets(ev, panel, diagnostics or {}, fig_dir, out_results, min_history, cold_start_days))
    quality = data_quality(panel)
    quality.to_csv(out_results / "data_quality.csv", index=False)
    head["data_quality"] = quality.set_index("metric")["value"].to_dict()
    (out_results / "summary.json").write_text(json.dumps(head, indent=2, default=float))
    return head
