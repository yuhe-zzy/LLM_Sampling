"""Collect the per-dataset outputs into report figures and tables.

    python -m hodge.report reports/2026-09-25

Reads outputs/<dataset>/{panels,summary,population,null}.csv and outputs/extras,
writes figures and compact CSV/Markdown tables into the report directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated: dataviz validate_palette.js, light
MARKERS = ["o", "s", "^", "D"]

STATIC_ROWS = [
    ("HelpSteer", "helpsteer", "single_score_hard", "identity", "hard labels on mean score (repo IPO), identity"),
    ("HelpSteer", "helpsteer", "single_score_bt", "identity", "one BT annotator, identity (link mismatch)"),
    ("HelpSteer", "helpsteer", "attribute_vote_smoothed", "logit", "attributes as annotators, logit"),
    ("UltraFeedback", "ultrafeedback", "single_score_hard", "identity", "hard labels on mean score, identity"),
    ("UltraFeedback", "ultrafeedback", "single_score_bt", "identity", "one BT annotator, identity (link mismatch)"),
    ("UltraFeedback", "ultrafeedback", "attribute_vote_smoothed", "logit", "aspects as annotators, logit"),
    ("MT-Bench", "mt_bench_human", "votes_smoothed", "logit", "human votes per question, logit"),
    ("MT-Bench", "mt_bench_gpt4", "votes_smoothed", "logit", "GPT-4 votes per question, logit"),
    ("MT-Bench", "mt_bench_human_pooled", "votes_smoothed", "logit", "human votes pooled by model, logit"),
]

FRONTIER_SERIES = [
    ("UltraFeedback, aspects as annotators", "ultrafeedback", "attribute_vote_smoothed"),
    ("HelpSteer, attributes as annotators", "helpsteer", "attribute_vote_smoothed"),
    ("MT-Bench, human votes per question", "mt_bench_human", "votes_smoothed"),
    ("MT-Bench, GPT-4 votes per question", "mt_bench_gpt4", "votes_smoothed"),
]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def static_table():
    rows = []
    for group, ds, construction, link, label in STATIC_ROWS:
        panels = pd.read_csv(OUT / ds / "panels.csv")
        x = panels[(panels.construction == construction) & (panels.link == link)]
        share = x.cyclic_share
        null_median = np.nan
        if (OUT / ds / "null.csv").exists() and construction == "votes_smoothed":
            null_median = pd.read_csv(OUT / ds / "null.csv").null_median.median()
        rows.append(dict(group=group, dataset=ds, construction=construction, link=link, label=label,
                         panels=len(x), K=int(x.K.median()), completeness=x.completeness.mean(),
                         share_p25=share.quantile(0.25), share_median=share.median(),
                         share_p90=share.quantile(0.9), frac_share_gt_005=(share > 0.05).mean(),
                         condorcet_rate=x.has_condorcet_cycle.mean(), null_median=null_median))
    arena = json.loads((OUT / "extras" / "arena_clique.json").read_text())
    for key, label in (("min_votes=1", "34 models, every pair compared"),
                       ("min_votes=20", "14 models, >= 20 votes per pair")):
        a = arena[key]
        rows.append(dict(group="Arena 55k", dataset="arena55k", construction="votes_smoothed", link="logit",
                         label=f"human votes, {label}, logit", panels=1, K=a["K"], completeness=1.0,
                         share_p25=a["cyclic_share"], share_median=a["cyclic_share"], share_p90=a["cyclic_share"],
                         frac_share_gt_005=float(a["cyclic_share"] > 0.05),
                         condorcet_rate=float(a["condorcet_triples"] > 0), null_median=a["null_share_median"]))
    return pd.DataFrame(rows)


def figure_static(table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(table)
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * n + 1.3), facecolor=SURFACE)
    _style(ax)
    y = np.arange(n)[::-1]
    ax.hlines(y, table.share_p25, table.share_p90, color=SERIES[0], linewidth=2, zorder=2)
    ax.scatter(table.share_median, y, s=42, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5, zorder=3,
               label="observed: median (dot), 25th to 90th percentile (line)")
    has_null = table.null_median.notna()
    ax.scatter(table.null_median[has_null], y[has_null.to_numpy()] - 0.22, s=42, marker="D", color=SERIES[1],
               edgecolor=SURFACE, linewidth=1.5, zorder=4, label="Bradley-Terry null: median")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g}: {l}" for g, l in zip(table.group, table.label)], fontsize=8, color=INK)
    for boundary in np.flatnonzero(table.group.to_numpy()[1:] != table.group.to_numpy()[:-1]):
        ax.axhline(y[boundary] - 0.5, color=GRID, linewidth=0.8)
    ax.set_xlim(0, 0.75)
    ax.set_xlabel(r"cyclic energy share  $\|C\|_F^2 / \|A\|_F^2$", fontsize=9, color=INK2)
    ax.set_title("How much of the transformed preference flow is cyclic", fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper right", labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def frontier_table():
    rows = []
    for label, ds, construction in FRONTIER_SERIES:
        pop = pd.read_csv(OUT / ds / "population.csv")
        x = pop[(pop.construction == construction) & (pop.link == "logit") & (pop.lam == 1.0)]
        for _, r in x.iterrows():
            rows.append(dict(series=label, dataset=ds, construction=construction, alpha=r.alpha,
                             beta_lambda=r.beta, frac_beyond_frontier=r.frac_beyond_frontier,
                             gamma_median=r.gamma_median, entropy_median=r.entropy_median, panels=r.panels))
    return pd.DataFrame(rows)


def figure_frontier(table, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gains = (2.0, 10.0)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True, facecolor=SURFACE)
    for ax, gain in zip(axes, gains):
        _style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        for k, (label, *_ ) in enumerate(FRONTIER_SERIES):
            line = table[(table.series == label) & (table.beta_lambda == gain)].sort_values("alpha")
            ax.plot(line.alpha, line.frac_beyond_frontier, color=SERIES[k], linewidth=2, marker=MARKERS[k],
                    markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.2, label=label)
            if k == 2:
                last = line.iloc[-1]
                ax.annotate(f"{last.frac_beyond_frontier:.2f}", (last.alpha, last.frac_beyond_frontier),
                            xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK2)
        ax.set_title(fr"$\beta\lambda = {gain:g}$" + ("  (DPO $\\beta = 0.1$)" if gain == 10 else ""),
                     fontsize=9, color=INK, loc="left")
        ax.set_xlabel(r"reference refresh $\alpha$", fontsize=9, color=INK2)
        ax.set_xticks([0.5, 0.8, 0.9, 0.95, 0.99])
        ax.text(0.99, 0.012 if gain == 2.0 else 0.045, "UltraFeedback and HelpSteer stay at or below 0.002",
                fontsize=7.5, color=INK2, ha="right", va="bottom")
    axes[0].set_ylabel(r"share of panels with $\alpha^2 + \gamma^2 > 1$", fontsize=9, color=INK2)
    axes[0].set_ylim(-0.01, 0.4)
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left", labelcolor=INK2)
    fig.suptitle("Exact population recursion at the unique fixed point, uniform reference and coverage",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    dest = Path(argv[0]) if argv else ROOT / "reports" / "latest"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "figures").mkdir(exist_ok=True)
    static = static_table()
    static.to_csv(dest / "static_summary.csv", index=False)
    figure_static(static, dest / "figures" / "cyclic_share.png")
    frontier = frontier_table()
    frontier.to_csv(dest / "frontier_summary.csv", index=False)
    figure_frontier(frontier, dest / "figures" / "frontier.png")
    for ds in ("helpsteer", "helpsteer_quality3", "ultrafeedback", "mt_bench_human", "mt_bench_gpt4",
               "mt_bench_human_pooled", "arena55k"):
        pd.read_csv(OUT / ds / "summary.csv").to_csv(dest / f"summary_{ds}.csv", index=False)
    pd.read_csv(OUT / "extras" / "near_tie.csv").to_csv(dest / "near_tie.csv", index=False)
    for name in ("arena_clique.json",):
        (dest / name).write_text((OUT / "extras" / name).read_text())
    (dest / "helpsteer2_heterogeneity.json").write_text(
        (OUT / "helpsteer2_preference" / "heterogeneity.json").read_text())
    print(static.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
