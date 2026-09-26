"""Run the Hodge diagnostics on one dataset.

    python -m hodge.run helpsteer --out outputs/helpsteer

Writes panels.csv (one row per panel x construction x link), summary.csv,
population.csv (exact population index at the unique fixed point, uniform
reference and coverage), null.csv for vote datasets, figures, and report.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constructions import rating_constructions, vote_matrix
from .core import bt_null_cyclic_share, flow, hodge, panel_summary
from .datasets import LOADERS
from .population import ordinary_stability

ALPHAS = (0.5, 0.8, 0.9, 0.95, 0.99)
BETAS = (1.0, 2.0, 5.0, 10.0)
LAMBDAS = (0.5, 1.0)
SHARE_THRESHOLDS = (0.05, 0.2)


def constructions_for(panel, bt_scale, pseudo):
    if panel.ratings is not None:
        return rating_constructions(panel.ratings, bt_scale=bt_scale, pseudo=pseudo)
    return {"votes": (vote_matrix(panel.wins), {"identity"}),
            "votes_smoothed": (vote_matrix(panel.wins, pseudo), {"identity", "logit"})}


def panel_rows(panels, bt_scale, pseudo):
    rows = []
    for panel in panels:
        for name, (P, links) in constructions_for(panel, bt_scale, pseudo).items():
            for link in sorted(links):
                rows.append(dict(dataset=panel.dataset, panel_id=panel.panel_id, construction=name,
                                 **panel_summary(P, link)))
    return pd.DataFrame(rows)


def summarize(df):
    out = []
    for (construction, link), x in df.groupby(["construction", "link"], sort=False):
        share = x.cyclic_share
        row = dict(construction=construction, link=link, panels=len(x),
                   K_median=x.K.median(), K_max=x.K.max(), d_median=x.d.median(),
                   completeness_mean=x.completeness.mean(),
                   share_median=share.median(), share_p90=share.quantile(0.9))
        for t in SHARE_THRESHOLDS:
            row[f"frac_share_gt_{t}"] = (share > t).mean()
        row.update(
            condorcet_rate=x.has_condorcet_cycle.mean(),
            cyclic_without_condorcet=((share > SHARE_THRESHOLDS[0]) & ~x.has_condorcet_cycle).mean(),
            sst_violation_mean=x.sst_violation_rate.mean(),
            a_median=x.a.median(), norm_u_median=x.norm_u.median(),
            L_C_median=x.L_C.median(), C_norm2_median=x.C_norm2.median(),
            omega_uniform_median=x.omega_uniform.median(), omega_uniform_p90=x.omega_uniform.quantile(0.9),
            fact_bounds_hold=bool(((x.L_C <= x.fact_L_C_bound + 1e-9) & (x.C_norm2 <= x.fact_C2_bound + 1e-9)).all()))
        out.append(row)
    return pd.DataFrame(out)


def population(panels, bt_scale, pseudo, sample, seed):
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(panels), size=min(sample, len(panels)), replace=False)
    rows = []
    for idx in chosen:
        panel = panels[idx]
        for name, (P, links) in constructions_for(panel, bt_scale, pseudo).items():
            if name in ("single_score_hard", "attribute_vote", "votes"):
                continue  # unsmoothed variants duplicate their smoothed or soft counterparts
            for link in sorted(links):
                h = hodge(flow(P, link))
                for alpha in ALPHAS:
                    for beta in BETAS:
                        for lam in LAMBDAS:
                            s = ordinary_stability(h, alpha, beta, lam)
                            rows.append(dict(construction=name, link=link, panel_id=panel.panel_id,
                                             alpha=alpha, beta=beta, lam=lam, gamma=s["gamma"],
                                             index=s["index"], entropy=s["entropy"], min_prob=s["min_prob"]))
    df = pd.DataFrame(rows)
    grouped = df.groupby(["construction", "link", "alpha", "beta", "lam"])
    table = grouped.agg(panels=("index", "size"),
                        frac_beyond_frontier=("index", lambda v: float(np.mean(v > 1))),
                        gamma_median=("gamma", "median"), gamma_p90=("gamma", lambda v: v.quantile(0.9)),
                        entropy_median=("entropy", "median")).reset_index()
    return table


def null_tests(panels, pseudo, draws, seed):
    rows, null_means = [], []
    for panel in panels:
        if panel.wins is None:
            return pd.DataFrame(), {}
        observed = panel_summary(vote_matrix(panel.wins, pseudo), "logit")
        if observed["edges"] < 3 or observed["observed_triples"] == 0:
            continue
        draws_ = bt_null_cyclic_share(panel.wins, "logit", pseudo, draws=draws, seed=seed)
        rows.append(dict(panel_id=panel.panel_id, votes=int(panel.wins.sum()), edges=observed["edges"],
                         cyclic_share=observed["cyclic_share"], null_median=float(np.median(draws_)),
                         p_value=float((1 + np.sum(draws_ >= observed["cyclic_share"])) / (1 + draws))))
        null_means.append(draws_)
    df = pd.DataFrame(rows)
    if df.empty:
        return df, {}
    pooled_null = np.mean(np.vstack(null_means), axis=0)
    pooled = dict(panels=len(df), observed_mean_share=float(df.cyclic_share.mean()),
                  null_mean_share_median=float(np.median(pooled_null)),
                  pooled_p_value=float((1 + np.sum(pooled_null >= df.cyclic_share.mean())) / (1 + draws)),
                  frac_panels_p_lt_0_05=float(np.mean(df.p_value < 0.05)))
    return df, pooled


def figures(df, pop, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    combos = list(df.groupby(["construction", "link"], sort=False).groups)
    fig, axes = plt.subplots(1, len(combos), figsize=(3.2 * len(combos), 2.8), squeeze=False)
    for ax, (construction, link) in zip(axes[0], combos):
        share = df[(df.construction == construction) & (df.link == link)].cyclic_share
        ax.hist(share, bins=np.linspace(0, 1, 41), color="#4C72B0")
        ax.set_title(f"{construction}\n{link}", fontsize=8)
        ax.set_xlabel(r"$\|C\|_F^2/\|A\|_F^2$", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "cyclic_share.png", dpi=150)
    plt.close(fig)
    if pop is None or pop.empty:
        return
    combos = list(pop.groupby(["construction", "link"], sort=False).groups)
    fig, axes = plt.subplots(1, len(combos), figsize=(3.2 * len(combos), 2.8), squeeze=False, sharey=True)
    for ax, (construction, link) in zip(axes[0], combos):
        sub = pop[(pop.construction == construction) & (pop.link == link) & (pop.lam == 1.0)]
        for beta, line in sub.groupby("beta"):
            ax.plot(line.alpha, line.frac_beyond_frontier, marker="o", label=fr"$\beta\lambda={beta:g}$")
        ax.set_title(f"{construction}\n{link}, " + r"$\lambda=1$", fontsize=8)
        ax.set_xlabel(r"$\alpha$")
    axes[0][0].set_ylabel(r"fraction with $\alpha^2+\gamma^2>1$")
    axes[0][-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "population_frontier.png", dpi=150)
    plt.close(fig)


def markdown(title, summary, pop, null_df, pooled, meta):
    lines = [f"# {title}", "", "```", json.dumps(meta, indent=1), "```", "", "## Static Hodge diagnostics", "",
             summary.to_markdown(index=False, floatfmt=".3g"), ""]
    if pop is not None and not pop.empty:
        focus = pop[(pop.lam == 1.0)].pivot_table(index=["construction", "link", "beta"], columns="alpha",
                                                  values="frac_beyond_frontier")
        lines += ["## Fraction of panels beyond the ordinary frontier (lambda = 1, uniform reference)", "",
                  focus.to_markdown(floatfmt=".3f"), ""]
    if pooled:
        lines += ["## Bradley--Terry null test (logit link, smoothed votes)", "", "```",
                  json.dumps(pooled, indent=1), "```", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=list(LOADERS))
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N panels")
    parser.add_argument("--bt-scale", type=float, default=1.0, help="logit per rating point")
    parser.add_argument("--pseudo", type=float, default=0.5, help="pseudo-count for logit-link shares")
    parser.add_argument("--pop-sample", type=int, default=1000)
    parser.add_argument("--null-draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    out = Path(args.out or Path(__file__).resolve().parents[1] / "outputs" / args.dataset)
    out.mkdir(parents=True, exist_ok=True)

    panels = LOADERS[args.dataset](args.root)
    if args.limit:
        panels = panels[: args.limit]
    meta = dict(dataset=args.dataset, panels=len(panels), bt_scale=args.bt_scale, pseudo=args.pseudo,
                pop_sample=args.pop_sample, seed=args.seed, alphas=ALPHAS, betas=BETAS, lambdas=LAMBDAS,
                reference="uniform on the panel", coverage="uniform on the panel")
    df = panel_rows(panels, args.bt_scale, args.pseudo)
    df.to_csv(out / "panels.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out / "summary.csv", index=False)
    pop = population(panels, args.bt_scale, args.pseudo, args.pop_sample, args.seed) if args.pop_sample else None
    if pop is not None:
        pop.to_csv(out / "population.csv", index=False)
    null_df, pooled = null_tests(panels, args.pseudo, args.null_draws, args.seed)
    if not null_df.empty:
        null_df.to_csv(out / "null.csv", index=False)
    figures(df, pop, out)
    (out / "report.md").write_text(markdown(args.dataset, summary, pop, null_df, pooled, meta))
    print((out / "report.md").read_text())


if __name__ == "__main__":
    main()
