"""Follow-up analyses for the report.

1. Near-tie stratification: the cyclic share and the population frontier on the
   panels with the weakest directional signal ||u|| (the regime singled out after
   prop:cyclic-sources, where IPO sees a small cycle and a weak direction).
2. Arena on a fully compared clique: the pooled arena panel is 64% complete, so its
   residual depends on the indifference fill; restricting to a set of models that
   were all compared with each other removes that dependence.

    python -m hodge.extras
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constructions import rating_constructions, vote_matrix
from .core import bt_null_cyclic_share, flow, hodge, panel_summary
from .datasets import LOADERS
from .population import ordinary_stability

ROOT = Path(__file__).resolve().parents[1]
GRID = [(a, b) for a in (0.5, 0.8, 0.9, 0.95, 0.99) for b in (1.0, 2.0, 5.0, 10.0)]


def near_tie(dataset, construction="attribute_vote_smoothed", link="logit", quantiles=(0.1, 0.25, 1.0),
             sample=1500, seed=0):
    panels = LOADERS[dataset](ROOT / "data")
    rows = []
    for p in panels:
        P = rating_constructions(p.ratings)[construction][0]
        h = hodge(flow(P, link))
        rows.append((p, h, float(np.linalg.norm(h.u)), float(np.sum(h.C ** 2) / max(np.sum(h.A ** 2), 1e-300))))
    norms = np.array([r[2] for r in rows])
    rng = np.random.default_rng(seed)
    out = []
    for q in quantiles:
        cut = np.quantile(norms, q)
        idx = np.flatnonzero(norms <= cut)
        idx = rng.choice(idx, size=min(sample, len(idx)), replace=False)
        shares = np.array([rows[i][3] for i in idx])
        for alpha, beta in GRID:
            idxs = [ordinary_stability(rows[i][1], alpha, beta, 1.0)["index"] for i in idx]
            out.append(dict(dataset=dataset, construction=construction, link=link, u_quantile=q,
                            u_cut=float(cut), panels=len(idx), share_median=float(np.median(shares)),
                            share_p90=float(np.quantile(shares, 0.9)), alpha=alpha, beta_lambda=beta,
                            frac_beyond_frontier=float(np.mean(np.array(idxs) > 1))))
    return pd.DataFrame(out)


def arena_clique(min_votes=1, draws=500, pseudo=0.5, seed=0):
    panel = LOADERS["arena55k"](ROOT / "data")[0]
    W = panel.wins
    N = W + W.T
    order = np.argsort(-N.sum(axis=1))
    clique = []
    for m in order:
        if all(N[m, k] >= min_votes for k in clique):
            clique.append(m)
    clique = sorted(clique)
    Wc = W[np.ix_(clique, clique)]
    observed = panel_summary(vote_matrix(Wc, pseudo), "logit")
    null = bt_null_cyclic_share(Wc, "logit", pseudo, draws=draws, seed=seed)
    stability = {f"alpha={a},beta_lambda={b}": round(ordinary_stability(hodge(flow(vote_matrix(Wc, pseudo), "logit")),
                                                                        a, b, 1.0)["index"], 4)
                 for a, b in GRID}
    return dict(models=[panel.labels[k] for k in clique], K=len(clique), min_votes_per_pair=int(N[np.ix_(clique, clique)][np.triu_indices(len(clique), 1)].min()),
                votes=float(Wc.sum()), cyclic_share=observed["cyclic_share"],
                condorcet_triples=observed["condorcet_triples"], sst_violation_rate=observed["sst_violation_rate"],
                null_share_median=float(np.median(null)),
                p_value=float((1 + np.sum(null >= observed["cyclic_share"])) / (1 + draws)),
                stability_index=stability)


def main():
    out = ROOT / "outputs" / "extras"
    out.mkdir(parents=True, exist_ok=True)
    tables = [near_tie(ds) for ds in ("helpsteer", "ultrafeedback")]
    near = pd.concat(tables)
    near.to_csv(out / "near_tie.csv", index=False)
    arena = {f"min_votes={m}": arena_clique(min_votes=m) for m in (1, 20)}
    (out / "arena_clique.json").write_text(json.dumps(arena, indent=1))
    focus = near[near.beta_lambda == 10.0].pivot_table(index=["dataset", "u_quantile", "share_median"],
                                                       columns="alpha", values="frac_beyond_frontier")
    print(focus.to_string())
    print(json.dumps({k: {kk: v for kk, v in d.items() if kk != "models"} for k, d in arena.items()}, indent=1))


if __name__ == "__main__":
    main()
