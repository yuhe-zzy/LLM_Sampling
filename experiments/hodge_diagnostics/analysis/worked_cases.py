"""The four worked cases cited in the report and handoff.

UltraFeedback (aspects as annotators, logit, 0.5 pseudo-votes, uniform reference):
  typical panel     the panel whose cyclic share is the dataset median
  exception         the most cyclic panel with four distinct rating rows
  plus the exact count of panels beyond the frontier at three settings.
MT-Bench: the two human per-question panels with the largest index at alpha = 0.9,
beta lambda = 10, each contrasted with the GPT-4 judge on the same question and turn.
Trajectories iterate the exact population recursion from the uniform policy.
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from _common import DATA, OUT
from hodge.constructions import attribute_vote, vote_matrix
from hodge.core import bt_null_cyclic_share, fit_bt, flow, hodge, omega_max, ordinal_stats
from hodge.datasets import LOADERS, ULTRAFEEDBACK_ASPECTS, _rating
from hodge.population import fixed_point, forcing, ordinary_stability, simulate, softmax

np.set_printoptions(precision=3, suppress=True, linewidth=150)
record = {}


def dynamics(h, K, settings, T=3000):
    rows = []
    for alpha, gain in settings:
        b, g = forcing(h, alpha, gain, 1.0)
        x = fixed_point(h.C, b, alpha, g)
        pi = softmax(x)
        gamma = g * omega_max(h.C, pi)
        xs = simulate(h.C, b, g, np.zeros(K), T, (alpha,))
        tail = np.array([softmax(v) for v in xs[-300:]])
        rows.append(dict(alpha=alpha, beta_lambda=gain, pi_star=np.round(pi, 3).tolist(), gamma=float(gamma),
                         index=float(alpha ** 2 + gamma ** 2), gamma_if_uniform=float(g * omega_max(h.C, np.full(K, 1 / K))),
                         tail_range=np.round(tail.max(0) - tail.min(0), 3).tolist()))
    return rows


# UltraFeedback ------------------------------------------------------------------
panels = LOADERS["ultrafeedback"](DATA)
flows = [flow(attribute_vote(p.ratings, pseudo=0.5), "logit") for p in panels]
parts = [hodge(A) for A in flows]
share = np.array([np.sum(h.C ** 2) / max(np.sum(A ** 2), 1e-300) for h, A in zip(parts, flows)])
typical = int(np.argmin(np.abs(share - np.median(share))))
exception = next(int(k) for k in np.argsort(-share)
                 if len({tuple(r) for r in panels[k].ratings}) == 4 and not np.isnan(panels[k].ratings).any())
texts, n = {}, 0
for path in sorted((DATA / "ultrafeedback").glob("train-*.parquet")):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2000, columns=["instruction", "completions"]):
        for row in batch.to_pylist():
            comps = row["completions"] or []
            R = np.array([[_rating((c.get("annotations") or {}).get(a)) for a in ULTRAFEEDBACK_ASPECTS]
                          for c in comps]).reshape(len(comps), 4)
            if len(comps) < 3 or np.all(np.isnan(R)):
                continue
            if n in (typical, exception):
                texts[n] = row["instruction"][:160].replace("\n", " ")
            n += 1
for label, k in (("typical", typical), ("exception", exception)):
    p, h, A = panels[k], parts[k], flows[k]
    case = dict(panel=p.panel_id, source=p.meta["source"], instruction=texts[k],
                ratings={m: r.tolist() for m, r in zip(p.labels, p.ratings)}, aspects=ULTRAFEEDBACK_ASPECTS,
                u=np.round(h.u, 3).tolist(), cyclic_share=float(share[k]), ordinal=ordinal_stats(np.array(
                    attribute_vote(p.ratings, pseudo=0.5))),
                dynamics=dynamics(h, 4, [(0.9, 1.0), (0.9, 10.0), (0.99, 1.0), (0.99, 10.0)]))
    record[f"ultrafeedback_{label}"] = case
counts = {}
for alpha, gain in ((0.9, 10.0), (0.99, 1.0), (0.9, 2.0)):
    c = sum(ordinary_stability(h, alpha, gain, 1.0)["index"] > 1 for h in parts)
    counts[f"alpha={alpha}, beta_lambda={gain}"] = f"{c} of {len(parts)}"
record["ultrafeedback_beyond_frontier_counts"] = counts

# MT-Bench -----------------------------------------------------------------------
human = {p.panel_id: p for p in LOADERS["mt_bench_human"](DATA)}
gpt4 = {p.panel_id: p for p in LOADERS["mt_bench_gpt4"](DATA)}
raw = pd.read_parquet(DATA / "mt_bench" / "human.parquet")
ranked = sorted(((ordinary_stability(hodge(flow(vote_matrix(p.wins, 0.5), "logit")), 0.9, 10.0, 1.0)["index"], pid)
                 for pid, p in human.items()), reverse=True)
record["mt_bench_human_beyond_frontier_alpha0.9_bl10"] = f"{sum(i > 1 for i, _ in ranked)} of {len(ranked)}"
for index, pid in ranked[:2]:
    q, turn = pid.split("-")
    conv = raw[(raw.question_id == int(q)) & (raw.turn == int(turn))].conversation_a.iloc[0]
    conv = ast.literal_eval(conv) if isinstance(conv, str) else conv
    case = dict(question=int(q), turn=int(turn), prompt=str(conv[0]["content"])[:160].replace("\n", " "))
    for judge, pool in (("human", human), ("gpt4", gpt4)):
        p = pool[pid]
        P = vote_matrix(p.wins, 0.5)
        h = hodge(flow(P, "logit"))
        null = bt_null_cyclic_share(p.wins, "logit", 0.5, draws=500, seed=0)
        obs = float(np.sum(h.C ** 2) / np.sum(h.A ** 2))
        case[judge] = dict(models=p.labels, wins=p.wins.tolist(), votes=float(p.wins.sum()), cyclic_share=obs,
                           null_p=float((1 + np.sum(null >= obs)) / 501), bt_scores=np.round(fit_bt(p.wins), 2).tolist(),
                           ordinal=ordinal_stats(P), dynamics=dynamics(h, 6, [(0.9, 2.0), (0.9, 10.0), (0.99, 10.0)], T=4000))
    record[f"mt_bench_q{q}_turn{turn}"] = case

(OUT / "worked_cases.json").write_text(json.dumps(record, indent=1, default=float))
print(json.dumps(record, indent=1, default=float)[:6000])
