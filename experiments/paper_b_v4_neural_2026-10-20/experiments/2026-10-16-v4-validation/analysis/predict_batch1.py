"""Quantitative predictions for batch 1 (H1–H4), computed from the panels before any training run.

Writes prereg/predictions_batch1.json with, per configuration: population entropy and top-1 trajectories, the target
entropy at B = beta/(1-alpha), dominance statistics, and for H1 the pairwise KL between targets.
Usage: python predict_batch1.py <panels.json> <out.json>
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from population import condorcet_winner, entropy, hodge, link, margins, omega_max, sm, target, trajectory  # noqa: E402

D = json.load(open(sys.argv[1]))
panels = D['panels']
N, K = len(panels), len(panels[0]['responses'])
RM = 'Skywork-Reward-Llama-3.1-8B-v0.2-d4117fbfd81b'
R = np.array([p['rm'][RM] for p in panels])
s_ref = np.array([p['s0'] for p in panels])
gaps = R[:, :, None] - R[:, None, :]
iu = np.triu_indices(K, 1)
T_soft = 0.5 * float(np.std(gaps[:, iu[0], iu[1]]))   # half the std of within-panel reward gaps
P_soft = 1 / (1 + np.exp(-gaps / T_soft))
P_hard = (gaps > 0) + 0.5 * (gaps == 0)
out = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), N=N, K=K, T_soft=T_soft, configs={})


def summary(name, P, lk, alpha, beta, lam, T):
    A = link(P, lk)
    tr = trajectory(A, s_ref, alpha, beta, lam, T)
    d = dict(link=lk, alpha=alpha, beta=beta, lam=lam, T=T,
             entropy=np.round(entropy(tr).mean(-1), 4).tolist(), top=np.round(sm(tr).max(-1).mean(-1), 4).tolist())
    if alpha < 1:
        B = beta / (1 - alpha)
        X, res = target(A, s_ref, B, lam)
        d.update(B=B, target_entropy=float(entropy(X).mean()), target_top=float(sm(X).max(-1).mean()), target_resid=float(res.max()))
        gB = np.array([B * lam * omega_max(hodge(A[n])[1], sm(X[n])) for n in range(N)])
        rad = np.sqrt(alpha ** 2 + ((1 - alpha) * gB) ** 2)
        d.update(gammaB_median=float(np.median(gB)), gammaB_max=float(gB.max()), radius_max=float(rad.max()), share_unstable=float((rad > 1).mean()))
        d['_X'] = X
    out['configs'][name] = d
    return d


# H1: same target at B = 4 (IPO, soft oracle, lam = 0.8); control B = 1
for nm, (a, b) in {'H1_a0_b8': (0.0, 8.0), 'H1_a05_b4': (0.5, 4.0), 'H1_a08_b16': (0.8, 1.6), 'H1_a09_b08': (0.9, 0.8), 'H1_a05_b1': (0.5, 1.0)}.items():
    summary(nm, P_soft, 'identity', a, b, 0.8, 40)
X4 = out['configs']['H1_a0_b8']['_X']; X1 = out['configs']['H1_a05_b1']['_X']
kl = lambda X, Y: float((sm(X) * (np.log(sm(X)) - np.log(sm(Y)))).sum(-1).mean())
out['H1_KL_B4_vs_B1'] = kl(X4, X1)
# H2: plateau set by B (IPO, soft, beta = 1, lam = 0.8)
for a in (0.5, 0.8, 0.9, 0.95):
    summary(f'H2_a{a}', P_soft, 'identity', a, 1.0, 0.8, 80)
# H3: full refresh, IPO hard labels, lam = 1
d3 = summary('H3_a1_b1', P_hard, 'identity', 1.0, 1.0, 1.0, 60)
cw = np.array([condorcet_winner(P_hard[n]) for n in range(N)])
tr3 = trajectory(link(P_hard, 'identity'), s_ref, 1.0, 1.0, 1.0, 60)
d3['pop_top_is_condorcet_final'] = float(np.mean(sm(tr3[-1]).argmax(-1) == cw))
G = np.array([margins(link(P_hard[n], 'identity'), 1.0)[0] for n in range(N)])
pos = np.array([all(G[n, cw[n], j] > 0 for j in range(K) if j != cw[n]) for n in range(N)])
d3['share_positive_worst_margins'] = float(pos.mean())
# H4: link test at alpha = 0.8, beta = 1, lam = 0.8
summary('H4_ipo_hard', P_hard, 'identity', 0.8, 1.0, 0.8, 80)
summary('H4_dpo_soft', P_soft, 'logit', 0.8, 1.0, 0.8, 80)
out['configs']['H4_dpo_hard'] = dict(note='logit of deterministic labels is infinite: no finite target; prediction is continued concentration (entropy at t=80 well below t=30) limited only by the inner optimization')
for d in out['configs'].values():
    d.pop('_X', None)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk in ('B', 'target_entropy', 'target_top', 'pop_top_is_condorcet_final', 'share_positive_worst_margins', 'gammaB_median', 'gammaB_max', 'radius_max', 'share_unstable')} for k, v in out['configs'].items()}))
print('H1 KL(B=8 target || B=2 target) =', round(out['H1_KL_B4_vs_B1'], 4), ' T_soft =', round(T_soft, 3))
