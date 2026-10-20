"""Frozen population prediction for the E3' rerun (refresh until the drift budget is used, then anchor), 2026-10-18.
Usage: python predict_switch.py <v4 experiment dir> <prereg dir>"""
import json, os, sys, time
import numpy as np
V4, PRE = sys.argv[1:3]
sys.path.insert(0, os.path.join(V4, 'analysis'))
from population import link, sm, target, entropy
cen = lambda x: x - x.mean(-1, keepdims=True)
kl = lambda p, q: (p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum(-1)
Z = np.load(os.path.join(V4, 'results', 'runs_sampled', 'S1n512_dpo_a0_b8', 'run.npz')); s0 = Z['s'][0].astype(float); A = link(Z['P'].astype(float), 'logit')
D = np.load(os.path.join(PRE, 'drift_targets_B8.npz'))['D']; X8, _ = target(A, s0, 8.0, 0.8)
s = s0.copy(); a = np.ones(len(s0)); pi0 = sm(s0); switch_round = np.full(len(s0), -1)
for t in range(1, 61):
    pi = sm(s); d = kl(pi, pi0); newly = (d >= D) & (a == 1.0); a = np.where(d >= D, 0.875, a); switch_round[newly] = t
    mu = 0.2 / 4 + 0.8 * pi
    s = cen(a[:, None] * s + (1 - a[:, None]) * s0 + 1.0 * np.einsum('nij,nj->ni', A, mu))
k = kl(sm(s), sm(X8))
pred = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), run='C3b_switch', H_t60=float(entropy(s).mean()), H_target_B8=float(entropy(X8).mean()),
            KL_to_B8_median=float(np.median(k)), KL_to_B8_mean=float(k.mean()), panels_KL_gt_01=int((k > 0.1).sum()),
            switch_round_median=float(np.median(switch_round[switch_round > 0])), never_switched=int((switch_round < 0).sum()),
            criteria='C3b_switch: median KL to the B=8 target <= 0.02, mean entropy within 0.05 of the target entropy (the frozen E3 criterion); also report panels with KL > 0.1 (C3_drift had 16 of 64).')
json.dump(pred, open(os.path.join(PRE, 'predictions_switch.json'), 'w'), indent=1); print(json.dumps(pred, indent=1))
