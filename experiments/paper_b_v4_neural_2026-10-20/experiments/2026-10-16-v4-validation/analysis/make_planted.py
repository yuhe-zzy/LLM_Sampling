"""Planted-cycle oracle on the real panels (labelled control for H5/H6) and its frozen predictions.

For each panel, a random permutation assigns roles (1, 2, 3, 4) to the four responses. On roles 1-3,
A = g 1^T - 1 g^T + q R_cyc with g = (Delta, 0, -Delta); role 4 loses to every other role with A = -e.
P* = 1/2 + A (identity link). Half the panels are 'balanced' (Delta = 0.05 < q/2: majority cycle 1>2>3>1,
maximal lottery on the face of roles 1-3) and half are 'decisive' (Delta = 0.2 > q/2: role 1 is a Condorcet winner).

Predictions (IPO, identity link, lambda = 1) for each configuration (alpha, beta): per panel the target at
B = beta/(1-alpha), gamma_B, the local spectral radius sqrt(alpha^2 + ((1-alpha) gamma_B)^2), and whether the
target is predicted unstable; plus the exact population trajectory summary.
Usage: python make_planted.py <panels.json> <oracle_out.npz> <pred_out.json>
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from population import condorcet_winner, entropy, hodge, maximal_lottery, omega_max, sm, target, trajectory  # noqa: E402

q, e, D_BAL, D_DEC = 0.3, 0.3, 0.05, 0.2
rng = np.random.default_rng(20261016)
D = json.load(open(sys.argv[1]))
panels = D['panels']
N, K = len(panels), 4
s_ref = np.array([p['s0'] for p in panels])
R3 = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], float)
kind = np.array(['balanced'] * (N // 2) + ['decisive'] * (N - N // 2)); rng.shuffle(kind)
roles = np.array([rng.permutation(K) for _ in range(N)])   # roles[n][r] = response index holding role r
P = np.zeros((N, K, K))
for n in range(N):
    Dl = D_BAL if kind[n] == 'balanced' else D_DEC
    g = np.array([Dl, 0, -Dl])
    Ar = np.zeros((K, K))
    Ar[:3, :3] = g[:, None] - g[None, :] + q * R3
    Ar[3, :3] = -e; Ar[:3, 3] = e
    idx = roles[n]
    A = np.zeros((K, K))
    for a in range(K):
        for b in range(K):
            A[idx[a], idx[b]] = Ar[a, b]
    P[n] = 0.5 + A
assert P.min() >= -1e-12 and P.max() <= 1 + 1e-12
np.savez(sys.argv[2], ids=np.array([p['id'] for p in panels]), P=P, kind=kind, roles=roles)

A = P - 0.5
cw = np.array([condorcet_winner(P[n]) for n in range(N)])
ml = np.array([maximal_lottery(A[n]) for n in range(N)])
out = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), q=q, e=e, delta_balanced=D_BAL, delta_decisive=D_DEC,
           n_balanced=int((kind == 'balanced').sum()), n_decisive=int((kind == 'decisive').sum()),
           condorcet_winner_exists={k: float(np.mean(cw[kind == k] >= 0)) for k in ('balanced', 'decisive')},
           ml_support={k: np.bincount((ml[kind == k] > 1e-6).sum(1), minlength=K + 1).tolist() for k in ('balanced', 'decisive')},
           configs={})
CONFIGS = {  # H5: fixed beta = 4, growing alpha; H6: fixed target B = 12, growing alpha
    'H5_b4_a05': (0.5, 4.0), 'H5_b4_a09': (0.9, 4.0), 'H5_b4_a095': (0.95, 4.0),
    'H6_B12_a02': (0.2, 9.6), 'H6_B12_a05': (0.5, 6.0), 'H6_B12_a08': (0.8, 2.4), 'H6_B12_a09': (0.9, 1.2)}
TR = 80


def signatures(tr, X):
    """Late-window signatures: top-response switches in the last 30 rounds, and the ratio of the distance to the
    target between round T and round T-20 (well below 1 = converging; about 1 or above = not converging)."""
    top = sm(tr).argmax(-1)                                      # [T+1, N]
    switches = (np.diff(top[-31:], axis=0) != 0).sum(0)
    c = lambda x: x - x.mean(-1, keepdims=True)
    dist = np.abs(c(tr) - c(X)[None]).max(-1)                    # [T+1, N]
    ratio = dist[-1] / np.maximum(dist[-21], 1e-12)
    return switches, ratio, dist[-1]


for name, (a, b) in CONFIGS.items():
    B = b / (1 - a)
    X, res = target(A, s_ref, B, 1.0)
    gB = np.array([B * omega_max(hodge(A[n])[1], sm(X[n])) for n in range(N)])
    rad = np.sqrt(a ** 2 + ((1 - a) * gB) ** 2)
    tr = trajectory(A, s_ref, a, b, 1.0, TR)
    sw, ratio, dfin = signatures(tr, X)
    d = dict(alpha=a, beta=b, B=B, rounds=TR, target_resid_max=float(res.max()))
    for k in ('balanced', 'decisive'):
        m = kind == k
        d[k] = dict(gammaB_median=float(np.median(gB[m])), radius_median=float(np.median(rad[m])), share_unstable=float((rad[m] > 1).mean()),
                    pop_switch_share=float((sw[m] > 0).mean()), pop_ratio_median=float(np.median(ratio[m])), pop_final_dist_median=float(np.median(dfin[m])),
                    target_top_median=float(np.median(sm(X[m]).max(-1))), target_to_ML_L1_median=float(np.median(np.abs(sm(X[m]) - ml[m]).sum(-1))))
    out['configs'][name] = d
json.dump(out, open(sys.argv[3], 'w'), indent=1)
print(json.dumps(out, indent=1)[:3500])
