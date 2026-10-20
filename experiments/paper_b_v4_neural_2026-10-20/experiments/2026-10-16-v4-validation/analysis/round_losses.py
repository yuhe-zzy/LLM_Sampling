"""Per-round training losses of the expected-label runs, reconstructed from the saved panel scores (2026-10-17).

Each outer round t minimizes L_t(h) = mean over panels of sum_{i != j} W_ij l(h_i - h_j), with h = s_theta - r_t,
r_t = alpha s_t + (1 - alpha) s_0, W = mu_t mu_t^T (off-diagonal, normalized per panel), and l the IPO square loss
(d - beta A_ij)^2 or the DPO soft-label logistic loss (vm/panel_po.py). The script reports, per round:
  logged_first, logged_last : the training loss logged at inner steps 1 and 40 (results/.../run.npz)
  start  : L_t at the start of the round (h = s_t - r_t), recomputed; should equal logged_first
  end    : L_t at the trained model after the round (h = s_{t+1} - r_t), recomputed
  opt    : the exact minimum of L_t over panel scores (IPO: weighted least squares; DPO: convex, Newton)
Usage: python round_losses.py <out json>"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from population import sm  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
RUNS = [('runs', r) for r in ('H1_a0_b8', 'H1_a05_b4', 'H1_a08_b16', 'H1_a09_b08', 'H1_a05_b1', 'H2_a0.8', 'H2_a0.9', 'H2_a0.95', 'H3_a1_b1',
                              'H4_dpo_soft', 'H4_ipo_hard', 'H4_dpo_hard', 'H5_b4_a05', 'H5_b4_a09', 'H5_b4_a095',
                              'H6_B12_a02', 'H6_B12_a05', 'H6_B12_a08', 'H6_B12_a09')] + \
       [('runs_q15', r) for r in ('R1_q15_a0_b8', 'R1_q15_a08_b16', 'R2_q15_dpo_soft', 'R3_q15_a02', 'R3_q15_a09')]
K = 4
off = 1 - np.eye(K)


def panel_loss(h, P, W, beta, obj):
    d = h[:, None] - h[None, :]
    if obj == 'ipo':
        m = (d - beta * (P - 0.5)) ** 2
    else:
        z = d / beta
        m = -(P * -np.logaddexp(0, -z) + (1 - P) * -np.logaddexp(0, z))
    return float((W * m * off).sum())


def panel_opt(P, W, beta, obj, h0):
    if obj == 'ipo':                                   # weighted least squares on pairwise differences
        rows, tg = [], []
        for i in range(K):
            for j in range(K):
                if i != j and W[i, j] > 0:
                    e = np.zeros(K); e[i], e[j] = 1, -1
                    rows.append(np.sqrt(W[i, j]) * e); tg.append(np.sqrt(W[i, j]) * beta * (P[i, j] - 0.5))
        x, *_ = np.linalg.lstsq(np.array(rows), np.array(tg), rcond=None)
        return panel_loss(x, P, W, beta, obj)
    f = lambda x: panel_loss(np.r_[0.0, x], P, W, beta, obj)
    r = minimize(f, (h0 - h0[0])[1:], method='BFGS', options={'gtol': 1e-10, 'maxiter': 500})
    return float(r.fun)


out = {}
for d, r in RUNS:
    Z = np.load(os.path.join(RES, d, r, 'run.npz')); cfg = json.load(open(os.path.join(RES, d, r, 'config.json')))
    s = Z['s'].astype(np.float64); P = Z['P'].astype(np.float64); N = s.shape[1]
    a, b, lam, obj = cfg['alpha'], cfg['beta'], cfg['lam'], cfg['objective']
    hard_dpo = obj == 'dpo' and np.all((P * off == 0) | (P * off == 1))
    rows = dict(logged_first=Z['loss_first'].tolist(), logged_last=Z['loss_last'].tolist(), start=[], end=[], opt=[])
    for t in range(s.shape[0] - 1):
        st, s0, s1 = s[t], s[0], s[t + 1]
        mu = (1 - lam) / K + lam * sm(st)
        rt = a * st + (1 - a) * s0
        st_, en_, op_ = 0.0, 0.0, 0.0
        for n in range(N):
            W = np.outer(mu[n], mu[n]) * off; W = W / W.sum()
            st_ += panel_loss(st[n] - rt[n], P[n], W, b, obj); en_ += panel_loss(s1[n] - rt[n], P[n], W, b, obj)
            op_ += 0.0 if hard_dpo else panel_opt(P[n], W, b, obj, s1[n] - rt[n])
        rows['start'].append(st_ / N); rows['end'].append(en_ / N); rows['opt'].append(op_ / N)
    rows['cfg'] = dict(alpha=a, beta=b, lam=lam, objective=obj, hard_dpo=bool(hard_dpo))
    out[f'{d}/{r}'] = rows
    ex_s = np.array(rows['start']) - np.array(rows['opt']); ex_e = np.array(rows['end']) - np.array(rows['opt'])
    print(f"{r:16s} logged_first vs start max rel diff {np.max(np.abs(np.array(rows['logged_first']) - rows['start']) / (np.abs(rows['start']) + 1e-12)):.3g}; "
          f"excess start median {np.median(ex_s):.3g}, end median {np.median(ex_e):.3g}, ratio {np.median(ex_e / np.maximum(ex_s, 1e-30)):.3g}", flush=True)
json.dump(out, open(sys.argv[1], 'w'))
