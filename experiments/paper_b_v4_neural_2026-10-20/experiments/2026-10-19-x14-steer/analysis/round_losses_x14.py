"""Per-round training losses of the X14 runs (2026-10-19; copy of experiments/2026-10-17-controls/analysis/round_losses_ctrl.py
with the D5 label correction of steer.json added: from the loop round of the D5 decision on, the labels' log-odds carry
- eps_hat (s_i - s_j) of the current policy). Also used for the Sycamore 3B runs (no steering).

Original description: per-round training losses of the control runs, reconstructed from the saved panel scores (2026-10-18).
Mirrors the expected-label loss of vm/panel_po_ctrl.py round by round: reference r_t (alpha, or the per-panel switch alphas),
judge-biased labels P_eff, replay sampler m_t, per-panel step beta_t (panels with beta_t = 0 are left out; the loss is still
divided by all N panels), and the IPO square or DPO soft-label logistic loss. Per round it reports
  logged_first, logged_last : the loss logged at inner steps 1 and 40
  start : L_t at h = s_t - r_t (should equal logged_first)
  end   : L_t at the trained model, h = s_{t+1} - r_t
  opt   : the exact minimum of L_t over panel scores
Usage: python round_losses_ctrl.py <runs dir> <out json>"""
import json, os, sys
import numpy as np
from scipy.optimize import minimize
RUNS, OUT = sys.argv[1:3]
K = 4; off = 1 - np.eye(K)
sm = lambda x: np.exp(x - x.max(-1, keepdims=True)) / np.exp(x - x.max(-1, keepdims=True)).sum(-1, keepdims=True)
sig = lambda x: 1 / (1 + np.exp(-x))


def panel_loss(h, P, W, beta, obj):
    d = h[:, None] - h[None, :]
    if obj == 'ipo':
        m = (d - beta * (P - 0.5)) ** 2
    else:
        z = d / beta; m = -(P * -np.logaddexp(0, -z) + (1 - P) * -np.logaddexp(0, z))
    return float((W * m * off).sum())


def panel_opt(P, W, beta, obj, h0):
    if obj == 'ipo':
        rows, tg = [], []
        for i in range(K):
            for j in range(K):
                if i != j and W[i, j] > 0:
                    e = np.zeros(K); e[i], e[j] = 1, -1; rows.append(np.sqrt(W[i, j]) * e); tg.append(np.sqrt(W[i, j]) * beta * (P[i, j] - 0.5))
        x, *_ = np.linalg.lstsq(np.array(rows), np.array(tg), rcond=None); return panel_loss(x, P, W, beta, obj)
    # DPO soft labels: minimize over z = d/beta, i.e. the minimum does not depend on beta; scale h0 accordingly
    f = lambda x: panel_loss(np.r_[0.0, x], P, W, 1.0, obj)
    # two starts (the trained model, and zero), keep the lower minimum: at tiny beta the scaled start can stall BFGS
    best = min(minimize(f, x0, method='BFGS', options={'gtol': 1e-10, 'maxiter': 500}).fun for x0 in (((h0 - h0[0]) / beta)[1:], np.zeros(K - 1)))
    return float(best)


out = {}
for r in sorted(os.listdir(RUNS)):
    if not os.path.exists(os.path.join(RUNS, r, 'run.npz')): continue
    Z = np.load(os.path.join(RUNS, r, 'run.npz')); cfg = json.load(open(os.path.join(RUNS, r, 'config.json')))
    s = Z['s'].astype(np.float64); P = Z['P'].astype(np.float64); N = s.shape[1]; T = s.shape[0] - 1
    K = s.shape[2]; off = 1 - np.eye(K)   # panel size per run (K = 4 in X14 and 3B; K = 3 in X15)
    obj, lam, rho, eps = cfg['objective'], cfg['lam'], cfg.get('replay', 1.0), cfg.get('judge_bias', 0.0)
    betas = Z['betas'].astype(np.float64); alphas = Z['alphas'].astype(np.float64)
    logitP = np.log(np.clip(P, 1e-6, 1 - 1e-6) / (1 - np.clip(P, 1e-6, 1 - 1e-6)))
    stj = os.path.join(RUNS, r, 'steer.json'); st_info = json.load(open(stj)) if os.path.exists(stj) else {}
    d5 = (st_info or {}).get('d5') or {}; corr, corr_from = d5.get('bias_correction', 0.0) or 0.0, d5.get('round', 10 ** 9)
    m = sm(s[0]); rows = dict(logged_first=Z['loss_first'].tolist(), logged_last=Z['loss_last'].tolist(), start=[], end=[], opt=[])
    for t in range(T):
        st, s0, s1 = s[t], s[0], s[t + 1]
        if betas.ndim == 1: betas = np.repeat(betas[:, None], N, 1)
        pi = sm(st); m = (1 - rho) * m + rho * pi; mu = (1 - lam) / K + lam * m
        a = alphas[t] if alphas.ndim == 2 else np.full(N, alphas[t])
        rt = a[:, None] * st + (1 - a[:, None]) * s0
        sb = s0 if cfg.get('judge_frozen') else st
        c = corr if t + 1 >= corr_from else 0.0                # loop round t+1 (1-based); correction from the D5 round on
        Pe = sig(logitP + eps * (sb[:, :, None] - sb[:, None, :]) - c * (st[:, :, None] - st[:, None, :])) * off + 0.5 * np.eye(K) if eps else P
        b = betas[t]; st_ = en_ = op_ = 0.0
        for n in range(N):
            if b[n] <= 1e-6: continue
            W = np.outer(mu[n], mu[n]) * off; W = W / W.sum()
            st_ += panel_loss(st[n] - rt[n], Pe[n], W, b[n], obj); en_ += panel_loss(s1[n] - rt[n], Pe[n], W, b[n], obj)
            op_ += panel_opt(Pe[n], W, b[n], obj, s1[n] - rt[n])
        rows['start'].append(st_ / N); rows['end'].append(en_ / N); rows['opt'].append(op_ / N)
    rows['cfg'] = {k: cfg.get(k) for k in ('objective', 'alpha', 'beta', 'lam', 'judge_bias', 'judge_frozen', 'beta_ctrl', 'replay', 'adaptive_W', 'B')}
    out[r] = rows
    f, st_, en_, op_ = (np.array(rows[k]) for k in ('logged_first', 'start', 'end', 'opt'))
    print(f'{r:12s} logged vs start max rel diff {np.max(np.abs(f - st_) / (np.abs(st_) + 1e-12)):.3g}; excess start median {np.median(st_ - op_):.3g}, '
          f'end median {np.median(en_ - op_):.3g}, end median t>=5 {np.median((en_ - op_)[4:]):.3g}, end max {np.max(en_ - op_):.3g}', flush=True)
json.dump(out, open(OUT, 'w'))
