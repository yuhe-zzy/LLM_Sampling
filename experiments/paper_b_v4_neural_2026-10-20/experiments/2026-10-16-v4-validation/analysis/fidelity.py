"""Compare a neural panel run with the exact population recursion.

  one-step residual  e_t = || Pi s_{t+1} - Pi F(s_t, s_{t-1}) ||_inf   (F = exact round map from the *neural* state)
  open-loop gap      g_t = || Pi s_t - Pi s_t^pop ||_inf                (population trajectory from the same start)
Also reports panel entropy and top-1 mass over rounds for both.

Usage: python fidelity.py <run_dir> [<run_dir> ...]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from population import entropy, link, sm, trajectory  # noqa: E402


def load(run):
    Z = np.load(os.path.join(run, 'run.npz'))
    cfg = json.load(open(os.path.join(run, 'config.json')))
    return Z, cfg


def one_step(cfg, A, s_t, s_prev, s_ref):
    K = s_t.shape[-1]
    mu = (1 - cfg['lam']) / K + cfg['lam'] * sm(s_t)
    a, nu, b = cfg['alpha'], cfg.get('nu', 0.0), cfg['beta']
    return (a - nu) * s_t + nu * s_prev + (1 - a) * s_ref + b * np.einsum('nij,nj->ni', A, mu)


def center(x):
    return x - x.mean(-1, keepdims=True)


def analyse(run):
    Z, cfg = load(run)
    s = Z['s'].astype(np.float64)                    # [T+1, N, K]
    P = Z['P']
    lk = 'identity' if cfg['objective'] == 'ipo' else 'logit'
    A = link(P, lk)
    s_ref = s[0]
    res = []
    for t in range(1, s.shape[0]):
        s_prev = s[t - 2] if t >= 2 else s[0]
        pred = one_step(cfg, A, s[t - 1], s_prev, s_ref)
        res.append(np.abs(center(s[t]) - center(pred)).max(-1))
    res = np.array(res)                              # [T, N]
    pop = trajectory(A, s_ref, cfg['alpha'], cfg['beta'], cfg['lam'], s.shape[0] - 1, nu=cfg.get('nu', 0.0))
    gap = np.abs(center(s) - center(pop)).max(-1)    # [T+1, N]
    out = dict(run=os.path.basename(run.rstrip('/')), cfg={k: cfg[k] for k in ('objective', 'oracle', 'alpha', 'beta', 'lam', 'inner', 'lr', 'form') if k in cfg},
               N=int(s.shape[1]), rounds=int(s.shape[0] - 1),
               resid_median_by_round=np.round(np.median(res, 1), 4).tolist(),
               resid_median=float(np.median(res)), resid_q90=float(np.quantile(res, 0.9)),
               gap_median_final=float(np.median(gap[-1])), gap_q90_final=float(np.quantile(gap[-1], 0.9)),
               entropy_neural=np.round(entropy(s).mean(-1), 4).tolist(), entropy_pop=np.round(entropy(pop).mean(-1), 4).tolist(),
               top_neural=np.round(sm(s).max(-1).mean(-1), 4).tolist(), top_pop=np.round(sm(pop).max(-1).mean(-1), 4).tolist())
    return out


if __name__ == '__main__':
    for run in sys.argv[1:]:
        o = analyse(run)
        print(json.dumps({k: v for k, v in o.items() if k not in ('entropy_neural', 'entropy_pop', 'top_neural', 'top_pop', 'resid_median_by_round')}))
        print('  resid by round (first 10):', o['resid_median_by_round'][:10])
        print('  entropy neural/pop at t=0,5,10,last:', [o['entropy_neural'][i] for i in (0, 5, 10, -1) if i < len(o['entropy_neural'])],
              [o['entropy_pop'][i] for i in (0, 5, 10, -1) if i < len(o['entropy_pop'])])
