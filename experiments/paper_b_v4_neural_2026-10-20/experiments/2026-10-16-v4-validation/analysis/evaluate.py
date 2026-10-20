"""Score the neural runs against the frozen predictions (H0-H6).

Usage: python evaluate.py <results/runs dir> <panels.json> <out.json>
Reads every run found; skips missing ones. Uses the exact population recursion of population.py with the
same oracle the run used.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fidelity import analyse  # noqa: E402
from population import condorcet_winner, entropy, link, sm, target, trajectory  # noqa: E402

runs_dir, panels_path, out_path = sys.argv[1:4]
D = json.load(open(panels_path))
runs = {os.path.basename(r): r for r in sorted(glob.glob(os.path.join(runs_dir, '*'))) if os.path.exists(os.path.join(r, 'run.npz'))}
res = dict(runs={})
c = lambda x: x - x.mean(-1, keepdims=True)
kl = lambda X, Y: (sm(X) * (np.log(sm(X) + 1e-300) - np.log(sm(Y) + 1e-300))).sum(-1)


def load(name):
    Z = np.load(os.path.join(runs[name], 'run.npz'))
    cfg = json.load(open(os.path.join(runs[name], 'config.json')))
    return Z['s'].astype(np.float64), Z['P'], cfg


for name in runs:
    s, P, cfg = load(name)
    lk = 'identity' if cfg['objective'] == 'ipo' else 'logit'
    A = link(P, lk)
    fid = analyse(runs[name])
    d = dict(cfg={k: cfg[k] for k in ('objective', 'oracle', 'alpha', 'beta', 'lam')}, resid_median=fid['resid_median'], resid_q90=fid['resid_q90'],
             entropy_neural=fid['entropy_neural'], entropy_pop=fid['entropy_pop'], top_neural=fid['top_neural'])
    if cfg['alpha'] < 1 and not (cfg['objective'] == 'dpo' and cfg['oracle'] == 'hard'):
        B = cfg['beta'] / (1 - cfg['alpha'])
        X, _ = target(A, s[0], B, cfg['lam'])
        d.update(B=B, target_entropy=float(entropy(X).mean()), final_entropy=float(entropy(s[-1]).mean()),
                 KL_final_to_target_mean=float(kl(s[-1], X).mean()), KL_final_to_target_median=float(np.median(kl(s[-1], X))))
        res['_X_' + name] = X
    res['runs'][name] = d

# H1: same target at B = 8
h1 = [n for n in ('H1_a0_b8', 'H1_a05_b4', 'H1_a08_b16', 'H1_a09_b08') if n in runs]
if len(h1) >= 2:
    F = {n: load(n)[0][-1] for n in h1}
    pair = {f'{a}|{b}': float(kl(F[a], F[b]).mean()) for i, a in enumerate(h1) for b in h1[i + 1:]}
    ctl = load('H1_a05_b1')[0][-1] if 'H1_a05_b1' in runs else None
    res['H1'] = dict(pairwise_KL_B8=pair, max_pairwise_KL_B8=max(pair.values()),
                     KL_B8_vs_B2_control={n: float(kl(F[n], ctl).mean()) for n in h1} if ctl is not None else None)
# H2: entropy against M_t
h2 = {n: r for n, r in runs.items() if n.startswith('H2_') or n == 'H1_a05_b1'}
if h2:
    out = {}
    for n in h2:
        s, P, cfg = load(n)
        a = cfg['alpha']; t = np.arange(s.shape[0]); M = (1 - a ** t) / (1 - a)
        H = entropy(s).mean(-1)
        out[n] = dict(alpha=a, M=M.round(3).tolist(), H=H.round(4).tolist(), H_at_M={m: float(np.interp(m, M, H)) for m in (1, 2, 3, 4, 5, 8) if M.max() >= m})
    res['H2'] = out
# H3: full refresh
if 'H3_a1_b1' in runs:
    s, P, cfg = load('H3_a1_b1')
    cw = np.array([condorcet_winner(P[n]) for n in range(P.shape[0])])
    top = sm(s).argmax(-1)
    pw = sm(s)[np.arange(s.shape[0])[:, None], np.arange(s.shape[1])[None], cw[None]]   # [T+1, N]
    res['H3'] = dict(share_top_is_CW_final=float(np.mean(top[-1] == cw)), mean_CW_mass=pw.mean(1).round(4).tolist(),
                     entropy=entropy(s).mean(-1).round(4).tolist(), share_CW_mass_ge_095_final=float(np.mean(pw[-1] >= 0.95)))
# H4: link
h4 = {n: load(n) for n in ('H4_dpo_hard', 'H4_dpo_soft', 'H4_ipo_hard') if n in runs}
if h4:
    res['H4'] = {n: dict(entropy=entropy(v[0]).mean(-1).round(5).tolist(), top=sm(v[0]).max(-1).mean(-1).round(5).tolist()) for n, v in h4.items()}
# H5/H6: planted cycles
pl = {n: r for n, r in runs.items() if n.startswith(('H5_', 'H6_'))}
if pl:
    Zp = np.load(os.path.join(os.path.dirname(panels_path), 'oracle_planted.npz'))
    kind_by_id = {int(i): k for i, k in zip(Zp['ids'], Zp['kind'])}
    out = {}
    for n in pl:
        s, P, cfg = load(n)
        kind = np.array([kind_by_id[int(i)] for i in np.load(os.path.join(pl[n], 'run.npz'))['ids']])
        A = P - 0.5
        B = cfg['beta'] / (1 - cfg['alpha'])
        X, _ = target(A, s[0], B, 1.0)
        top = sm(s).argmax(-1)
        sw = (np.diff(top[-31:], axis=0) != 0).sum(0)
        dist = np.abs(c(s) - c(X)[None]).max(-1)
        ratio = dist[-1] / np.maximum(dist[-21], 1e-12)
        def pamp(traj, T_=30):  # detrended amplitude of the panel probabilities over the last T_ rounds
            tt = np.arange(T_) - (T_ - 1) / 2; x = sm(traj[-T_:]); slope = (tt[:, None, None] * (x - x.mean(0))).sum(0) / (tt ** 2).sum()
            return (x - x.mean(0) - slope[None] * tt[:, None, None]).std(0).max(-1)
        amp = pamp(s); amp_pop = pamp(trajectory(A, s[0], cfg['alpha'], cfg['beta'], 1.0, s.shape[0] - 1))
        THR = 0.0536
        out[n] = {k: dict(osc_share=float((amp[kind == k] > THR).mean()), amp_median=float(np.median(amp[kind == k])),
                          pop_osc_share=float((amp_pop[kind == k] > THR).mean()), pop_amp_median=float(np.median(amp_pop[kind == k])),
                          switch_share=float((sw[kind == k] > 0).mean()), ratio_median=float(np.median(ratio[kind == k])),
                          final_dist_median=float(np.median(dist[-1][kind == k])), final_top_median=float(np.median(sm(s[-1])[kind == k].max(-1))))
                  for k in ('balanced', 'decisive')}
        out[n]['resid_median'] = analyse(pl[n])['resid_median']
    res['H56'] = out
for k in [k for k in res if k.startswith('_X_')]:
    res.pop(k)
json.dump(res, open(out_path, 'w'), indent=1)
summary = {n: {k: v for k, v in d.items() if k in ('resid_median', 'B', 'target_entropy', 'final_entropy', 'KL_final_to_target_mean')} for n, d in res['runs'].items()}
print(json.dumps(summary, indent=1))
for k in ('H1', 'H3', 'H56'):
    if k in res:
        print(k, json.dumps(res[k] if k != 'H3' else {kk: vv for kk, vv in res[k].items() if not isinstance(vv, list)}, indent=1)[:2500])
