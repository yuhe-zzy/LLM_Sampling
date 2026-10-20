"""X10 diagnosis (post hoc, 2026-10-19): per-panel speed and late oscillation of the lagged runs against the ordinary run H6_B12_a08."""
import json, os, sys
import numpy as np
sys.path.insert(0, '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/experiments/2026-10-16-v4-validation/analysis')
from population import link, sm, target, trajectory
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
Zp = np.load('/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/experiments/2026-10-16-v4-validation/data/oracle_planted.npz'); KB = {int(i): k for i, k in zip(Zp['ids'], Zp['kind'])}
kl = lambda p, q: (p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum(-1)


def amp(p, W=30):
    p = p[-W:]; t = np.arange(W) - (W - 1) / 2; sl = (t[:, None, None] * (p - p.mean(0))).sum(0) / (t ** 2).sum()
    return (p - p.mean(0) - sl * t[:, None, None]).std(0).max(-1)


out = {}
for name, path in (('H6_B12_a08', '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/experiments/2026-10-16-v4-validation/results/runs/H6_B12_a08'),
                   ('X10_a04', os.path.join(ROOT, 'results', 'runs', 'X10_a04')), ('X10_a045', os.path.join(ROOT, 'results', 'runs', 'X10_a045'))):
    Z = np.load(os.path.join(path, 'run.npz')); cfg = json.load(open(os.path.join(path, 'config.json'))); s = Z['s'].astype(float); P = Z['P'].astype(float)
    bal = np.array([KB[int(i)] == 'balanced' for i in Z['ids']]); A = link(P, 'identity'); X, _ = target(A, s[0], 12.0, 1.0); PS = sm(X)
    K_ = kl(sm(s), PS[None])[:, bal]
    stay = [next((t for t in range(K_.shape[0]) if np.all(K_[t:, n] < 1e-2)), None) for n in range(K_.shape[1])]
    a = amp(sm(s))[bal]
    pop = trajectory(A, s[0], cfg['alpha'], cfg['beta'], 1.0, s.shape[0] - 1, nu=cfg.get('nu', 0.0)); ap = amp(sm(pop))[bal]
    out[name] = dict(per_panel_stay_1e2=stay, never=int(sum(x is None for x in stay)), median_stay=float(np.median([x if x is not None else 99 for x in stay])),
                     osc_share_neural=float((a > 0.0536).mean()), osc_share_pop=float((ap > 0.0536).mean()),
                     median_KL_t=[float(np.median(K_[t])) for t in (3, 5, 8, 10, 15, 20, 40, 79)])
    print(name, {k: v for k, v in out[name].items() if k != 'per_panel_stay_1e2'})
json.dump(out, open(os.path.join(ROOT, 'results', 'diagnose_x10.json'), 'w'), indent=1)
