"""Post hoc (2026-10-19): recompute X1's population DPO predictions with the fixed bt_fit (dpo_map.py, row-scaled Newton) and compare
with the stored trajectories made by the original solver (results/pop_X1_*.npz). Balanced panels, the v4 oscillation rule."""
import json, os, sys
import numpy as np
sys.path.insert(0, '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/reviews/2026-10-18-jmlr-review/theory')
from dpo_map import dpo_round, sm, Pi
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
Zp = np.load('/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/experiments/2026-10-16-v4-validation/data/oracle_planted.npz'); kb = {int(i): k for i, k in zip(Zp['ids'], Zp['kind'])}
THR = 0.0536
smk = lambda x: np.exp(x - x.max(-1, keepdims=True)) / np.exp(x - x.max(-1, keepdims=True)).sum(-1, keepdims=True)


def amp(p, W=30):
    p = p[-W:]; t = np.arange(W) - (W - 1) / 2; sl = (t[:, None, None] * (p - p.mean(0))).sum(0) / (t ** 2).sum()
    return (p - p.mean(0) - sl * t[:, None, None]).std(0).max(-1)


out = {}
for name, a, b in (('X1_B4_a08', 0.8, 0.8), ('X1_B2_a03', 0.3, 1.4)):
    Z = np.load(os.path.join(ROOT, 'results', 'runs', name, 'run.npz')); s = Z['s'].astype(float); P = Z['P'].astype(float); ids = Z['ids']; T = s.shape[0] - 1
    bal = np.array([kb[int(i)] == 'balanced' for i in ids])
    new = np.zeros_like(s)
    for n in range(s.shape[1]):
        r = Pi(s[0, n]); x = r.copy(); new[0, n] = x
        for t in range(T): x = dpo_round(x, r, P[n], a, b, 1.0, np.ones(4) / 4); new[t + 1, n] = x
    old = np.load(os.path.join(ROOT, 'results', f'pop_{name}.npz'))['dpo']
    an, ao = amp(smk(new)), amp(smk(old)); neural = amp(smk(s))
    out[name] = dict(share_new=float((an[bal] > THR).mean()), share_old=float((ao[bal] > THR).mean()), share_neural=float((neural[bal] > THR).mean()),
                     max_abs_traj_diff=float(np.abs(smk(new) - smk(old)).max()), agree_panels_new=float(((an[bal] > THR) == (neural[bal] > THR)).mean()))
    print(name, out[name], flush=True)
json.dump(out, open(os.path.join(ROOT, 'results', 'recheck_solver.json'), 'w'), indent=1)
