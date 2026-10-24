"""Post hoc identity-link prediction for X15_c100_A_ipo (Claude, 2026-10-20, after scoring; not preregistered).
predict_x15.py and evaluate_x15.py use dpo_map.logit_round (logit link) for the IPO arm, while PREREG.md specifies the
identity-link master recursion. This recomputes the population trajectory with the identity link, A = P - 1/2 (the map that
matched the v4 IPO runs), from the run's own initial scores, and scores it with the same oscillation rule.
Usage: python ipo_identity_posthoc.py <run dir> <path to 2026-10-16-v4-validation/analysis> <out json>"""
import json, sys
import numpy as np
RUN, POP, OUT = sys.argv[1:4]
sys.path.insert(0, POP)
from population import trajectory, sm
def amp(s, W=30):
    p = sm(s[-W:]); t = np.arange(W) - (W - 1) / 2; sl = (t[:, None, None] * (p - p.mean(0))).sum(0) / (t ** 2).sum()
    return (p - p.mean(0) - sl * t[:, None, None]).std(0).max(-1)
Z = np.load(f'{RUN}/run.npz'); cfg = json.load(open(f'{RUN}/config.json'))
s = Z['s'].astype(float); P = Z['P'].astype(float); T = s.shape[0] - 1
for n in range(P.shape[0]): np.fill_diagonal(P[n], 0.5)
tr = trajectory(P - 0.5, s[0], cfg['alpha'], cfg['beta'], cfg['lam'], T)
a_n, a_p = amp(s), amp(tr)
res = dict(share_pop_identity=float((a_p > 0.0536).mean()), share_neural=float((a_n > 0.0536).mean()),
           agreement=float(((a_p > 0.0536) == (a_n > 0.0536)).mean()), median_amp_neural=float(np.median(a_n)), median_amp_pop=float(np.median(a_p)))
json.dump(res, open(OUT, 'w'), indent=1); print(res)
