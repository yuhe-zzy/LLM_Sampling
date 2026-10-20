"""Amendment (2026-10-16, after 2 of 7 batch-2 runs): noise-robust oscillation signature.
A panel counts as oscillating when the late detrended amplitude -- the largest over answers of the standard deviation
of the centered panel logits over the last 30 of 80 rounds after removing a linear trend in t -- exceeds THR, four times
the noise floor measured the same way on the converged batch-1 runs (computed in this script). Predicted shares come from the noise-free population recursion on the same subset.
Usage: python predict_batch2_amplitude.py <panels.json> <oracle.npz> <subset_pred.json> <out.json>"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from population import trajectory  # noqa: E402
def detamp(win):
    T = win.shape[0]; t = np.arange(T) - (T - 1) / 2
    x = win - win.mean(-1, keepdims=True)
    slope = (t[:, None, None] * (x - x.mean(0))).sum(0) / (t ** 2).sum()
    resid = x - x.mean(0) - slope[None] * t[:, None, None]
    return resid.std(0).max(-1)


RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'runs')
floor = []
for r, W in (('H1_a0_b8', 20), ('H1_a05_b4', 15), ('H4_dpo_soft', 30), ('H4_ipo_hard', 30), ('H2_a0.8', 10), ('H1_a05_b1', 15)):
    s_ = np.load(os.path.join(RUNS, r, 'run.npz'))['s'].astype(float)
    floor.append(float(np.median(detamp(s_[-W:]))))
NOISE = float(np.median(floor)); THR = 4 * NOISE
print('noise floor (detrended) %.3f -> threshold %.3f' % (NOISE, THR))
D = json.load(open(sys.argv[1])); Z = np.load(sys.argv[2]); S = json.load(open(sys.argv[3]))
idx = {int(i): n for n, i in enumerate(Z['ids'])}; pid = {p['id']: p for p in D['panels']}
sel = [idx[i] for i in S['ids']]
A = Z['P'][sel] - 0.5; kind = Z['kind'][sel]; s0 = np.array([pid[i]['s0'] for i in S['ids']])
c = lambda x: x - x.mean(-1, keepdims=True)
out = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), threshold=THR, noise_floor=NOISE, configs={})
for name, cfg in S['configs'].items():
    tr = trajectory(A, s0, cfg['alpha'], cfg['beta'], 1.0, 80)
    amp = detamp(tr[-30:])
    out['configs'][name] = {k: dict(osc_share=float((amp[kind == k] > THR).mean()), amp_median=float(np.median(amp[kind == k]))) for k in ('balanced', 'decisive')}
json.dump(out, open(sys.argv[4], 'w'), indent=1)
for n, v in out['configs'].items():
    print(n, {k: {kk: round(vv, 3) for kk, vv in x.items()} for k, x in v.items()})
