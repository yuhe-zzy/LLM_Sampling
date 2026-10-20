"""Post hoc diagnostic for X9 (not preregistered): does the per-round move scale with the exact round move (GD-like partial step)
or stay roughly fixed in size (Adam-like normalized step)? Log-log slope of |achieved move| on |exact move| over all rounds and
balanced panels, and the fraction f binned by |exact move|. Usage: python diagnose_x9.py <runs dir> <out json>"""
import json, os, sys
import numpy as np
ROOT = '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO'
sys.path.insert(0, f'{ROOT}/experiments/2026-10-16-v4-validation/analysis')
from population import sm
Zp = np.load(f'{ROOT}/experiments/2026-10-16-v4-validation/data/oracle_planted.npz'); kb = {int(i): k for i, k in zip(Zp['ids'], Zp['kind'])}
RUNS, OUT = sys.argv[1:3]
a, b = 0.5, 6.0
Pi = lambda v: v - v.mean(-1, keepdims=True)
paths = {'k40': f'{ROOT}/experiments/2026-10-16-v4-validation/results/runs/H6_B12_a05'} | {f'k{k}': os.path.join(RUNS, f'X9_k{k}') for k in (10, 3, 1)}
out = {}
for name, p in paths.items():
    Z = np.load(os.path.join(p, 'run.npz')); s = Pi(Z['s'].astype(float)); A = Z['P'].astype(float) - 0.5
    kind = np.array([kb[int(i)] for i in Z['ids']])
    row = {}
    for grp in ('balanced', 'decisive'):
        idx = np.where(kind == grp)[0]; M, D, Fr = [], [], []
        for t in range(s.shape[0] - 1):
            for n in idx:
                x = s[t, n]; m = Pi(a * x + (1 - a) * s[0, n] + b * (A[n] @ sm(x))) - x; d = s[t + 1, n] - x
                # measure on the three highest-probability answers' logits would be similar; use probability-weighted norm to damp wandering logits
                w = np.sqrt(sm(x)); M.append(np.linalg.norm(w * m)); D.append(np.linalg.norm(w * d)); Fr.append(float(d @ m / (m @ m)))
        M, D, Fr = map(np.array, (M, D, Fr)); ok = M > 1e-4
        sl = np.polyfit(np.log(M[ok]), np.log(D[ok] + 1e-12), 1)[0]
        q = np.quantile(M[ok], [0, 0.25, 0.5, 0.75, 1]); bins = []
        for lo, hi in zip(q[:-1], q[1:]):
            sel = ok & (M >= lo) & (M <= hi); bins.append(dict(exact_move=[float(lo), float(hi)], median_achieved=float(np.median(D[sel])), median_f=float(np.median(Fr[sel]))))
        row[grp] = dict(loglog_slope=float(sl), quartile_bins=bins)
    out[name] = row; print(name, {g: round(row[g]['loglog_slope'], 3) for g in row}, [round(x['median_achieved'], 3) for x in row['balanced']['quartile_bins']], [round(x['median_f'], 2) for x in row['balanced']['quartile_bins']])
json.dump(out, open(OUT, 'w'), indent=1)
