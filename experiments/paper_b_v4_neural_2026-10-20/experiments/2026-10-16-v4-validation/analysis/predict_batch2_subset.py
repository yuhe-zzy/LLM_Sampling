"""Frozen predictions for batch 2 on a fixed subset of 64 panels (32 balanced, 32 decisive), chosen before any
batch-2 run by taking the first 32 panel ids of each kind. Same oracle (data/oracle_planted.npz), same metrics as
make_planted.py. Usage: python predict_batch2_subset.py <panels.json> <oracle.npz> <out.json>"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from population import hodge, maximal_lottery, omega_max, sm, target, trajectory  # noqa: E402
D = json.load(open(sys.argv[1])); Z = np.load(sys.argv[2])
ids_all = Z['ids']; kind_all = Z['kind']
sel = np.concatenate([np.where(kind_all == 'balanced')[0][:32], np.where(kind_all == 'decisive')[0][:32]])
sel.sort()
pid = {p['id']: p for p in D['panels']}
s_ref = np.array([pid[int(ids_all[i])]['s0'] for i in sel]); A = Z['P'][sel] - 0.5; kind = kind_all[sel]; N = len(sel)
ml = np.array([maximal_lottery(A[n]) for n in range(N)])
CONFIGS = {'H5_b4_a05': (0.5, 4.0), 'H5_b4_a09': (0.9, 4.0), 'H5_b4_a095': (0.95, 4.0),
           'H6_B12_a02': (0.2, 9.6), 'H6_B12_a05': (0.5, 6.0), 'H6_B12_a08': (0.8, 2.4), 'H6_B12_a09': (0.9, 1.2)}
out = dict(created=time.strftime('%Y-%m-%d %H:%M:%S'), ids=[int(ids_all[i]) for i in sel], configs={})
c = lambda x: x - x.mean(-1, keepdims=True)
for name, (a, b) in CONFIGS.items():
    B = b / (1 - a); X, res = target(A, s_ref, B, 1.0)
    gB = np.array([B * omega_max(hodge(A[n])[1], sm(X[n])) for n in range(N)]); rad = np.sqrt(a ** 2 + ((1 - a) * gB) ** 2)
    tr = trajectory(A, s_ref, a, b, 1.0, 80)
    top = sm(tr).argmax(-1); sw = (np.diff(top[-31:], axis=0) != 0).sum(0)
    dist = np.abs(c(tr) - c(X)[None]).max(-1); ratio = dist[-1] / np.maximum(dist[-21], 1e-12)
    out['configs'][name] = dict(alpha=a, beta=b, B=B, **{k: dict(gammaB_median=float(np.median(gB[kind == k])), radius_median=float(np.median(rad[kind == k])),
        share_unstable=float((rad[kind == k] > 1).mean()), pop_switch_share=float((sw[kind == k] > 0).mean()), pop_ratio_median=float(np.median(ratio[kind == k])))
        for k in ('balanced', 'decisive')})
json.dump(out, open(sys.argv[3], 'w'), indent=1)
for n, cfg in out['configs'].items():
    print(n, {k: {kk: round(vv, 3) for kk, vv in cfg[k].items()} for k in ('balanced', 'decisive')})
