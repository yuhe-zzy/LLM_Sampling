"""Post hoc diagnosis of X14_c24 (census alpha 0.95 at B = 24), which failed S3-c: is the residual motion a sustained rotation or a
slow approach plus the inner-optimization noise floor? Compares, per balanced panel: the detrended amplitude in rounds 21-50
against 51-80; the distance to the population target over rounds; the population trajectory's own amplitude; and the same
quantities for X14_c12 (passed) and the 0.5B H6 alpha = 0.9 run (B = 12, settled)."""
import json, os, sys
import numpy as np
EXP = '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/experiments'
sys.path.insert(0, f'{EXP}/2026-10-16-v4-validation/analysis')
from population import sm, target, trajectory
HERE = os.path.dirname(os.path.abspath(__file__))
Zp = np.load(f'{EXP}/2026-10-16-v4-validation/data/oracle_planted.npz'); pm = {int(i): n for n, i in enumerate(Zp['ids'])}
kl = lambda p, q: (p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum(-1)


def amp_w(s, a, b):
    p = sm(s[a:b]); W = b - a; t = np.arange(W) - (W - 1) / 2; sl = (t[:, None, None] * (p - p.mean(0))).sum(0) / (t ** 2).sum()
    return (p - p.mean(0) - sl * t[:, None, None]).std(0).max(-1)


out = {}
for name, path, B, a, b in (('X14_c24', f'{EXP}/2026-10-19-x14-steer/results/runs/X14_c24', 24.0, 0.95, 1.2),
                            ('X14_c12', f'{EXP}/2026-10-19-x14-steer/results/runs/X14_c12', 12.0, 0.8748141797660436, 1.502229842807477),
                            ('H6_B12_a09', f'{EXP}/2026-10-16-v4-validation/results/runs/H6_B12_a09', 12.0, 0.9, 1.2)):
    Z = np.load(os.path.join(path, 'run.npz')); s = Z['s'].astype(float); ids = [int(i) for i in Z['ids']]
    A = np.stack([Zp['P'][pm[i]] for i in ids]) - 0.5; bal = np.array([str(Zp['kind'][pm[i]]) == 'balanced' for i in ids])
    X, _ = target(A, s[0], B, 1.0); tr = trajectory(A, s[0], a, b, 1.0, s.shape[0] - 1)
    d = np.array([kl(sm(X[bal]), sm(s[t][bal])) for t in range(s.shape[0])]); dp = np.array([kl(sm(X[bal]), sm(tr[t][bal])) for t in range(s.shape[0])])
    row = dict(amp_21_50=float(np.median(amp_w(s, 21, 51)[bal])), amp_51_80=float(np.median(amp_w(s, 51, 81)[bal])),
               pop_amp_51_80=float(np.median(amp_w(tr, 51, 81)[bal])), share_amp_gt_thr_51_80=float((amp_w(s, 51, 81)[bal] > 0.0536).mean()),
               median_kl_target_by_round={t: float(np.median(d[t])) for t in (10, 20, 40, 60, 80)},
               pop_median_kl_target_by_round={t: float(np.median(dp[t])) for t in (10, 20, 40, 60, 80)},
               top_switches_last30_median=float(np.median((np.diff(sm(s[-31:])[:, bal].argmax(-1), axis=0) != 0).sum(0))),
               pop_top_switches_last30_median=float(np.median((np.diff(sm(tr[-31:])[:, bal].argmax(-1), axis=0) != 0).sum(0))))
    # mean step size of the policy per round in the last 30 rounds (noise floor proxy)
    row['median_step_last30'] = float(np.median(np.abs(np.diff(sm(s[-31:])[:, bal], axis=0)).max(-1).mean(0)))
    row['pop_median_step_last30'] = float(np.median(np.abs(np.diff(sm(tr[-31:])[:, bal], axis=0)).max(-1).mean(0)))
    out[name] = row; print(name, json.dumps(row))
json.dump(out, open(os.path.join(HERE, '..', 'results', 'diagnose_c24.json'), 'w'), indent=1)
