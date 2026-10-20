"""X10 and X5 evaluation (2026-10-19), as preregistered in prereg/PREREG.md (predictions.json frozen, sha256 prefix dd62875cb77c).
X10: first round at which the median KL(pi_t || pi*_B12) over balanced planted panels falls below 0.01 and stays below; pass if both X10 runs
are at most half the same statistic of the ordinary-refresh v4 run H6_B12_a08. X5: first round with median KL to the B = 8 target below 1e-3;
pass if refresh-then-anchor is within 2 rounds of ceil(8/beta_bar) and at least 1.8x sooner than the constant anchor."""
import json, math, os, sys
import numpy as np
sys.path.insert(0, '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/experiments/2026-10-16-v4-validation/analysis')
from population import link, sm, target, trajectory
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); RUNS = os.path.join(ROOT, 'results', 'runs')
V4 = '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/experiments/2026-10-16-v4-validation/results/runs'
CT = '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/experiments/2026-10-17-controls/results/runs'
Zp = np.load('/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/experiments/2026-10-16-v4-validation/data/oracle_planted.npz'); KB = {int(i): k for i, k in zip(Zp['ids'], Zp['kind'])}
kl = lambda p, q: (p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum(-1)


def load(path):
    Z = np.load(os.path.join(path, 'run.npz')); return {k: Z[k] for k in Z.files} | {'cfg': json.load(open(os.path.join(path, 'config.json')))}


def first_stay(series, thr):
    for t in range(len(series)):
        if np.all(series[t:] < thr): return int(t)
    return None


def first_below(series, thr):
    idx = np.where(series < thr)[0]; return int(idx[0]) if len(idx) else None


out = {'X10': {}, 'X5': {}, 'verdicts': {}}
# ---------------- X10 ----------------
for name, path in (('H6_B12_a08', os.path.join(V4, 'H6_B12_a08')), ('X10_a04', os.path.join(RUNS, 'X10_a04')), ('X10_a045', os.path.join(RUNS, 'X10_a045'))):
    if not os.path.exists(os.path.join(path, 'run.npz')): continue
    d = load(path); s = d['s'].astype(float); P = d['P'].astype(float); cfg = d['cfg']; ids = d['ids']
    bal = np.array([KB[int(i)] == 'balanced' for i in ids]); A = link(P, 'identity')
    X, res = target(A, s[0], 12.0, 1.0); PS = sm(X)
    K_ = kl(sm(s), PS[None]); med = np.median(K_[:, bal], 1); q90 = np.quantile(K_[:, bal], 0.9, 1); medd = np.median(K_[:, ~bal], 1)
    pop = trajectory(A, s[0], cfg['alpha'], cfg['beta'], 1.0, s.shape[0] - 1, nu=cfg.get('nu', 0.0)); Kp = kl(sm(pop), PS[None]); medp = np.median(Kp[:, bal], 1)
    out['X10'][name] = dict(alpha=cfg['alpha'], nu=cfg.get('nu', 0.0), beta=cfg['beta'], rounds=int(s.shape[0] - 1), target_residual_max=float(res.max()),
                            stay_1e2_median=first_stay(med, 1e-2), stay_1e2_q90=first_stay(q90, 1e-2), stay_1e3_median=first_stay(med, 1e-3),
                            pop_stay_1e2_median=first_stay(medp, 1e-2), pop_stay_1e3_median=first_stay(medp, 1e-3),
                            floor_last20_median=float(np.median(med[-20:])), decisive_stay_1e2_median=first_stay(medd, 1e-2),
                            median_KL_rounds_0_5_10_20=[float(med[t]) for t in (0, 5, 10, 20) if t < len(med)],
                            series_median=[float(v) for v in med], series_pop_median=[float(v) for v in medp])
    print(name, {k: v for k, v in out['X10'][name].items() if not k.startswith('series')}, flush=True)
if all(k in out['X10'] for k in ('H6_B12_a08', 'X10_a04', 'X10_a045')):
    ref = out['X10']['H6_B12_a08']['stay_1e2_median']; lim = math.floor(ref / 2)
    ok = all(out['X10'][k]['stay_1e2_median'] is not None and out['X10'][k]['stay_1e2_median'] <= lim for k in ('X10_a04', 'X10_a045'))
    out['verdicts']['X10'] = dict(reference=ref, limit=lim, PASS=bool(ok))
# ---------------- X5 ----------------
pairs = {0.5: ('X5_sw05', 'X5_an05', RUNS), 2.0: ('X5_sw2', 'X5_an2', RUNS), 1.0: ('C3b_switch', 'C2_anchorB8', CT)}
for bb, (sw, an, base) in pairs.items():
    row = {}
    for role, name in (('switch', sw), ('anchor', an)):
        path = os.path.join(base, name)
        if not os.path.exists(os.path.join(path, 'run.npz')): continue
        d = load(path); s = d['s'].astype(float); P = d['P'].astype(float)
        X8, res = target(link(P, 'logit'), s[0], 8.0, 0.8); K_ = kl(sm(s), sm(X8)[None]); med = np.median(K_, 1)
        row[role] = dict(run=name, first_below_1e3=first_below(med, 1e-3), stay_1e3=first_stay(med, 1e-3), floor_last10=float(np.median(med[-10:])),
                         target_residual_max=float(res.max()), series_median=[float(v) for v in med])
    if 'switch' in row and 'anchor' in row:
        a_, b_ = row['switch']['first_below_1e3'], row['anchor']['first_below_1e3']; goal = math.ceil(8 / bb)
        row['goal'] = goal; row['ratio'] = (b_ / a_) if (a_ and b_) else None
        row['PASS'] = bool(a_ is not None and b_ is not None and abs(a_ - goal) <= 2 and b_ >= 1.8 * a_) if bb != 1.0 else None
    out['X5'][f'beta_bar={bb}'] = row
    print(bb, {r: ({k: v for k, v in row[r].items() if k != 'series_median'} if isinstance(row[r], dict) else row[r]) for r in row}, flush=True)
if all(out['X5'].get(f'beta_bar={bb}', {}).get('PASS') is not None for bb in (0.5, 2.0)):
    out['verdicts']['X5'] = dict(PASS=bool(all(out['X5'][f'beta_bar={bb}']['PASS'] for bb in (0.5, 2.0))))
print('VERDICTS', out['verdicts'])
json.dump(out, open(os.path.join(ROOT, 'results', 'evaluation_x10_x5.json'), 'w'), indent=1)
