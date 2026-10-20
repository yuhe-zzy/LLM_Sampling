"""Score X1 (actual DPO on planted cycles) and X9 (inner steps) against prereg/predictions_x1_x9.json (frozen 2026-10-18 17:38 EDT).
Scoring details fixed in the PREREG log at 18:10 EDT, before any result was read.
Usage: python evaluate_x1_x9.py <runs dir> <out json>"""
import json, os, sys
import numpy as np
ROOT = '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO'
sys.path.insert(0, f'{ROOT}/experiments/2026-10-16-v4-validation/analysis')
sys.path.insert(0, f'{ROOT}/reviews/2026-10-18-jmlr-review/theory')
from population import sm, target
from dpo_map import dpo_round, logit_round, Pi
RUNS, OUT = sys.argv[1:3]
HERE = os.path.dirname(os.path.abspath(__file__))
PRED = json.load(open(os.path.join(HERE, '..', 'prereg', 'predictions_x1_x9.json')))
ONSET = json.load(open(os.path.join(HERE, '..', 'prereg', 'onset_predictions.json')))['sweep']
Zp = np.load(f'{ROOT}/experiments/2026-10-16-v4-validation/data/oracle_planted.npz'); kb = {int(i): k for i, k in zip(Zp['ids'], Zp['kind'])}
kl = lambda p, q: (p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum(-1)
THR = 0.0536


def load(path):
    if not os.path.exists(os.path.join(path, 'run.npz')): return None
    Z = np.load(os.path.join(path, 'run.npz')); return {k: Z[k] for k in Z.files} | {'cfg': json.load(open(os.path.join(path, 'config.json')))}


def amp(s, W=30):
    """Detrended probability amplitude per panel over the last W states (the v4 rule)."""
    p = sm(s[-W:]); t = np.arange(W) - (W - 1) / 2; sl = (t[:, None, None] * (p - p.mean(0))).sum(0) / (t ** 2).sum()
    return (p - p.mean(0) - sl * t[:, None, None]).std(0).max(-1)


def population(fround, P, s0, a, b, T):
    """Exact population trajectories (centered logits) from the run's own initial scores; [T+1, N, K]."""
    out = np.zeros((T + 1,) + s0.shape)
    for n in range(s0.shape[0]):
        r = Pi(s0[n]); x = r.copy(); out[0, n] = x
        for t in range(T): x = fround(x, r, P[n], a, b, 1.0, np.ones(4) / 4); out[t + 1, n] = x
    return out


out = {'X1': {}, 'X9': {}, 'verdicts': {}}
# ---------------- X1 ----------------
for name, pr in PRED['X1']['runs'].items():
    d = load(os.path.join(RUNS, name))
    if d is None: continue
    s = d['s'].astype(float); P = d['P'].astype(float); ids = d['ids']; a, b = pr['alpha'], pr['beta']; T = s.shape[0] - 1
    bal = np.array([kb[int(i)] == 'balanced' for i in ids])
    am = amp(s); popD = population(dpo_round, P, s[0], a, b, T); popL = population(logit_round, P, s[0], a, b, T)
    amD = amp(popD); amL = amp(popL)
    # post hoc: one-step error of each map applied to the model's own state, rounds 1..T (centered logits, balanced panels)
    eD, eL = [], []
    for t in range(T):
        for n in np.where(bal)[0]:
            x = Pi(s[t, n]); r = Pi(s[0, n]); y = Pi(s[t + 1, n])
            eD.append(np.linalg.norm(y - dpo_round(x, r, P[n], a, b, 1.0, np.ones(4) / 4)))
            eL.append(np.linalg.norm(y - logit_round(x, r, P[n], a, b, 1.0, np.ones(4) / 4)))
    share = float((am[bal] > THR).mean())
    out['X1'][name] = dict(alpha=a, beta=b, rounds=T, share_balanced=share, share_decisive=float((am[~bal] > THR).mean()),
                           amp_balanced_median=float(np.median(am[bal])),
                           pred_DPO=pr['pred_share_balanced_DPO'], pred_logit=pr['pred_share_balanced_logit'],
                           pop_own_s0_share_DPO=float((amD[bal] > THR).mean()), pop_own_s0_share_logit=float((amL[bal] > THR).mean()),
                           agree_panels_DPO=float(((am[bal] > THR) == (amD[bal] > THR)).mean()),
                           agree_panels_logit=float(((am[bal] > THR) == (amL[bal] > THR)).mean()),
                           onestep_err_median_DPO=float(np.median(eD)), onestep_err_median_logit=float(np.median(eL)),
                           KL_last_to_popDPO_balanced_median=float(np.median(kl(sm(s[-1][bal]), sm(popD[-1][bal])))),
                           pass_criterion=bool(abs(share - pr['pred_share_balanced_DPO']) <= 0.15 and abs(share - pr['pred_share_balanced_logit']) >= 0.35))
    np.savez_compressed(os.path.join(os.path.dirname(OUT), f'pop_{name}.npz'), dpo=popD, logit=popL)
if len(out['X1']) == 2:
    out['verdicts']['X1'] = dict(PASS=all(v['pass_criterion'] for v in out['X1'].values()))

# ---------------- X9 ----------------
def share_at(alpha):
    xs = sorted(float(k) for k in ONSET); ys = [ONSET[[k for k in ONSET if float(k) == x][0]]['share_osc'] for x in xs]
    if alpha >= 0.62: return 0.0
    if alpha <= 0.45: return 0.9375
    return float(np.interp(alpha, xs, ys))


a, b, B = 0.5, 6.0, 12.0
runs = {'k40_ref': f'{ROOT}/experiments/2026-10-16-v4-validation/results/runs/H6_B12_a05'} | {f'k{k}': os.path.join(RUNS, f'X9_k{k}') for k in (10, 3, 1)}
F = lambda x, r, A, pi: Pi(a * x + (1 - a) * r + b * (A @ pi))
for name, path in runs.items():
    d = load(path)
    if d is None: continue
    s = d['s'].astype(float); P = d['P'].astype(float); ids = d['ids']; bal = np.array([kb[int(i)] == 'balanced' for i in ids])
    A = P - 0.5; X, res = target(A, s[0], B, 1.0); am = amp(s)
    fs = []
    for t in range(5):
        for n in np.where(bal)[0]:
            x = Pi(s[t, n]); r = Pi(s[0, n]); m = F(x, r, A[n], sm(x)) - x; dlt = Pi(s[t + 1, n]) - x
            fs.append(float(dlt @ m / (m @ m)))
    f = float(np.median(fs)); rho = 1 - f; aeff = a + (1 - a) * rho
    p = sm(s); step = np.abs(np.diff(p[-31:], axis=0)).max(-1)[:, bal].mean()
    pav = p[-30:].mean(0)
    row = dict(rounds=s.shape[0] - 1, share_balanced=float((am[bal] > THR).mean()), share_decisive=float((am[~bal] > THR).mean()),
               amp_balanced_median=float(np.median(am[bal])), f_rounds1to5_median=f, f_iqr=[float(np.quantile(fs, 0.25)), float(np.quantile(fs, 0.75))],
               rho=rho, alpha_eff=aeff, pred_share_calibrated=share_at(aeff), mean_step_last30=float(step), drifting=bool(step >= 0.01),
               KL_avg30_to_target_balanced_median=float(np.median(kl(pav[bal], sm(X[bal])))),
               KL_avg30_to_target_decisive_median=float(np.median(kl(pav[~bal], sm(X[~bal])))),
               KL_last_to_target_balanced_median=float(np.median(kl(p[-1][bal], sm(X[bal])))), target_solver_res_max=float(res.max()))
    out['X9'][name] = row
if all(k in out['X9'] for k in ('k40_ref', 'k10', 'k3', 'k1')):
    sh = [out['X9'][k]['share_balanced'] for k in ('k40_ref', 'k10', 'k3', 'k1')]
    ordinal = all(sh[i] >= sh[i + 1] for i in range(3)) and min(sh) <= 0.2
    tgt = {k: v['KL_avg30_to_target_balanced_median'] <= 0.02 for k, v in out['X9'].items() if not v['drifting']}
    out['verdicts']['X9'] = dict(shares_k40_k10_k3_k1=sh, ordinal_PASS=ordinal, target_PASS=all(tgt.values()) if tgt else None, target_scored=tgt,
                                 calibrated=[(k, out['X9'][k]['alpha_eff'], out['X9'][k]['pred_share_calibrated'], out['X9'][k]['share_balanced']) for k in ('k40_ref', 'k10', 'k3', 'k1')])
json.dump(out, open(OUT, 'w'), indent=1)
print(json.dumps(out['verdicts'], indent=1))
