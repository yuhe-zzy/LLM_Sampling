"""Score X15 runs against prereg/predictions_x15.json (frozen 2026-10-20 00:00 EDT). Usage: python evaluate_x15.py <runs dir> <out json>
Scoring: the v4 oscillation rule (detrended probability amplitude over the last 30 rounds > 0.0536); the population trajectory of
each arm is recomputed from the run's own initial scores (same as the frozen predictions, which used the panel file's s0)."""
import json, os, sys
import numpy as np
ROOT = '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO'
sys.path.insert(0, f'{ROOT}/reviews/2026-10-18-jmlr-review/theory')
from dpo_map import dpo_round, logit_round, sm as sm1, Pi
RUNS, OUT = sys.argv[1:3]
HERE = os.path.dirname(os.path.abspath(__file__))
PRED = json.load(open(os.path.join(HERE, '..', 'prereg', 'predictions_x15.json')))
THR = PRED['THR']
sm = lambda x: np.exp(x - x.max(-1, keepdims=True)) / np.exp(x - x.max(-1, keepdims=True)).sum(-1, keepdims=True)
kl = lambda p, q: (p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum(-1)


def load(path):
    f = os.path.join(path, 'run.npz')
    if not os.path.exists(f): return None
    Z = np.load(f); return {k: Z[k] for k in Z.files} | {'cfg': json.load(open(os.path.join(path, 'config.json')))}


def amp(s, W=30):
    p = sm(s[-W:]); t = np.arange(W) - (W - 1) / 2; sl = (t[:, None, None] * (p - p.mean(0))).sum(0) / (t ** 2).sum()
    return (p - p.mean(0) - sl * t[:, None, None]).std(0).max(-1)


def population(fround, P, s0, a, b, T):
    K = s0.shape[1]; out = np.zeros((T + 1,) + s0.shape)
    for n in range(s0.shape[0]):
        r = Pi(s0[n]); x = r.copy(); out[0, n] = x
        for t in range(T): x = fround(x, r, P[n], a, b, 1.0, np.ones(K) / K); out[t + 1, n] = x
    return out


def target(fround, P, s0, a, b):
    """Target of sharpness B = b/(1-a) by a damped, slowly refreshed loop (alpha=0.95, beta=0.05 B), 3000 rounds."""
    K = s0.shape[1]; B = b / (1 - a) if a < 1 else None; out = np.zeros_like(s0)
    for n in range(s0.shape[0]):
        r = Pi(s0[n]); x = r.copy()
        for _ in range(3000): x = fround(x, r, P[n], 0.95, 0.05 * B, 1.0, np.ones(K) / K)
        out[n] = x
    return out


res = {'arms': {}, 'criteria': {}}
for arm, pr in PRED['arms'].items():
    d = load(os.path.join(RUNS, arm))
    if d is None: res['arms'][arm] = {'status': 'missing'}; continue
    s = d['s'].astype(float); P = d['P'].astype(float); ids = [int(i) for i in d['ids']]; cfg = d['cfg']
    a, b = cfg['alpha'], cfg['beta']; T = s.shape[0] - 1; K = s.shape[2]
    fr = dpo_round if cfg['objective'] == 'dpo' else logit_round
    am = amp(s); neural_share = float((am > THR).mean())
    pred_sel = {r['id']: r for r in pr['per_panel']}; pred_share_sel = float(np.mean([pred_sel[i]['osc'] for i in ids]))
    pop = population(fr, P, s[0], a, b, T); amP = amp(pop); pop_share = float((amP > THR).mean())
    agree = float(np.mean((am > THR) == (amP > THR)))
    tgt = target(fr, P, s[0], a, b); pavg = sm(s[-10:]).mean(0); kl_avg = kl(sm(tgt), pavg)
    step_kl = np.median([kl(sm(s[t + 1]), sm(s[t])).mean() for t in range(T - 30, T)])
    # one-step error under the DPO and logit maps (DPO arms only), rounds 1..T, all panels
    e = {}
    if cfg['objective'] == 'dpo':
        for nm, f2 in (('dpo', dpo_round), ('logit', logit_round)):
            errs = []
            for t in range(T):
                for n in range(len(ids)):
                    x = Pi(s[t, n]); r = Pi(s[0, n]); y = Pi(s[t + 1, n])
                    errs.append(np.linalg.norm(y - f2(x, r, P[n], a, b, 1.0, np.ones(K) / K)))
            e[nm] = float(np.median(errs))
    res['arms'][arm] = dict(alpha=a, beta=b, objective=cfg['objective'], rounds=T, n_panels=len(ids), sampled=cfg.get('sampled', 0),
                            neural_share=neural_share, pred_share_frozen_selected=pred_share_sel, pop_share_own_s0=pop_share,
                            panel_agreement_with_pop=agree, median_amp=float(np.median(am)), median_amp_pop=float(np.median(amP)),
                            median_kl_target_vs_last10avg=float(np.median(kl_avg)), median_step_kl_last30=float(step_kl),
                            one_step_err=e, per_panel_amp={int(i): float(v) for i, v in zip(ids, am)})
    print(arm, 'neural', round(neural_share, 3), 'pred', round(pred_share_sel, 3), 'pop(own s0)', round(pop_share, 3), 'agree', round(agree, 2),
          'KL avg->target', round(float(np.median(kl_avg)), 4), 'err', {k: round(v, 3) for k, v in e.items()}, flush=True)

A = res['arms']; g = lambda n, k: A.get(n, {}).get(k)
within = lambda n: (g(n, 'neural_share') is not None) and abs(g(n, 'neural_share') - g(n, 'pop_share_own_s0')) <= 0.15
shares = [g(n, 'neural_share') for n in ('X15_c0_A', 'X15_c25_A', 'X15_c50_A', 'X15_c100_A')]
res['criteria']['1_direction'] = dict(shares=shares, nondecreasing=all(x is not None for x in shares) and all(shares[i] <= shares[i + 1] + 1e-9 for i in range(3)),
                                      c0_le_005=shares[0] is not None and shares[0] <= 0.05,
                                      c100_ge_05=shares[3] is not None and (shares[3] >= 0.5 or g('X15_c100_A', 'pop_share_own_s0') < 0.5))
res['criteria']['2_within_015'] = {n: within(n) for n in A if 'neural_share' in A[n]}
res['criteria']['4_remedy'] = dict(within=within('X15_c100_C'), drop_vs_A=(g('X15_c100_A', 'neural_share') or 0) - (g('X15_c100_C', 'neural_share') or 1),
                                   kl_ok=(g('X15_c100_C', 'median_kl_target_vs_last10avg') or 9) <= 0.01)
res['criteria']['5_random'] = dict(share=g('X15_rand_A', 'neural_share'), share_le_01=(g('X15_rand_A', 'neural_share') or 1) <= 0.1,
                                   step_kl_rand=g('X15_rand_A', 'median_step_kl_last30'), step_kl_c0=g('X15_c0_A', 'median_step_kl_last30'),
                                   noisier=(g('X15_rand_A', 'median_step_kl_last30') or 0) > (g('X15_c0_A', 'median_step_kl_last30') or 9))
res['criteria']['6_map'] = {n: (A[n]['one_step_err'].get('dpo', 9) < A[n]['one_step_err'].get('logit', 0)) for n in ('X15_c25_A', 'X15_c50_A', 'X15_c100_A', 'X15_c100_C') if n in A and A[n].get('one_step_err')}
ons = PRED['onset']
def slope(xs, ys):
    xs, ys = np.log(xs), np.log(ys); return float(np.polyfit(xs, ys, 1)[0])
try:
    res['criteria']['3_onset_slopes'] = dict(vs_c_m1=slope([0.5, 1.0], [ons['K3_m1.0_owner_c50']['median'], ons['K3_m1.0_owner_c100']['median']]),
                                            vs_m_c100=slope([0.5, 1.0], [ons['K3_m0.5_owner_c100']['median'], ons['K3_m1.0_owner_c100']['median']]),
                                            vs_m_c50=slope([0.5, 1.0], [ons['K3_m0.5_owner_c50']['median'], ons['K3_m1.0_owner_c50']['median']]),
                                            note='c<=0.25 has no finite onset (Condorcet winner on most panels); the m-slope at c=0 is undefined for the same reason')
except Exception as ex:
    res['criteria']['3_onset_slopes'] = str(ex)
json.dump(res, open(OUT, 'w'), indent=1)
print(json.dumps(res['criteria'], indent=1))
