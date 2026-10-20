"""Score the control runs against the frozen criteria in prereg/predictions_controls.json (2026-10-17).
Usage: python evaluate_controls.py <v4 experiment dir> <runs dir> <out json>"""
import json, os, sys
import numpy as np
V4, RUNS, OUT = sys.argv[1:4]
sys.path.insert(0, os.path.join(V4, 'analysis'))
from population import link, sm, target, entropy
kl = lambda p, q: (p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum(-1)
PRED = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'prereg', 'predictions_controls.json')))['runs']
_ps = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'prereg', 'predictions_switch.json')
if os.path.exists(_ps): PRED['C3b_switch'] = json.load(open(_ps))
Zp = np.load(os.path.join(V4, 'data', 'oracle_planted.npz')); kb = {int(i): k for i, k in zip(Zp['ids'], Zp['kind'])}


def L(r):
    p = os.path.join(RUNS, r, 'run.npz')
    if not os.path.exists(p): return None
    Z = np.load(p); return {k: Z[k] for k in Z.files} | {'cfg': json.load(open(os.path.join(RUNS, r, 'config.json')))}


def amp(s, W=30):
    p = sm(s[-W:]); t = np.arange(W) - (W - 1) / 2; sl = (t[:, None, None] * (p - p.mean(0))).sum(0) / (t ** 2).sum()
    return (p - p.mean(0) - sl * t[:, None, None]).std(0).max(-1)


def top2(x):
    xs = np.sort(x, -1); return float(np.median(xs[:, -1] - xs[:, -2]))


out = {'runs': {}, 'verdicts': {}}
R = out['runs']
for r in ('C1_e0', 'C1_e03', 'C1_e045', 'C1_e055', 'C1_f055', 'C2_eta0125', 'C2_eta025', 'C2_anchorB8', 'C2_anchorB4', 'C3_drift', 'C3_perround', 'C3b_switch'):
    d = L(r)
    if d is None: continue
    s = d['s'].astype(float)
    R[r] = dict(H_t60=float(entropy(s[-1]).mean()), H_t30=float(entropy(s[30]).mean()), top2_t30=top2(s[30]), top2_t60=top2(s[-1]),
                KL0_t60=float(kl(sm(s[-1]), sm(s[0])).mean()), pred=PRED[r])
    if r in ('C3_drift', 'C3b_switch'):
        X8, _ = target(link(d['P'], 'logit'), s[0], 8.0, 0.8); k8 = kl(sm(s[-1]), sm(X8))
        R[r].update(KL_to_B8_median=float(np.median(kl(sm(s[-1]), sm(X8)))), KL_to_B8_mean=float(kl(sm(s[-1]), sm(X8)).mean()),
                    H_target_B8=float(entropy(X8).mean()), B_cum_median=float(np.median(d['betas'].sum(0))),
                    frac_panels_zero_beta_last10=float((d['betas'][-10:] < 1e-6).mean()), n_KL_over_01=int((k8 > 0.1).sum()),
                    KL_to_B8_q90=float(np.quantile(k8, 0.9)), KL_to_B8_max=float(k8.max()))
    if r == 'C3b_switch':
        al = d['alphas']; sw = (al < 0.999).argmax(0).astype(float); sw[(al >= 0.999).all(0)] = np.nan
        R[r].update(switch_round_median=float(np.nanmedian(sw) + 1), n_never_switched=int(np.isnan(sw).sum()), H_t60_abs_gap=abs(R[r]['H_t60'] - R[r]['H_target_B8']))
    if r == 'C3_perround':
        R[r]['B_cum'] = float(d['betas'][:, 0].sum())
for r in ('C4_a05_r02', 'C4_a08_r02', 'C5_adapt', 'C6_wavg'):
    d = L(r)
    if d is None: continue
    s = d['s'].astype(float); ids = d['ids']; bal = np.array([kb[int(i)] == 'balanced' for i in ids]); cfg = d['cfg']
    am = amp(s)
    Xp, _ = target(d['P'] - 0.5, s[0], 12.0, 1.0)
    row = dict(osc_share_balanced=float((am[bal] > 0.0536).mean()), amp_balanced_median=float(np.median(am[bal])),
               osc_share_decisive=float((am[~bal] > 0.0536).mean()), KL_to_target_balanced_median=float(np.median(kl(sm(s[-1][bal]), sm(Xp[bal])))),
               pred=PRED.get(r))
    if r == 'C5_adapt':
        row['alpha_path'] = [float(a) for a in d['alphas'][[0, 20, 40, 60, 80, 100, -1]]]; row['alpha_final'] = float(d['alphas'][-1])  # same indices as the prediction (rounds 1, 21, ..., 101, 150)
    if r == 'C6_wavg':
        pav = sm(s[-30:]).mean(0); pw = sm(d['s_wavg'].astype(float))
        row.update(KL_last_balanced_median=float(np.median(kl(sm(s[-1][bal]), sm(Xp[bal])))),
                   KL_policyavg30_balanced_median=float(np.median(kl(pav[bal], sm(Xp[bal])))),
                   KL_weightavg_balanced_median=float(np.median(kl(pw[bal], sm(Xp[bal])))), wavg_count=int(d['wavg_count']))
    R[r] = row
V = out['verdicts']
if all(k in R for k in ('C1_e0', 'C1_e03', 'C1_e045', 'C1_e055', 'C1_f055')):
    ok_conv = all(abs(R[k]['H_t60'] - PRED[k]['H_t60']) <= 0.05 for k in ('C1_e0', 'C1_e03', 'C1_e045', 'C1_f055'))
    order = R['C1_e0']['H_t60'] > R['C1_e03']['H_t60'] > R['C1_e045']['H_t60']
    div = R['C1_e055']['top2_t60'] >= 1.2 * R['C1_e055']['top2_t30'] and R['C1_e055']['H_t60'] < 0.05
    V['E1'] = dict(converging_within_005=ok_conv, ordered=order, divergence=div, PASS=ok_conv and order and div)
if all(k in R for k in ('C2_eta0125', 'C2_eta025', 'C2_anchorB8', 'C2_anchorB4')):
    ok = all(abs(R[k]['H_t60'] - PRED[k]['H_t60']) <= 0.05 for k in ('C2_eta0125', 'C2_eta025', 'C2_anchorB8', 'C2_anchorB4'))
    order = R['C2_eta0125']['KL0_t60'] > R['C2_anchorB8']['KL0_t60'] and R['C2_eta025']['KL0_t60'] > R['C2_anchorB4']['KL0_t60']
    V['E2'] = dict(entropies_within_005=ok, KL0_ordering=order, PASS=ok and order)
if all(k in R for k in ('C3_drift', 'C3_perround')):
    a = R['C3_drift']['KL_to_B8_median'] <= 0.02 and abs(R['C3_drift']['H_t60'] - R['C3_drift']['H_target_B8']) <= 0.05
    b = R['C3_perround']['H_t60'] < 0.05
    V['E3'] = dict(drift_reaches_target=a, perround_collapses=b, PASS=a and b)
if all(k in R for k in ('C3b_switch', 'C3_perround')):
    a = R['C3b_switch']['KL_to_B8_median'] <= 0.02 and abs(R['C3b_switch']['H_t60'] - R['C3b_switch']['H_target_B8']) <= 0.05
    V['E3_switch'] = dict(switch_reaches_target=a, perround_collapses=R['C3_perround']['H_t60'] < 0.05, n_KL_over_01=R['C3b_switch']['n_KL_over_01'],
                          PASS=a and R['C3_perround']['H_t60'] < 0.05)
if all(k in R for k in ('C4_a05_r02', 'C4_a08_r02')):
    a = R['C4_a05_r02']['osc_share_balanced'] <= 0.2; b = R['C4_a08_r02']['osc_share_balanced'] >= 0.25
    V['E4'] = dict(stabilizes_fast_refresh=a, destabilizes_slow_refresh=b, PASS=a and b)
if 'C5_adapt' in R:
    c = R['C5_adapt']; V['E5'] = dict(PASS=c['alpha_final'] >= 0.6 and c['osc_share_balanced'] <= 0.1 and c['KL_to_target_balanced_median'] <= 0.02)
if 'C6_wavg' in R:
    c = R['C6_wavg']; V['E6'] = dict(PASS=c['KL_weightavg_balanced_median'] <= 2 * c['KL_policyavg30_balanced_median'])
json.dump(out, open(OUT, 'w'), indent=1, default=float)
print(json.dumps(V, indent=1)); print(json.dumps({k: {a: b for a, b in v.items() if a != 'pred'} for k, v in R.items()}, indent=1, default=float)[:6000])
