"""Score the X14 runs against the frozen predictions and pass criteria of prereg/PREREG.md (2026-10-19).
Usage: python evaluate_x14.py   (reads ../results/runs/X14_*, ../prereg/predictions.json; writes ../results/evaluation_x14.json)"""
import json, os, sys
import numpy as np
EXP = '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/experiments'
sys.path.insert(0, f'{EXP}/2026-10-16-v4-validation/analysis')
from population import sm, target
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, '..', 'results', 'runs')
P = json.load(open(os.path.join(HERE, '..', 'prereg', 'predictions.json'))); IDS = P['ids']
ent = lambda s: -(sm(s) * np.log(sm(s) + 1e-300)).sum(-1)
kl = lambda p, q: (p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum(-1)
cen = lambda x: x - x.mean(-1, keepdims=True)
pan = {p['id']: p for p in json.load(open(f'{EXP}/2026-10-16-v4-validation/data/panels_q05.json'))['panels']}
RM = 'Skywork-Reward-Llama-3.1-8B-v0.2-d4117fbfd81b'; U = np.array([pan[i]['rm'][RM] for i in IDS]) / 3.168
Zp = np.load(f'{EXP}/2026-10-16-v4-validation/data/oracle_planted.npz'); pm = {int(i): n for n, i in enumerate(Zp['ids'])}
A_PL = np.stack([Zp['P'][pm[i]] for i in IDS]).astype(float) - 0.5; BAL = np.array([str(Zp['kind'][pm[i]]) == 'balanced' for i in IDS])


def load(name, root=RUNS):
    d = os.path.join(root, name); Z = np.load(os.path.join(d, 'run.npz')); assert [int(i) for i in Z['ids']] == IDS
    st = json.load(open(os.path.join(d, 'steer.json'))) if os.path.exists(os.path.join(d, 'steer.json')) else None
    return Z['s'].astype(float), st


def amp(traj, W=30):
    p = sm(traj[-W:]); t = np.arange(W) - (W - 1) / 2; sl = (t[:, None, None] * (p - p.mean(0))).sum(0) / (t ** 2).sum()
    return (p - p.mean(0) - sl * t[:, None, None]).std(0).max(-1)


def top2_step(s):
    xs = np.sort(s[-11:], -1); g = xs[..., -1] - xs[..., -2]; return float(np.median(np.abs(np.diff(g, axis=0)).mean(0)))


def interp(grid, eh, key):
    ks = sorted(float(k) for k in grid); eh = min(max(eh, ks[0]), ks[-1])
    lo = max(k for k in ks if k <= eh); hi = min(k for k in ks if k >= eh)
    vlo, vhi = grid[f'{lo:.2f}'][key], grid[f'{hi:.2f}'][key]
    return vlo if hi == lo else vlo + (vhi - vlo) * (eh - lo) / (hi - lo)


out = {}; mark = lambda b: 'PASS' if b else 'FAIL'
# ---------------- S1
planned = P['S1']['planned_entropy_eps0']; s1 = {}
base = {'C1_e045': load('C1_e045', f'{EXP}/2026-10-17-controls/results/runs')[0], 'C1_e055': load('C1_e055', f'{EXP}/2026-10-17-controls/results/runs')[0]}
s1['naive'] = {k: dict(end_entropy=float(ent(v[-1]).mean()), last10_top2_step=top2_step(v)) for k, v in base.items()}
for name, eps, key in (('X14_db45', 0.45, 'debias_entropy'), ('X14_db55', 0.55, 'debias_entropy'), ('X14_cp55', 0.55, 'comp_entropy')):
    s, st = load(name); d5 = st['d5']; H = float(ent(s[-1]).mean()); pred = interp(P['S1'][f'eps={eps}']['by_eps_hat'], d5['eps_hat'], key)
    row = dict(eps_true=eps, eps_hat=d5['eps_hat'], se=d5['se'], acted=d5['acted'], alpha_after=d5['alpha'], beta_after=d5['beta'], end_entropy=H,
               predicted_entropy=pred, last10_top2_step=top2_step(s), entropy_by_round=[float(x) for x in ent(s).mean(-1)])
    if key == 'debias_entropy':
        row['pass'] = mark(d5['acted'] and abs(H - pred) <= 0.05 and abs(H - planned) <= 0.10)
    else:
        row['pass'] = mark(d5['acted'] and abs(H - pred) <= 0.05 and row['last10_top2_step'] <= 0.05)
    s1[name] = row
s1['S1-c'] = mark(all(s1[n]['end_entropy'] >= 0.3 for n in ('X14_db45', 'X14_db55', 'X14_cp55')))
out['S1'] = s1
# ---------------- S2
s2 = {}
s, _ = load('X14_fr'); s2['X14_fr'] = dict(end_entropy=float(ent(s[-1]).mean()), predicted=P['S2']['naive_entropy_t60'])
s2['X14_fr']['pass'] = mark(s2['X14_fr']['end_entropy'] < 0.06)
s, st = load('X14_sw'); sw = st['switch']; S0 = s[0]
if sw:
    Bsw = sw['B']; X = S0 + Bsw * cen(U); Ht = float(ent(X).mean()); K_ = float(np.median(kl(sm(X), sm(s[-1])))); H = float(ent(s[-1]).mean())
    s2['X14_sw'] = dict(switch_round=sw['round'], B_switch=Bsw, entropy_at_switch=sw['mean_entropy'], end_entropy=H, target_entropy_at_B_switch=Ht,
                        median_kl_target_policy=K_, predicted_switch_round=P['S2']['switch_round'], entropy_by_round=[float(x) for x in ent(s).mean(-1)])
    s2['X14_sw']['pass'] = mark(sw['round'] <= 20 and abs(H - Ht) <= 0.05 and K_ <= 0.01 and H >= 0.25)
else:
    s2['X14_sw'] = dict(switch_round=None, pass_='FAIL')
s, _ = load('X14_cen'); H = float(ent(s[-1]).mean())
s2['X14_cen'] = dict(B_hat=P['S2']['census_B_hat'], end_entropy=H, predicted=P['S2']['census_entropy_true_at_B_hat'], entropy_by_round=[float(x) for x in ent(s).mean(-1)])
s2['X14_cen']['pass'] = mark(abs(H - P['S2']['census_entropy_true_at_B_hat']) <= 0.05 and H >= 0.25)
s2['X14_fr']['entropy_by_round'] = [float(x) for x in ent(load('X14_fr')[0]).mean(-1)]
out['S2'] = s2
# ---------------- S3
s3 = {}
for name, B, a, crit in (('X14_c12', 12.0, P['S3']['alpha_choice'], 'census'), ('X14_n24', 24.0, 0.5, 'naive'), ('X14_c24', 24.0, P['S3b']['alpha_choice'], 'census')):
    s, _ = load(name); am = amp(s); X, _ = target(A_PL, s[0], B, 1.0); Kb = float(np.median(kl(sm(X[BAL]), sm(s[-1][BAL]))))
    row = dict(B=B, alpha=a, osc_share_balanced=float((am[BAL] > 0.0536).mean()), osc_share_all=float((am > 0.0536).mean()),
               median_amp_balanced=float(np.median(am[BAL])), median_kl_target_policy_balanced=Kb)
    key = 'S3' if B == 12 else 'S3b'; pk = 'census' if crit == 'census' else 'naive_a0.5'
    row['predicted'] = {k: P[key][pk][k] for k in ('osc_share_balanced', 'osc_share_all', 'median_kl_to_target_t80_balanced')}
    if name == 'X14_c12':
        row['pass'] = mark(row['osc_share_balanced'] <= 0.10 and Kb <= 0.01)
    elif name == 'X14_n24':
        row['pass'] = mark(row['osc_share_balanced'] >= 0.75)
    else:
        row['pass'] = mark(row['osc_share_balanced'] <= 0.10 and Kb <= 0.02)
    s3[name] = row
s, _ = load('H6_B12_a02', f'{EXP}/2026-10-16-v4-validation/results/runs'); am = amp(s)
s3['naive_H6_B12_a02'] = dict(osc_share_balanced=float((am[BAL] > 0.0536).mean()))
out['S3'] = s3
supported = (s1['S1-c'] == 'PASS' and all(s2[k]['end_entropy'] >= 0.25 for k in ('X14_sw', 'X14_cen') if 'end_entropy' in s2[k])
             and s3['X14_c12']['osc_share_balanced'] <= 0.10 and s3['X14_c24']['osc_share_balanced'] <= 0.10
             and s2['X14_fr']['pass'] == 'PASS' and s3['X14_n24']['pass'] == 'PASS')
out['verdict_supported'] = bool(supported)
json.dump(out, open(os.path.join(HERE, '..', 'results', 'evaluation_x14.json'), 'w'), indent=1)
for sec in ('S1', 'S2', 'S3'):
    for k, v in out[sec].items():
        print(sec, k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in (v.items() if isinstance(v, dict) else [('v', v)]) if 'by_round' not in kk})
print('verdict supported:', supported)
