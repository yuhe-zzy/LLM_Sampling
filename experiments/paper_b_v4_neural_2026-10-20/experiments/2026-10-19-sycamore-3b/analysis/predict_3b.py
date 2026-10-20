"""Sycamore 3B dynamics runs (2026-10-19): frozen population predictions, computed from the 3B model's own initial panel
scores (s_0 from the timing job) before any 3B training run.

Five configurations on the same 64 panels, each the exact counterpart of a 0.5B v4 run:
  D3B_conv_rm   IPO, soft reward-model labels (T = std of within-panel reward gaps over all 3B panels), alpha 0.8, beta 1.6 (B = 8), lambda 0.8, 30 rounds   <-> H1_a08_b16
  D3B_pol_full  IPO, hard reward-model labels, full refresh alpha 1, beta 1, lambda 1, 40 rounds                 <-> H3_a1_b1
  D3B_osc_fast  IPO, planted cycle, alpha 0.2, beta 9.6 (B = 12), lambda 1, 80 rounds                           <-> H6_B12_a02
  D3B_conv_pl   IPO, planted cycle, alpha 0.8, beta 2.4 (B = 12), lambda 1, 80 rounds                           <-> H6_B12_a08
  D3B_osc_slow  IPO, planted cycle, alpha 0.9, beta 4 (B = 40), lambda 1, 80 rounds                             <-> H5_b4_a09
Usage: python predict_3b.py <scoring run.npz> <panels.json> <oracle_planted.npz> <soft-label temperature> [out.json]
(the scoring run gives the model's exact initial panel scores s_0 on the training panels)"""
import json, os, sys
import numpy as np
EXP = '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/experiments'
sys.path.insert(0, f'{EXP}/2026-10-16-v4-validation/analysis')
from population import sm, trajectory, target, hodge, omega_max, condorcet_winner
HERE = os.path.dirname(os.path.abspath(__file__)); PANELS, ORACLE, TEMP = sys.argv[2], sys.argv[3], float(sys.argv[4])
OUT = sys.argv[5] if len(sys.argv) > 5 else os.path.join(HERE, '..', 'prereg', 'predictions_3b.json')
Zt = np.load(sys.argv[1]); IDS = [int(i) for i in Zt['ids']]; S0 = Zt['s'][0].astype(float); K = 4
pan = {p['id']: p for p in json.load(open(PANELS))['panels']}
RM = 'Skywork-Reward-Llama-3.1-8B-v0.2-d4117fbfd81b'; R = np.array([pan[i]['rm'][RM] for i in IDS]); G = R[:, :, None] - R[:, None, :]
P_SOFT = 1 / (1 + np.exp(-G / TEMP)); P_HARD = (G > 0) + 0.5 * (G == 0)
Zp = np.load(ORACLE); pm = {int(i): n for n, i in enumerate(Zp['ids'])}
P_PL = np.stack([Zp['P'][pm[i]] for i in IDS]).astype(float); KIND = np.array([str(Zp['kind'][pm[i]]) for i in IDS]); BAL = KIND == 'balanced'
for P in (P_SOFT, P_HARD, P_PL):
    for n in range(len(IDS)): np.fill_diagonal(P[n], 0.5)
ent = lambda s: -(sm(s) * np.log(sm(s) + 1e-300)).sum(-1)
kl = lambda p, q: (p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum(-1)


def amp(traj, W=30):
    p = sm(traj[-W:]); t = np.arange(W) - (W - 1) / 2; sl = (t[:, None, None] * (p - p.mean(0))).sum(0) / (t ** 2).sum()
    return (p - p.mean(0) - sl * t[:, None, None]).std(0).max(-1)


CFG = {'D3B_conv_rm': (P_SOFT, 0.8, 1.6, 0.8, 30), 'D3B_pol_full': (P_HARD, 1.0, 1.0, 1.0, 40), 'D3B_osc_fast': (P_PL, 0.2, 9.6, 1.0, 80),
       'D3B_conv_pl': (P_PL, 0.8, 2.4, 1.0, 80), 'D3B_osc_slow': (P_PL, 0.9, 4.0, 1.0, 80)}
out = dict(ids=IDS, s0_source=os.path.abspath(sys.argv[1]), panels=os.path.abspath(PANELS), oracle=os.path.abspath(ORACLE), temp=TEMP, init_mean_entropy=float(ent(S0).mean()), init_mean_top=float(sm(S0).max(-1).mean()), runs={})
for name, (P, a, b, lam, T) in CFG.items():
    A = P - 0.5; tr = trajectory(A, S0, a, b, lam, T); am = amp(tr) if T >= 30 else np.zeros(len(IDS))
    row = dict(alpha=a, beta=b, lam=lam, rounds=T, end_mean_entropy=float(ent(tr[-1]).mean()), end_mean_top=float(sm(tr[-1]).max(-1).mean()),
               entropy_by_round=[float(x) for x in ent(tr).mean(-1)])
    if P is P_PL:
        row.update(osc_share_all=float((am > 0.0536).mean()), osc_share_balanced=float((am[BAL] > 0.0536).mean()), osc_share_decisive=float((am[~BAL] > 0.0536).mean()),
                   median_amp_balanced=float(np.median(am[BAL])))
        if a < 1:
            X, _ = target(A, S0, b / (1 - a), lam); row['median_kl_target_end_balanced'] = float(np.median(kl(sm(X[BAL]), sm(tr[-1][BAL]))))
            g = np.array([b / (1 - a) * lam * omega_max(hodge(A[n])[1], sm(X[n])) for n in range(len(IDS))])
            row['median_gamma_B_balanced'] = float(np.median(g[BAL])); row['alpha_loc_median_balanced'] = float(np.median(np.where(g[BAL] > 1, (g[BAL] ** 2 - 1) / (g[BAL] ** 2 + 1), 0)))
    if name == 'D3B_pol_full':
        cw = np.array([condorcet_winner(P[n]) for n in range(len(IDS))]); has = cw >= 0
        row['share_with_cw'] = float(has.mean()); row['mass_on_cw_end_mean'] = float(np.mean([sm(tr[-1][n])[cw[n]] for n in range(len(IDS)) if has[n]]))
    if name == 'D3B_conv_rm':
        X, _ = target(A, S0, 8.0, lam); row['target_mean_entropy'] = float(ent(X).mean()); row['median_kl_target_end'] = float(np.median(kl(sm(X), sm(tr[-1]))))
    out['runs'][name] = row
    print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k != 'entropy_by_round'})
os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(out, open(OUT, 'w'), indent=1); print('written', OUT)
