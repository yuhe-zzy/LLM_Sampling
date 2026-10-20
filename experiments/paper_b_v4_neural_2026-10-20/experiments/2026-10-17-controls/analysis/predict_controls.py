"""Frozen population predictions for the control experiments (2026-10-17), computed before any training run.
Mirrors vm/panel_po_ctrl.py exactly at the population level (exact round optimum): judge bias in label log-odds, per-panel
drift controller, per-round KL controller, replay sampler, adaptive refresh. Panels: the 64-panel subset used since batch 2.
Usage: python predict_controls.py <v4 experiment dir> <out dir>"""
import json, os, sys, time
import numpy as np
V4, OUT = sys.argv[1:3]
sys.path.insert(0, os.path.join(V4, 'analysis'))
from population import link, sm, target, entropy, condorcet_winner
cen = lambda x: x - x.mean(-1, keepdims=True)
kl = lambda p, q: (p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum(-1)
K = 4


def load(d, r):
    Z = np.load(os.path.join(V4, 'results', d, r, 'run.npz')); return Z['s'][0].astype(float), Z['P'].astype(float), Z['ids']


def simulate(A0, s0, alpha, beta, lam, T, link_kind, eps=0.0, frozen=False, ctrl='none', kappa=2.0, Dstar=None, kl_round=0.02,
             rho=1.0, adaptive_W=0, B=0.0):
    """Population version of panel_po_ctrl.py. A0: payoff of the unbiased labels (logit or identity link)."""
    N = s0.shape[0]; s = s0.copy(); m = sm(s0); pi0 = sm(s0)
    a = alpha; b = np.full(N, beta); traj = [s.copy()]; alphas = []; betas = []; steps = []
    if adaptive_W: b[:] = (1 - a) * B
    logitP = A0 if link_kind == 'logit' else None
    for t in range(1, T + 1):
        pi = sm(s); m = (1 - rho) * m + rho * pi; mu = (1 - lam) / K + lam * m
        if ctrl == 'drift':
            b = kappa * np.maximum(Dstar - kl(pi, pi0), 0.0)
        sb = s0 if frozen else s
        if eps != 0.0:
            assert link_kind == 'logit'
            A = logitP + eps * (sb[:, :, None] - sb[:, None, :])
        else:
            A = A0
        g = np.einsum('nij,nj->ni', A, mu)
        s_new = cen(a * s + (1 - a) * s0 + b[:, None] * g)
        alphas.append(a); betas.append(b.copy())
        pin = sm(s_new)
        if ctrl == 'perround':
            k = float(kl(pin, pi).mean()); fac = 1.5 if k < kl_round / 1.5 else (1 / 1.5 if k > kl_round * 1.5 else 1.0)
            b = np.clip(b * fac, 1e-3, 8.0)
        if adaptive_W:
            steps.append(float(np.abs(pin - pi).max(-1).mean()))
            if t % adaptive_W == 0 and t >= 2 * adaptive_W and np.mean(steps[-adaptive_W:]) > 0.8 * np.mean(steps[-2 * adaptive_W:-adaptive_W]):
                a = min(0.95, a + 0.3 * (1 - a)); b[:] = (1 - a) * B
        s = s_new; traj.append(s.copy())
    return np.array(traj), np.array(alphas), np.array(betas)


def amp(traj, W=30):
    p = sm(traj[-W:]); t = np.arange(W) - (W - 1) / 2; sl = (t[:, None, None] * (p - p.mean(0))).sum(0) / (t ** 2).sum()
    return (p - p.mean(0) - sl * t[:, None, None]).std(0).max(-1)


def top2(x):
    xs = np.sort(x, -1); return float(np.median(xs[:, -1] - xs[:, -2]))


os.makedirs(OUT, exist_ok=True)
s0, P, ids = load('runs_sampled', 'S1n512_dpo_a0_b8')            # 64 panels, soft Bradley-Terry labels (T = 3.168)
A_logit = link(P, 'logit'); lam = 0.8
X8, _ = target(A_logit, s0, 8.0, lam); D8 = kl(sm(X8), sm(s0))
np.savez(os.path.join(OUT, 'drift_targets_B8.npz'), ids=ids, D=D8)
pred = {'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'panels': [int(i) for i in ids], 'runs': {}}
R = pred['runs']
# E1: judge bias, DPO soft, alpha = 0.5, beta = 1, 60 rounds
for name, eps, fr in (('C1_e0', 0.0, False), ('C1_e03', 0.3, False), ('C1_e045', 0.45, False), ('C1_e055', 0.55, False), ('C1_f055', 0.55, True)):
    tr, _, _ = simulate(A_logit, s0, 0.5, 1.0, lam, 60, 'logit', eps=eps, frozen=fr)
    R[name] = dict(H_t60=float(entropy(tr[-1]).mean()), H_t30=float(entropy(tr[30]).mean()), top2_t30=top2(tr[30]), top2_t60=top2(tr[60]))
# E2: entropy bonus (eps = -eta) under full refresh vs anchors, DPO soft, beta = 1, 60 rounds
for name, a, eps in (('C2_eta0125', 1.0, -0.125), ('C2_eta025', 1.0, -0.25), ('C2_anchorB8', 0.875, 0.0), ('C2_anchorB4', 0.75, 0.0)):
    tr, _, _ = simulate(A_logit, s0, a, 1.0, lam, 60, 'logit', eps=eps)
    R[name] = dict(H_t60=float(entropy(tr[-1]).mean()), KL0_t60=float(kl(sm(tr[-1]), sm(s0)).mean()))
# E3: controllers under full refresh, DPO soft, 60 rounds
tr, _, bt = simulate(A_logit, s0, 1.0, 1.0, lam, 60, 'logit', ctrl='drift', kappa=2.0, Dstar=D8)
R['C3_drift'] = dict(H_t60=float(entropy(tr[-1]).mean()), H_target_B8=float(entropy(X8).mean()), KL_to_B8_median=float(np.median(kl(sm(tr[-1]), sm(X8)))),
                     KL_to_B8_mean=float(kl(sm(tr[-1]), sm(X8)).mean()), B_cum_median=float(np.median(bt.sum(0))))
tr, _, bt = simulate(A_logit, s0, 1.0, 1.0, lam, 60, 'logit', ctrl='perround')
R['C3_perround'] = dict(H_t60=float(entropy(tr[-1]).mean()), B_cum=float(bt[:, 0].sum()))
# planted panels for E4-E6
s0p, Pp, idsp = load('runs', 'H6_B12_a02'); Apl = Pp - 0.5
Zp = np.load(os.path.join(V4, 'data', 'oracle_planted.npz')); kb = {int(i): k for i, k in zip(Zp['ids'], Zp['kind'])}
bal = np.array([kb[int(i)] == 'balanced' for i in idsp]); Xp, _ = target(Apl, s0p, 12.0, 1.0)
for name, a, rho, T in (('C4_a05_r02', 0.5, 0.2, 150), ('C4_a08_r02', 0.8, 0.2, 150), ('C4_a05_r1', 0.5, 1.0, 150), ('C4_a08_r1', 0.8, 1.0, 150)):
    tr, _, _ = simulate(Apl, s0p, a, (1 - a) * 12, 1.0, T, 'identity', rho=rho)
    am = amp(tr)
    R[name] = dict(osc_share_balanced=float((am[bal] > 0.0536).mean()), amp_balanced_median=float(np.median(am[bal])),
                   osc_share_decisive=float((am[~bal] > 0.0536).mean()))
tr, al, _ = simulate(Apl, s0p, 0.2, 9.6, 1.0, 150, 'identity', adaptive_W=10, B=12.0)
am = amp(tr)
R['C5_adapt'] = dict(alpha_path=[float(al[i]) for i in (0, 20, 40, 60, 80, 100, 149)], alpha_final=float(al[-1]),
                     osc_share_balanced=float((am[bal] > 0.0536).mean()), KL_to_target_balanced_median=float(np.median(kl(sm(tr[-1][bal]), sm(Xp[bal])))))
tr, _, _ = simulate(Apl, s0p, 0.2, 9.6, 1.0, 80, 'identity')
R['C6_wavg'] = dict(policy_avg30_KL_balanced_median_population=float(np.median(kl(sm(tr[-30:]).mean(0)[bal], sm(Xp[bal])))),
                    note='v4 neural H6_B12_a02: last iterate 0.444, 30-round policy average 0.0077 (balanced median KL to target)')
pred['criteria'] = {
 'E1': 'C1_e0, C1_e03, C1_e045, C1_f055: |H_t60 - predicted H_t60| <= 0.05, and end entropy ordered e0 > e03 > e045. C1_e055: median top-two gap at t60 >= 1.2 x the gap at t30 and H_t60 < 0.05.',
 'E2': 'each run |H_t60 - predicted| <= 0.05; KL to the initial policy: C2_eta0125 > C2_anchorB8 and C2_eta025 > C2_anchorB4.',
 'E3': 'C3_drift: median KL to the B=8 target <= 0.02 and mean entropy within 0.05 of the target entropy; C3_perround: H_t60 < 0.05.',
 'E4': 'oscillating share of balanced panels (v4 amplitude measure, threshold 0.0536): C4_a05_r02 <= 0.2 (v4 neural at rho = 1: 0.94); C4_a08_r02 >= 0.25 (v4 neural at rho = 1: 0.00).',
 'E5': 'C5_adapt: final alpha >= 0.6, oscillating share of balanced panels over the last 30 rounds <= 0.1, median KL to the B=12 target <= 0.02.',
 'E6': 'C6_wavg: balanced-panel median KL of the weight-averaged model (rounds 51-80) to the target <= 2 x that of the run\'s own 30-round policy average.'}
json.dump(pred, open(os.path.join(OUT, 'predictions_controls.json'), 'w'), indent=1)
print(json.dumps(R, indent=1))
