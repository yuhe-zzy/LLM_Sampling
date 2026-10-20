"""X14 (2026-10-19): frozen pipeline decisions and population predictions, computed before any X14 training run.

X14 tests whether interventions chosen ONLY from P-free diagnostics keep the trained loop from collapsing, over-sharpening or
rotating. The controller never reads P. It reads the model's own panel log-probabilities, sampled judgments (Bernoulli draws
from the label source, standing in for judge calls), and nothing else. P is used here only to compute the predictions that
the runs are scored against.

Decisions fixed here (pre-training census, seeded):
  S2-census  soft reward-model labels, fixed reference (alpha = 0): 3 judgments per ordered pair (36 calls per prompt), a
             Bradley-Terry fit with ridge 0.1, and the sharpness B_hat whose predicted mean panel entropy is the budget 0.3.
  S3-census  planted cycles, planned B = 12: 3 judgments per ordered pair, the noise-corrected gain census, and
             alpha = min(0.95, max_n alpha_loc(gamma_hat_n) + 0.05), beta = (1 - alpha) * 12.
Decisions taken online by the training script (rules fixed here, outcomes logged by the run):
  S1-debias / S1-comp  judge biased toward the current policy by eps (unknown to the controller): at round 1, D5 with
             16 judgments per pair at s_0 and s_1 on all 64 panels (12,288 calls), conditional-likelihood eps_hat; act only
             if |eps_hat| > 1.96 SE. debias: labels' log-odds get - eps_hat (s_i - s_j) from round 2 on. comp: refresh is
             lowered so that alpha + beta*eps_hat equals the planned 0.5 (alpha = 0.5 - eps_hat if >= 0, else alpha = 0 and
             beta = 0.5 / eps_hat).
  S2-switch  full refresh, beta = 0.5: when the mean panel entropy of the current policy first falls to 0.3 or below,
             switch every panel to the anchor alpha = 1 - beta / B_t, B_t = beta * (rounds completed).
Usage: python predict_x14.py   (writes ../prereg/predictions.json)"""
import json, math, os, sys
import numpy as np
EXP = '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/experiments'
sys.path.insert(0, f'{EXP}/2026-10-16-v4-validation/analysis')
from population import sm, target, hodge, omega_max
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, '..', 'prereg', 'predictions.json')
K = 4; IU = np.triu_indices(K, 1)
IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
       35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64]
cen = lambda x: x - x.mean(-1, keepdims=True)
ent = lambda s: -(sm(s) * np.log(sm(s) + 1e-300)).sum(-1)
kl = lambda p, q: (p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum(-1)
pan = {p['id']: p for p in json.load(open(f'{EXP}/2026-10-16-v4-validation/data/panels_q05.json'))['panels']}
RM = 'Skywork-Reward-Llama-3.1-8B-v0.2-d4117fbfd81b'; T = 3.168
R = np.array([pan[i]['rm'][RM] for i in IDS]); U_TRUE = R / T; LOGITP = U_TRUE[:, :, None] - U_TRUE[:, None, :]
Zr = np.load(f'{EXP}/2026-10-17-controls/results/runs/C1_e0/run.npz'); assert list(Zr['ids']) == IDS; S0_RM = Zr['s'][0].astype(float)
Zp = np.load(f'{EXP}/2026-10-16-v4-validation/data/oracle_planted.npz'); pm = {int(i): n for n, i in enumerate(Zp['ids'])}
P_PL = np.stack([Zp['P'][pm[i]] for i in IDS]).astype(float); KIND = np.array([str(Zp['kind'][pm[i]]) for i in IDS])
Zh = np.load(f'{EXP}/2026-10-16-v4-validation/results/runs/H6_B12_a02/run.npz'); assert list(Zh['ids']) == IDS; S0_PL = Zh['s'][0].astype(float)
assert np.allclose(S0_PL, S0_RM, atol=1e-3)
rng = np.random.default_rng(14)
out = dict(ids=IDS, budget_entropy=0.3)


def simulate(A0, s0, alpha, beta, lam, T_, link_kind, eps=0.0, sched=None):
    """Population MRS-PsiPO, mirroring the training script; sched(t, s, s0) may return (alpha, beta, residual eps) per round."""
    s = s0.copy(); a, b, e = alpha, beta, eps; traj = [s.copy()]; info = []
    for t in range(1, T_ + 1):
        if sched is not None:
            a, b, e = sched(t, s, a, b, e)
        mu = (1 - lam) / K + lam * sm(s)
        A = A0 + e * (s[:, :, None] - s[:, None, :]) if e != 0.0 else A0
        g = np.einsum('nij,nj->ni', A, mu)
        s = cen(a * s + (1 - a) * s0 + b * g); traj.append(s.copy()); info.append((a, b, e))
    return np.array(traj), info


def amp(traj, W=30):
    p = sm(traj[-W:]); t = np.arange(W) - (W - 1) / 2; sl = (t[:, None, None] * (p - p.mean(0))).sum(0) / (t ** 2).sum()
    return (p - p.mean(0) - sl * t[:, None, None]).std(0).max(-1)


def judged(P, n_per_order):
    """Sampled judgments: n per presentation order (the synthetic judges have no order effect), so 2n per unordered pair."""
    w = rng.binomial(2 * n_per_order, P[:, IU[0], IU[1]]); return w, 2 * n_per_order


# ------------------------------------------------------------------ S1: judge bias, planned alpha = 0.5, beta = 1 (B = 2)
s1 = {}
tr, _ = simulate(LOGITP, S0_RM, 0.5, 1.0, 0.8, 60, 'logit'); s1['planned_entropy_eps0'] = float(ent(tr[-1]).mean())
for eps in (0.45, 0.55):
    tr, _ = simulate(LOGITP, S0_RM, 0.5, 1.0, 0.8, 60, 'logit', eps=eps); naive = float(ent(tr[-1]).mean())
    grid = {}
    for eh in np.round(np.arange(eps - 0.15, eps + 0.151, 0.01), 3):
        def sch_db(t, s, a, b, e, eh=eh, eps=eps):
            return (0.5, 1.0, eps if t == 1 else eps - eh)
        def sch_cp(t, s, a, b, e, eh=eh, eps=eps):
            if t == 1:
                return (0.5, 1.0, eps)
            return (0.5 - eh, 1.0, eps) if 0.5 - eh >= 0 else (0.0, 0.5 / eh, eps)
        tdb, _ = simulate(LOGITP, S0_RM, 0.5, 1.0, 0.8, 60, 'logit', eps=eps, sched=sch_db)
        tcp, _ = simulate(LOGITP, S0_RM, 0.5, 1.0, 0.8, 60, 'logit', eps=eps, sched=sch_cp)
        top = lambda tr_: float(np.median(np.abs(np.diff(np.sort(tr_[-11:], -1)[..., -1] - np.sort(tr_[-11:], -1)[..., -2], axis=0)).mean(0)))
        grid[f'{eh:.2f}'] = dict(debias_entropy=float(ent(tdb[-1]).mean()), comp_entropy=float(ent(tcp[-1]).mean()),
                                 debias_last10_top2_step=top(tdb), comp_last10_top2_step=top(tcp))
    s1[f'eps={eps}'] = dict(naive_entropy_t60=naive, by_eps_hat=grid)
    print('S1', eps, 'naive', round(naive, 4), 'debias@eps', grid[f'{eps:.2f}']['debias_entropy'], 'comp@eps', grid[f'{eps:.2f}']['comp_entropy'])
out['S1'] = s1

# ------------------------------------------------------------------ S2: soft reward-model labels, budget mean entropy 0.3
s2 = {}
Hb = lambda B, u: float(ent(S0_RM + B * cen(u)).mean())
# naive full refresh beta = 0.5 for 60 rounds: B_60 = 30
s2['naive_entropy_t60'] = Hb(30.0, U_TRUE)
# switch: loop round t sees the policy after t-1 updates (B = 0.5 (t-1)); switch when its mean entropy <= 0.3
for t in range(1, 61):
    if Hb(0.5 * (t - 1), U_TRUE) <= 0.3:
        s2['switch_round'] = t; s2['switch_B'] = 0.5 * (t - 1); s2['switch_entropy_target'] = Hb(0.5 * (t - 1), U_TRUE); break
# census: 36 calls per prompt, BT fit with ridge 0.1, B_hat with predicted mean entropy 0.3
w, nn = judged(1 / (1 + np.exp(-LOGITP)), 3)
uh = np.zeros((len(IDS), K))
for n in range(len(IDS)):
    u = np.zeros(K)
    for _ in range(100):
        d = u[IU[0]] - u[IU[1]]; p = 1 / (1 + np.exp(-d)); gpair = w[n] - nn * p; hpair = nn * p * (1 - p)
        g = np.zeros(K); H = 0.1 * np.eye(K)
        for e_, (i, j) in enumerate(zip(*IU)):
            g[i] += gpair[e_]; g[j] -= gpair[e_]
            H[i, i] += hpair[e_]; H[j, j] += hpair[e_]; H[i, j] -= hpair[e_]; H[j, i] -= hpair[e_]
        g -= 0.1 * u; step = np.linalg.solve(H, g); u += step
        if np.abs(step).max() < 1e-10: break
    uh[n] = u - u.mean()
lo, hi = 0.0, 60.0
for _ in range(80):
    mid = (lo + hi) / 2
    (lo, hi) = (mid, hi) if Hb(mid, uh) > 0.3 else (lo, mid)
s2['census_B_hat'] = (lo + hi) / 2; s2['census_entropy_true_at_B_hat'] = Hb(s2['census_B_hat'], U_TRUE)
s2['census_calls_per_prompt'] = 36
print('S2', s2)
out['S2'] = s2

# ------------------------------------------------------------------ S3: planted cycles, planned B = 12, IPO lambda = 1
s3 = {}; A_PL = P_PL - 0.5; B = 12.0
w, nn = judged(P_PL, 3)
ph = w / nn; var = ph * (1 - ph) / nn
Ah = np.zeros((len(IDS), K, K)); Ah[:, IU[0], IU[1]] = ph - 0.5; Ah[:, IU[1], IU[0]] = 0.5 - ph
gam_hat = np.zeros(len(IDS)); Ac = Ah.copy()
for n in range(len(IDS)):
    u, C = hodge(Ah[n]); bias = (K - 2) / K * var[n].sum(); f = math.sqrt(max(0.0, 1 - bias / max((C[IU] ** 2).sum(), 1e-12)))
    Ac[n] = (u[:, None] - u[None, :]) + f * C
X, _ = target(Ac, S0_PL, B, 1.0)
for n in range(len(IDS)):
    gam_hat[n] = B * omega_max(hodge(Ac[n])[1], sm(X[n]))
aloc = np.where(gam_hat > 1, (gam_hat ** 2 - 1) / (gam_hat ** 2 + 1), 0.0)
a_star = float(min(0.95, aloc.max() + 0.05)) if (gam_hat > 1).any() else 0.2
Xt, _ = target(A_PL, S0_PL, B, 1.0); gam_true = np.array([B * omega_max(hodge(A_PL[n])[1], sm(Xt[n])) for n in range(len(IDS))])
s3.update(gamma_hat=gam_hat.tolist(), alpha_loc_hat_max=float(aloc.max()), alpha_choice=a_star, beta_choice=(1 - a_star) * B,
          gamma_true=gam_true.tolist(), alpha_loc_true_max=float(np.where(gam_true > 1, (gam_true ** 2 - 1) / (gam_true ** 2 + 1), 0).max()),
          share_flagged=float((gam_hat > 1).mean()), share_true_gt1=float((gam_true > 1).mean()))
bal = KIND == 'balanced'
for name, a in (('naive_a0.2', 0.2), ('census', a_star)):
    tr, _ = simulate(A_PL, S0_PL, a, (1 - a) * B, 1.0, 80, 'identity'); am = amp(tr)
    s3[name] = dict(alpha=a, beta=(1 - a) * B, osc_share_balanced=float((am[bal] > 0.0536).mean()), osc_share_all=float((am > 0.0536).mean()),
                    median_kl_to_target_t80=float(np.median(kl(sm(Xt), sm(tr[-1])))), median_kl_to_target_t80_balanced=float(np.median(kl(sm(Xt[bal]), sm(tr[-1][bal])))))
print('S3', {k: v for k, v in s3.items() if not isinstance(v, list)})
out['S3'] = s3

# ------------------------------------------------------------------ benign checks (no training): would any intervention fire?
ben = {}
w, nn = judged(1 / (1 + np.exp(-LOGITP)), 3)          # DPO acts on the logit scale: empirical logits with a half-count correction
ph = (w + 0.5) / (nn + 1); lg = np.log(ph / (1 - ph)); var = 1 / ((nn + 1) * ph * (1 - ph))
Ah = np.zeros((len(IDS), K, K)); Ah[:, IU[0], IU[1]] = lg; Ah[:, IU[1], IU[0]] = -lg
for n in range(len(IDS)):
    u, C = hodge(Ah[n]); bias = (K - 2) / K * var[n].sum(); f = math.sqrt(max(0.0, 1 - bias / max((C[IU] ** 2).sum(), 1e-12)))
    Ah[n] = (u[:, None] - u[None, :]) + f * C
Xb, _ = target(Ah, S0_RM, 2.0, 0.8)
gb = np.array([2.0 * 0.8 * omega_max(hodge(Ah[n])[1], sm(Xb[n])) for n in range(len(IDS))])
ben['rm_labels_gain_census_B2_max_gamma'] = float(gb.max()); ben['rm_labels_gain_census_B2_flagged'] = int((gb > 1).sum())
out['benign'] = ben
print('benign', ben)

# ------------------------------------------------------------------ S3b: planted cycles at a sharper planned target, B = 24 (no earlier runs)
s3b = {}; B2 = 24.0
w, nn = judged(P_PL, 3)
ph = w / nn; var = ph * (1 - ph) / nn
Ah = np.zeros((len(IDS), K, K)); Ah[:, IU[0], IU[1]] = ph - 0.5; Ah[:, IU[1], IU[0]] = 0.5 - ph; Ac = Ah.copy()
for n in range(len(IDS)):
    u, C = hodge(Ah[n]); bias = (K - 2) / K * var[n].sum(); f = math.sqrt(max(0.0, 1 - bias / max((C[IU] ** 2).sum(), 1e-12)))
    Ac[n] = (u[:, None] - u[None, :]) + f * C
X, _ = target(Ac, S0_PL, B2, 1.0); gh = np.array([B2 * omega_max(hodge(Ac[n])[1], sm(X[n])) for n in range(len(IDS))])
al = np.where(gh > 1, (gh ** 2 - 1) / (gh ** 2 + 1), 0.0); a2 = float(min(0.95, al.max() + 0.05)) if (gh > 1).any() else 0.5
Xt2, _ = target(A_PL, S0_PL, B2, 1.0); gt2 = np.array([B2 * omega_max(hodge(A_PL[n])[1], sm(Xt2[n])) for n in range(len(IDS))])
s3b.update(gamma_hat=gh.tolist(), alpha_loc_hat_max=float(al.max()), alpha_choice=a2, beta_choice=(1 - a2) * B2, gamma_true=gt2.tolist(),
           alpha_loc_true_max=float(np.where(gt2 > 1, (gt2 ** 2 - 1) / (gt2 ** 2 + 1), 0).max()), share_flagged=float((gh > 1).mean()), share_true_gt1=float((gt2 > 1).mean()))
for name, a in (('naive_a0.5', 0.5), ('census', a2)):
    tr, _ = simulate(A_PL, S0_PL, a, (1 - a) * B2, 1.0, 80, 'identity'); am = amp(tr)
    s3b[name] = dict(alpha=a, beta=(1 - a) * B2, osc_share_balanced=float((am[bal] > 0.0536).mean()), osc_share_all=float((am > 0.0536).mean()),
                     median_kl_to_target_t80=float(np.median(kl(sm(Xt2), sm(tr[-1])))), median_kl_to_target_t80_balanced=float(np.median(kl(sm(Xt2[bal]), sm(tr[-1][bal])))))
print('S3b', {k: v for k, v in s3b.items() if not isinstance(v, list)})
out['S3b'] = s3b
json.dump(out, open(OUT, 'w'), indent=1)
print('written', OUT)
