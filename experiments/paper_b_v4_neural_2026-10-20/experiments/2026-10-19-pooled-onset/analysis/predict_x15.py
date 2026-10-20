"""X15 population predictions (2026-10-19): pooled labels from three transitive criteria on specialist panels, owner routing with
share c, population DPO map (dpo_map.dpo_round) and the identity-link master recursion for IPO. Writes oracle npz files and the
frozen predictions. Usage: python predict_x15.py <data_dir with panels3.json, panels4.json> <out_dir>"""
import json, os, sys
import numpy as np
sys.path.insert(0, '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/reviews/2026-10-18-jmlr-review/theory')
from dpo_map import dpo_round, logit_round, sm, Pi, bt_fit, dphi_formula
data_dir, out_dir = sys.argv[1:3]; os.makedirs(out_dir, exist_ok=True)
THR = 0.0536; SKY = 'Skywork-Reward-Llama-3.1-8B-v0.2-d4117fbfd81b'
sig = lambda z: 1 / (1 + np.exp(-z)); logit = lambda p: np.log(p / (1 - p))
RECIPES = {'A': (0.0, 10.0), 'C': (0.9, 1.0)}


def load(K):
    D = json.load(open(os.path.join(data_dir, f'panels{K}.json'))); pan = D['panels']
    crit = {'Q': np.array([p['rm'][SKY] for p in pan]), 'C': -np.array([p['chars'] for p in pan], float), 'N': -np.array([p['s0'] for p in pan])}
    return pan, crit


def scales(crit, K):
    IU = np.triu_indices(K, 1); T = {}
    for k, s in crit.items():
        g = s[:, :, None] - s[:, None, :]; T[k] = float(np.std(g[:, IU[0], IU[1]]))
    return T


def labels(crit, T, mt):
    Z = {k: mt * (s[:, :, None] - s[:, None, :]) / T[k] for k, s in crit.items()}
    return Z, {k: sig(z) for k, z in Z.items()}


def owner_weights(Z):
    """Owner of an unordered pair by role (responses are stored in role order Q, C, N[, L]): the cycle edge {Q,C} belongs to Q
    (its top pair), {C,N} to C, {N,Q} to N, as in cor:pooled-onset. Pairs involving the fourth answer (K=4), on which all
    criteria agree, go to the criterion with the largest standardized gap. W[s, n, i, j] in {0,1}, symmetric in (i,j)."""
    ks = list(Z); S = len(ks); A = np.stack([Z[k] for k in ks]); N, K = A.shape[1:3]
    W = np.zeros((S, N, K, K)); role = {'Q': 0, 'C': 1, 'N': 2}
    for (i, j), k in (((0, 1), 'Q'), ((1, 2), 'C'), ((2, 0), 'N')):
        W[ks.index(k), :, i, j] = 1; W[ks.index(k), :, j, i] = 1
    if K == 4:
        sal = np.abs(A).argmax(0)
        for i in range(3):
            W[:, :, i, 3] = 0; W[:, :, 3, i] = 0
            for s in range(S):
                W[s, :, i, 3] = (sal[:, i, 3] == s); W[s, :, 3, i] = (sal[:, i, 3] == s)
    return W


def pool(Z, P, rule, c=None):
    ks = list(P); S = len(ks); N, K = P[ks[0]].shape[:2]
    soft = np.mean([P[k] for k in ks], 0)
    if rule == 'soft': Pp = soft
    elif rule == 'owner':
        W = owner_weights(Z); Pown = sum(W[s] * P[k] for s, k in enumerate(ks)); Pp = c * Pown + (1 - c) * soft
    elif rule == 'salience':
        A = np.abs(np.stack([Z[k] for k in ks])); W = (A == A.max(0, keepdims=True)).astype(float); W = W / W.sum(0, keepdims=True)
        Pp = sum(W[s] * P[k] for s, k in enumerate(ks))
    for n in range(N): np.fill_diagonal(Pp[n], .5)
    assert np.allclose(Pp + np.transpose(Pp, (0, 2, 1)), 1, atol=1e-9)
    return Pp


def hodge_stats(Pp):
    N, K = Pp.shape[:2]; IU = np.triu_indices(K, 1); out = []
    for n in range(N):
        A = logit(np.clip(Pp[n], 1e-9, 1 - 1e-9)); r = A.mean(1); C = A - (r[:, None] - r[None, :])
        tot = (A[IU] ** 2).sum(); share = float((C[IU] ** 2).sum() / tot) if tot > 0 else 0.0
        curl = float(A[0, 1] + A[1, 2] + A[2, 0]); cw = any(all(Pp[n][w, j] > .5 for j in range(K) if j != w) for w in range(K))
        out.append(dict(cyc_share=share, curl012=curl, condorcet_winner=bool(cw)))
    return out


def amp(tr, W=30):
    p = np.array([sm(x) for x in tr[-W:]]); t = np.arange(W) - (W - 1) / 2
    sl = (t[:, None] * (p - p.mean(0))).sum(0) / (t ** 2).sum()
    return float((p - p.mean(0) - sl * t[:, None]).std(0).max())


def run(P, s, a, b, rnd, T=120):
    K = len(s); r = Pi(np.array(s)); x = r.copy(); tr = [x]
    for _ in range(T): x = rnd(x, r, P, a, b, 1.0, np.ones(K) / K); tr.append(x)
    return tr


def target_newton(P, r, beta, rnd_name, x0):
    """Fixed point of x = r + beta * Pi(phi(mu(x))) (DPO) or x = r + beta * Pi(A mu(x)) (logit), by damped Newton from x0."""
    K = len(r); pi0 = np.ones(K) / K
    A = logit(np.clip(P, 1e-9, 1 - 1e-9))
    def F(x):
        mu = sm(x)
        return (bt_fit(P, mu) if rnd_name == 'dpo' else Pi(A @ mu))
    def DF(x):
        mu = sm(x); J = np.diag(mu) - np.outer(mu, mu)
        return (dphi_formula(P, mu) @ J) if rnd_name == 'dpo' else (A @ J)
    x = x0.copy()
    for it in range(100):
        g = x - r - beta * Pi(F(x)); gn = np.abs(g).max()
        if gn < 1e-10: break
        M = np.eye(K) - beta * DF(x) + np.ones((K, K)) / K      # the ones-block pins the centering gauge
        step = -np.linalg.solve(M, g); t = 1.0
        while t > 1e-6:
            xn = Pi(x + t * step); gnn = np.abs(xn - r - beta * Pi(F(xn))).max()
            if gnn < gn: x = xn; break
            t /= 2
        else: break
    return x, np.abs(x - r - beta * Pi(F(x))).max()


def onset_beta(P, s, rnd_name):
    """First beta on a log grid at which the alpha=0 loop's multiplier modulus at the target of sharpness beta reaches 1."""
    K = len(s); r = Pi(np.array(s)); A = logit(np.clip(P, 1e-9, 1 - 1e-9)); x = r.copy()
    grid = np.exp(np.linspace(np.log(0.05), np.log(400), 160)); prev = None
    for beta in grid:
        x, res = target_newton(P, r, beta, rnd_name, x)
        if res > 1e-6:
            x, res = target_newton(P, r, beta, rnd_name, r.copy())
            if res > 1e-6: continue       # skip this grid point; the scan continues
        mu = sm(x); J = np.diag(mu) - np.outer(mu, mu)
        M = beta * ((dphi_formula(P, mu) @ J) if rnd_name == 'dpo' else (A @ J))
        rad = float(np.abs(np.linalg.eigvals(M)).max())
        if rad >= 1:
            if prev is None: return float(beta)
            b0, r0 = prev; return float(np.exp(np.log(b0) + (1 - r0) / (rad - r0) * (np.log(beta) - np.log(b0))))
        prev = (beta, rad)
    return np.inf


out = dict(THR=THR, recipes=RECIPES, panels={}, arms={}, hodge={}, onset={})
oracles = {}
for K in (3, 4):
    if not os.path.exists(os.path.join(data_dir, f'panels{K}.json')): continue
    pan, crit = load(K); N = len(pan); ids = np.array([p['id'] for p in pan]); S0 = [p['s0'] for p in pan]
    if K == 3: T = scales(crit, 3); out['T'] = T
    out['panels'][K] = dict(N=N, ids=ids.tolist())
    for mt in (1.0, 0.5):
        Z, P = labels(crit, T, mt)
        mats = {f'owner_c{int(c*100)}': pool(Z, P, 'owner', c) for c in (0, 0.25, 0.5, 1.0)}
        mats['salience'] = pool(Z, P, 'salience')
        for name, M in mats.items():
            tag = f'K{K}_m{mt}_{name}'; np.savez_compressed(os.path.join(out_dir, f'oracle_{tag}.npz'), ids=ids, P=M); oracles[tag] = M
            hs = hodge_stats(M); out['hodge'][tag] = dict(mean_cyc_share=float(np.mean([h['cyc_share'] for h in hs])),
                                                        mean_curl=float(np.mean([h['curl012'] for h in hs])),
                                                        no_cw=float(np.mean([not h['condorcet_winner'] for h in hs])))
            if K == 3 and name.startswith('owner'):
                ons = [onset_beta(M[n], S0[n], 'dpo') for n in range(N)]
                out['onset'][tag] = dict(median=float(np.median(ons)), q25=float(np.quantile(ons, .25)), q75=float(np.quantile(ons, .75)), per_panel=ons)
            print(tag, out['hodge'][tag], out['onset'].get(tag, {}).get('median'), flush=True)

ARMS = {
    'X15_c0_A': ('K3_m1.0_owner_c0', 'A', 'dpo'), 'X15_c25_A': ('K3_m1.0_owner_c25', 'A', 'dpo'),
    'X15_c50_A': ('K3_m1.0_owner_c50', 'A', 'dpo'), 'X15_c100_A': ('K3_m1.0_owner_c100', 'A', 'dpo'),
    'X15_c100_C': ('K3_m1.0_owner_c100', 'C', 'dpo'), 'X15_c50_C': ('K3_m1.0_owner_c50', 'C', 'dpo'),
    'X15_rand_A': ('K3_m1.0_owner_c0', 'A', 'dpo'), 'X15_c100_A_ipo': ('K3_m1.0_owner_c100', 'A', 'ipo'),
    'X15_m05_c100_A': ('K3_m0.5_owner_c100', 'A', 'dpo'),
    'X15_K4_c100_A': ('K4_m1.0_owner_c100', 'A', 'dpo'), 'X15_K4_c0_A': ('K4_m1.0_owner_c0', 'A', 'dpo'),
}
for arm, (tag, rec, obj) in ARMS.items():
    if tag not in oracles: continue
    K = int(tag[1]); pan, _ = load(K); M = oracles[tag]; a, b = RECIPES[rec]; rnd = dpo_round if obj == 'dpo' else logit_round
    res = []
    for n, p in enumerate(pan):
        tr = run(M[n], p['s0'], a, b, rnd); A_ = amp(tr); res.append(dict(id=p['id'], amp=A_, osc=bool(A_ > THR), end=sm(tr[-1]).tolist()))
    out['arms'][arm] = dict(oracle=tag, recipe=rec, objective=obj, osc_share=float(np.mean([r['osc'] for r in res])),
                            n_osc=int(sum(r['osc'] for r in res)), per_panel=res)
    print(arm, 'osc share', out['arms'][arm]['osc_share'], 'n', out['arms'][arm]['n_osc'], flush=True)
json.dump(out, open(os.path.join(out_dir, 'predictions_x15.json'), 'w'), indent=1)
