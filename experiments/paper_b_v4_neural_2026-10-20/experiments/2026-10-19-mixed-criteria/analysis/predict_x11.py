"""X11 population predictions (2026-10-19): pooled labels from three individually transitive criteria on the 130 v4 panels.
Criteria (scalar scores per response, higher preferred): Q quality = Skywork-Reward-Llama-3.1-8B score; C concision = -characters;
N originality = -s0 (total log-probability under the base policy). Each criterion's soft labels P^k = sigma(gap / T_k), T_k the std of
its within-panel gaps over all panels (the framework's `soft` rule). Pooling rules: single criterion; soft = uniform mixture;
salience = each pair labelled by the criterion with the largest standardized gap |gap|/T_k (a judge that decides on the most salient
difference); filtered = each criterion keeps its most decisive third of pairs, uniform fallback. Recipes: actual population DPO (T1),
lambda = 1, reference = base policy, 120 rounds from log p0; oscillating = detrended amplitude over the last 30 rounds > 0.0536."""
import json, os, sys
import numpy as np
sys.path.insert(0, '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/reviews/2026-10-18-jmlr-review/theory')
from dpo_map import dpo_round, sm, Pi, bt_fit
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PAN = '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/experiments/2026-10-16-v4-validation/data/panels_q05.json'
RM = 'Skywork-Reward-Llama-3.1-8B-v0.2-d4117fbfd81b'
THR = 0.0536; K = 4; IU = np.triu_indices(K, 1)
sig = lambda z: 1 / (1 + np.exp(-z)); logit = lambda p: np.log(p / (1 - p))
panels = json.load(open(PAN))['panels']; ids = np.array([p['id'] for p in panels]); N = len(panels)
crit = {'Q': np.array([p['rm'][RM] for p in panels]),
        'C': -np.array([[len(r) for r in p['responses']] for p in panels], float),
        'N': -np.array([p['s0'] for p in panels])}
jury = {n: np.array([p['rm'][n] for p in panels]) for n in sorted(panels[0]['rm'])}
s0 = np.array([p['s0'] for p in panels])


def labels(scores):
    Z, P = {}, {}
    for k, s in scores.items():
        g = s[:, :, None] - s[:, None, :]; T = float(np.std(g[:, IU[0], IU[1]])); Z[k] = g / T; P[k] = sig(Z[k])
    return Z, P


def pool(Z, P, rule):
    ks = list(P); S = len(ks)
    if rule.startswith('single:'): return P[rule[7:]]
    if rule == 'soft': return np.mean([P[k] for k in ks], 0)
    A = np.abs(np.stack([Z[k] for k in ks]))                                  # [S,N,K,K]
    if rule == 'salience':
        W = (A == A.max(0, keepdims=True)).astype(float)
    elif rule == 'filtered':
        W = np.stack([(A[s] >= np.quantile(A[s][:, IU[0], IU[1]], 2 / 3)).astype(float) for s in range(S)])
        none = W.sum(0, keepdims=True) == 0; W = np.where(none, 1.0, W)
    W = W / W.sum(0, keepdims=True)
    Pp = sum(W[s] * P[k] for s, k in enumerate(ks))
    for n in range(Pp.shape[0]): np.fill_diagonal(Pp[n], .5)
    return Pp


def hodge_cyc_share(A):
    """Cyclic share of an antisymmetric K x K matrix under uniform pair weights (complete graph)."""
    r = A.mean(1); G = r[:, None] - r[None, :]; C = A - G
    tot = (A[IU] ** 2).sum(); return float((C[IU] ** 2).sum() / tot) if tot > 0 else 0.0, C


def condorcet_cycle(P):
    M = P > .5
    for i in range(K):
        for j in range(K):
            for l in range(K):
                if len({i, j, l}) == 3 and M[i, j] and M[j, l] and M[l, i]: return True
    return False


def amp(tr, W=30):
    p = np.array([sm(x) for x in tr[-W:]]); t = np.arange(W) - (W - 1) / 2; sl = (t[:, None] * (p - p.mean(0))).sum(0) / (t ** 2).sum()
    return float((p - p.mean(0) - sl * t[:, None]).std(0).max())


def run(P, s, a, b, T=120):
    r = Pi(s); x = r.copy(); tr = [x]
    for _ in range(T): x = dpo_round(x, r, P, a, b, 1.0, np.ones(K) / K); tr.append(x)
    return tr


RECIPES = {'A_fixedref': (0.0, 10.0), 'B_a05': (0.5, 5.0), 'C_a09': (0.9, 1.0)}
res = {'criteria_T': {}, 'cells': {}, 'per_panel': {}}
for fam, scores in (('style', crit), ('rm_jury', jury)):
    Z, P = labels(scores)
    res['criteria_T'][fam] = {k: float(np.std((s[:, :, None] - s[:, None, :])[:, IU[0], IU[1]])) for k, s in scores.items()}
    # agreement between criteria: mean Kendall tau over panels
    ks = list(scores); tau = {}
    for a_ in range(len(ks)):
        for b_ in range(a_ + 1, len(ks)):
            sa, sb = scores[ks[a_]], scores[ks[b_]]
            t_ = [np.mean(np.sign(sa[n][IU[0]] - sa[n][IU[1]]) * np.sign(sb[n][IU[0]] - sb[n][IU[1]])) for n in range(N)]
            tau[f'{ks[a_]}-{ks[b_]}'] = float(np.mean(t_))
    res['criteria_T'][fam + '_kendall_tau'] = tau
    rules = [f'single:{k}' for k in ks] + ['soft', 'salience', 'filtered']
    for rule in rules:
        Pp = pool(Z, P, rule)
        cyc = np.array([hodge_cyc_share(logit(Pp[n]))[0] for n in range(N)]); cond = np.array([condorcet_cycle(Pp[n]) for n in range(N)])
        cell = dict(cyc_share_logit_mean=float(cyc.mean()), cyc_share_logit_median=float(np.median(cyc)), share_cyc_gt_0p1=float((cyc > .1).mean()),
                    condorcet_cycle_share=float(cond.mean()))
        osc = {}
        for name, (a, b) in RECIPES.items():
            am = np.array([amp(run(Pp[n], s0[n], a, b)) for n in range(N)]); osc[name] = am
            cell[f'osc_share_{name}'] = float((am > THR).mean())
        res['cells'][f'{fam}|{rule}'] = cell
        res['per_panel'][f'{fam}|{rule}'] = dict(cyc=cyc.tolist(), condorcet=cond.tolist(), **{f'amp_{k}': v.tolist() for k, v in osc.items()})
        print(fam, rule, {k: round(v, 3) for k, v in cell.items()}, flush=True)
        if fam == 'style' and rule in ('soft', 'salience', 'filtered'):
            np.savez_compressed(os.path.join(ROOT, 'data', f'oracle_{fam}_{rule}.npz'), ids=ids, P=Pp)
res['ids'] = ids.tolist()
json.dump(res, open(os.path.join(ROOT, 'prereg', 'population_predictions.json'), 'w'))
