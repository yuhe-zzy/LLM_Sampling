"""X11 stage 1 predictions (2026-10-19): sources = rubric contexts of one judge (reward model) on the 130 v4 panels.
Same label rule, pooling rules, recipes and oscillation rule as predict_x11.py (stage 0); see prereg/PREREG.md.
Usage: python stage1_predict.py <rubric_scores.json>"""
import json, os, sys
import numpy as np
sys.path.insert(0, '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/reviews/2026-10-18-jmlr-review/theory')
from dpo_map import dpo_round, sm, Pi
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PAN = '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/experiments/2026-10-16-v4-validation/data/panels_q05.json'
MAIN = 'Skywork-Reward-Llama-3.1-8B-v0.2-d4117fbfd81b'
THR = 0.0536; K = 4; IU = np.triu_indices(K, 1); CTX = ['rigor', 'creativity', 'concision']
sig = lambda z: 1 / (1 + np.exp(-z)); logit = lambda p: np.log(p / (1 - p))
panels = json.load(open(PAN))['panels']; S0 = np.array([p['s0'] for p in panels]); N = len(panels)
RS = json.load(open(sys.argv[1])); assert RS['ids'] == [p['id'] for p in panels]


def labels(scores):
    Z, P = {}, {}
    for k, s in scores.items():
        g = s[:, :, None] - s[:, None, :]; T = float(np.std(g[:, IU[0], IU[1]])); Z[k] = g / T; P[k] = sig(Z[k])
    return Z, P


def pool(Z, P, rule):
    ks = list(P); S = len(ks)
    if rule.startswith('single:'): return P[rule[7:]]
    if rule == 'soft': return np.mean([P[k] for k in ks], 0)
    A = np.abs(np.stack([Z[k] for k in ks]))
    if rule == 'salience':
        W = (A == A.max(0, keepdims=True)).astype(float)
    elif rule == 'filtered':
        W = np.stack([(A[s] >= np.quantile(A[s][:, IU[0], IU[1]], 2 / 3)).astype(float) for s in range(S)])
        W = np.where(W.sum(0, keepdims=True) == 0, 1.0, W)
    W = W / W.sum(0, keepdims=True); Pp = sum(W[s] * P[k] for s, k in enumerate(ks))
    for n in range(Pp.shape[0]): np.fill_diagonal(Pp[n], .5)
    return Pp


def cyc_share(A):
    r = A.mean(1); C = A - (r[:, None] - r[None, :]); tot = (A[IU] ** 2).sum()
    return float((C[IU] ** 2).sum() / tot) if tot > 0 else 0.0


def no_cw(P): return not any(all(P[w, j] > .5 for j in range(K) if j != w) for w in range(K))


def amp(tr, W=30):
    p = np.array([sm(x) for x in tr[-W:]]); t = np.arange(W) - (W - 1) / 2; sl = (t[:, None] * (p - p.mean(0))).sum(0) / (t ** 2).sum()
    return float((p - p.mean(0) - sl * t[:, None]).std(0).max())


def run(P, s, a, b, T=120):
    r = Pi(s); x = r.copy(); tr = [x]
    for _ in range(T): x = dpo_round(x, r, P, a, b, 1.0, np.ones(K) / K); tr.append(x)
    return tr


RECIPES = {'A_fixedref': (0.0, 10.0), 'B_a05': (0.5, 5.0), 'C_a09': (0.9, 1.0)}
out = dict(cells={}, per_panel={}, kendall={}, ids=RS['ids'], plain_check={})
for judge, sc in RS['scores'].items():
    scores = {c: np.array(sc[c]) for c in CTX}
    plain = np.array(sc['plain']); ref = np.array([p['rm'][judge] for p in panels])
    out['plain_check'][judge] = float(np.abs(plain - ref).max())
    # how much each context moves the ranking relative to the plain prompt, and agreement between contexts
    sgn = lambda s: np.sign(s[:, IU[0]] - s[:, IU[1]])
    out['kendall'][judge] = {f'{a}-{b}': float(np.mean(sgn(scores[a]) * sgn(scores[b]))) for i, a in enumerate(CTX) for b in CTX[i + 1:]}
    out['kendall'][judge].update({f'plain-{a}': float(np.mean(sgn(plain) * sgn(scores[a]))) for a in CTX})
    Z, P = labels(scores)
    for rule in [f'single:{c}' for c in CTX] + ['soft', 'salience', 'filtered']:
        Pp = pool(Z, P, rule)
        cs = np.array([cyc_share(logit(Pp[n])) for n in range(N)]); nc = np.array([no_cw(Pp[n]) for n in range(N)])
        cell = dict(cyc_share_mean=float(cs.mean()), share_cyc_gt_0p1=float((cs > .1).mean()), no_condorcet_winner=float(nc.mean()))
        pp = dict(cyc=cs.tolist(), no_cw=nc.tolist())
        if judge == MAIN or rule in ('soft', 'salience', 'filtered'):
            for name, (a, b) in RECIPES.items():
                am = np.array([amp(run(Pp[n], S0[n], a, b)) for n in range(N)])
                cell[f'osc_{name}'] = float((am > THR).mean()); pp[f'amp_{name}'] = am.tolist()
        out['cells'][f'{judge}|{rule}'] = cell; out['per_panel'][f'{judge}|{rule}'] = pp
        print(judge[:22], rule, {k: round(v, 3) for k, v in cell.items()}, flush=True)
        if judge == MAIN and rule in ('soft', 'salience', 'filtered'):
            (lambda *a, **k: None)(os.path.join(ROOT, 'data', f'oracle_rubric_{rule}.npz'), ids=np.array(RS['ids']), P=Pp)
print('plain-score reproduction (max abs diff vs panels file):', out['plain_check'])
print('kendall', json.dumps(out['kendall'], indent=0))
json.dump(out, open(os.path.join(ROOT, 'results', 'recheck_stage1_predictions.json'), 'w'))
