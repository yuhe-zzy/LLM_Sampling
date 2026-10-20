"""X12 evaluation (2026-10-19): pairwise LLM judge (Qwen2.5-7B-Instruct), order-averaged labels, on the v4 panels and the style panels.
Rule (prereg): if >= 10% of the 130 v4 panels lack a Condorcet winner under the order-averaged labels, Part II gets more weight."""
import json, os, sys
import numpy as np
sys.path.insert(0, '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/reviews/2026-10-18-jmlr-review/theory')
from dpo_map import dpo_round, sm, Pi
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
PANELS = {'/home/azureuser/v4val/panels_q05.json': '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/experiments/2026-10-16-v4-validation/data/panels_q05.json',
          '/home/azureuser/mix/style_panels.json': '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/experiments/2026-10-19-mixed-criteria/data/style_panels.json'}
THR = 0.0536; K = 4; IU = np.triu_indices(K, 1); logit = lambda p: np.log(p / (1 - p))
RECIPES = {'A_fixedref': (0.0, 10.0), 'B_a05': (0.5, 5.0), 'C_a09': (0.9, 1.0)}
D = json.load(open(os.path.join(ROOT, 'data', 'llm_judge.json')))


def no_cw(P): return not any(all(P[w, j] > .5 for j in range(K) if j != w) for w in range(K))


def three_cycle(P):
    M = P > .5
    return any(M[i, j] and M[j, l] and M[l, i] for i in range(K) for j in range(K) for l in range(K) if len({i, j, l}) == 3)


def cyc_share(A):
    r = A.mean(1); C = A - (r[:, None] - r[None, :]); tot = (A[IU] ** 2).sum(); return float((C[IU] ** 2).sum() / tot) if tot > 0 else 0.0


def amp(tr, W=30):
    p = np.array([sm(x) for x in tr[-W:]]); t = np.arange(W) - (W - 1) / 2; sl = (t[:, None] * (p - p.mean(0))).sum(0) / (t ** 2).sum()
    return float((p - p.mean(0) - sl * t[:, None]).std(0).max())


def run(P, s, a, b, T=120):
    r = Pi(s); x = r.copy(); tr = [x]
    for _ in range(T): x = dpo_round(x, r, P, a, b, 1.0, np.ones(K) / K); tr.append(x)
    return tr


out = {'judge': D['judge'], 'files': {}}
for f, res in D['files'].items():
    pan = {p['id']: p for p in json.load(open(PANELS[f]))['panels']}
    Q = np.array([r['q'] for r in res]); ids = [r['id'] for r in res]; N = len(ids)
    letter = np.mean([[t in ('A', 'B') for i, row in enumerate(r['top']) for j, t in enumerate(row) if i != j] for r in res])
    bias = np.mean([Q[n][i, j] + Q[n][j, i] - 1 for n in range(N) for i in range(K) for j in range(K) if i < j])   # > 0: slot A favoured
    variants = {}
    Pavg = 0.5 * (Q + 1 - np.transpose(Q, (0, 2, 1)))
    Pfirst = np.where(np.triu(np.ones((K, K)), 1)[None] > 0, Q, 1 - np.transpose(Q, (0, 2, 1)))      # i < j always shown in slot A
    Psecond = np.where(np.triu(np.ones((K, K)), 1)[None] > 0, 1 - np.transpose(Q, (0, 2, 1)), Q)     # i < j always shown in slot B
    for name, PP in (('order_averaged', Pavg), ('lower_index_first', Pfirst), ('lower_index_second', Psecond)):
        PP = np.clip(PP, 1e-4, 1 - 1e-4).copy()
        for n in range(N): np.fill_diagonal(PP[n], .5)
        nc = np.array([no_cw(PP[n]) for n in range(N)]); tc = np.array([three_cycle(PP[n]) for n in range(N)])
        cs = np.array([cyc_share(logit(PP[n])) for n in range(N)])
        cell = dict(n=N, no_condorcet_winner=float(nc.mean()), no_cw_count=int(nc.sum()), majority_three_cycle=float(tc.mean()),
                    cyc_share_logit_mean=float(cs.mean()), cyc_share_gt_0p1=float((cs > .1).mean()), no_cw_ids=[ids[n] for n in np.where(nc)[0]])
        if name == 'order_averaged':
            s0 = np.array([pan[i]['s0'] for i in ids])
            for rn, (a, b) in RECIPES.items():
                am = np.array([amp(run(PP[n], s0[n], a, b)) for n in range(N)]); cell[f'osc_{rn}'] = float((am > THR).mean()); cell[f'osc_{rn}_count'] = int((am > THR).sum())
            # agreement with the plain Skywork-8B ranking
            rm = np.array([pan[i]['rm']['Skywork-Reward-Llama-3.1-8B-v0.2-d4117fbfd81b'] if 'rm' in pan[i] else [0] * K for i in ids])
            sg = np.sign(rm[:, IU[0]] - rm[:, IU[1]]); sj = np.sign(PP[:, IU[0], IU[1]] - .5); cell['sign_agreement_with_skywork8b'] = float((sg * sj).mean())
            cell['mean_abs_margin'] = float(np.abs(PP[:, IU[0], IU[1]] - .5).mean())
        variants[name] = cell
    out['files'][os.path.basename(f)] = dict(letter_top_share=float(letter), position_bias=float(bias), **variants)
    print(os.path.basename(f), 'letters %.3f bias %+.3f' % (letter, bias), {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'no_cw_ids'} for k, v in variants.items()})
main = out['files']['panels_q05.json']['order_averaged']
out['rule'] = dict(threshold=0.10, share=main['no_condorcet_winner'], triggered=bool(main['no_condorcet_winner'] >= 0.10))
print('RULE', out['rule'])
json.dump(out, open(os.path.join(ROOT, 'results', 'x12_evaluation.json'), 'w'), indent=1)
