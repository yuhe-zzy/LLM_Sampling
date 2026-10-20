"""Post-hoc evaluation (2026-10-17) of three preregistered sub-predictions not scored during the runs:
H1 approach rate ~ alpha per round; H2 sandwich bounds on the log odds; H3 dominance round of cor:dominance.
Usage: python check_subpredictions.py  (run from the experiment folder); writes results/eval_subpredictions.json"""
import json
import sys

import numpy as np

sys.path.insert(0, 'analysis')
from population import condorcet_winner, link, margins, target, trajectory  # noqa: E402


def L(r):
    Z = np.load(f'results/runs/{r}/run.npz')
    return Z['s'].astype(float), Z['P'], json.load(open(f'results/runs/{r}/config.json'))


cen = lambda x: x - x.mean(-1, keepdims=True)
out = {'H1_rate': {}, 'H2_sandwich': {}, 'H3_cor8': {}}
for r in ['H1_a05_b4', 'H1_a08_b16', 'H1_a09_b08']:
    s, P, c = L(r); A = link(P, 'identity'); X, _ = target(A, s[0], c['beta'] / (1 - c['alpha']), c['lam'])
    pop = trajectory(A, s[0], c['alpha'], c['beta'], c['lam'], s.shape[0] - 1)
    row = {}
    for name, tr in (('neural', s), ('population', pop)):
        md = np.median(np.linalg.norm(cen(tr) - cen(X)[None], axis=-1), axis=1)
        ok = [t for t in range(1, len(md)) if md[t] > 0.5]          # well above the neural noise floor (~0.15-0.2)
        row[name] = {'rate': float(np.exp(np.mean(np.log([md[t] / md[t - 1] for t in ok])))), 'rounds': len(ok), 'floor_final': float(md[-1])}
    out['H1_rate'][r] = {'alpha': c['alpha'], **row}
for r in ['H1_a05_b1', 'H2_a0.8', 'H2_a0.9', 'H2_a0.95']:
    s, P, c = L(r); A = link(P, 'identity'); B = c['beta'] / (1 - c['alpha']); X, _ = target(A, s[0], B, c['lam'])
    inX = inF = tot = 0; viol = []; dev = []; width = []
    for n in range(P.shape[0]):
        G, Gb = margins(A[n], c['lam'])
        for w in range(4):
            for j in range(4):
                if w == j:
                    continue
                lo = s[0][n, w] - s[0][n, j] + B * G[w, j]; hi = s[0][n, w] - s[0][n, j] + B * Gb[w, j]
                f = s[-1][n, w] - s[-1][n, j]; x = X[n, w] - X[n, j]
                tot += 1; inX += lo - 1e-9 <= x <= hi + 1e-9; inF += lo - 1e-9 <= f <= hi + 1e-9
                viol.append(max(lo - f, f - hi, 0.0)); dev.append(abs(f - x)); width.append(hi - lo)
    out['H2_sandwich'][r] = {'B': B, 'pairs': tot, 'target_inside': int(inX), 'neural_inside': int(inF), 'viol_q99': float(np.quantile(viol, 0.99)),
                             'dev_from_target_median': float(np.median(dev)), 'dev_q99': float(np.quantile(dev, 0.99)), 'width_median': float(np.median(width))}
s, P, c = L('H3_a1_b1'); A = link(P, 'identity')
pos = sum(min(margins(A[n], 1.0)[0][condorcet_winner(P[n]), j] for j in range(4) if j != condorcet_winner(P[n])) > 0 for n in range(P.shape[0]))
out['H3_cor8'] = {'panels': int(P.shape[0]), 'panels_all_worst_case_margins_positive': int(pos)}
json.dump(out, open('results/eval_subpredictions.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
