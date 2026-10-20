"""Post hoc recheck of X12's population predictions with the robust DPO solver (robust_dpo.py). Same panels, labels, recipes and
oscillation rule as evaluate_x12.py; also reports the original solver's residual on the moment equations along its trajectories."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/reviews/2026-10-18-jmlr-review/theory')
from robust_dpo import dpo_round_robust, bt_fit_robust, sm, Pi
from dpo_map import dpo_round, bt_fit
K = 4; THR = 0.0536
RECIPES = {'A_fixedref': (0.0, 10.0), 'B_a05': (0.5, 5.0), 'C_a09': (0.9, 1.0)}
D = json.load(open(os.path.join(ROOT, 'data', 'llm_judge.json')))
PAN = {'/home/azureuser/v4val/panels_q05.json': '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/experiments/2026-10-16-v4-validation/data/panels_q05.json',
       '/home/azureuser/mix/style_panels.json': '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/experiments/2026-10-19-mixed-criteria/data/style_panels.json'}


def amp(tr, W=30):
    p = np.array([sm(x) for x in tr[-W:]]); t = np.arange(W) - (W - 1) / 2; sl = (t[:, None] * (p - p.mean(0))).sum(0) / (t ** 2).sum()
    return float((p - p.mean(0) - sl * t[:, None]).std(0).max())


def resid(P, mu, phi):
    S = 1 / (1 + np.exp(-(phi[:, None] - phi[None, :]))); np.fill_diagonal(S, .5)
    return float(np.abs((mu[None, :] * (S - P)).sum(1)).max())


out = {}
for f, res in D['files'].items():
    pan = {p['id']: p for p in json.load(open(PAN[f]))['panels']}; cell = {}; worst_orig = 0.0; worst_rob = 0.0
    for rn, (a, b) in RECIPES.items():
        n_osc = 0
        for r_ in res:
            q = np.array(r_['q']); P = np.clip(0.5 * (q + 1 - q.T), 1e-4, 1 - 1e-4); np.fill_diagonal(P, .5)
            r = Pi(np.array(pan[r_['id']]['s0'])); x = r.copy(); tr = [x]
            for t in range(120):
                mu = sm(x)
                if rn == 'A_fixedref' and t % 20 == 0:
                    worst_orig = max(worst_orig, resid(P, mu, bt_fit(P, mu))); worst_rob = max(worst_rob, resid(P, mu, bt_fit_robust(P, mu)))
                x = dpo_round_robust(x, r, P, a, b, 1.0, np.ones(K) / K); tr.append(x)
            n_osc += amp(tr) > THR
        cell[f'osc_{rn}'] = n_osc / len(res); cell[f'osc_{rn}_count'] = int(n_osc)
    cell['max_moment_residual_original_solver'] = worst_orig; cell['max_moment_residual_robust_solver'] = worst_rob
    out[os.path.basename(f)] = cell; print(os.path.basename(f), cell, flush=True)
json.dump(out, open(os.path.join(ROOT, 'results', 'x12_recheck_robust.json'), 'w'), indent=1)
