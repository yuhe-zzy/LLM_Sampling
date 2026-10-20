"""X12 diagnosis (post hoc, 2026-10-19): why does population DPO with a fixed reference (alpha 0, beta 10) oscillate on many panels
that have a Condorcet winner? Per panel: Condorcet winner and its smallest margin, the DPO fixed point at B = 10 (Newton), its top mass,
the local modulus there, and the oscillation amplitude over the last 30 of 300 rounds."""
import json, os, sys
import numpy as np
from scipy.optimize import fsolve
sys.path.insert(0, '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/reviews/2026-10-18-jmlr-review/theory')
from dpo_map import dpo_round, bt_fit, dphi_formula, sm, Pi
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
K = 4; Q4 = np.linalg.qr(np.vstack([np.ones(K), np.eye(K)[:K - 1]]).T)[0][:, 1:]
D = json.load(open(os.path.join(ROOT, 'data', 'llm_judge.json')))
res = D['files']['/home/azureuser/v4val/panels_q05.json']
pan = {p['id']: p for p in json.load(open('/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/experiments/2026-10-16-v4-validation/data/panels_q05.json'))['panels']}


def fixed(P, r, B):
    F = lambda z: Q4.T @ (Pi(r + B * bt_fit(P, sm(Q4 @ z))) - Q4 @ z)
    best = None
    for z0 in [Q4.T @ r] + [np.random.default_rng(k).normal(0, 2, K - 1) for k in range(6)]:
        z, info, ier, _ = fsolve(F, z0, full_output=True, xtol=1e-12)
        if ier == 1 and np.abs(F(z)).max() < 1e-8:
            x = Q4 @ z
            if best is None: best = x
    return best


def modulus(P, x, a, b):
    pi = sm(x); J = np.diag(pi) - np.outer(pi, pi); M = a * np.eye(K) + b * dphi_formula(P, pi) @ J
    return float(np.abs(np.linalg.eigvals(Q4.T @ M @ Q4)).max()), np.linalg.eigvals(Q4.T @ M @ Q4)


rows = []
for r_ in res:
    q = np.array(r_['q']); P = np.clip(0.5 * (q + 1 - q.T), 1e-4, 1 - 1e-4); np.fill_diagonal(P, .5)
    s0 = np.array(pan[r_['id']]['s0']); r = Pi(s0)
    cw = [w for w in range(K) if all(P[w, j] > .5 for j in range(K) if j != w)]
    marg = float(min(P[cw[0], j] - .5 for j in range(K) if j != cw[0])) if cw else None
    x = fixed(P, r, 10.0)
    if x is None: rows.append(dict(id=r_['id'], cw=bool(cw), solved=False)); continue
    mod, ev = modulus(P, x, 0.0, 10.0); pi = sm(x)
    xx = r.copy(); tr = []
    for t in range(300): xx = dpo_round(xx, r, P, 0.0, 10.0, 1.0, np.ones(K) / K); tr.append(sm(xx))
    tr = np.array(tr[-30:]); amp = float(tr.std(0).max())
    rows.append(dict(id=r_['id'], cw=bool(cw), cw_margin=marg, cw_is_top=bool(cw and int(np.argmax(pi)) == cw[0]), top_mass=float(pi.max()),
                     modulus=mod, real_part_max=float(np.abs(ev.real).max()), imag_max=float(np.abs(ev.imag).max()), amp300=amp))
R = [r for r in rows if r.get('solved', True)]
osc = [r for r in R if r['amp300'] > 0.0536]
print('solved', len(R), 'of', len(rows), '; oscillating at 300 rounds', len(osc))
print('oscillating with a CW:', sum(r['cw'] for r in osc), '; modulus>1 among oscillating:', sum(r['modulus'] > 1 for r in osc), '; modulus>1 overall:', sum(r['modulus'] > 1 for r in R))
for key in ('cw_margin', 'top_mass', 'modulus', 'real_part_max', 'imag_max'):
    a = [r[key] for r in osc if r.get(key) is not None]; b = [r[key] for r in R if r not in osc and r.get(key) is not None]
    print(key, 'osc median %.3f' % np.median(a) if a else '', '| settle median %.3f' % np.median(b) if b else '')
json.dump(rows, open(os.path.join(ROOT, 'results', 'x12_diagnosis.json'), 'w'), indent=1)
