"""Robustness subsection for v4: practice-like sampled pairs (0.5B) and the 1.5B replication, numbers from run data.
Usage: python write_robustness.py <results dir> <out.tex>"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from population import entropy, link, sm, target, trajectory  # noqa: E402

RES, OUT = sys.argv[1:3]
klp = lambda p, q: (p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum(-1)


def L(d, r):
    Z = np.load(os.path.join(RES, d, r, 'run.npz'))
    return Z['s'].astype(float), Z['P'], json.load(open(os.path.join(RES, d, r, 'config.json'))), Z['ids']


def pamp(traj, T=30):
    t = np.arange(T) - (T - 1) / 2; x = sm(traj[-T:]); sl = (t[:, None, None] * (x - x.mean(0))).sum(0) / (t ** 2).sum()
    return (x - x.mean(0) - sl[None] * t[:, None, None]).std(0).max(-1)


def same_target(d, r0, r8, lk, B=8.0):
    o = {}
    for r in (r0, r8):
        s, P, cfg, _ = L(d, r); X, _ = target(link(P, lk), s[0], B, cfg['lam'])
        o[r] = float(klp(sm(s[-10:]).mean(0), sm(X)).mean())
    return o[r0], o[r8]


def link_test(d, rh, rs):
    sh, _, _, _ = L(d, rh); ss, P, cfg, _ = L(d, rs)
    X, _ = target(link(P, 'logit'), ss[0], cfg['beta'] / (1 - cfg['alpha']), cfg['lam'])
    return float(entropy(sh[1]).mean()), float(entropy(ss[-1]).mean()), float(entropy(X).mean())


def cycles(d, r2, r9, oracle):
    Zp = np.load(oracle); kb = {int(i): k for i, k in zip(Zp['ids'], Zp['kind'])}
    out = []
    for r in (r2, r9):
        s, P, cfg, ids = L(d, r); kind = np.array([kb[int(i)] for i in ids])
        a = pamp(s); ap = pamp(trajectory(P - 0.5, s[0], cfg['alpha'], cfg['beta'], 1.0, s.shape[0] - 1))
        out.append((float(np.median(a[kind == 'balanced'])), float(np.median(ap[kind == 'balanced']))))
    return out


rows = []
data_dir = os.path.join(os.path.dirname(RES), 'data')
st_s = same_target('runs_sampled', 'S1n512_dpo_a0_b8', 'S1n512_dpo_a08_b16', 'logit')
lk_s = link_test('runs_sampled', 'S2_dpo_hard', 'S2_dpo_bern')
cy_s = cycles('runs_sampled', 'S3n512_B12_a02', 'S3n512_B12_a09', os.path.join(data_dir, 'oracle_planted.npz'))
fl = []
for r in ('S1n512_dpo_a0_b8', 'S1n512_dpo_a08_b16'):
    s, _, _, _ = L('runs_sampled', r); p = sm(s[-10:]); fl.append(float(np.median(klp(p, p.mean(0)[None]).mean(0))))
have4 = os.path.exists(os.path.join(RES, 'runs_q15', 'R3_q15_a09', 'run.npz'))
if have4:
    st_b = same_target('runs_q15', 'R1_q15_a0_b8', 'R1_q15_a08_b16', 'identity')
    lk_b = link_test('runs_q15', 'R2_q15_dpo_hard', 'R2_q15_dpo_soft')
    cy_b = cycles('runs_q15', 'R3_q15_a02', 'R3_q15_a09', os.path.join(data_dir, 'oracle_planted_q15.npz'))
f3 = lambda x: f'{x:.3f}'
cell = lambda v: v if have4 else '--'
T = [r"""
\subsection{Sampled pairs and a larger model}
\label{sec:exp-robust}
The runs above optimize the expected pairwise loss. In practice each round sees finitely many sampled comparisons with random labels, and models are larger. We repeat the three main tests on 64 of the panels with ordered pairs drawn from $\bmu_t\otimes\bmu_t$ and labels drawn from $P^*$ (512 pairs per prompt and round for the same-target and cycle tests, 32 for the link test), and with Qwen2.5-1.5B-Instruct on its own 390 panels (64 used), again with predictions fixed before training (\Cref{tab:exp-robust}). Sampling adds stationary noise but leaves the targets in place: at the same $B=8$, the time-averaged policies of $\alpha=0$ and $\alpha=0.8$ both lie at the target, and their fluctuation around it differs by a factor of """ + f'{fl[0] / fl[1]:.1f}' + r""" in KL, against the factor $(1+\alpha)/(1-\alpha)=9$ that the stationary variance of a linear noise model predicts at a fixed target. Slower refresh at the same target therefore also averages sampling noise. Hard-label DPO still collapses in the first round, and a planted majority cycle still oscillates with the amplitude of the deterministic prediction at $\alpha=0.2$ and settles at $\alpha=0.9$.

\begin{table}[t]
\centering
\caption{Robustness of the main predictions: practice-like sampled pairs (0.5B model) and a larger model (1.5B, expected labels). Same target: KL from the policy averaged over the last 10 rounds to the population target at $B=8$, for $\alpha=0$ and $\alpha=0.8$. Link: panel entropy of hard-label DPO after one round, and final entropy of soft-label DPO against its target, at $\alpha=0.8$. Cycles: median detrended oscillation amplitude of answer probabilities over the last 30 rounds on balanced planted-cycle panels at $B=12$, neural against population.}
\label{tab:exp-robust}
\small
\begin{tabular}{lll}
\toprule
Test & 0.5B, sampled pairs & 1.5B, expected labels\\
\midrule
Same target, $\alpha=0$ / $0.8$ & """ + f"{st_s[0]:.4f} / {st_s[1]:.4f}" + ' & ' + cell(f"{st_b[0]:.4f} / {st_b[1]:.4f}" if have4 else '') + r"""\\
Hard-label DPO entropy after one round & """ + f3(lk_s[0]) + ' & ' + cell(f3(lk_b[0]) if have4 else '') + r"""\\
Soft-label DPO entropy (target) & """ + f"{f3(lk_s[1])} ({f3(lk_s[2])})" + ' & ' + cell(f"{f3(lk_b[1])} ({f3(lk_b[2])})" if have4 else '') + r"""\\
Cycle amplitude, $\alpha=0.2$ (population) & """ + f"{f3(cy_s[0][0])} ({f3(cy_s[0][1])})" + ' & ' + cell(f"{f3(cy_b[0][0])} ({f3(cy_b[0][1])})" if have4 else '') + r"""\\
Cycle amplitude, $\alpha=0.9$ (population) & """ + f"{f3(cy_s[1][0])} ({f3(cy_s[1][1])})" + ' & ' + cell(f"{f3(cy_b[1][0])} ({f3(cy_b[1][1])})" if have4 else '') + r"""\\
\bottomrule
\end{tabular}
\end{table}
"""]
open(OUT, 'w').write(''.join(T))
print('written', OUT, 'have 1.5B:', have4)
