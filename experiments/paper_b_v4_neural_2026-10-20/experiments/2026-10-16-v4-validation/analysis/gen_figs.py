"""PGFPlots figures for JMLR-B-v4 Section 8 from the neural runs and the exact population recursion.

Every population curve is computed from the run's own initial scores s[0] and configuration.
Usage: python gen_figs.py <results/runs> <panels.json> <outdir>   (writes exp_*.tex; compile with pdflatex)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from population import condorcet_winner, entropy, link, sm, target, trajectory  # noqa: E402

runs_dir, panels_path, outdir = sys.argv[1:4]
os.makedirs(outdir, exist_ok=True)
HEAD = r"""\documentclass[tikz,border=2pt]{standalone}
\usepackage{amsmath}\usepackage{pgfplots}\usepgfplotslibrary{groupplots}\pgfplotsset{compat=1.18}
\definecolor{cA}{HTML}{2166AC}\definecolor{cB}{HTML}{1B9E77}\definecolor{cC}{HTML}{D95F02}\definecolor{cD}{HTML}{C51B7D}\definecolor{cE}{HTML}{7570B3}
\begin{document}\begin{tikzpicture}
"""
TAIL = r"""\end{tikzpicture}\end{document}
"""
AX = (r"tick label style={font=\scriptsize}, label style={font=\small}, title style={font=\small\bfseries}, "
      r"grid=major, grid style={draw=gray!22, line width=0.25pt}, axis line style={black, line width=0.6pt}, "
      r"legend style={font=\scriptsize, draw=none, fill=white, fill opacity=0.85, text opacity=1, cells={anchor=west}, row sep=-2pt}")


def have(n):
    return os.path.exists(os.path.join(runs_dir, n, 'run.npz'))


def load(n):
    Z = np.load(os.path.join(runs_dir, n, 'run.npz'))
    cfg = json.load(open(os.path.join(runs_dir, n, 'config.json')))
    return Z['s'].astype(np.float64), Z['P'], cfg


def pop(n, T=None):
    s, P, cfg = load(n)
    lk = 'identity' if cfg['objective'] == 'ipo' else 'logit'
    T = T or s.shape[0] - 1
    return trajectory(link(P, lk), s[0], cfg['alpha'], cfg['beta'], cfg['lam'], T)


def pts(x, y, every=1):
    return ' '.join(f'({a:.5g},{b:.5g})' for i, (a, b) in enumerate(zip(x, y)) if i % every == 0 or i == len(x) - 1)


# ---------------- Figure E1: target sharpness (H1, H2)
H1 = [('H1_a0_b8', r'$\alpha=0,\ \beta=8$', 'cA'), ('H1_a05_b4', r'$\alpha=0.5,\ \beta=4$', 'cB'),
      ('H1_a08_b16', r'$\alpha=0.8,\ \beta=1.6$', 'cC'), ('H1_a09_b08', r'$\alpha=0.9,\ \beta=0.8$', 'cD'),
      ('H1_a05_b1', r'$\alpha=0.5,\ \beta=1$ ($B=2$)', 'cE')]
H2 = [('H1_a05_b1', 0.5, 'cE'), ('H2_a0.8', 0.8, 'cC'), ('H2_a0.9', 0.9, 'cD'), ('H2_a0.95', 0.95, 'cA')]
body = [r"\begin{groupplot}[group style={group size=2 by 1, horizontal sep=1.5cm}, width=7.0cm, height=5.4cm, " + AX + "]"]
body.append(r"\nextgroupplot[title={(a) Same $B$, same target}, xlabel={outer round $t$}, ylabel={mean panel entropy (nats)}, legend style={at={(0.98,0.62)}, anchor=east}]")
for n, lab, col in H1:
    if not have(n):
        continue
    s, P, cfg = load(n)
    H = entropy(s).mean(-1); Hp = entropy(pop(n)).mean(-1); t = np.arange(len(H))
    body.append(rf"\addplot[{col}, line width=0.9pt, mark=*, mark size=0.9pt, mark repeat=3] coordinates {{{pts(t, H)}}}; \addlegendentry{{{lab}}}")
    body.append(rf"\addplot[{col}, dashed, line width=0.6pt, forget plot] coordinates {{{pts(t, Hp)}}};")
body.append(r"\nextgroupplot[title={(b) Entropy against the horizon}, xlabel={$M_t(\alpha)$}, xmode=log, ylabel={mean panel entropy (nats)}, legend style={at={(0.02,0.02)}, anchor=south west}]")
for n, a, col in H2:
    if not have(n):
        continue
    s, P, cfg = load(n)
    H = entropy(s).mean(-1); Hp = entropy(pop(n)).mean(-1); t = np.arange(len(H)); M = (1 - a ** t) / (1 - a)
    body.append(rf"\addplot[{col}, line width=0.9pt, mark=*, mark size=0.9pt, mark repeat=3] coordinates {{{pts(M[1:], H[1:])}}}; \addlegendentry{{$\alpha={a}$}}")
    body.append(rf"\addplot[{col}, dashed, line width=0.6pt, forget plot] coordinates {{{pts(M[1:], Hp[1:])}}};")
body.append(r"\end{groupplot}")
open(os.path.join(outdir, 'exp_sharpness.tex'), 'w').write(HEAD + '\n'.join(body) + '\n' + TAIL)

# ---------------- Figure E2: link (H4) and full refresh (H3)
body = [r"\begin{groupplot}[group style={group size=2 by 1, horizontal sep=1.5cm}, width=7.0cm, height=5.4cm, " + AX + "]"]
body.append(r"\nextgroupplot[title={(a) Link and labels, $\alpha=0.8$}, xlabel={outer round $t$}, ylabel={mean panel entropy (nats)}, ymin=0, legend style={at={(0.98,0.98)}, anchor=north east}]")
for n, lab, col in [('H4_dpo_soft', 'DPO, soft labels', 'cA'), ('H4_ipo_hard', 'IPO, hard labels', 'cB'), ('H4_dpo_hard', 'DPO, hard labels', 'cD')]:
    if not have(n):
        continue
    s, P, cfg = load(n)
    H = entropy(s).mean(-1); t = np.arange(len(H))
    body.append(rf"\addplot[{col}, line width=0.9pt, mark=*, mark size=0.9pt, mark repeat=3] coordinates {{{pts(t, H)}}}; \addlegendentry{{{lab}}}")
    if n != 'H4_dpo_hard':
        Hp = entropy(pop(n)).mean(-1)
        body.append(rf"\addplot[{col}, dashed, line width=0.6pt, forget plot] coordinates {{{pts(t, Hp)}}};")
body.append(r"\nextgroupplot[title={(b) Full refresh, hard labels}, xlabel={outer round $t$}, ylabel={mean mass on the Condorcet winner}, ymin=0, ymax=1.02, legend style={at={(0.98,0.02)}, anchor=south east}]")
if have('H3_a1_b1'):
    s, P, cfg = load('H3_a1_b1')
    cw = np.array([condorcet_winner(P[n]) for n in range(P.shape[0])])
    w = sm(s)[:, np.arange(s.shape[1]), cw].mean(-1); wp = sm(pop('H3_a1_b1'))[:, np.arange(s.shape[1]), cw].mean(-1); t = np.arange(len(w))
    body.append(rf"\addplot[cA, line width=0.9pt, mark=*, mark size=0.9pt, mark repeat=3] coordinates {{{pts(t, w)}}}; \addlegendentry{{neural}}")
    body.append(rf"\addplot[cA, dashed, line width=0.6pt] coordinates {{{pts(t, wp)}}}; \addlegendentry{{population}}")
body.append(r"\end{groupplot}")
open(os.path.join(outdir, 'exp_link_fullrefresh.tex'), 'w').write(HEAD + '\n'.join(body) + '\n' + TAIL)

# ---------------- Figure E3: cycles (H5, H6)
CFG = [('H5_b4_a05', r'$\beta{=}4$, $\alpha{=}0.5$'), ('H5_b4_a09', r'$\beta{=}4$, $\alpha{=}0.9$'), ('H5_b4_a095', r'$\beta{=}4$, $\alpha{=}0.95$'),
       ('H6_B12_a02', r'$B{=}12$, $\alpha{=}0.2$'), ('H6_B12_a05', r'$B{=}12$, $\alpha{=}0.5$'), ('H6_B12_a08', r'$B{=}12$, $\alpha{=}0.8$'),
       ('H6_B12_a09', r'$B{=}12$, $\alpha{=}0.9$')]
pred_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prereg', 'predictions_batch2_amplitude.json')
pred = json.load(open(pred_path)) if os.path.exists(pred_path) else None
Zp = np.load(os.path.join(os.path.dirname(panels_path), 'oracle_planted.npz'))
kind_by_id = {int(i): k for i, k in zip(Zp['ids'], Zp['kind'])}
obs = {}
for n, _ in CFG:
    if not have(n):
        continue
    s, P, cfg = load(n)
    ids = np.load(os.path.join(runs_dir, n, 'run.npz'))['ids']
    kind = np.array([kind_by_id[int(i)] for i in ids])
    def pamp(traj, T_=30):
        tt = np.arange(T_) - (T_ - 1) / 2; x = sm(traj[-T_:]); slope = (tt[:, None, None] * (x - x.mean(0))).sum(0) / (tt ** 2).sum()
        return (x - x.mean(0) - slope[None] * tt[:, None, None]).std(0).max(-1)
    amp = pamp(s); amp_pop = pamp(pop(n))
    obs[n] = {k: float((amp[kind == k] > 0.0536).mean()) for k in ('balanced', 'decisive')}
    obs[n].update({'pop_' + k: float((amp_pop[kind == k] > 0.0536).mean()) for k in ('balanced', 'decisive')})
body = [r"\begin{groupplot}[group style={group size=2 by 1, horizontal sep=1.6cm}, width=7.0cm, height=5.4cm, " + AX + "]"]
xt = ','.join(str(i) for i in range(len(CFG)))
xl = ','.join('{' + lab + '}' for _, lab in CFG)
body.append(r"\nextgroupplot[title={(a) Persistent oscillation}, ylabel={share of panels}, ymin=-0.03, ymax=1.05, xtick={" + xt + r"}, xticklabels={" + xl +
            r"}, x tick label style={rotate=40, anchor=east, font=\tiny}, legend style={at={(0.98,0.55)}, anchor=east}]")
for k, col, mk in (('balanced', 'cD', '*'), ('decisive', 'cA', 'square*')):
    xs = [i for i, (n, _) in enumerate(CFG) if n in obs]
    c_obs = ' '.join('({},{:.3f})'.format(i, obs[CFG[i][0]][k]) for i in xs)
    body.append(rf"\addplot[only marks, {col}, mark={mk}, mark size=2.2pt] coordinates {{{c_obs}}}; \addlegendentry{{{k}, neural}}")
    if obs:
        c_pred = ' '.join('({},{:.3f})'.format(i, obs[CFG[i][0]]['pop_' + k]) for i in xs)
        body.append(rf"\addplot[only marks, {col}, mark=o, mark size=3.4pt, thick] coordinates {{{c_pred}}}; \addlegendentry{{{k}, population}}")
body.append(r"\nextgroupplot[title={(b) One balanced panel at $B=12$}, xlabel={outer round $t$}, ylabel={probability}, ymin=0, ymax=1, legend style={at={(0.98,0.98)}, anchor=north east}]")
if have('H6_B12_a02') and have('H6_B12_a09'):
    s2, P2, _ = load('H6_B12_a02'); s9, _, _ = load('H6_B12_a09')
    ids = np.load(os.path.join(runs_dir, 'H6_B12_a02', 'run.npz'))['ids']
    nb = [i for i, x in enumerate(ids) if kind_by_id[int(x)] == 'balanced'][0]
    t = np.arange(s2.shape[0])
    p2, p9 = sm(s2[:, nb]), sm(s9[:, nb])
    j = int(np.argmax(p2[0]))
    body.append(rf"\addplot[cD, line width=0.9pt] coordinates {{{pts(t, p2[:, j])}}}; \addlegendentry{{$\alpha=0.2$ (predicted unstable)}}")
    body.append(rf"\addplot[cA, line width=0.9pt] coordinates {{{pts(t, p9[:, j])}}}; \addlegendentry{{$\alpha=0.9$ (predicted stable)}}")
body.append(r"\end{groupplot}")
open(os.path.join(outdir, 'exp_cycles.tex'), 'w').write(HEAD + '\n'.join(body) + '\n' + TAIL)
print('figures written to', outdir, '; batch-2 runs found:', sorted(obs))
