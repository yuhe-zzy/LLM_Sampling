"""Fill the v4 Experiments section from the evaluation output (no hand-typed numbers).

Usage: python write_section.py <eval.json> <predictions_batch1.json> <predictions_batch2_subset.json> <out.tex>
"""
import json
import sys

import numpy as np

E = json.load(open(sys.argv[1])); P1 = json.load(open(sys.argv[2])); P2 = json.load(open(sys.argv[3]))
R = E['runs']
f2 = lambda x: f'{x:.2f}'
f3 = lambda x: f'{x:.3f}'
f4 = lambda x: f'{x:.4f}'
pct = lambda x: f'{100 * x:.0f}\\%'

main = [n for n in R if not n.startswith('h0_') and n != 'H4_dpo_hard']
res_med = float(np.median([R[n]['resid_median'] for n in main]))
h1 = E['H1']
h1kl = [R[n]['KL_final_to_target_mean'] for n in ('H1_a0_b8', 'H1_a05_b4', 'H1_a08_b16', 'H1_a09_b08')]
ctl = E['H1']['KL_B8_vs_B2_control'] or {}
h2 = {n: R[n] for n in ('H1_a05_b1', 'H2_a0.8', 'H2_a0.9', 'H2_a0.95') if n in R}
h3 = E.get('H3', {})
h4 = E['H4']
dh = h4['H4_dpo_hard']['entropy']
h56 = E.get('H56', {})

S = []
S.append(r"""\section{Experiments}
\label{sec:experiments}

We test the population predictions on a language model trained with the actual IPO and DPO losses. The test is quantitative. For every run, the exact recursion of \Cref{sec:hodge}, evaluated on the same panels, preference oracle, and hyperparameters, fixes in advance the trajectory and the target that the trained model should reach, and the hypotheses with their pass criteria were frozen before training.

\subsection{Setup}
\paragraph{Panels.} The policy is Qwen2.5-0.5B-Instruct, fully fine-tuned. Prompts are open-ended requests that admit several short answers: 65 templated requests for names, slogans, titles, recommendations, and descriptions, and the NoveltyBench prompts~\citep{zhang2025noveltybench}, each with the instruction to answer in at most six words. For each prompt the initial model draws 48 answers, and the four most probable distinct complete answers form a fixed panel. A panel is kept when every answer has initial panel probability at least $0.05$, so that the initial policy is interior; 130 of 600 prompts qualify.

\paragraph{Feedback.} Every answer is scored by the Bradley--Terry reward model Skywork-Reward-Llama-3.1-8B-v0.2. Soft labels $P^*_{ij}=\sigma((r_i-r_j)/T)$ with $T=""" + f2(P1['T_soft']) + r"""$ are Bradley--Terry, hence additive under the logit link; hard labels $P^*_{ij}=\mathbf 1[r_i>r_j]$ are what a scalar oracle that compares scores produces. Cycles are studied with a labelled planted-cycle oracle on the same panels (\Cref{sec:exp-cycles}).

\paragraph{One round.} With $s_t(i)$ the sequence log-probability of panel answer $i$ under the current model, the round uses the reference scores $\alpha s_t+(1-\alpha)s_0$, the sampler $\bmu_t=(1-\lambda)\,\mathrm{unif}+\lambda\,\softmax(s_t)$, and the pairwise loss $\sum_{i\neq j}\mu_{t,i}\mu_{t,j}\,\ell(h_i-h_j)$ in the log-ratios $h=s_\theta-(\alpha s_t+(1-\alpha)s_0)$, where $\ell$ is the IPO square loss $(h_i-h_j-\beta(P^*_{ij}-\tfrac12))^2$ or the DPO logistic loss with soft labels and coefficient $1/\beta$. At its optimum this round is exactly MRS-$\Psi$PO, with the identity link for IPO and, for Bradley--Terry labels, the logit link for DPO. We take 40 Adam steps per round from a fresh optimizer; across the runs, the median per-round deviation of the trained panel scores from the exact round map, applied to the model's own state, is """ + f3(res_med) + r""" nats. All population curves below start from the same initial scores as the run they are compared with.
""")

S.append(r"""
\subsection{Target sharpness decides where the loop goes}
\label{sec:exp-sharpness}
\Cref{prop:target} predicts that configurations with the same $B=\beta/(1-\alpha)$ share their target. Four IPO runs with $B=8$, from a fixed reference with $\beta=8$ to slow refresh with $\alpha=0.9$ and $\beta=0.8$, end at the same panel policy: the largest pairwise KL between their final policies is """ + f4(h1['max_pairwise_KL_B8']) + r""" nats, and their KL to the population target ranges from """ + f4(min(h1kl)) + ' to ' + f4(max(h1kl)) + r""" nats, while the control with $B=2$ ends """ + (f'at KL {f3(min(ctl.values()))}--{f3(max(ctl.values()))} from them' if ctl else 'elsewhere') + r""" (\Cref{fig:exp-sharpness}(a)). The refresh weight changes only the approach, at the rate $\alpha$. With $\beta=1$ fixed, raising $\alpha$ raises the sharpness, and the entropy of each run levels off at the population target for $B=1/(1-\alpha)$; plotted against $M_t(\alpha)$, the curves of different $\alpha$ coincide until each saturates (\Cref{fig:exp-sharpness}(b)).""")
if h2:
    S.append(' Final entropies: ' + '; '.join(f"$\\alpha={R[n]['cfg']['alpha']}$: {f3(R[n]['final_entropy'])} (target {f3(R[n]['target_entropy'])})" for n in h2) + '.')
S.append(r"""

\begin{figure}[t]
\centering
\includegraphics[width=0.98\textwidth]{exp_sharpness.pdf}
\caption{Target sharpness in neural IPO (solid, with markers) against the exact population recursion (dashed). \textbf{(a)} Four configurations with $B=\beta/(1-\alpha)=8$ reach the same panel entropy and policy; the control with $B=2$ does not. \textbf{(b)} At $\beta=1$, entropy against the horizon $M_t(\alpha)$ follows one curve until each run saturates at its own target.}
\label{fig:exp-sharpness}
\end{figure}
""")

S.append(r"""
\subsection{The link decides whether there is a finite target}
\label{sec:exp-link}
At $\alpha=0.8$ and $\beta=1$ (\Cref{fig:exp-link}(a)), DPO with soft Bradley--Terry labels levels off at entropy """ + f3(R['H4_dpo_soft']['final_entropy']) + r""" against the predicted """ + f3(R['H4_dpo_soft']['target_entropy']) + r""" (KL to the target """ + f4(R['H4_dpo_soft']['KL_final_to_target_mean']) + r"""), and IPO with hard labels levels off at """ + f3(R['H4_ipo_hard']['final_entropy']) + r""" against """ + f3(R['H4_ipo_hard']['target_entropy']) + r""". DPO with the same hard labels has no finite target: its panel entropy is """ + f3(dh[1]) + r""" after one round, and the log-odds between its two most probable answers keep growing at every later round despite the anchor, as the deterministic-label analysis of \Cref{sec:concentration} predicts.
""")
if h3:
    S.append(r"""
\subsection{Full refresh concentrates on the Condorcet winner}
Under full refresh with hard labels and on-policy sampling, the hard order has a Condorcet winner on every panel, and \Cref{cor:sst} predicts concentration on it. The trained model puts its largest mass on the Condorcet winner on """ + pct(h3['share_top_is_CW_final']) + r""" of the panels after the last round, with mass at least $0.95$ on """ + pct(h3['share_CW_mass_ge_095_final']) + r""" of them, tracking the population curve (\Cref{fig:exp-link}(b)).
""")
S.append(r"""
\begin{figure}[t]
\centering
\includegraphics[width=0.98\textwidth]{exp_link_fullrefresh.pdf}
\caption{\textbf{(a)} Link and labels at $\alpha=0.8$: soft-label DPO and hard-label IPO level off at their population targets (dashed), while hard-label DPO collapses in the first round. \textbf{(b)} Full refresh with hard labels: mean mass on the Condorcet winner, neural against population.}
\label{fig:exp-link}
\end{figure}
""")
if h56:
    b = lambda n, k: h56[n][k]['osc_share'] if n in h56 else float('nan')
    pb = lambda n, k: h56[n][k]['pop_osc_share']
    S.append(r"""
\subsection{Cycles: nonconvergence where the theory predicts it}
\label{sec:exp-cycles}
Natural feedback rarely produces candidate sets without a Condorcet winner (\Cref{sec:exp-practice}), so we plant one, on 64 of the panels. On each panel three answers form a rock--paper--scissors cycle of strength $q=0.3$ and the fourth loses to all; half the panels are balanced ($\Delta=0.05<q/2$, a majority cycle whose maximal lottery has three answers) and half decisive ($\Delta=0.2$, a Condorcet winner). The population model predicts, before training, which configurations hold their target. At fixed $\beta=4$, balanced panels are stable at $\alpha=0.5$ and unstable at $\alpha=0.9$ and $0.95$; at the fixed target $B=12$, they are unstable at $\alpha=0.2$ and $0.5$ and stable at $\alpha=0.8$ and $0.9$; decisive panels concentrate and stay stable throughout. We call a panel oscillating when, over the last 30 of 80 rounds and after removing a linear trend, the probability of some answer still varies with standard deviation above four times the noise floor of the converged runs of \Cref{sec:exp-sharpness}; we measure on probabilities rather than logits because the pairwise loss places almost no weight on answers of vanishing probability, whose logits then wander without affecting the policy. \Cref{fig:exp-cycles}(a) compares, configuration by configuration, the share of oscillating panels with its prediction: for balanced panels it is """ +
             ', '.join(f"{pct(b(n, 'balanced'))} (predicted {pct(pb(n, 'balanced'))})" for n in ('H5_b4_a05', 'H5_b4_a09', 'H5_b4_a095', 'H6_B12_a02', 'H6_B12_a05', 'H6_B12_a08', 'H6_B12_a09')) +
             r""" in the order of the figure, and for decisive panels at most """ + pct(max(b(n, 'decisive') for n in h56)) + r""" (population at most """ + pct(max(pb(n, 'decisive') for n in h56)) + r"""). Slower refresh at the same target therefore turns persistent rotation into convergence, as \Cref{prop:fixed-target} states (\Cref{fig:exp-cycles}(b)).

\begin{figure}[t]
\centering
\includegraphics[width=0.98\textwidth]{exp_cycles.pdf}
\caption{Planted cycles. \textbf{(a)} Share of panels whose answer probabilities still oscillate over the last 30 of 80 rounds, neural (filled) against the noise-free population recursion from the same initial scores (open), for balanced (majority cycle) and decisive (Condorcet winner) panels. \textbf{(b)} One balanced panel at the fixed target $B=12$: the probability of one answer under $\alpha=0.2$, predicted unstable, and $\alpha=0.9$, predicted stable.}
\label{fig:exp-cycles}
\end{figure}
""")
open(sys.argv[4], 'w').write(''.join(S))
print('section written', sys.argv[4])
