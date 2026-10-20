"""Insert the Experiments results into JMLR-B-v4/jmlr_main.tex (mirror) and keep the plan as 'Further experiments (TODO)'.

Usage: python integrate_v4.py <mirror v4 main.tex> <exp_section.tex>
- The generated section (write_section.py) becomes \\section{Experiments} with label sec:experiments.
- A fixed subsection on how common cycles are in practice is appended (jury, SPPO PairRM probe, Oracle1 runs);
  the Oracle1 figure moves there from the plan.
- The old plan section is demoted to \\subsection{Further experiments (TODO)} (label sec:exp-plan), keeping every
  review block verbatim (asserted).
"""
import collections
import pathlib
import sys

sys.path.insert(0, '/Users/fanyao/Desktop/UNC/Research/Iterative_PsiPO/paper/reviews/2026-10-15-concentration-reframe/draft')
from blocks import macro_blocks  # noqa: E402

main_path, sec_path = map(pathlib.Path, sys.argv[1:3])
src = main_path.read_text(encoding='utf-8')
doc = src
new_sec = sec_path.read_text(encoding='utf-8')


def once(a):
    assert doc.count(a) == 1, (doc.count(a), a[:70])


# 1. take the Oracle1 figure out of the plan
fs = doc.index('\\begin{figure}[t]\n    \\centering\n    \\includegraphics[width=0.95\\textwidth]{oracle1_entropy_vs_horizon.pdf}')
fe = doc.index('\\end{figure}', fs) + len('\\end{figure}')
oracle_fig = doc[fs:fe]
doc = doc[:fs] + doc[fe:]
oracle_fig = oracle_fig.replace('\\caption{\\textcolor{todocolor}{Preliminary evidence (G1a.1): re-analysis', '\\caption{{Re-analysis').replace('\\label{fig:oracle1-horizon}', '\\label{fig:oracle1-horizon}')
assert '\\caption{{Re-analysis' in oracle_fig

practice = r"""
\subsection{How often do cycles arise in practice?}
\label{sec:exp-practice}
The planted cycles show that the stability theory predicts nonconvergence when a majority cycle is present; the remaining question is how often natural feedback produces one. On our 130 panels, a jury of three reward models (Skywork-Reward-Llama-3.1-8B-v0.2, Skywork-Reward-V2-Qwen3-1.7B, and GRM-Llama3.2-3B) that labels each pair by its vote share leaves a single panel without a Condorcet winner, although the three models agree on only 68--75\% of pairwise orderings. On 1{,}200 training panels released with SPPO~\citep{wu2025sppo}, five on-policy answers per prompt with their complete PairRM win-probability matrices, no panel contains a majority three-cycle. Under such feedback the theory predicts concentration rather than cycling, and existing multi-round runs behave accordingly: in iterative IPO and DPO with a frozen scalar reward model, panel entropy against the horizon $M_t(\alpha)$ follows a single curve for IPO and collapses for hard-label DPO (\Cref{fig:oracle1-horizon}).

""" + oracle_fig + '\n'

# 2. demote the plan
old_head = '\\section{\\textcolor{todocolor}{Experimental Plan (TODO)}}\n\\label{sec:experiments}'
once(old_head)
doc = doc.replace(old_head, new_sec.rstrip('\n') + '\n' + practice + '\\subsection{\\textcolor{todocolor}{Further experiments (TODO)}}\n\\label{sec:exp-plan}')
once('\\todomark{This section is the experimental plan of JMLR-B-v4 (2026-10-16)')
doc = doc.replace('\\todomark{This section is the experimental plan of JMLR-B-v4 (2026-10-16), written after the reframing around concentration; results will replace it.',
                  '\\todomark{The same-target test (G2.1), the link test (G2.3), the full-refresh test, and the fixed-$\\beta$ and fixed-target stability tests (G2.5) are reported above with a planted-cycle control in place of G1b.2. The items below remain planned.')

cs = collections.Counter(b for _, b in macro_blocks(src)); cd = collections.Counter(b for _, b in macro_blocks(doc))
assert cs == cd, 'review blocks changed'
main_path.write_text(doc, encoding='utf-8')
print('integrated; review blocks', sum(cd.values()))
