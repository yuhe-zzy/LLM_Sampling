# Hodge diagnostics for Paper B (JMLR-B-v3)

GPU-free diagnostic layer of the v3 experimental roadmap (Section 7, Part 0, item 2,
and the exact-population checks of A1). It computes, on real preference panels, the
quantities the theorems speak about, so that the paper's modeling assumptions and the
neural experiment's parameters can be justified from data rather than asserted.

Nothing here is a proof, and nothing here is an observed training result. Numbers are
diagnostics of preference data under stated constructions.

Start with [`HANDOFF.md`](HANDOFF.md) (state, findings, open decisions, next actions), then
the report in [`reports/2026-09-25/REPORT.md`](reports/2026-09-25/REPORT.md).

## What each diagnostic supports

| Paper claim or assumption | Diagnostic | Where |
|---|---|---|
| A cyclic residual arises from **link mismatch** even for a transitive Bradley--Terry population (`prop:cyclic-sources`, Section 3 example) | `single_score_bt`: `cyclic_share` is 0 under logit and positive under identity | `summary.csv` |
| A cyclic residual arises generically from **heterogeneous annotators** under the logit link | `attribute_bt`, `attribute_vote_smoothed` (each rated attribute as one annotator); `votes_smoothed` on human votes | `summary.csv` |
| Whether the residual in human votes **exceeds sampling noise** | parametric bootstrap under a fitted BT truth (logit link, where BT has C = 0 exactly) | `null.csv`, `report.md` |
| `C != 0` need not imply a Condorcet cycle; SST need not imply `C = 0` | `cyclic_without_condorcet`, `sst_violation_mean`; `single_score_hard` is SST with `C != 0` | `summary.csv` |
| Comparisons are sparse (`ass:d-sparse`), and the constants obey the Fact after `thm:kl-contraction` | `d`, `completeness`, `fact_bounds_hold` | `summary.csv` |
| Whether the frontier `alpha^2 + gamma^2 = 1` is reached at practical parameters | realized gain `gamma = beta lambda omega_max(pi*)` at the unique fixed point, fraction of panels beyond the frontier | `population.csv`, `population_frontier.png` |
| What the current neural protocol actually feeds the dynamics | `single_score_hard` on HelpSteer with the five attributes averaged (as `scripts/build_pairs.py`), and the repo's cyclic tournament (tests) | `outputs/helpsteer` |

## Definitions and conventions

- `A = Psi(P*) - Psi(1/2)`, `u = A 1 / K`, `C = A - (u 1^T - 1 u^T)` on the complete,
  uniformly weighted graph; unobserved pairs are filled at indifference (`ass:d-sparse`).
  On incomplete panels this is the `C` of the recursion but not an identified quantity;
  `max_abs_observed_curl` reports the curls of fully observed triangles, which are identified.
- `cyclic_share = ||C||_F^2 / ||A||_F^2`, the fraction of flow energy that no scalar score explains.
- `L_C`, `||C||_2`, `omega_uniform = ||C||_2 / K` (the frequency at the uniform policy),
  `a = max |A_ij|`, `d` the maximum comparison degree.
- Population layer: centered recursion `x' = alpha x + b + beta lambda C softmax(x)` with
  **uniform reference and uniform coverage on the panel**, so `b = beta u`. The paper's
  `beta` multiplies the payoff: DPO's KL coefficient is `1/beta` (DPO `beta = 0.1` is
  `beta = 10` here); the IPO loss `(h - 1/(2 tau))^2` gives `beta = 1/tau`, and
  `LLM_Sampling`'s `beta_train` is `tau`.
- Logit-link shares use pseudo-counts (`--pseudo`, default 0.5 per side) so unanimous
  judgments stay finite; identity-link shares are unsmoothed except in `*_smoothed`.
- `attribute_bt` and `single_score_bt` use `sigma(scale * rating difference)` with
  `--bt-scale` (default 1 logit per rating point); the scale changes `A` and must be reported.

## Datasets

Panels need `K >= 3` candidates per prompt: any `K = 2` panel has `C = 0` identically, so
pairwise-only corpora (Anthropic HH-RLHF, HelpSteer2 pairs, PKU-SafeRLHF, binarized
UltraFeedback) cannot exhibit a per-prompt cycle. They are structurally `d = 1`; cycles
appear once an iterative pipeline samples several candidates per prompt.

| Key | Source | License | Panels | Size |
|---|---|---|---|---|
| `helpsteer` | `nvidia/HelpSteer` | CC-BY-4.0 | responses per prompt, 5 attribute ratings (0--4) | 17 MB |
| `ultrafeedback` | `openbmb/UltraFeedback` (HF parquet conversion) | MIT | 4 completions per instruction, 4 GPT-4 aspect ratings (1--5) | 323 MB |
| `mt_bench` | `lmsys/mt_bench_human_judgments` | CC-BY-4.0 | 6 models x 80 questions x 2 turns, human and GPT-4 pairwise votes | 1.3 MB |
| `arena55k` | `lmarena-ai/arena-human-preference-55k` | Apache-2.0 | one pooled model-level panel of human votes | 102 MB |
| `helpsteer2_preference` | `nvidia/HelpSteer2` | CC-BY-4.0 | pairwise with several annotators (heterogeneity only, `hodge/heterogeneity.py`) | 15 MB |

## Usage

Run from this directory (`experiments/hodge_diagnostics/`):

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q                      # 23 tests, includes the paper's worked numbers
.venv/bin/python -m hodge.fetch --list
.venv/bin/python -m hodge.fetch --yes helpsteer mt_bench
.venv/bin/python -m hodge.run helpsteer            # outputs/helpsteer/{report.md,summary.csv,...}
.venv/bin/python -m hodge.run synthetic            # no download needed
```

The 2026-09-25 report and its figures are in `reports/2026-09-25/REPORT.md`; `data/` and
`outputs/` are not committed.

`data/MANIFEST.json` records URL, byte size, and SHA-256 of every fetched file.

## Layout

- `hodge/core.py`: links, Hodge decomposition, curls, constants, frequencies, ordinal
  checks, graph statistics, BT fit and parametric null.
- `hodge/constructions.py`: panel matrices from ratings or votes.
- `hodge/population.py`: exact population recursion, unique fixed point, realized gain,
  the thresholds `Gamma_kappa(alpha)`, `kappa_exp`, `R_nu(alpha)`, and the
  history-weighted simulator (for A1/B1).
- `hodge/datasets.py`, `hodge/fetch.py`, `hodge/run.py`; `hodge/heterogeneity.py`,
  `hodge/extras.py` (near-tie strata, Arena cliques), `hodge/report.py` (report figures and tables),
  `hodge/coherence.py` (cross-prompt coherence under parameter sharing: compressed cyclic
  frequency with a within-prompt shuffle null, label-level HodgeRank, transitive surrogates,
  and the exact dynamics of a policy that shares one score per model or length rank).
- `analysis/`: scripts behind specific statements (pilot frontier table, worked cases,
  block-diagonal stacking, shared-parameter probes); run as `python analysis/<name>.py`.
- `tests/`: the Section 3 worked examples, the tanh formula, the three-response frequency,
  the three-cycle frontier by simulation, both threshold formulas against characteristic
  roots, the same-limit-point property, and the additive horizons `M_T` and `M_{ceil(T/2)}`.
