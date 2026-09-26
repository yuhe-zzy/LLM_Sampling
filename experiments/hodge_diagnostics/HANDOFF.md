# Handoff: Hodge diagnostics for Paper B

Branch `hodge-diagnostics`, 2026-09-26. Prepared by Claude at Fan Yao's request for Yu He,
Fan, and any agent that continues the Paper B experiments. The detailed evidence is in
[`reports/2026-09-25/REPORT.md`](reports/2026-09-25/REPORT.md). This file is the entry
point: state, findings, open decisions, and the next actions.

## 1. State

- **Scope.** `experiments/hodge_diagnostics/` is the GPU-free diagnostic layer of the
  JMLR-B-v3 experimental roadmap (Section 7, Part 0 item 2, plus the exact-population check
  of A1). It is independent of the neural pipeline and does not modify it.
- **Tests.** 23 pytest tests pass. They cover:
  - the paper's worked numbers: Section 3 examples, the tanh formula, three-response
    frequencies;
  - the three-cycle frontier by simulation;
  - both stabilizer thresholds, checked against characteristic roots;
  - the same-limit-point property and the additive horizons;
  - the coherence statistic and the exact restricted map, which reproduces the tabular map
    when realizable.

  These tests are not wired into the repository CI, which only runs `compileall` on this
  directory.
- **Data.** Five public datasets, fetched on demand (`python -m hodge.fetch`, about
  440 MB); SHA-256 hashes are in `data/MANIFEST.json`. Neither `data/` nor `outputs/` is
  committed.
- **Status of claims.** Everything here is a diagnostic of preference data or a numerical
  check. None of it is a neural training result or a verified theorem. The protocol-freeze
  gate for neural runs (2026-09-22 meeting) is unchanged.

## 2. Code map

| Path | Purpose |
|---|---|
| `hodge/core.py` | links, complete-graph Hodge decomposition, curls, `L_C`, `\|\|C\|\|_2`, cyclic frequencies, ordinal checks, graph statistics, BT fit, parametric BT null |
| `hodge/constructions.py` | panel matrices from ratings (hard single score, single BT, attribute votes, attribute BT mixture) or from votes |
| `hodge/population.py` | exact population recursion, unique fixed point (Newton with continuation in `1 - alpha`), realized gain, `Gamma_kappa`, `kappa_exp`, `R_nu`, history-weighted simulator for A1/B1 |
| `hodge/datasets.py`, `hodge/fetch.py` | loaders and the download manifest |
| `hodge/run.py` | per-dataset static diagnostics, population frontier grid, BT null (`outputs/<dataset>/`) |
| `hodge/heterogeneity.py`, `hodge/extras.py` | HelpSteer2 annotator disagreement; near-tie strata and Arena cliques |
| `hodge/coherence.py` | cross-prompt coherence under parameter sharing, label-level HodgeRank, transitive surrogates, exact dynamics of a maximally shared policy |
| `hodge/report.py` | report figures and tables |
| `analysis/` | scripts behind specific statements: `pilot_frontier.py`, `worked_cases.py`, `block_diagonal.py`, `kernel_probe.py`, `kernel_coherence_toy.py` |
| `reports/2026-09-25/` | report, figures, summary tables, coherence outputs |

Reproduction commands are in the report. The full pipeline takes about 45 minutes on one
CPU, most of it in `hodge.coherence`.

## 3. Findings

1. **The sources of cyclic residual in `prop:cyclic-sources` exist in real data, and the
   residuals are cardinal, not ordinal.**
   - Link mismatch and heterogeneous criteria produce `C != 0`, almost never with a
     Condorcet cycle: 0 of 8,321 HelpSteer panels and 20 of 63,966 UltraFeedback panels
     have one.
   - In HelpSteer2, 41.6% of multiply annotated pairs have annotators on opposite sides.
2. **In curated multi-response data the residual is small.** The median cyclic energy
   share is 0.9% on HelpSteer and 3.1% on UltraFeedback.
3. **Large residuals measured on human votes are sampling noise.** MT-Bench per-question
   shares (about 37%) match a fitted BT null (p = 0.92 pooled). Model-level MT-Bench and
   Arena pools are also BT-consistent.
4. **At the population level, ordinary refresh almost never crosses the frontier on these
   panels.** The directional part dominates, the fixed point concentrates, `J(pi*) -> 0`,
   and the gain vanishes. The exceptions are near-tied answers with conflicting criteria,
   0.09% of UltraFeedback panels.
5. **Frozen sampling noise crosses the tabular frontier**: 24--36% of human MT-Bench panels
   at `beta lambda = 10`. It is incoherent across prompts, however.
6. **Parameter sharing filters incoherent cycles.**
   - The residual that stays aligned across prompts is 10--41% of the per-prompt magnitude.
   - Transitive surrogates with saturated labels reproduce and exceed it, so it comes from
     label discretization, not from intransitive preference. Soft BT labels shrink it
     tenfold.
   - A maximally shared policy converges in every configuration tested.
7. **The current protocol's hard labels** add a structural identity-link residual (share
   0.10), make DPO's logit targets infinite, and create exactly the kind of coherent
   residual that survives sharing.
8. **The cyclic-history pilot as planned is predicted to converge.** At `alpha = 0.9`,
   `lambda = 0.8`, `beta_train = 1` the index is 0.863, and sampling from raw sequence-sum
   softmax concentrates the panel policy further. Neither stabilizer can show an effect at
   that point. See `analysis/pilot_frontier.py` for the parameter table.

## 4. Implications (proposals; framing decisions belong to Fan)

- **Theory.** The v3 frontier, the certificates, both stabilizers, and the same-limit-point
  comparison are unaffected. The adaptation--stability insight generalizes (Section 5):
  inner-loop budget and expressivity are further knobs, each trading adaptation for
  stability.
- **Empirical positioning.** The data support "cyclic residuals are generic but typically
  small". They do not support "oscillation is common". A defensible framing is a
  diagnostic that explains why most refresh schedules are stable and predicts when they are
  not: coherent cyclic structure, amplified noise, or inner-loop over-relaxation.
- **Experiments** (Yu He, roadmap Parts A and B):
  1. The controlled cyclic oracle (roadmap oracle iii) is required, and its cycle must be
     coherent across prompts. The pilot's identical tournament is the right positive
     control.
  2. Add a matched arm that shuffles the response order per prompt. Per-prompt gains stay
     the same, and theory predicts convergence.
  3. Choose pilot parameters beyond the frontier (table in `analysis/pilot_frontier.py`).
     Record the sampler's reference convention: raw sequence sums, or scores relative to
     the initial policy, which amounts to a uniform panel reference.
  4. Freeze soft labels (BT probabilities or label smoothing) and record the link.
- **Hypotheses for the ICML-era DPO fluctuations** (to test in A5; not findings):
  - per-round comparison and optimization noise (the stochastic appendix);
  - coherent residuals created by discrete labels;
  - inner-loop over-relaxation, since with hard labels the DPO step is set by the
    optimizer budget.

  Frozen per-prompt noise is a weaker candidate under parameter sharing.

## 5. Candidate theory for shared parameters (unverified)

None of these is in the manuscript. The project rules keep LoRA/projection calculations in
review notes until authors decide otherwise. In the lazy linearization with shared features
`Phi` and NTK `Theta`:

- **R1, exact restricted optimization.** The local map on the accessible subspace is
  `alpha I + beta lambda P_Phi C J`. Its multipliers are `alpha + i beta lambda omega~`,
  and `omega~ <= max_x omega_x` (Cauchy interlacing), so sharing compresses the cyclic
  gain. The adaptation horizon acts on the representable potential: the additive path is
  `r + beta M_T P_Phi u`.
  - **Correction:** this holds exactly only in the realizable case. When the shared class
    cannot represent the tabular target, a curvature term proportional to the projection
    residual enters. Contraction faster than `alpha` was observed, for example 0.70 at
    `alpha = 0.9` on MT-Bench human. `hodge/coherence.py` therefore iterates the exact map
    instead of using this formula.
- **R2, finite kernel steps.** Write the update as `z' = z + Theta W J (T(z) - z)`,
  similar to `I - H (I - T*)` with `H = D Theta D`. The Lyapunov function
  `V = z^T H^{-1} z` gives `Delta V <= -[2(1-alpha) - h((1-alpha)^2 + gamma^2)] ||z||^2`
  for `H <= h I`.
  - So the ordinary condition stays sufficient without over-relaxation (`h <= 1`).
    Under-relaxation enlarges the region to `gamma^2 < 2(1-alpha)/h - (1-alpha)^2`, at the
    price of slower adaptation (zero-frequency multipliers `1 - (1-alpha) h_i`).
  - `n` GD steps with `eta ||H^|| <= 1` satisfy `h <= 1` for every `n`. The full version's
    structured under-relaxation calculation is the scalar case.
- **R3, coherence filter.** With purely shared features the compressed operator is the
  average `sum_x Q^T S_x Q / N`. Identical cycles survive, and randomly oriented cycles
  decay like `N^{-1/2}`.

Numerical checks: `analysis/kernel_probe.py` (R1 matrix and R2 over 4,000 random
instances; no R2 violation), `analysis/kernel_coherence_toy.py`, and `hodge/coherence.py`.

**Next step:** a review note with derivations for author and coauthor verification. Before
any novelty claim, check it against:
- Ren and Sutherland, *Learning Dynamics of LLM Finetuning* (single-step eNTK analysis and
  the DPO squeezing effect);
- Razin et al., likelihood displacement in DPO;
- lazy-training/NTK results.

## 6. Open items and next actions

| Item | Owner | Note |
|---|---|---|
| Choose pilot parameters beyond the frontier; add the shuffled-order arm | Yu He | table in `analysis/pilot_frontier.py` |
| Protocol freeze: soft labels, link, sampler reference, seeds | Yu He | adds to the existing exp-002 gate |
| Framing of the empirical claim; whether R1/R2 enter v3 | Fan | proposals in Sections 4--5 |
| Review note for R1--R3 and verification | Fan / coauthors | not started |
| eNTK measurement of the LoRA kernel on panel logits (predict `omega~` for the neural pilot) | next GPU step | one A100 on university cloud is being provisioned; not yet available |
| Coherence in richer shared features (embeddings, style) | diagnostics | only model identity and length rank were tested |
| Project records outside this repo (`experiments/REGISTRY.yaml`, `PROJECT.md`, `HANDOFF.md` in the project workspace) | Fan / agents | not updated with these results |

## 7. Conventions and pitfalls

- The paper's `beta` multiplies the payoff. DPO's KL coefficient is `1/beta` (DPO
  `beta = 0.1` is `beta = 10`). IPO `(h - 1/(2 tau))^2` gives `beta = 1/tau`; this repo's
  `beta_train` is `tau`.
- Default panel conventions:
  - uniform reference and coverage;
  - unobserved pairs filled at indifference;
  - logit shares smoothed with 0.5 pseudo-votes;
  - BT constructions at 1 logit per rating point.

  All of these change magnitudes and are recorded in the outputs.
- Near `alpha = 1` fixed points saturate, and probabilities of order 1e-13 limit
  double-precision solves. The population solver uses continuation. The shared-policy code
  uses Fisher-preconditioned Gauss--Newton steps and roundoff-tolerant line search, and
  iterates the exact map rather than solving the fixed-point equation.
- Stacking prompts into one block-diagonal matrix and applying the complete-graph formula
  inflates `C` to nearly 100% (`analysis/block_diagonal.py`). Decompose per prompt.
