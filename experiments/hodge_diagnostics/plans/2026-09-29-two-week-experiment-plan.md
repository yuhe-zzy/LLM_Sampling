---
date: 2026-09-29
project_id: "iterative-psipo"
status: proposed (revised the same day to build on the 2026-09-28 calibration)
---

# Plan — Two-week neural validation of the Paper B frontier and stabilizers (2 x H100)

Drafted by Claude at the project lead's request. The plan builds on four sources:
- the 2026-09-22 roadmap (v3 Section 7);
- the Hodge diagnostics (`HANDOFF.md`, `reports/2026-09-25/REPORT.md`);
- Fan's review of 2026-09-26 (`HANDOFF_TO_YU_HE_2026-09-26.md`);
- Yu He's calibration of 2026-09-28 (`HANDOFF_TO_FAN_AND_YU_HE_2026-09-28.md`,
  `../cyclic_history/CALIBRATED_PROTOCOL.md`).

Nothing below is an observed result. Every radius and threshold is a theory prediction:
- the E3 table: `hodge/population.py`, uniform-reference design illustration;
- all run-time predictions: `cyclic_history/population_calibration.py`, the actual IPO and
  DPO operators.

Every run time is an estimate to be replaced by the first measured runs.

## Where we start

- **The diagnostics make curated data a negative control.** Curated preference data
  (HelpSteer, UltraFeedback, MT-Bench, Arena) sit inside the frontier, so they cannot show
  instability on their own.
- **The pilot's two arms disagree.** The prepared six-run pilot at `alpha = 0.9`,
  `lambda = 0.8`, `beta_train = 1` differs by objective. The IPO arm is predicted to
  converge (radius 0.929 with a flat reference). The **actual DPO operator predicts
  instability** (radius 1.289), and the prepared `nu = 0.45` and `kappa = 0.25` do not
  stabilize it (1.021 and 1.292). The earlier diagnostics statement that "the pilot is
  predicted to converge" held for IPO only.
- **Four constraints fixed by the 26 Sep review.** They apply to every experiment below:
  1. predictions for DPO use the actual pairwise BT optimizer, not entrywise `logit(P)`;
  2. the raw sampler `mu = 0.2 Unif + 0.8 softmax(sequence sum)` stays, and a
     relative-to-initial sampler would be a different protocol;
  3. the current `kappa` arm is named **oracle-assisted feedback extrapolation**, which is
     distinct from the manuscript's two-sampler estimator;
  4. beyond the frontier the claim is generic nonconvergence and an oscillation signature,
     not a periodic cycle.
- **What the 28 Sep calibration supplies.**
  - Actual IPO/BT fixed points and Jacobians, augmented-history spectra, and exact
    population trajectories.
  - A balanced soft four-cycle oracle over fixed roles: adjacent 0.8/0.2, opposite pairs
    0.5, so the directional component is zero.
  - A staged 10-arm micropilot on six selected panels, ready to launch after approval.
  - The bottleneck: only 8 of 500 existing panels have every initial raw probability of at
    least 0.005.

## Objective and success criteria

The goal is to pre-empt the review "the contribution is only theoretical". Section 7
should show that the paper's quantitative predictions hold in iterative preference
optimization of a language model (LoRA, SGD, shared parameters), and that the diagnostics
say when they matter.

Success means four decisive figures with preregistered predictions and at least two seeds
per cell:

1. **Adaptation horizon.** Neural limit points equal `beta u / (1 - alpha)` and the
   approach follows the discounted horizon.
2. **Frontier.** Across panels and parameters, runs show the preregistered oscillation
   signature exactly when the predicted index exceeds one. Measured growth per round
   tracks the predicted radius and measured rotation tracks the predicted argument.
   Every panel is a point with its own realized gain.
3. **Stabilizers.** A preregistered matrix of predicted stable or unstable outcomes is
   confirmed, including cells where extrapolation destabilizes a run that is stable without
   it. Stabilized runs approach the ordinary fixed point `pi*(alpha)` computed by the
   solver, and maximal lag costs about twice the rounds.
4. **When it matters.**
   - Curated data stay inside the frontier.
   - Persona heterogeneity with conflicting style preferences crosses it.
   - Role-aligned and mixed-orientation cycles are compared under shared LoRA parameters.

A failed prediction is reported with its diagnosis: inner-update error (relaxation `h`,
fixed-point residuals) or accessible modes under parameter sharing.

## Testbed

This is panel-restricted iterative PsiPO: the paper's finite-response model with a real
language model. It is the calibrated `experiments/cyclic_history` protocol.

**Frozen environment** (from `CALIBRATED_PROTOCOL.md`):
- Qwen2.5-1.5B with sequence sums including EOS.
- Raw sampler `mu = 0.2 Unif + 0.8 softmax(sequence sum)`.
- Full-pair soft-label mode (all pairs, soft labels, no pair subsampling).
- Cached log-score reference `(1-alpha) s_0 + (alpha-nu) s_t + nu s_{t-1}`.
- Optimizer reset each round.
- Fixed-point residuals and per-prompt cyclic-mode amplitude and phase logged.

**Additions in this plan:**
- **Panels with interior support, Phase 0.** For 256 training and 64 held-out prompts,
  sample 24 candidates per prompt from the base model (temperature 1, at most 192 new
  tokens) and score them.
  - Select K = 4 candidates whose initial raw panel probabilities are all at least 0.05,
    preferring near-equal token lengths.
  - Assign roles by ascending length, the calibrated convention.
  - This keeps the raw sampler and removes the 8-of-500 bottleneck. The panels become the
    base model's own samples rather than UltraFeedback responses, and the report says so.
  - If fewer than 128 prompts qualify, lower the threshold toward 0.005 and report it.
- **Predictions.** Per panel and objective, `population_calibration.py` gives the
  fixed point, realized gain, ordinary and history radii, and exact population
  trajectories, i.e. the shadow runs. Failed roots stay "unresolved".
- **Relaxation.** `h_t = <Delta x_t, T(x_t) - x_t> / ||T(x_t) - x_t||^2` is logged. If
  `h` differs from 1, the relaxed map `x + h (T(x) - x)` is reported beside the exact one.
- **Oracles:**
  - `O1`: Bradley--Terry from a scalar score (C = 0 under the logit link; actual DPO
    equals entrywise-logit PsiPO here);
  - `O2`: the calibrated balanced soft four-cycle;
  - `O2m`: the same cycle with reversed orientation on half the prompts, i.e. mixed
    orientation;
  - `O3`: `O2` plus a potential;
  - `O4`: a persona-mixture LLM judge over style-diverse responses;
  - `O5`: an UltraFeedback aspect mixture (curated control).
- **Metrics.** All are computed by script:
  - log-amplitude slope (measured radius) and rotation per round of the cyclic mode;
  - the preregistered oscillating, converged or concentrated labels;
  - step norm and fixed-point residual;
  - KL to the computed `pi*(alpha)`;
  - rounds to 90% of the terminal log-odds;
  - last-iterate exploitability;
  - held-out quality.

## Phase 1: calibrated micropilot (Days 1--3; prepared by Yu He)

Six panels (54, 251, 612, 737, 867, 945), `alpha = 0.9`, `lambda = 0.8`, 30 rounds,
10 inner epochs, one seed, run only after approval:

- **Stage A** (4 runs): per objective, an ordinary arm predicted unstable (IPO
  `beta_train = 0.2`, DPO 0.8) and a stable control (0.4, 1.6).
- **Stage B** (4 runs, if Stage A shows the signature): `nu = 0.45` and oracle-assisted
  `kappa = 0.5` at the unstable `beta`. Predicted radii: ordinary 1.05--1.08,
  reference 0.96, feedback 0.92--0.96.
- **Stage C** (2 runs): mixed orientation, ordinary.

**Gate G0.** In Stage A, the unstable arms show the signature and the stable controls do
not, for both objectives. If they fail, measure inner-update error and accessible modes
before any scale-up. Phase 1 also gives the first measured wall time.

## Phase 2 experiments (Days 2--12)

### E1 Adaptation horizon and relaxation calibration (Tier 1)

- **Setup.** `O1` on the Phase 0 panels, DPO with soft labels, and IPO as the link
  comparison. `alpha in {0, 0.5, 0.8, 0.9, 0.95, 1}`, 30 rounds. Qwen2.5-0.5B with
  3 seeds, and 1.5B with 2 seeds.
- **Predictions.** Limit `beta (u_i - u_j) / (1 - alpha)` and linear growth at
  `alpha = 1`. With relaxation `h`, the approach is `h beta u sum_k (1 - h(1-alpha))^k`.
- **Gate G1.** Limit points within 10% for `alpha <= 0.9`.

### E2 Frontier (Tier 1)

- **Setup.** `O2` on the Phase 0 panels, with IPO and actual DPO,
  `alpha in {0.5, 0.7, 0.8, 0.9, 0.95}`.
  - `beta_train` per objective is set so that the median panel's predicted index sits at
    0.6, 0.85, 1.25 and 1.6 times its boundary. Panel heterogeneity then fills the
    `(alpha, gamma)` plane within each run.
  - 60 rounds, and 80 at `alpha = 0.95`. `alpha = 0.99` is excluded: its onset period is
    about 44 rounds and growth only 0.7% per round.
- **Runs.** 0.5B with 2 seeds on the full grid; 1.5B with 2 seeds on the `alpha = 0.8`
  and 0.9 columns.
- **Controls.**
  - Gain collapse only where `u = 0` with matched reference and coverage, then how it
    breaks as `O3` adds a potential.
  - A random, unselected panel set at one cell: predicted mostly stable because the
    initial mass is concentrated.
- **Analysis.** Panel-level predicted index against observed amplitude (ROC); measured
  against predicted radius and rotation.
- **Gate G2.** At least 80% agreement outside a boundary band of 10%, and correlation of
  at least 0.8 in log radius.

### E3 Stabilizers at the same limit point (Tier 1)

- **Design.** Five configurations chosen, in the uniform-reference design illustration
  below, so every cell's radius is at least 0.02 from 1.
- **Before preregistration** the cells are recomputed per panel with the actual operators
  on the Phase 0 panels. `beta` is re-set so that the median panel reproduces the target
  radii. Cells whose recomputed margin falls below 0.02 are replaced.
- **Schemes:**
  - ordinary;
  - oracle-assisted `kappa in {0.25, 0.5, 1, kappa*}`;
  - `nu in {alpha/2, alpha}`;
  - if the authors choose, the two-sampler estimator at S1.
- **Runs.** 26 cells x 2 seeds on 0.5B; S1 and S2 x 2 seeds on 1.5B.

Design illustration (predicted spectral radius on a balanced cycle with uniform reference;
cells marked * grow):

| Config | ordinary | kappa=0.25 | kappa=0.5 | kappa=1 | kappa* | nu=alpha/2 | nu=alpha |
|---|---|---|---|---|---|---|---|
| S1 alpha=0.9, gamma=0.62 | 1.093* | 1.041* | 0.978 | 1.118* | 0.961 (k*=0.58) | 0.965 | 0.949 |
| S2 alpha=0.7, gamma=0.62 | 0.935 | 0.867 | -- | **1.174*** | -- | -- | 0.837 |
| S3 alpha=0.45, gamma=0.78 | 0.900 | 0.955 | **1.142*** | -- | -- | -- | 0.671 |
| S4 alpha=0.9, gamma=0.90 | 1.273* | -- | -- | 1.727* | 1.387* | 1.015* | 0.949 |
| S5 alpha=0.95, gamma=0.45 | 1.051* | -- | 0.958 | 0.747 | 0.902 (k*=0.69) | 0.983 | 0.975 |

- **Same target.** On `O3`, endpoints are compared with `pi*(alpha)` and with the time
  average of the ordinary orbit.
- **Cost.** On `O1`, `nu = alpha` needs about twice the rounds to reach 90% of the
  terminal log-odds.

### E4 When it matters (Tier 2)

- **Persona oracle `O4`.**
  - Three style variants per prompt (concise, step-by-step, bulleted), judged in both
    orders by Qwen2.5-7B-Instruct under three personas with cyclic style preferences.
    Probabilities come from the choice tokens.
  - Validate with `hodge.coherence` and the actual operators before training.
  - Runs: 4 `alpha` x 2 seeds, plus `nu = alpha` and `kappa*` at one unstable cell.
- **Role-aligned against mixed orientation.** `O2` against `O2m` at matched per-panel
  norms and zero potential, at S1 and at one E2 cell, 2 seeds, on 0.5B and 1.5B. This
  extends the calibrated Stage C to many panels. It enters Section 7 only if the authors
  agree, and only as a measurement of shared-parameter behavior.
- **Curated control.** `O5` at `alpha in {0.9, 0.95}`, 2 seeds.

### E5 DPO labels (Tier 2)

- **Setup.** `O1` with transitive hard labels (an acyclic win graph, so the BT optimum is
  infinite) against soft labels, `alpha in {0.8, 0.95}`, 2 seeds, on 0.5B.
- **Prediction.** Soft labels reach the E1 limit; hard labels drift at a rate set by the
  optimizer budget. On a strongly connected cyclic oracle, hard labels do have a finite
  DPO optimum (26 Sep review), so the comparison is transitive-specific.

### Scale check (Tier 2) and E6 on-policy loop (Tier 3, stretch)

- **Scale check.** S1 with ordinary refresh, `nu = alpha`, `kappa*` and `kappa = 1`,
  x 2 seeds, on Qwen2.5-3B.
- **E6.** Each round, generate K = 4 responses per prompt and judge them with the `O4`
  personas; read out the style shares.
  - Ordinary refresh at `alpha in {0.5, 0.95}` and `nu = alpha` at 0.95, 2 seeds, 1.5B.
  - Go/no-go on Day 10.

## Compute budget

These are estimates. Phase 1 supplies the first measured wall time.

| Item | GPU-hours |
|---|---|
| Phase 0 panel construction (generation and scoring) | about 4 |
| Phase 1 micropilot (10 arms, 6 panels, 30 rounds) | about 5--10 |
| Tier 1: E1, E2, E3 | about 150 |
| Tier 2: E4, E5, scale, judging | about 70 |
| Tier 3: E6 | about 25 |
| **Total** against 432 available | **about 260** |

- **If runs are twice as slow:** keep 2 seeds, drop Tier 3 and the 3B check, and run the
  E2 grid on 0.5B only.
- **If runs are faster:** raise seeds on the headline cells.

## Milestones

| Milestone | Evidence of completion | Target date | Status |
|---|---|---|---|
| Stage A approved and launched; panel construction started | job IDs; generation running | 2026-09-30 | proposed |
| G0; Stage B/C launched; Phase 0 panels scored | Stage A signature table; panel support table | 2026-10-01 | proposed |
| E1 done (G1); Phase 2 preregistration | horizon figure, fitted h; timestamped `prereg/` commit with per-panel actual-operator predictions | 2026-10-02 | proposed |
| E2 done (G2) | panel-level phase diagram, radius and rotation agreement | 2026-10-06 | proposed |
| E3 done | confusion matrix, same-target and cost results | 2026-10-08 | proposed |
| E4, E5, scale | persona, orientation, curated and label results | 2026-10-10 | proposed |
| E6 go/no-go and run | decision on Day 10; runs if go | 2026-10-11 | proposed |
| Section 7 draft and artifacts | figures, tables, reproducibility notes, code tag | 2026-10-13 | proposed |

## Risks and mitigations

- **Phase 0 finds too few panels with interior support.** Lower the threshold toward 0.005,
  shorten the responses, or sample more candidates. Report the selection, since it is
  theory-guided rather than representative, and keep a random-panel arm.
- **SGD noise near the frontier.** Use the boundary band, full pairs, and seeds.
- **Relaxation `h` far from 1 or anisotropic.** Report relaxed predictions and per-panel
  `h` in the shadow runs.
- **LoRA capacity or sharing damps panel-specific dynamics.** Read the orientation arm;
  measure accessible modes before adding seeds.
- **Persona judges too consistent.** Validate before training; sharpen the personas or
  fall back to `O2` on style roles.
- **"Synthetic oracle."** Real prompts and model-generated responses; E4, E6 and the
  public-data diagnostics.
- **Schedule slips.** Tier 1 carries the core claim. Cut Tiers 3 and 2 first.

## Decisions needed

1. **Approve Stage A of the prepared micropilot now** (Fan with Yu He). Stages B and C
   follow the G0 rule.
2. **Approve Phase 0 panel construction** as the way to scale while keeping the raw
   sampler. A relative-to-initial sampler would be a separate protocol and is not
   proposed.
3. **DPO scope.** The actual pairwise DPO operator (recommended, since it matches
   training), entrywise-logit PsiPO, or both with separate boundaries.
4. **Sampling intervention.** The oracle-assisted feedback extrapolation (mechanism test,
   implemented), the two-sampler estimator (practical, needs implementation), or both at
   S1.
5. Whether the role-aligned/mixed arm may enter Section 7.
6. **Division of work.** Proposed:
   - Yu He: pipeline, Phase 0/1 runs, launches;
   - Claude: panel-construction and persona scripts, preregistered predictions from the
     actual operators, analysis and figures;
   - Fan: decisions and Section 7.

## Self-review iterations

- **v0.** The 22 Sep roadmap grid was too expensive, and most of its cells are predicted
  stable.
- **v1.** Controlled cycles and exact pair enumeration, so E2 and E3 test the
  deterministic theorems.
- **v2.** Relaxation calibration, shadow population runs, preregistration.
- **v3.** Stabilizer configurations chosen by spectral separation; `alpha = 0.99` dropped;
  counter-intuitive cells added.
- **v4.** Realism: persona mixture, orientation arm, curated control, gated E6, model
  tiers.
- **v5 (same day).** Reconciled with the 26 Sep review and the 28 Sep calibration:
  - kept the raw sampler, and replaced the relative-to-initial sampler with constructed
    panels that have interior support;
  - made the actual IPO and DPO operators the prediction source;
  - put the prepared micropilot first as Phase 1 with gate G0;
  - renamed the `kappa` arm and the orientation arms;
  - corrected the IPO-only "pilot is stable" statement;
  - restricted gain collapse to `u = 0`.

## Project-level next action

The project lead and Yu He approve Stage A of the prepared micropilot and the Phase 0
panel construction, so that both start on 2026-09-30 on the two H100s.

## Review date

2026-10-02 (after G0 and G1)
