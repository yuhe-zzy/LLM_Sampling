# Preregistration: X15, onset of rotation under pair-dependent pooling (written 2026-10-19 23:35 EDT, before any measurement)

Approved by Fan on 2026-10-19 ("use the criteria; K = 3 main arms plus K = 4 side arms; accept the semi-synthetic design; add m̃ = 0.5; run on the VM; start after preregistration", translated from Chinese). Design: `../DESIGN.md`. Theory: v4 §5b (prop:pooled, cor:pooled-onset) and `reviews/2026-10-19-mixed-criteria-cycles/theory.md`.

## Question
cor:pooled-onset says that when comparisons are assigned to the source that owns them (pair-dependent routing with share c), the step at which a fixed-reference DPO loop starts to rotate falls from order m⁻³ (uniform mixture) to order (cm)⁻¹. Does a trained language model rotate accordingly, on panels whose candidates three transitive criteria rank in rotation, and does slower refresh at the same target stop it?

## What is semi-synthetic, stated once
The three criteria are real and each is transitive (a scalar per answer). The panels are selected from the base model's own samples so that the three criteria rank the three candidates in rotation (a Latin square). X11 (2026-10-19) showed that such panels are rare among sampled candidates and that rubric contexts do not create them; this experiment therefore tests the conditional claim of §5b ("requires candidates that different criteria rank in a rotation"), not the prevalence of such panels.

## Sources
- Q (quality): Skywork-Reward-Llama-3.1-8B-v0.2 score on the plain prompt.
- C (concision): minus the number of characters.
- N (originality): minus the base-policy sequence log-probability s0.
- Labels of source k: P^k_ij = σ(m̃ · gap^k_ij / T_k), T_k = the standard deviation of criterion k's within-panel gaps over all kept panels (computed once on the K = 3 panel set, reused for K = 4). m̃ ∈ {1 (main), 0.5}.

## Panels (`vm/build_specialist_panels.py`)
- Prompt pool and suffix as in v4 (`build_panels.py`: 65 short + NoveltyBench curated + WildChat, up to 600 prompts). The base policy samples M = 48 answers per prompt at temperature 1; distinct complete answers with at most 8 words are scored by exact sequence log-probability; candidates with normalized probability ≥ 0.05 among the distinct answers of the prompt are eligible (the v4 floor, applied to the candidate set).
- K = 3 (main): among eligible candidates, all triples (a_Q, a_C, a_N) in which criterion k ranks a_k first and the three rankings form a rotation (Q: a_Q ≻ a_C ≻ a_N, C: a_C ≻ a_N ≻ a_Q, N: a_N ≻ a_Q ≻ a_C, or the mirror rotation). If several, keep the triple with the largest minimum standardized gap (standardization by a provisional T_k from all eligible candidates). The panel's s0/p0 are renormalized over the three answers; every p0 must be ≥ 0.05.
- K = 4 (secondary): the same triple plus a fourth eligible answer that every criterion ranks below all three (if none exists the prompt gives no K = 4 panel).
- Target: at least 64 K = 3 panels. If fewer, the floor is lowered to 0.02 and reported; if still fewer than 32, the stage stops and is reported.
- Each kept answer is scored by the three reward models as in v4 (only Skywork-8B is used for Q).

## Pooled labels (`analysis/predict_x15.py`, written to `data/oracle_*.npz` with P[N,K,K])
- Owner of an oriented edge (i, j) along the rotation: the criterion that ranks i above j with the largest standardized gap among those that do; by construction of the Latin square each cycle edge has exactly two criteria ranking it forward (gaps m-like and 2m-like in the corollary's idealization) and the owner is the one with the larger standardized gap. For K = 4 the edges to the fourth answer have all three criteria agreeing; the owner is the largest-gap criterion.
- owner(c): P̄_e = c · P^{owner(e)}_e + (1 − c) · (1/3) Σ_k P^k_e, for c ∈ {0, 0.25, 0.5, 1}. c = 0 is the uniform mixture.
- salience: each edge labelled by the criterion with the largest standardized |gap| (X11's rule).
- Per matrix we record the logit-scale cyclic share, the curl of the cycle triangle, and whether a Condorcet winner exists.

## Testbed (unchanged from X14)
`panel_po_steer.py` on the VM: Qwen2.5-0.5B-Instruct, full fine-tuning, λ = 1, uniform coverage, 40 inner Adam steps at lr 2e-5, fp32, 120 rounds, seed 0, 64 panels (all K = 3 panels if fewer). Labels from `--oracle npz:`.

## Arms (B = 10 throughout; A: α = 0, β = 10; C: α = 0.9, β = 1)
1. X15_c0_A: owner c = 0, DPO, A.
2. X15_c25_A: owner c = 0.25, DPO, A.
3. X15_c50_A: owner c = 0.5, DPO, A.
4. X15_c100_A: owner c = 1, DPO, A.
5. X15_c100_C: owner c = 1, DPO, C.
6. X15_c50_C: owner c = 0.5, DPO, C (run only if arm 3 is predicted to oscillate on ≥ 8 panels).
7. X15_rand_A: random assignment: `--oracle npz:` with the c = 0 matrix and `--sampled 512` (a mixture of Bernoullis equals a Bernoulli of the mixture, so drawing a criterion per pair per round and labelling by it is exactly sampled labels from the uniform mixture), DPO, A.
8. X15_c100_A_ipo: owner c = 1, IPO, A (identity link).
9. X15_m05_c100_A: owner c = 1 with m̃ = 0.5, DPO, A.
10. X15_K4_c100_A: K = 4 panels, owner c = 1, DPO, A.
11. X15_K4_c0_A: K = 4 panels, owner c = 0, DPO, A.
Arms 1–5 and 7–11 are fixed; arm 6 is conditional as stated. 4 runs in parallel.

## Predictions (fixed before training, `prereg/predictions_x15.json`)
For every panel and arm, the population recursion from the model's own initial scores: population DPO (`dpo_map.dpo_round`, the solver fixed on 2026-10-19 14:40 EDT) for DPO arms, the identity-link master recursion for the IPO arm; 120 rounds; oscillating = detrended probability amplitude over the last 30 rounds > 0.0536 (the v4 rule). Also recorded: each panel's onset β*(c) at α = 0 from the DPO-map multipliers, and the median onset against c and against m̃ on a grid (population only).

## Pass criteria
1. Direction: the neural oscillating share is nondecreasing in c over arms 1–4 (ties allowed), ≤ 0.05 at c = 0, and ≥ 0.5 at c = 1 if the population prediction at c = 1 is ≥ 0.5. If the population model predicts fewer than 8 oscillating panels at c = 1 under A, training is not started; the population result is reported and m̃ is reconsidered with a logged amendment.
2. Quantitative: every arm's neural oscillating share within 0.15 of its population prediction.
3. Onset scaling (population layer): the log–log slope of the median onset β* against c over c ∈ {0.25, 0.5, 1} is within [−1.3, −0.7]; against m̃ at c = 0 within [−3.5, −2.5] and at c = 1 within [−1.3, −0.7].
4. Remedy: arm 5's oscillating share is within 0.15 of its population prediction and below arm 4's by at least 0.3, and its median KL(target ‖ last-10 policy average) ≤ 0.01. (Amended 2026-10-20 00:05 EDT, before predictions were computed: a smoke test on synthetic panels showed that a strong enough planted-like cycle can still rotate at α = 0.9, β = 1, so the remedy criterion is stated relative to the population prediction rather than as an absolute ≤ 0.10.)
5. Random assignment: arm 7 oscillating share ≤ 0.10; its mean per-round KL(π_{t+1} ‖ π_t) over the last 30 rounds exceeds arm 1's.
6. Map choice: in arms 2–5 the median one-step error of the trained model is smaller under the DPO map than under the logit-link map.
7. m̃ = 0.5 (arm 9) and K = 4 (arms 10–11): criterion 2 applies.

## Decision rule
Passes are reported; failures are diagnosed and reported; any change is logged here with a timestamp before a rerun. The VM is deallocated after the batch. No manuscript edit before the scorecard is written.

## Cost
Panel construction: one VM pass (sampling 600 × 48, scoring with three reward models), about 25 minutes. Training: 10–11 runs of 120 rounds on ≤ 64 panels, 4 in parallel; by the X14 batch (9 runs, 580 rounds, 2.1 h at 3 in parallel) about 4–5 VM hours, $30–35.

## Log
- 2026-10-19 23:35 EDT: written before any measurement. VM started 23:31 EDT; panel construction started 23:50 EDT.
- 2026-10-20 00:05 EDT: criterion 4 amended (see its text); onset solver in `predict_x15.py` replaced by Newton continuation after a smoke test on synthetic panels. No real panel had been measured.
- 2026-10-20 00:20 EDT, panel construction at the 0.05 floor gave 15 K = 3 panels and 1 K = 4 panel (saved in `data/floor05/`). Per the panel rule the floor is lowered to 0.02; in addition M is raised from 48 to 96 samples per prompt to enlarge the candidate pool (amendment, before any prediction or training). Second builder pass started 00:20 EDT into `data02/`.
- **2026-10-20 00:00 EDT, predictions frozen and batch launched.** Second pass gave 76 K = 3 panels (p0 min 0.021) and 4 K = 4 panels; 64 K = 3 panels selected with seed 0 (`vm/batch_x15_selection.json`); all 4 K = 4 panels used. `prereg/predictions_x15.json` (script `analysis/predict_x15.py`, log `prereg/predict_x15.log`), computed before any run: oscillating shares under A: c = 0: 0.0, c = 0.25: 0.066, c = 0.5: 0.474, c = 1: 0.947; c = 1 under C: 0.118; c = 0.5 under C: 0.0 (arm 6 runs, since arm 3 has 36 predicted panels); random assignment: 0.0; IPO c = 1: 0.947; m̃ = 0.5, c = 1: 0.618; K = 4: c = 1: 1.0, c = 0: 0.0. Median onset β* at α = 0: c = 1: 4.15, c = 0.5: 10.05, c ≤ 0.25: ∞ (a Condorcet winner survives on most panels); m̃ = 0.5: c = 1: 8.16, c = 0.5: 24.4. Logit-scale cyclic share at c = 1: 0.76; no Condorcet winner on 100% (c = 1), 68% (c = 0.5), 26% (c = 0.25), 0% (c = 0). Batch of 9 K = 3 runs (4 in parallel) started 23:57 EDT on the VM, followed by the 2 K = 4 runs.
- 2026-10-20 00:12 EDT, scoring clarification written before any run was read (`analysis/evaluate_x15.py`): criterion 3's slope in c is taken over c ∈ {0.5, 1}, because at c ≤ 0.25 most panels keep a Condorcet winner and the onset is infinite (frozen predictions); the slope in m̃ is taken at c = 1 and c = 0.5 for the same reason, and the "−3 at c = 0" check is replaced by the observation that the c = 0 onset is infinite at every m̃ (no panel oscillates). The population comparison in criterion 2 uses the run's own training-time initial scores (as in v4 §8), which can differ slightly from the panel file's s0 after re-tokenization; the frozen shares are reported alongside.
- **2026-10-20 00:30 EDT, launch error found and corrected (reported by Fan's Opus session; verified).** The batch files pointed `--panels` at the VM's first-pass file `data/panels3.json` (15 panels, floor 0.05) while the oracles and the id selection came from the second pass `data02/panels3.json` (76 panels). The trainer kept only the 12 selected ids present in the 15-panel file, and their labels belonged to other prompts. The first four runs (c = 0, 0.25, 0.5, 1 under A; shares 0 / 8% / 50% / 83% on those 12 mismatched panels) are void for this preregistration and are kept on the VM as `runs_bad_wrongpanels/` for the record only. The batch was stopped at 00:28 EDT, the paths corrected to `data02/`, and the batch relaunched at 00:31 EDT with the same frozen predictions, oracles and selection. No prediction was changed.
- 2026-10-20 00:40 EDT: the relaunch at 00:34 EDT started before the killed processes had released GPU memory; the first four runs (c = 0, 0.25, 0.5, 1 under A) and X15_rand_A died with CUDA out-of-memory within seconds and are rerun automatically after the current chain (`x15_followup.sh`; the runner skips finished runs). With 64 K = 3 panels a run takes 18–28 GB, so the parallelism is set to 3 (the mis-pointed first batch had only 12 panels, which is why 4 fitted). No prediction or arm definition changed.
- 2026-10-20 01:45 EDT: the rerun pass at parallelism 3 hit out-of-memory again on an empty GPU (a 64-panel K = 3 run takes about 35 GB), so the parallelism is 2 from here; the two remedy arms and the two K = 4 arms finished in the first chain (01:29 and 01:41 EDT). Rerun of the seven remaining arms started 01:43 EDT. Expected end about 05:00 EDT.
