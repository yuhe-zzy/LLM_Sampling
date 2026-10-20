# Preregistration: X10 (lagged leash), X5 (step limit) and X12 (pairwise LLM judge), written 2026-10-19 before any run

Approved by Fan on 2026-10-19 (translated: "2: run X10 + X5; 3: choose a + b; 4: as recommended; do X12"). Predictions are frozen in `predictions.json` (sha256 prefix recorded in the log). The narrative they serve (v4 spine plus a section on where cycles come from) was adopted the same day.

## Testbed
As in `experiments/2026-10-18-dpo-inner/`: script `panel_po_ctrl.py` (unchanged; the lag is its `--nu` option, reference r_t = (α − ν)s_t + ν s_{t−1} + (1 − α)s_ref), Qwen2.5-0.5B-Instruct, full fine-tuning, the 64-panel subset, 40 inner Adam steps at lr 2e-5, fp32. Batch file `vm/batch_x10_x5.json`, two runs at a time.

## X10: does the lagged leash reach a cycling target faster? (Part II, lag theorem)
- Planted oracle, IPO, λ = 1, B = 12; maximal lag ν = α at α = 0.4 (β = 7.2) and α = 0.45 (β = 6.6); 80 rounds.
- Measure: KL(π_t ‖ π*) to the B = 12 population target computed from the run's own initial scores (identity link). For each run, the first round at which the median over the 32 balanced panels falls below 0.01 and stays below for the rest of the run.
- Reference: the v4 run H6_B12_a08 (ordinary refresh α = 0.8, β = 2.4, same target), statistic 17 in the storyline analysis; it is recomputed by the same evaluation script.
- Population predictions (balanced, from accel_panels.py): 4 and 5 rounds (ordinary α = 0.8: 11); 90th percentile 7 and 6; below 1e-3 at 7 and 8 (ordinary 19).
- Pass: both X10 runs at most half the reference statistic (at most 8 if the reference is 17).

## X5: does refresh-then-anchor reach the target in about B/β̄ rounds? (Part II, T5)
- Reward-model panels, soft-label DPO (Skywork-8B, temperature 3.168), λ = 0.8, B = 8, as in E3′.
- Step caps β̄ = 0.5 and 2. Refresh-then-anchor (α = 1 until each panel's drift reaches its target drift D*, then α = 1 − β̄/8) against a constant anchor (α = 1 − β̄/8 from round 1).
- Measure (the plan's): the first round at which the median KL to the B = 8 target (logit link, λ = 0.8, run's own initial scores) is below 1e-3.
- Frozen population predictions (`reviews/2026-10-18-jmlr-review/theory/schedule_predictions.json`): 15 vs 37 rounds at β̄ = 0.5, 4 vs 9 at β̄ = 2. Already observed at β̄ = 1: 8 vs 16.
- Pass: for each β̄, refresh-then-anchor within 2 rounds of ⌈8/β̄⌉ (16 and 4) and at least 1.8× sooner than the constant anchor.

## X12: does a pairwise LLM judge produce candidate sets without a Condorcet winner?
- Judge: Qwen2.5-7B-Instruct, downloaded from Hugging Face to the VM (approved with X12). Script `vm/llm_judge.py`, bf16, greedy next-token probabilities over the letters A and B.
- Panels: the 130 v4 panels (primary) and the 25 style panels of X11 stage 1b (secondary). Every ordered pair (12 per panel) is judged; order-averaged labels p_ij = (q_ij + 1 − q_ji)/2.
- Rule: if at least 10% of the 130 v4 panels lack a Condorcet winner under the order-averaged labels, the paper's "How fast / whether" part is moved forward and given more weight, and a neural follow-up is proposed (not run without approval). Otherwise the share enters the prevalence table and the structure stays.
- Reported: no-Condorcet-winner share; majority three-cycles; logit cyclic share; position bias; the share of top tokens that are a letter; single-order variants; population DPO oscillation under the X11 recipes A/B/C.

## Decision rule
As in the earlier batches: passes are reported; failures are diagnosed and reported; any change is logged here with a timestamp before a rerun. The VM is deallocated after the batch.

## Log
- 2026-10-19: predictions frozen, predictions.json sha256 prefix dd62875cb77c.
- **2026-10-19 14:40 EDT, X12 scored.** Judge pass on the VM 14:24–14:30 EDT (download 2 min, 130 + 25 panels in 3.3 min). Data `data/llm_judge.json` (sha256 prefix e240cd75d146361f). Script `analysis/evaluate_x12.py`, output `results/x12_evaluation.json`. Every top token was a letter.
  - Rule (v4 panels, order-averaged labels): 4 of 130 panels (3.1%) lack a Condorcet winner, below the 10% threshold, so the rule is **not triggered**. Majority three-cycle 2.3%.
  - Labels: mean logit cyclic share 0.141 (48.5% of panels above 0.1), mean |p − ½| 0.33, sign agreement with the Skywork-8B ranking only 0.37. Position bias q_ij + q_ji − 1 = −0.10 (slot B favoured). Single-order labels: 8.5% and 6.2% of panels without a Condorcet winner.
  - Style panels (secondary): 1 of 25 without a Condorcet winner.
  - **Solver correction (post hoc, before this entry).** The first evaluation predicted population DPO oscillation under recipe A (fixed reference, β = 10) on 73 of 130 panels. Diagnosis (`analysis/diagnose_x12.py`): on those panels the fixed point is a vertex (top mass ≈ 1), the Condorcet winner's margin is large (median 0.43), and the reported multipliers were real (≈ 5), which T1(3) rules out at λ = 1. The shared solver `dpo_map.bt_fit` solved the μ-weighted Bradley–Terry equations through a Hessian with entries μ_iμ_j and left the low-mass coordinates unsolved near vertices (moment residual up to 0.92). It was replaced by Newton on the row-scaled equations (residual ≤ 2e-16; identical to 1e-13 on interior points; old version kept as `bt_fit_lstsq`, file backup `dpo_map.prev-2026-10-19.py`). With it: recipe A 3 of 130 (2.3%), B 3 (2.3%), C 2 (1.5%); style panels A 2 of 25. The earlier numbers are kept in `results/x12_evaluation.oldsolver.json`.
  - Impact on earlier results, rechecked with the fixed solver: X1 population predictions unchanged (trajectory difference 2e-12 and 5e-11; shares 0.00 and 0.28; `experiments/2026-10-18-dpo-inner/results/recheck_solver.json`); mixed-criteria theory checks unchanged (99 of 99, 16 of 16, toy table identical); X11 stages unchanged except one stage-1 single-criterion panel (0.8% → 0%) that had been a solver artefact (`experiments/2026-10-19-mixed-criteria/results/recheck_*.json`).
- **2026-10-19 15:05 EDT, X10 scored: FAIL.** Both runs ended at 15:00 EDT (exit 0); copied to `results/runs/`. Script `analysis/evaluate_x10_x5.py`, output `results/evaluation_x10_x5.json`; predictions file hash unchanged (dd62875cb77c).
  - The reference statistic recomputed by the same script is 17 for H6_B12_a08, as in the storyline analysis, so the limit is 8.
  - Median KL over balanced panels first stays below 0.01 at round 11 (X10_a04) and 8 (X10_a045). Population predictions were 4 and 5. a045 meets the limit and a04 does not, so the preregistered criterion (both) **fails**.
  - Secondary: neither run nor the ordinary run reaches a median of 1e-3 to stay. Floors over the last 20 rounds are 0.0012 and 0.00086, against 0.0017 for ordinary refresh. Decisive panels stay below 0.01 at round 5 and 6, against 10.
  - Post hoc (`analysis/diagnose_x10.py`): the lag is faster early. Median KL at round 3 is 0.047 and 0.027 against 0.128 for ordinary; at round 5, 0.024 and 0.019 against 0.067; at round 10, 0.011 and 0.0045 against 0.026. But it plateaus near 0.005 between rounds 10 and 20, where the population map keeps falling. The lag runs take a step three times larger at the same target (β = 7.2 and 6.6 against 2.4), so their per-round training error is larger. That is consistent with T2's bias–variance trade-off, but not tested. a04 sits close to the lag's population stability edge (α ≤ 0.3 never converges). No rerun without approval.
- **2026-10-19 15:50 EDT, X5 scored: PASS.** The last runs ended at 15:43 EDT (exit 0); the VM was deallocated right after copying. Output: `results/evaluation_x10_x5.json`.
  - Step cap 0.5: refresh-then-anchor reaches median KL < 1e-3 at round 15 (goal 16, population 15), against 33 for the constant anchor (population 37). Ratio 2.2.
  - Step cap 2: 4 (goal 4, population 4) against 8 (population 9). Ratio 2.0.
  - Both within 2 rounds of ⌈8/β̄⌉ and at least 1.8× faster, so the preregistered criterion passes. Floors over the last 10 rounds: 2e-5 to 8e-5.
  - VM use for this batch: 14:20–15:44 EDT plus the X12 download; today's total over three passes is about 2.1 GPU-hours, about $15.
