# Preregistration: X14, steering the loop with P-free diagnostics (written 2026-10-19, before any X14 training run)

## Approval and purpose
Fan approved the practical-diagnostics report on 2026-10-19. His words, translated: "I approve this report's route. You can run experiments to support these tests. We need experimental results showing that such interventions can avoid possible non-convergence or over-sharpening without knowing P."

X14 tests the closed loop: an intervention chosen only from P-free diagnostics, applied to the trained model, should avoid three failures:
- collapse from hidden refresh;
- over-sharpening under full refresh;
- rotation under cyclic labels.

The diagnostics are those of `reviews/2026-10-19-practical-diagnostics/text/report.html` (D1–D5).

## What the controller may read
**May read:**
- the policy's own log-probabilities on the training panels;
- sampled judgments. These are Bernoulli draws from the run's label source, standing in for judge calls; the synthetic judges have no order effect.

**Never reads:** the preference matrix *P*.
- Pre-training decisions are computed in `analysis/predict_x14.py` from seeded samples.
- Online decisions are taken by `vm/panel_po_steer.py`.
- Only the scoring below uses *P*, through the population recursion.

## Testbed
Same as the controls and X5 (`experiments/2026-10-17-controls/`):
- Qwen2.5-0.5B-Instruct, full fine-tuning;
- 40 inner Adam steps at learning rate 2e-5 from a fresh optimizer, fp32;
- the same 64 panels, one seed.

`vm/panel_po_steer.py` is `panel_po_ctrl.py` plus two rules; all X14 options are off by default.

**Reward-model scenarios (S1, S2).**
- Soft-label DPO with the Skywork-8B labels (T = 3.168), λ = 0.8.

**Cycle scenarios (S3).**
- IPO with the planted oracle, λ = 1.

**Oscillation** is the v4 rule: detrended amplitude over the last 30 rounds above 0.0536.

## Scenarios, arms, decisions and predictions
The frozen numbers are in `prereg/predictions.json`, sha256 e860d847e0fca7ae5cec5ae518d27bc81cda79ac6ec260c14170e2c1346ef35c.

### S1: hidden refresh from a judge that follows the model
**Planned recipe.** α = 0.5, β = 1, so B = 2. The planned end entropy at ε = 0 is 0.638.

**The judge.** It adds ε(s_i − s_j) of the current policy to every label's log-odds. ε is unknown to the controller.

**Naive baselines (existing runs).**
- C1_e045: end entropy 0.061.
- C1_e055: collapses to 0.007, with the top-two gap still growing.

**Rule (D5, online).**
1. At the start of round 2, judge all 6 pairs of all 64 panels 16 times at s_0 and again at s_1. That is 12,288 calls.
2. Estimate ε by the conditional likelihood.
3. Act only if |ε̂| > 1.96 SE.

**Arms.**
- **X14_db45, X14_db55 (debias).** ε = 0.45 and 0.55. From round 2 on, the labels' log-odds get −ε̂(s_i − s_j) of the current policy.
  - Prediction: end entropy at the planned 0.638 when ε̂ = ε.
  - With ε̂ = ε ± 0.05: 0.589 / 0.683.
  - With ε̂ = ε ± 0.10: 0.538 / 0.724.
- **X14_cp55 (compensate).** ε = 0.55. Refresh is lowered so that α + βε̂ = 0.5:
  - α = 0.5 − ε̂ if that is ≥ 0;
  - otherwise α = 0 and β = 0.5/ε̂.

  The theory then predicts a finite target, sharper than planned because the reference is amplified.
  - Prediction: end entropy 0.560 at ε̂ = ε.
  - With ε̂ = ε ± 0.05: 0.492 / 0.617.
  - The top-two log-odds stop moving.

**Pass criteria.**
- **Pass S1-a, each debias arm.**
  - D5 acts.
  - End mean entropy within 0.05 of the population prediction at the logged ε̂ (grid in `predictions.json`, linear interpolation).
  - End mean entropy within 0.10 of the planned 0.638.
- **Pass S1-b, compensation arm.**
  - D5 acts.
  - End mean entropy within 0.05 of the prediction at the logged ε̂.
  - Converged: the median over panels of the mean |change in top-two log-odds| over the last 10 rounds is ≤ 0.05 nats per round.
- **Pass S1-c.** All three arms end with mean entropy ≥ 0.3, against 0.061 and 0.007 for the naive runs.

### S2: over-sharpening under full refresh
**Budget.** Mean panel entropy at least 0.3, stated in observable terms.

**Arms.**
- **X14_fr (naive).** Full refresh α = 1, β = 0.5, 60 rounds.
  - Predicted end entropy: 0.031.
  - Pass S2-a (baseline confirmed): end entropy < 0.06.
- **X14_sw (switch, online, no judge calls).** Same recipe until the mean entropy of the current policy first falls to 0.3 or below. Then every panel switches to α = 1 − β/B_t, with B_t = β × (rounds completed), and runs to 60 rounds.
  - Predicted switch: round 13, B = 6.
  - Predicted target entropy: 0.274.
- **X14_cen (census before training).**
  1. Take 36 judgments per prompt (3 per ordered pair).
  2. Fit Bradley–Terry with ridge 0.1.
  3. Choose B̂ = 5.552, whose estimated mean entropy is 0.3.
  4. Train a fixed reference α = 0 with β = B̂ for 40 rounds.
  - Predicted end entropy (true labels): 0.298.

**Pass criteria.**
- **Pass S2-b, switch arm.**
  - The switch fires by round 20.
  - End mean entropy within 0.05 of the target entropy at the logged B_t.
  - Median KL(target ‖ policy) at the end ≤ 0.01.
  - End mean entropy ≥ 0.25.
- **Pass S2-c, census arm.**
  - End mean entropy within 0.05 of 0.298.
  - End mean entropy ≥ 0.25.

### S3: rotation under cyclic labels
**Rule (D3, before training).**
1. Take 3 judgments per ordered pair, 36 calls per prompt.
2. Run the noise-corrected gain census at the planned B, using the model's initial log-probabilities.
3. Set α = min(0.95, max over panels of α_loc(γ̂) + 0.05) and β = (1 − α)B.

**Arms.**
- **X14_c12, planned B = 12.**
  - Census: 47% of panels flagged (true γ > 1: 59%).
  - Largest α̂_loc 0.825 (true 0.623), giving α = 0.875 and β = 1.502. 80 rounds.
  - Naive baseline: H6_B12_a02 (α = 0.2), with 100% of balanced panels oscillating.
  - Prediction: 0% of balanced panels oscillating; KL to the target about 0.
- **X14_n24, planned B = 24 (naive).** α = 0.5, β = 12, 80 rounds. No earlier run.
  - Prediction: 100% of balanced panels oscillating.
- **X14_c24, planned B = 24 (census).**
  - Largest α̂_loc 0.945 (true 0.890), giving α = 0.95 (the cap) and β = 1.2. 80 rounds.
  - Prediction: 0% oscillating; median KL to the target on balanced panels 0.0009.

**Pass criteria.**
- **Pass S3-a (X14_c12).**
  - Oscillating share of balanced panels ≤ 0.10.
  - Median KL(target ‖ policy) at round 80 on balanced panels ≤ 0.01.
- **Pass S3-b (X14_n24, baseline confirmed).** Oscillating share of balanced panels ≥ 0.75.
- **Pass S3-c (X14_c24).**
  - Oscillating share of balanced panels ≤ 0.10.
  - Median KL on balanced panels ≤ 0.02. This is looser because slow refresh approaches slowly.

### Benign check (no training)
The same rules must not fire where no risk exists:
- **Gain census.** On the reward-model labels at the S1 sharpness, it flags 0 of 64 panels (largest γ̂ 0.77).
- **D5 on the existing ε = 0 run.** It does not act: ε̂ = 0.063 against 1.96 SE = 0.073 for seed 5. Seed 7 gives 0.004.

## Verdict rule
**Supported.** The claim "P-free interventions avoid collapse, over-sharpening and rotation" is supported if two things hold:
- every guided arm meets its "avoid" criterion: S1-c; end entropy ≥ 0.25 in S2-b and S2-c; oscillating share ≤ 0.10 in S3-a and S3-c;
- both new naive baselines fail as predicted (S2-a, S3-b).

**Precision.** The quantitative criteria (within 0.05 of prediction, KL bounds) score how precisely the theory predicts where the steered loop lands.

**Failures.** They are diagnosed and reported. Any change to an arm is logged here, with a timestamp, before a rerun.

## Cost
- 9 runs, 580 outer rounds, 3 in parallel.
- About 1.5–2 VM hours on psipo-h100-1 at $6.98/h, about $10–14.
- The VM is deallocated after the batch.

## Log
- 2026-10-19: written before any X14 run. `predict_x14.py` ran locally; no GPU was used.
- **2026-10-19 23:10 EDT, batch scored.**
  - Run: batch 00:51–02:53 UTC, about 2.1 VM hours (about $15). The VM was deallocated at 23:00 EDT.
  - Scoring: `analysis/evaluate_x14.py`, output `results/evaluation_x14.json`.

  **S1, hidden refresh.** D5 acted in every arm. ε̂ = 0.500 (true 0.45), 0.614 (0.55) and 0.640 (0.55), each 1–1.7 SE above the truth.

  | Arm | End entropy | Predicted at the logged ε̂ | Planned | Last-10 top-two step | Verdict |
  |---|---|---|---|---|---|
  | X14_db45 | 0.683 | 0.683 | 0.638 | – | **PASS** |
  | X14_db55 | 0.696 | 0.694 | 0.638 | – | **PASS** |
  | X14_cp55 | 0.651 | 0.657 | – | 0.0574 nats/round against the preset 0.05 | **FAIL** (convergence threshold only) |

  - The debias arms' last-10 steps are 0.058 and 0.049, so the preset threshold sits at the inner-optimization noise floor.
  - Naive runs: ε = 0.45 steps 0.10 and ε = 0.55 steps 1.40.
  - **S1-c PASS:** all three arms end with entropy ≥ 0.3, against 0.061 and 0.007 for the naive runs.

  **S2, over-sharpening.** All three arms **PASS**.
  - X14_fr: end entropy 0.031 (predicted 0.031).
  - X14_sw: switched at round 12 (predicted 13), B = 5.5. End entropy 0.300 against 0.301 for the target at B = 5.5. Median KL(target ‖ policy) 0.0002.
  - X14_cen: B̂ = 5.55, end entropy 0.297 (predicted 0.298).

  **S3, rotation.**

  | Arm | Balanced panels oscillating | Median balanced KL to target | Verdict |
  |---|---|---|---|
  | X14_c12 (α = 0.875) | 0% | 0.0019 | **PASS** |
  | X14_n24 (naive, α = 0.5) | 100% | 1.31 | **PASS** (baseline confirmed) |
  | X14_c24 (α = 0.95) | 44% (predicted 0%) | 0.009 (≤ 0.02 passes) | **FAIL** |

  For X14_c24 the median amplitude is 0.051 against the threshold 0.0536.

  **Verdict by the preregistered rule: NOT fully supported**, because S3-c failed. Every other "avoid" criterion holds.
- **2026-10-19 23:15 EDT, post hoc diagnosis of X14_c24** (`analysis/diagnose_c24.py`, `results/diagnose_c24.json`).
  - **Slow approach.** Median balanced KL to the target falls monotonically: 0.157, 0.075, 0.031, 0.018, 0.009 at rounds 10, 20, 40, 60, 80. The population gives 0.122, 0.043, 0.011, 0.003, 0.0009.
  - **Decaying amplitude.** The detrended amplitude falls from 0.075 (rounds 21–50) to 0.051 (rounds 51–80). The population's own last-30 amplitude is 0.021: a decaying spiral at α = 0.95 registers under the amplitude rule even in the population model.
  - **Inner noise.** The neural per-round step is 0.022, against 0.015 for X14_c12 and the 0.5B H6 α = 0.9 run.
  - **Reading.** A slow inward spiral that is about 2–3 times slower than the population, not a sustained rotation. The census cap α = 0.95 makes the approach slow. This is a reading after the result, not a preregistered test. Confirming it would need more rounds (not run).
