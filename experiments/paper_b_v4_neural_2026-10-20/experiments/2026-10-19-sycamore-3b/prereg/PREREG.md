# Preregistration: Qwen2.5-3B dynamics on Sycamore (written 2026-10-19, before any 3B training run)

## Approval
Fan approved on 2026-10-19. Translated: "You can first run a 3B-model experiment on Sycamore using 2 GPUs. Start with the most basic experiment that compares dynamics under different parameters: choose different alpha, beta and gamma on the same model and dataset, and observe response probabilities oscillating, converging and polarizing."

## Setup
**Model and training.**
- Qwen2.5-3B-Instruct, full fine-tuning, fp32 master weights with bf16 autocast.
- 40 fresh Adam steps per round at lr 2e-5. Micro-batches of 32 sequences: the 128-sequence micro-batch of the 0.5B runs ran out of memory. Gradients accumulate, so the objective is unchanged.
- Script `src/panel_po_steer.py`. It is the X14 script with `PANEL_CH` configurable; the X14 options are off.
- Slurm array `slurm/train_array.sbatch`: 5 tasks, one GPU each, at most 2 at a time, account `rc_fanyao_pi`, partition `h100_all`.

**Panels.** The 3B model's own panels.
- Built by `build_panels.py` from the v4 run, with the same prompt pool, K = 4, M = 48, min_prob 0.05, seed 0.
- 485 of 600 prompts kept; the training subset is panels 0–63.
- The first attempt reused the 0.5B panels. On them the 3B's initial distribution is near-degenerate (mean top probability 0.94), and the population model then predicts no oscillation at all. Those panels were therefore not used; nothing was trained on them.

**Labels.**
- Skywork-Reward-Llama-3.1-8B-v0.2 at revision d4117fbfd81b. Soft labels σ(gap/T) with T = 3.9643, the standard deviation of within-panel gaps over all 485 panels. Hard labels 1[gap > 0].
- Planted cycles from `make_planted.py`: the same construction and seed as v4. The training subset has 36 balanced and 28 decisive panels.

**Predictions.** `prereg/predictions_3b.json`, sha256 8e7d0dddfcad438f24d1dde97c7ef32b851ebe8a030da9a060c92cda71c92340.
- Computed by `analysis/predict_3b.py` with the exact population recursion.
- Starting point: the 3B's exact initial scores on the training panels (scoring job 4860495, `--rounds 0`).

## Runs, predictions and criteria

| Run | 0.5B counterpart | Labels, α, β, λ, rounds | Regime | Prediction | Pass |
|---|---|---|---|---|---|
| D3B_conv_rm | H1_a08_b16 | IPO soft, 0.8, 1.6 (B = 8), 0.8, 30 | converge to a graded mix | end entropy 0.968 (= target) | \|H − 0.968\| ≤ 0.05 and median KL(target ‖ policy) ≤ 0.02 |
| D3B_pol_full | H3_a1_b1 | IPO hard, 1.0, 1.0, 1.0, 40 | polarize on the Condorcet winner | mean mass on the Condorcet winner 1.000, entropy 0 | mean mass on the Condorcet winner ≥ 0.9, and it has the largest mass on ≥ 90% of panels |
| D3B_osc_fast | H6_B12_a02 | planted, 0.2, 9.6 (B = 12), 1.0, 80 | oscillate | balanced panels oscillating 100% (decisive 18%); median balanced γ_B 1.92, α_loc 0.57 | balanced share ≥ 0.85 |
| D3B_conv_pl | H6_B12_a08 | planted, 0.8, 2.4 (B = 12), 1.0, 80 | converge at the same target | 0% oscillating; KL to the target ≈ 0 | balanced share ≤ 0.15 and median balanced KL(target ‖ policy) ≤ 0.02 |
| D3B_osc_slow | H5_b4_a09 | planted, 0.9, 4.0 (B = 40), 1.0, 80 | slow oscillation | balanced 100%, decisive 0% (γ_B 6.6, α_loc 0.96 > 0.9) | balanced share ≥ 0.85 |

**Directional criterion.** At the same target B = 12, D3B_osc_fast oscillates on a larger balanced share than D3B_conv_pl.

**Oscillation rule.** The v4 rule: detrended amplitude over the last 30 rounds above 0.0536.

**Failures.** Reported with diagnosis. Any change is logged here before a rerun.

## Cost
- About 59 s per round per GPU (timing job 4860479): 320 rounds in all, about 5.3 GPU-hours.
- Wall time about 2.7 hours at 2 GPUs.

## Log
- 2026-10-19: environment job 4860468 (torch 2.13.0+cu130, transformers 5.13.1).
- 2026-10-19: timing job 4860476 ran out of memory; rerun as 4860479.
- 2026-10-19: panel job 4860484; scoring job 4860495.
- This file was written after the scoring job and before the training array.
- 2026-10-20 01:20 EDT: training array 4860498 finished (all five tasks COMPLETED; elapsed 1:36:48, 1:38:34, 1:36:58, 0:37:14, 0:49:27; 6.3 GPU-hours; peak memory 70.8 GiB; about 72 s per round). Runs and logs fetched to `results/`.

## Results (scored 2026-10-20 with `reviews/2026-10-19-x14-and-3b/analysis/d3b_eval.py`; `results/evaluation_3b.json`)

| Run | Prediction | Result | Pass |
|---|---|---|---|
| D3B_conv_rm | end entropy 0.968 | 0.968; median KL to target 0.0013 | PASS |
| D3B_pol_full | mass on the Condorcet winner 1.000 | 1.000; largest on 100% of panels | PASS |
| D3B_osc_fast | balanced 100% (decisive 18%) | 100% (decisive 18%); median amplitude 0.314 | PASS |
| D3B_conv_pl | 0% | 0%; median balanced KL 0.0015 | PASS |
| D3B_osc_slow | balanced 100% | 100%; median amplitude 0.306 | PASS |

Directional criterion (osc_fast > conv_pl at B = 12): PASS. Round losses (`results/round_losses_3b.json`): reconstruction within 3.9e-6 of every logged first-step loss.
