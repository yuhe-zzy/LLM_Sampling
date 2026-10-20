# Sources: X14 steering and the 3B dynamics runs (2026-10-19/20)

Paths are relative to the Iterative_PsiPO root. `X14` = `experiments/2026-10-19-x14-steer/`, `S3B` = `experiments/2026-10-19-sycamore-3b/`, `AN` = `reviews/2026-10-19-x14-and-3b/analysis/`.

Status: **confirmed** (read from the file or recomputed by the named script), **derived** (arithmetic on confirmed numbers), **inference** (ours).

## X14 design
| id | claim | source | status |
|---|---|---|---|
| D1 | Fan's approval (translated): "I approve this report's route. You can run experiments to support these tests. We need experimental results showing that such interventions can avoid possible non-convergence or over-sharpening without knowing P." | chat 2026-10-19; X14 PREREG "Approval and purpose" | confirmed |
| D2 | Testbed: Qwen2.5-0.5B-Instruct, full fine-tuning, 64 panels, 40 Adam steps per round, lr 2e-5, one seed. S1/S2: soft Skywork-8B labels (T = 3.168), DPO, λ = 0.8. S3: planted cycles, IPO, λ = 1. | X14 PREREG "Testbed"; `X14/vm/batch_x14.json` | confirmed |
| D3 | Decisions frozen before training: S2 census B̂ = 5.5517 (36 calls per prompt, BT ridge 0.1); S3 census α = 0.8748 (B = 12) and 0.95 (B = 24, cap; α̂_loc max 0.945); predictions sha256 e860d847… | `X14/prereg/predictions.json`; X14 PREREG | confirmed |
| D4 | Pass criteria and verdict rule as quoted in the report scorecard | X14 PREREG "Scenarios, arms, decisions and predictions", "Verdict rule" | confirmed |
| D5 | S3 census at B = 12: 47% of panels flagged; true share with γ_B > 1 59%; α̂_loc max 0.825 against true 0.623 | `X14/prereg/predictions.json` (S3) | confirmed |
| D6 | 36 balanced and 28 decisive planted panels among the 64 | `experiments/2026-10-16-v4-validation/data/oracle_planted.npz` (subset ids) | confirmed |

## X14 results (`X14/results/evaluation_x14.json`, `X14/analysis/evaluate_x14.py`)
| id | claim | value | status |
|---|---|---|---|
| R1 | D5 decisions (round 2, 12,288 calls each) | ε̂ = 0.4999 (SE 0.049, true 0.45), 0.6138 (0.053, 0.55), 0.6403 (0.054, 0.55); all acted; cp55 set α = 0, β = 0.7809 | confirmed (`steer.json`) |
| R2 | S1 end entropies (round 60) and predictions at the logged ε̂ | db45 0.6828 / 0.6825; db55 0.6955 / 0.6943; cp55 0.6511 / 0.6566; planned 0.638 | confirmed |
| R3 | Last-10-round median top-two log-odds step | db45 0.0582, db55 0.0487, cp55 0.0574; naive C1_e045 0.1018, C1_e055 1.400 | confirmed |
| R4 | Naive S1 end entropies | C1_e045 0.0610 (pop. 0.066), C1_e055 0.0067 (pop. 0.000) | confirmed (`AN/x14_series.json`) |
| R5 | S2 | fr 0.0308 (pred. 0.0311); sw switch round 12 (pred. 13), B = 5.5, entropy at switch 0.283, end 0.3001 vs target 0.3005, median KL 0.0002; cen end 0.2969 (pred. 0.2976) | confirmed |
| R6 | S3 | c12: 0% balanced oscillating, median balanced KL 0.0019; n24: 100%, KL 1.3117 (pred. 1.394); c24: 43.75% balanced (21.9% all), median amp 0.0506, KL 0.0090 (pred. 0% and 0.0009); naive H6_B12_a02 100% | confirmed |
| R7 | Verdict: not fully supported (S3-c failed); S1-b failed on the step threshold | evaluation_x14.json `verdict_supported` false | confirmed |
| R8 | c24 diagnosis: median balanced KL 0.157, 0.075, 0.031, 0.018, 0.009 at rounds 10, 20, 40, 60, 80 (pop. 0.122, 0.043, 0.011, 0.003, 0.0009); detrended amplitude 0.075 (rounds 21–50) and 0.051 (51–80); pop. amplitude 0.021; per-round step 0.022 (c12 0.015; 0.5B H6 α = 0.9 0.015) | `X14/results/diagnose_c24.json` | confirmed |
| R9 | "2–3 times slower than the population" | ratio of neural to population KL at rounds 20–40 (0.075/0.043, 0.031/0.011) | derived (approximate) |
| R10 | Round losses: reconstruction matches logged first-step losses within 1.74e-6; median excess start → end: fr 0.228 → 0.0152, n24 5.65 → 0.0211; share of rounds ending above their start: c12 0.83, sw 0.82, db45 0.93, db55 0.92, cp55 0.98, c24 0.58; end excess medians 5.4e-4 (db45) to 7.1e-3 (c24) | `X14/results/round_losses_x14.json`; `AN/x14_loss_summary.json` | confirmed |
| R11 | Noise floor of fresh Adam (fixed-size steps overshoot small moves) | MS §9.7 "Fewer inner steps" (X9) | confirmed (earlier result) |
| R12 | X14 cost: batch 00:51–02:53 UTC, VM deallocated 23:00 EDT; about 2.1 VM hours at $6.98/h, about $15 | `X14/results/batch_x14.log`; monitor output | confirmed (cost derived) |
| I1 | c24 is a slow inward spiral, not a sustained rotation | reading of R8, after the result | inference |
| I2 | cp55's step failure reflects the noise floor rather than non-convergence | R3 compared with the debias arms and the naive runs | inference |

## 3B runs (`S3B/results/evaluation_3b.json`, `AN/d3b_eval.py`)
| id | claim | value | status |
|---|---|---|---|
| B1 | Setup: Qwen2.5-3B-Instruct, full FT, fp32 + bf16 autocast, 40 Adam steps lr 2e-5, micro-batch 32; panels 485 of 600, training 0–63, 36 balanced / 28 decisive; T = 3.9643; predictions sha256 8e7d0ddd… | `S3B/prereg/PREREG.md` | confirmed |
| B2 | First attempt on 0.5B panels: mean top probability 0.94, no oscillation predicted; not trained | `S3B/prereg/PREREG.md` | confirmed |
| B3 | conv_rm: end entropy 0.9678 vs pred. 0.9683 (target 0.9679), median KL 0.0013; 0.5B H1_a08_b16 0.7845 vs 0.7820, KL 0.0011 | `S3B/results/evaluation_3b.json` | confirmed |
| B4 | pol_full: mass on CW 1.000, largest on 100%, CW on 100% of panels; 0.5B same | `evaluation_3b.json` | confirmed |
| B5 | osc_fast: balanced 100% (pred. 100%), decisive 17.9% (pred. 17.9%), median amp 0.3145 (pred. 0.3136), γ_B 1.92, α_loc 0.57; 0.5B 100%, 12.5%, 0.3167 | `evaluation_3b.json` | confirmed |
| B6 | conv_pl: 0% (pred. 0%), median KL 0.0015; 0.5B 0%, 0.0016 | `evaluation_3b.json` | confirmed |
| B7 | osc_slow: balanced 100% (pred. 100%), median amp 0.3060 (pred. 0.3026), γ_B 6.61, α_loc 0.955; 0.5B 100%, 0.3065 | `evaluation_3b.json` | confirmed |
| B8 | Directional check osc_fast > conv_pl: PASS | `evaluation_3b.json` | confirmed |
| B9 | Prompt rule (first three balanced ASCII panels), fixed before viewing; prompts 1, 3, 5 | `AN/d3b_eval.py` docstring; `AN/d3b_series.json` | confirmed |
| B10 | Round losses: reconstruction within 3.9e-6; first-step excess conv_rm 0.097 → 7.3e-4, conv_pl 0.393 → 1.08e-3 (last-10 medians); pol_full 0.25, osc_fast 3.5, osc_slow 0.52; end excess last-10 medians 2.3e-3 to 8.3e-3 | `S3B/results/round_losses_3b.json` (`X14/analysis/round_losses_x14.py`) | confirmed |
| B11 | Compute: array 4860498, tasks 1:36:48, 1:38:34, 1:36:58, 0:37:14, 0:49:27 (osc_fast, conv_pl, osc_slow, conv_rm, pol_full); 6.3 GPU-h; peak memory 70.8 GiB; timing estimate 59 s/round | `sacct`; `S3B/results/psipo3b-4860498_*.out`; PREREG | confirmed |

