# Preregistration: neural tests of feedback controls for collapse and oscillation (written 2026-10-17, before any training run)

Fan approved the runs on 2026-10-17 ("go ahead with 1 and 2", translated from Chinese), at about 5–6 VM hours and about $40. The theory and simulations are in `reviews/2026-10-17-collapse-oscillation-control/` (Propositions 1–5, `sims/`). The frozen numbers are in `prereg/predictions_controls.json`, computed by `analysis/predict_controls.py` with the exact population recursion on the same 64-panel subset (ids in the JSON). Its timestamp precedes every run.

## Testbed

Same as the v4 validation (`experiments/2026-10-16-v4-validation/`):
- Qwen2.5-0.5B-Instruct, full fine-tuning, the theory-matched round with expected labels, and 40 Adam steps at learning rate 2e-5 from a fresh optimizer.
- The 64-panel subset used since batch 2.

The script `vm/panel_po_ctrl.py` adds the controls; all are off by default, so the defaults reproduce `panel_po.py`.

**Reward-model runs (E1–E3).** DPO with soft Bradley–Terry labels from Skywork-8B (T = 3.168) and λ = 0.8.

**Planted-cycle runs (E4–E6).** IPO with the planted oracle and λ = 1.

Measures are those of v4:
- mean panel entropy;
- KL to population targets;
- the detrended probability amplitude over the last 30 rounds. A panel is oscillating when it exceeds 0.0536.

## Hypotheses, predictions and pass criteria

**E1: a judge biased toward the current policy is hidden refresh (Proposition 1).**
- Setup: label log-odds get + ε(s_i − s_j). α = 0.5, β = 1, 60 rounds.
- Runs: ε ∈ {0, 0.3, 0.45, 0.55} on-policy, plus ε = 0.55 from a frozen judge (bias on s_0).
- Predicted end entropy: 0.638 / 0.301 / 0.066 (ε = 0, 0.3, 0.45) and 0.527 (frozen 0.55).
- Predicted median top-two log-odds gap at ε = 0.55 (on-policy): 44 at t = 30, 235 at t = 60, so no target.
- Pass:
  - |H_t60 − prediction| ≤ 0.05 for the four converging runs;
  - end entropy ordered ε = 0 > 0.3 > 0.45;
  - for ε = 0.55 on-policy, the gap at t = 60 is ≥ 1.2 × the gap at t = 30, and H_t60 < 0.05.

**E2: entropy bonus vs anchor (Proposition 2).**
- Setup: full refresh (α = 1) with a popularity penalty (ε = −η), η ∈ {0.125, 0.25}, against partial refresh at B = 8 (α = 0.875) and B = 4 (α = 0.75). β = 1, 60 rounds.
- Predicted (entropy, KL to the initial policy):

  | Run | Entropy | KL to initial |
  |---|---|---|
  | η = 0.125 | 0.183 | 1.605 |
  | B = 8 | 0.196 | 1.255 |
  | η = 0.25 | 0.418 | 1.374 |
  | B = 4 | 0.403 | 0.936 |

- Pass:
  - each run's end entropy within 0.05 of its prediction;
  - KL to the initial policy larger for the bonus run than for its matched anchor run, in both pairs.

**E3: budget on drift vs per-round KL control (Proposition 4).**
- Setup: full refresh. The first run uses a per-panel drift budget equal to the drift of that panel's B = 8 target, with κ = 2. The second uses a PPO-style per-round KL controller (target 0.02, β₀ = 1). 60 rounds.
- Predicted:
  - drift run: median KL to the B = 8 target 5e-5, entropy 0.203 (target 0.196);
  - per-round run: end entropy 6e-5.
- Pass:
  - drift run: median KL to the B = 8 target ≤ 0.02, and mean entropy within 0.05 of the target entropy;
  - per-round run: H_t60 < 0.05.

**E4: replay cuts both ways (Proposition 5).**
- Setup: planted panels at B = 12, replay weight ρ = 0.2, 150 rounds. Runs at α = 0.5 (β = 6) and α = 0.8 (β = 2.4).
- Predicted oscillating share of balanced panels: 0.00 at α = 0.5 (0.91 without replay) and 0.53 at α = 0.8 (0.00 without replay). The α = 0.8 prediction sits near the threshold (median amplitude 0.056).
- Pass:
  - α = 0.5: share ≤ 0.2 (v4 neural at ρ = 1: 0.94);
  - α = 0.8: share ≥ 0.25 (v4 neural at ρ = 1: 0.00).

**E5: adaptive refresh at a fixed target.**
- Setup: planted panels, B = 12, starting at α = 0.2. Every 10 rounds, if the mean policy step did not fall by 20%, α moves 30% of the way to 1 (cap 0.95) and β = (1 − α)B. 150 rounds.
- Predicted: α path 0.2 → 0.44 → 0.61 → 0.73, then convergence.
- The neural optimization-noise floor may keep triggering the rule, so α may end higher than predicted. That is allowed.
- Pass:
  - final α ≥ 0.6;
  - oscillating share of balanced panels ≤ 0.1;
  - median KL to the B = 12 target ≤ 0.02.

**E6: deploying the average with weights.**
- Setup: rerun α = 0.2, β = 9.6 (B = 12), 80 rounds, keeping the running average of the weights after rounds 51–80.
- Known: the policy average over the last 30 rounds is within KL 0.0077 of the target (v4 neural); the population model gives 0.0074.
- Pass: the weight-averaged model's balanced-panel median KL to the target is ≤ 2× that of the same run's 30-round policy average.

## Decision rule

The decision rule follows v4:
- Results that pass are reported.
- A failure is diagnosed (optimization residual, noise floor, interference, a measurement artifact). Any change is written here with a timestamp before a rerun, and failed runs are kept.
- Whatever enters the paper goes to Fan as a report first.

## Log

- **2026-10-17.** Predictions frozen (`predictions_controls.json`).
- **2026-10-18 00:21 UTC: batch finished.** 15 runs; scores in `results/eval_controls.json`.
  - **Pass (E1, E2, E4, E5).**
    - E1: end entropies 0.638 / 0.300 / 0.061 (pred. 0.638 / 0.301 / 0.066). ε = 0.55 collapses: entropy 0.007, top-two gap 29 → 39. The neural gap grows much more slowly than the population gap (235); the criterion still passes. Frozen judge: 0.525 (pred. 0.527).
    - E2: entropies within 0.003. KL to the initial policy 1.60 vs 1.26 and 1.38 vs 0.93 (pred. 1.61 / 1.25, 1.37 / 0.94).
    - E4: balanced oscillating share 0.00 at α = 0.5 and 0.59 at α = 0.8 (pred. 0.00 / 0.53).
    - E5: α 0.2 → 0.44 → 0.61 → 0.73 → 0.91 (noise floor), oscillating share 0, KL 0.0009.
  - **Fail (E3).**
    - The per-round KL controller collapses as predicted (entropy 6e-6).
    - The drift-budget run puts the median panel at the target (KL 0.008 ≤ 0.02), but mean entropy is 0.095 against 0.196, so the run fails.
    - 16 of 64 panels overshoot their budget (drift/D* median 2.3, max 14).
  - **E3 diagnosis.** The step-only controller has no restoring force.
    - By round 60, 77% of panels have β = 0, yet they keep drifting through shared parameters.
    - DPO with a very small β (coefficient 1/β) produces exploding losses (step-40 loss up to 3,296; 52 rounds above 10) that push other panels.
  - **Fail (E6), refuted with diagnosis, no rerun.** Balanced-panel median KL to the target:
    - last iterate 0.479;
    - arithmetic policy average (last 30 rounds) 0.008;
    - geometric (log-probability) average 0.016;
    - weight average 0.052, i.e. 6.5× the policy average (criterion ≤ 2×).

    Averaging the weights of a nonlinear network reproduces neither average: KL(weight average, geometric average) = 0.031. The arithmetic policy average can itself be deployed by serving a uniformly random checkpoint from the last 30 rounds per request.
- **2026-10-18, condition change for E3 (decided before the rerun; Fan approved the rerun).**
  - New controller `--beta_ctrl switch`: full refresh with β = 1 until a panel's drift KL(π_t ‖ π_0) reaches its budget D*_n (the drift of its B = 8 target). From then on the panel uses partial refresh with α = 0.875, which has the same B = 8 target and pulls the policy back toward it. The switch is latched.
  - Frozen population prediction (`prereg/predictions_switch.json`): median KL to the B = 8 target ≈ 0, entropy 0.196 (target 0.196), no panel above KL 0.1, switch at round 9 (median).
  - Pass criterion: the frozen E3 criterion (median KL ≤ 0.02, entropy within 0.05). Panels with KL > 0.1 are also reported (16 of 64 before).
- **2026-10-18 02:55 UTC: E3′ (C3b_switch) finished, PASS.** 709 s on the VM; the VM was deallocated right after. Scores in `results/eval_controls.json` (`E3_switch`).
  - Median KL to the B = 8 target 3.6e-5 (criterion ≤ 0.02; pred. ≈ 0). Mean 4.1e-4, 90th percentile 1.1e-3, max 4.4e-3.
  - Mean entropy 0.197 against a target entropy of 0.196 (criterion within 0.05; pred. 0.196).
  - Panels with KL > 0.1: 0 of 64 (pred. 0; the step-budget run C3_drift had 16).
  - Switch round median 9, range 6–13 (pred. median 9). Every panel switched.
  - No loss blow-up: the largest step-40 loss is 0.52 (C3_drift: 3,296).
  - E3 as a whole now passes with the changed condition. The step-budget controller remains a reported failure.
  - **Post hoc, not preregistered.** The end state equals the plain B = 8 anchor (C2_anchorB8, KL to the initial policy 1.26 in both runs), but it arrives about twice as fast: median KL to the target first falls below 0.001 at round 8 against round 16 for the anchor run. The population model gives 8 vs 18.
