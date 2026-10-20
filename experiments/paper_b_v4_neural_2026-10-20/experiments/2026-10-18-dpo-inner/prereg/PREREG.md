# Preregistration: X1 (actual DPO on planted cycles) and X9 (inner steps), written 2026-10-18 before any run

Approved by Fan on 2026-10-18 (translated: "do step 1 of the text first, then run X1 and X9"). Predictions are frozen in `predictions_x1_x9.json` (sha256 prefix recorded in the log below); the theory is T1 and T6 in `reviews/2026-10-18-jmlr-review/theory/theorems.md`, and the prediction sources are `dpo_sweep.json`, `inner_steps.json` and `onset_predictions.json` (copied here).

## Testbed
As in `experiments/2026-10-17-controls/` (script `vm/panel_po_ctrl.py`, unchanged): Qwen2.5-0.5B-Instruct, full fine-tuning, the 64-panel subset (32 balanced, 32 decisive planted panels), expected labels from the planted oracle, fresh Adam optimizer each round at lr 2e-5, fp32.

## X1: which map does actual DPO follow on intransitive labels? (T1)
- Runs: DPO with soft planted labels, λ = 1, 120 rounds, 40 inner steps: X1_B4_a08 (α = 0.8, β = 0.8, B = 4) and X1_B2_a03 (α = 0.3, β = 1.4, B = 2).
- Predictions (share of balanced panels oscillating): actual population DPO 0.00 and 0.28; logit-link model 0.78 and 0.75.
- Pass: in both runs the share is within 0.15 of the DPO prediction and at least 0.35 from the logit prediction. Oscillating means a detrended probability amplitude above 0.0536 over the last 30 rounds (the v4 rule).

## X9: does a coarser inner solver stabilize the same target? (T6)
- Runs: IPO, λ = 1, α = 0.5, β = 6 (B = 12, unstable for exact rounds), 150 rounds, inner Adam steps k = 10, 3 and 1. The k = 40 reference is the v4 run H6_B12_a05 (share 0.94).
- Ordinal prediction: the oscillating share is non-increasing as k falls, and at least one k gives a share of 0.2 or less.
- Calibrated prediction: measure the fraction f of the exact round move achieved per round (rounds 1–5, median over balanced panels), set ρ_k = 1 − f and α_eff = 0.5 + 0.5ρ_k (T6 at a uniform target), and read the predicted share from the population sweep at B = 12 (`onset_predictions.json`, linear interpolation). Reported with its error; not a pass criterion, because Adam is not gradient descent.
- Target check: for every run that has stopped drifting (mean per-round step over the last 30 rounds below 0.01), the policy averaged over the last 30 rounds is within median KL 0.02 of the B = 12 target (fixed points unchanged).

## Decision rule
As in the earlier batches: results that pass are reported; failures are diagnosed and reported; any change is logged here with a timestamp before a rerun.

## Log
- 2026-10-18 17:38 EDT: predictions frozen, predictions_x1_x9.json sha256 prefix ead01ee52b8d.
- 2026-10-18 18:10 EDT, before any run finished and before any result was read: two quantities the X9 target check leaves unspecified are fixed as follows.
  - "Mean per-round step" is the mean, over balanced panels and the last 30 transitions, of max_i |p_{t+1,i} − p_{t,i}| in panel probabilities. Probabilities are used rather than logits because answers of vanishing probability have wandering logits.
  - "Median KL" is KL(time-averaged policy ‖ target), with the median over balanced panels, as in the C6 controls run. Decisive panels are reported alongside.
  - Population curves for figures start from each run's own initial scores. Two extra diagnostics are reported as post hoc and are not scored: the one-step error of the DPO and logit maps applied to the model's own state (X1), and the per-panel agreement between neural and population classifications.
- 2026-10-18 22:43 EDT, scored (batch ended 18:43 EDT, exit 0 for all five runs; VM deallocated after copying). Script `analysis/evaluate_x1_x9.py`, output `results/evaluation.json`; the predictions file hash still has prefix ead01ee52b8d.
  - **X1: PASS.**
    - Share of balanced panels oscillating: 0.094 at B = 4, α = 0.8 (DPO prediction 0.00, logit prediction 0.78) and 0.313 at B = 2, α = 0.3 (DPO 0.28, logit 0.75).
    - Post hoc: per-panel agreement with the DPO map 0.91 and 0.91, with the logit map 0.31 and 0.56. Median one-step error: DPO map 0.080 and 0.074, logit map 0.125 and 0.190.
  - **X9: ordinal prediction FAILS.**
    - Balanced share oscillating: 0.94 / 0.88 / 1.00 / 0.94 at k = 40 / 10 / 3 / 1. It is not monotone and never reaches 0.2.
    - Decisive panels oscillate more as k falls: 0.03 / 0.13 / 0.38 / 0.63.
    - Calibrated: α_eff = 0.51 / 0.59 / 0.69 / 0.39, predicted shares 0.87 / 0.26 / 0.00 / 0.94. Wrong for k = 10 and k = 3.
    - Target check: no run stopped moving (mean step 0.20–0.28 ≥ 0.01), so none is scored. Reported: the 30-round time-averaged policy is within median KL 0.0003 / 0.0007 / 0.0031 / 0.0064 of the B = 12 target.
  - **Diagnosis (post hoc, `analysis/diagnose_x9.py`).** With a fresh Adam optimizer each round, few steps do not take a fixed fraction of the exact move. The log-log slope of the achieved move against the exact move (probability-weighted norm, balanced panels) is 0.98 / 0.59 / 0.27 / 0.38. At k = 3 and 1, the fraction f is about 1.1 when the exact move is small (overshoot near the target) and 0.06–0.08 when it is large. A normalized step that does not shrink near the target keeps every panel moving, including decisive ones. T6 assumes gradient descent; its prediction does not carry over to fresh-Adam inner loops. No rerun is made without approval.
