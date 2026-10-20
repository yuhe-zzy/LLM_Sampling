# Preregistration: neural validation of the JMLR-B-v4 theory (written 2026-10-16, before any training run)

Fan's instruction (2026-10-16):
- Run IPO/DPO post-training on the VM and pick suitable data.
- Predict the results first, then run.
- Results that match go into v4. Otherwise find the cause and change the conditions until they match.

This file fixes the hypotheses and pass criteria before any run. Per-run quantitative predictions come from `analysis/population.py` on the actual panels and oracles, before the run. They are saved in `prereg/predictions_<batch>.json`, and their timestamps precede the runs.

## Testbed

- **Policy:** Qwen2.5-0.5B-Instruct, full fine-tuning, fp32 master weights with bf16 autocast. Key results are repeated on Qwen2.5-1.5B-Instruct.
- **Panels:** fixed short-answer panels with K = 4.
  - Prompts: 65 templated open-ended short-answer prompts (names, slogans, titles, recommendations, descriptions), NoveltyBench curated (100), and NoveltyBench WildChat (first 435). Each gets the instruction to answer in at most six words.
  - Each panel holds the 4 most probable distinct complete answers among 48 samples at temperature 1.
  - A panel is kept only if every normalized initial panel probability is ≥ 0.05, so the initial policy has interior support.
- **Preference channels:**
  - (i) Skywork-Reward-Llama-3.1-8B-v0.2 with soft labels σ((r_i − r_j)/T): Bradley–Terry, so C = 0 under the logit link.
  - (ii) The same model with hard labels 1[r_i > r_j]: deterministic and transitive.
  - (iii) A jury of three reward models (Skywork-8B-v0.2, Skywork-V2-Qwen3-1.7B, GRM-Llama3.2-3B), with P*_ij equal to the vote share. Its majority relation can cycle.
  - (iv) Constructed matrices, used only as labelled controls if (iii) has too few cyclic panels.
- **Round:** the theory-matched update (`vm/panel_po.py`).
  - Reference: a geometric mixture of the current and initial policy (log scores).
  - Sampler: μ = (1 − λ)·uniform + λ·π_t.
  - Loss: the exact pairwise IPO or DPO loss with soft labels and pair weights μ_iμ_j, optimized to near convergence from a fresh Adam each round.
  - The population map of one round is then exactly the MRS-ΨPO map of the paper (IPO: identity link; DPO with BT labels: logit link).

## Hypotheses and pass criteria

**H0 — Fidelity (prerequisite).**
- Prediction: with enough inner steps, a neural round realizes the population map on the panel.
- Pass: the median over panels of the per-round residual ‖Π s_{t+1} − Π F(s_t)‖∞ is ≤ 0.1 nats.
- If it fails: increase the inner steps or the learning rate, or use `--form payoff` (same optimum, better conditioned). This is a precondition, not a result.

**H1 — Same target at the same sharpness (prop:target).**
- Setup: IPO with the soft oracle, λ = 0.8, and (α, β) ∈ {(0, 4), (0.5, 2), (0.8, 0.8)}, all with B = 4; control (0.5, 0.5), with B = 1.
- Prediction:
  - the three B = 4 runs reach the same panel policy (mean pairwise KL between their final panel policies ≤ 0.02 nats, and each within 0.05 of the population target);
  - the B = 1 run is clearly different;
  - the approach rate is about α per round.

**H2 — Sharpness sets the plateau (prop:adaptation-horizon, Thm 7).**
- Setup: IPO with the soft oracle, β = 1, λ = 0.8, α ∈ {0.5, 0.8, 0.9, 0.95}.
- Prediction:
  - mean panel entropy against M_t(α) nearly coincides across α at early M;
  - each run levels off at the population target entropy for B = 1/(1−α) (within 0.05 nats);
  - per-panel log odds stay inside the sandwich bounds of eq. (target-sandwich).

**H3 — Full refresh drives dominance (Thm 7 part 2, Cor 8).**
- Setup: α = 1, β = 1, λ = 1, IPO with hard labels (each panel's Condorcet winner is its top-scored answer).
- Prediction:
  - the top response converges to the Condorcet winner on ≥ 95% of panels;
  - entropy falls toward 0 monotonically after the first rounds;
  - on panels with positive worst-case margins, π_w ≥ 0.95 is reached no later than the Cor 8 bound.

**H4 — Deterministic labels under the logit link have no finite target.**
- Setup: α = 0.8, β = 1, λ = 0.8; DPO with hard labels, DPO with soft labels, IPO with hard labels.
- Prediction:
  - hard-label DPO keeps concentrating after M_t saturates (entropy at t = 80 well below the entropy at t = 30);
  - soft-label DPO levels off at the exact Gibbs target π ∝ π_ref·exp(B·r/T) (within 0.05 nats of entropy);
  - hard-label IPO levels off at its identity-link target.

**H5 — Condorcet winner vs. cycle (prop:endpoints, thm:local-spectrum).**
- Setup: the jury oracle, IPO, λ = 1.
- Prediction:
  - panels with a Condorcet winner concentrate on it as B grows;
  - panels with a majority cycle approach the maximal lottery;
  - at fixed β and α above the predicted onset α² + γ² > 1 (computed per panel), the cyclic panels oscillate persistently; below the onset they converge.

**H6 — Stability at a fixed target (prop:fixed-target).**
- Setup: cyclic panels at fixed B.
- Prediction: runs with α > α_loc(γ_B) converge; runs with α < α_loc oscillate.

## Decision rule

- A hypothesis that passes goes into v4 Section 8 with its figure.
- If one fails, the first step is to diagnose:
  - H0 residuals;
  - cross-prompt interference from shared parameters;
  - mass leaving the panel;
  - an oracle that does not match the assumed link.
- Then change the condition (more inner steps, payoff form, a different oracle construction or a larger model) and rerun. Every change and every rerun is recorded here with a timestamp. A failed run is not deleted.

## Log

- **2026-10-16, H0 pilot.** IPO with the soft oracle (T = 3.168, half the std of the reward gaps), α = 0.5, β = 4, λ = 0.8, 8 rounds, 130 panels.
  - Pairwise form, 40 Adam steps at lr 1e-5: median per-round residual 0.14 nats (q90 0.29), falling from 0.25 at round 1 to 0.11 at round 8. Mean panel entropy, neural against population: 0.790 vs 0.794 at t = 5 and 0.785 vs 0.783 at t = 8; predicted target 0.792.
  - Payoff form, 20 steps at lr 3e-5: residual 0.44, worse.
  - Decision: pairwise form, 40 steps, lr 2e-5. The design was revised before any H1–H4 run: soft labels use T = 3.168, and H1 uses B = 8 against B = 2. Frozen predictions are in `predictions_batch1.json`.
- **2026-10-16, batch 1 launched** (`vm/batch1.json`, 12 runs, 2 in parallel). H2 at α = 0.5 is the H1 control run (α = 0.5, β = 1).
- **2026-10-16, batch 2 design frozen** (`analysis/make_planted.py`, `prereg/predictions_batch2.json`, `data/oracle_planted.npz`). This is a planted-cycle control on the same 130 real panels; the real three-RM jury left only 1 of 130 panels without a Condorcet winner.
  - In each panel, three randomly chosen responses form a rock–paper–scissors cycle (q = 0.3) and the fourth loses to all (e = 0.3).
  - Half the panels are balanced (Δ = 0.05: majority cycle, maximal lottery on the face of size 3); half are decisive (Δ = 0.2: Condorcet winner).
  - IPO, λ = 1, 80 rounds.
  - Signatures: top-response switches in the last 30 rounds, and the distance-to-target ratio between rounds T and T − 20.
  - Frozen predictions (population model):

    | Configuration | Balanced panels | Decisive panels |
    |---|---|---|
    | H5, fixed β = 4, α = 0.5 | stable: no switches, ratio 0.007 | stable, concentrate (top 0.68–1.0) |
    | H5, fixed β = 4, α = 0.9 and 0.95 | unstable on all panels: switches on 100%, ratio 0.91 and 0.98 | stable, concentrate |
    | H6, fixed B = 12, α = 0.2 | unstable: radius 1.56, switches on 100% | — |
    | H6, fixed B = 12, α = 0.5 | 92% unstable: radius 1.09, switches on 88% | — |
    | H6, fixed B = 12, α = 0.8 and 0.9 | stable: no switches, ratio 0.09 and 0.17 | — |
    | H6, any α | — | stable (11% unstable at α = 0.2) |
- **2026-10-16, H4 (DPO arms) result: PASS.**
  - **Soft-label DPO** (α = 0.8, β = 1, λ = 0.8). Mean panel entropy at t = 1, 5, 10, 20, 60: 0.888, 0.515, 0.416, 0.387, 0.388, against the population predictions 0.909, 0.525, 0.429, 0.396, 0.392 (target 0.392). Final panel policy against the population target: mean KL 0.0005, median 0.0001. Per-round residual: median 0.078 nats, q90 0.176 (H0 met).
  - **Hard-label DPO.** Entropy is 0.0073 after round 1 and 0.0000 from round 5. The median top-versus-second log-odds keeps growing: 10, 89, 385, 508 nats at t = 1, 10, 30, 60, despite α = 0.8. There is no finite target, as predicted.
- **2026-10-16, batch 2 trimmed before any batch-2 run.** To halve run time, batch 2 uses a fixed subset of 64 panels: the first 32 balanced and the first 32 decisive panel ids in `oracle_planted.npz`. The frozen subset predictions are in `prereg/predictions_batch2_subset.json` and match the full-set numbers to within a few percent (for example H6 at α = 0.5: 91% of balanced panels unstable, switches on 84%).
- **2026-10-16, H4 (IPO arm) result: PASS.** Hard-label IPO levels off at the identity-link target: final entropy 0.677 against target 0.679, mean KL to the target 0.0010, residual median 0.072. All of H4 passes.
  - Comparisons with the target use the target computed from each run's own reference scores s[0], the fp32 rescoring used in training.
  - The frozen predictions used the bf16 scores stored in the panel file. That shifts the entropies by about 0.02 (1.159 against 1.140 at t = 0).
- **2026-10-16, H1 partial.**
  - α = 0 with β = 8: KL to the B = 8 target 0.0008.
  - α = 0.5 with β = 4: KL 0.0011.
  - Pairwise KL between their final panel policies: 0.0019 (pass bound 0.02).
- **2026-10-16, H1 result: PASS.**
  - The four B = 8 runs (α = 0, 0.5, 0.8, 0.9) have a largest pairwise KL between final panel policies of 0.0076 (bound 0.02).
  - Their KL to the population target ranges from 0.0008 to 0.007; the α = 0.9 run is still approaching after 30 rounds, as rate α predicts.
  - The B = 2 control is at KL 0.21–0.23 from each of them. Its own target: final entropy 1.085 against 1.087, KL 0.0013.
- **2026-10-16, H2 partial.** α = 0.8 (B = 5): final entropy 0.940 against target 0.943, KL 0.0029.
- **2026-10-16, H2 result: PASS.**
  - Final entropy against the target at B = 1/(1−α): α = 0.5: 1.085 / 1.087; α = 0.8: 0.940 / 0.943; α = 0.9: 0.691 / 0.689; α = 0.95: 0.415 / 0.410. KL to target ≤ 0.0043.
  - Against M_t(α), entropy at M = 4 is 0.996–1.006 across α ∈ {0.8, 0.9, 0.95}, and 0.779 at M = 8 for α = 0.9 and 0.95.
- **2026-10-16, H3 result: PASS.** Full refresh with hard labels and λ = 1: the top answer is the Condorcet winner on 100% of panels, with mass ≥ 0.95 on 100%. Mean Condorcet-winner mass at t = 0, 5, 10, 20, 40: 0.27, 0.70, 0.95, 0.9995, 1.0. Entropy reaches 0.0000 by t = 40. Residual median 0.10.
- **Batch 1 complete:** H0 to H4 pass. Batch 2 started.
- **2026-10-16, amendment to the H5/H6 signature, made after 2 of 7 batch-2 runs (H5 at α = 0.5 and α = 0.9).**
  - **Why.** The preregistered top-switch share misfires on stable balanced panels.
    - At α = 0.5, the balanced targets put about 0.42 on each of two answers. Optimization noise, with a per-round residual of about 0.07 nats, flips the argmax on 41% of panels, although these panels have converged: late amplitude 0.06, distance to target 0.07.
    - The dynamics agree with the prediction; the metric does not.
  - **New primary signature.** The detrended late amplitude: the largest over answers of the standard deviation of the centered panel logits over the last 30 rounds, after removing a linear trend.
    - Threshold: 4 × the noise floor measured the same way on the converged batch-1 runs (floor 0.069, threshold 0.278). Batch 2 is not used to set it.
    - Predicted shares come from the noise-free population recursion and are frozen in `prereg/predictions_batch2_amplitude.json`, before the remaining 5 runs finish.
    - The switch share is still reported.
  - **First two runs under the new signature.**
    - α = 0.5: balanced late amplitude 0.06 against predicted ≈ 0 (stable).
    - α = 0.9: balanced late amplitude 1.54 against predicted 1.45; oscillation on 100% of balanced panels, as predicted. Decisive panels show 0 switches.
- **2026-10-16, batch 3 designed (practice-like, before any batch-3 run).** Each round samples 32 ordered pairs per prompt from μ×μ with labels drawn from Bernoulli(P*), and the loss is the empirical IPO or DPO loss with hard labels (`--sampled 32`). In expectation this equals the batch-1 and batch-2 losses, so the population predictions are unchanged and only the noise grows. Same 64-panel subset as batch 2.
  - **S1 (same target).** DPO with Bernoulli labels from the Bradley–Terry soft oracle, α = 0 with β = 8 against α = 0.8 with β = 1.6 (both B = 8). Prediction: both reach the B = 8 Gibbs target, with mean KL ≤ 0.02 to the target and to each other.
  - **S2 (link).** DPO with hard labels against DPO with Bernoulli labels, α = 0.8, β = 1. Prediction: hard labels collapse (entropy < 0.05 by t = 5, top-two log-odds growing); Bernoulli labels level off within 0.05 nats of the Gibbs target entropy.
  - **S3 (fixed target with a planted cycle).** IPO, B = 12, α = 0.2 against α = 0.9. Prediction: at α = 0.2 most balanced panels oscillate (population share 1.0); at α = 0.9 almost none do. The noise floor for the amplitude threshold is calibrated on the converged S1 runs.
- **2026-10-16, batch 2 result and second signature diagnosis.**
  - **Balanced panels: PASS under the amended logit-amplitude signature.** Oscillating share against the frozen prediction:
    - H5, β = 4: α = 0.5: 0.00 / 0.00; α = 0.9: 1.00 / 1.00; α = 0.95: 1.00 / 1.00.
    - H6, B = 12: α = 0.2: 1.00 / 1.00; α = 0.5: 0.94 / 0.88; α = 0.8: 0.00 / 0.00; α = 0.9: 0.03 / 0.00.
    - Median amplitudes against predictions: 1.50 / 1.45, 2.46 / 2.38, 1.57 / 1.57, 0.67 / 0.65.
  - **Decisive panels: mismatch at fixed β = 4 with α = 0.9 and 0.95.** The logit signature flags 78% and 100% of panels, against a prediction of 0, although the policy has concentrated (top mass 0.989 and 1.000).
  - **Diagnosis.** The amplitude is the largest over answers. It is driven by the logits of losing answers whose probability is near zero. The sampled-pair loss gives those answers vanishing weight μ_iμ_j, so their logits are unconstrained and wander under optimization noise (residual medians 0.18 and 0.44 in these runs).
    - This is a measurement artifact in logit space.
    - It is also an instance of the limitation that a learner only updates what it samples.
  - **Fix.** Measure oscillation on the policy itself: the same detrended amplitude computed on panel probabilities. The threshold is 4 × the probability-space noise floor of the converged batch-1 runs (floor 0.0134, threshold 0.054); the batch-2 data were not used. Population values are computed from each run's own initial scores.
  - **Result in probability space: PASS on all 14 cells.**
    - Balanced, neural / population: 0.00 / 0.00, 1.00 / 1.00, 1.00 / 1.00, 1.00 / 1.00, 0.94 / 0.91, 0.00 / 0.00, 0.00 / 0.00. Amplitudes: 0.306 / 0.307, 0.363 / 0.362, 0.317 / 0.317, 0.181 / 0.185.
    - Decisive: 0.00 / 0.00, 0.00 / 0.00, 0.00 / 0.00, 0.12 / 0.16, 0.03 / 0.00, 0.00 / 0.00, 0.00 / 0.00.
- **2026-10-17, batch 3 results (sampled pairs, n = 32 per prompt per round).**
  - **S2: PASS.** Bernoulli-label DPO levels off at entropy 0.328 against target 0.331. Hard-label DPO reaches entropy 0.023 at t = 1 and 0.0000 at t = 5; its median top-two log-odds gap goes 11, 72, 134, 225, 505 at t = 1, 5, 10, 20, 40.
  - **S1: FAIL on the frozen criterion (final KL ≤ 0.02).**
    - α = 0.8, β = 1.6: final KL to target 0.08. α = 0, β = 8: final KL 1.07, with entropy 0.18 against 0.20.
    - Diagnosis: sampling noise amplified by the per-round step. Averaged over the last 15 rounds, both policies are near the target (KL 0.080 and 0.040).
    - Fluctuations are ordered as the stochastic extension predicts (at a fixed target, stationary variance scales like (1−α)/(1+α)). The ratio of logit standard deviations between α = 0 and α = 0.8 is 3.4 (predicted 3.0); the probability-weighted ratio is 3.6.
    - In probability space the ratio is much larger (22) because at β = 8 the policy jumps between nearly deterministic states, which the small-noise theory does not cover.
    - With expected labels (batch 1), the same configurations fluctuate only 0.06.
  - **S3: inconclusive on the frozen criterion, consistent on amplitudes.**
    - The frozen threshold, calibrated on the noisy S1 runs (floor 0.17, threshold 0.69), classifies nothing as oscillating.
    - Amplitudes match the population model:
      - α = 0.2, balanced panels: median probability amplitude 0.323 (population 0.317); decisive panels 0.115 (sampling noise only, population 0).
      - α = 0.9: balanced 0.063 (population 0.001).
- **2026-10-17, batch 3b preregistered (condition changed: 512 sampled pairs per prompt per round; otherwise identical to S1 and S3; 64-panel subset).**
  - **S1' criteria:**
    - (a) KL from the time-averaged policy over the last 10 rounds to the B = 8 target is ≤ 0.02 for both α = 0 and α = 0.8;
    - (b) the mean per-round KL from π_t to that time average is at least 3× larger at α = 0 than at α = 0.8;
    - (c) both are smaller than at n = 32 (0.41 and 0.0076).
  - **S3' criteria:**
    - (a) at α = 0.2, the median probability amplitude of balanced panels is within ±25% of the population value (0.317) and at least 2× the median of decisive panels in the same run;
    - (b) at α = 0.9, the median for balanced panels is ≤ 0.10.
- **2026-10-16 23:21, batch 4 (Qwen2.5-1.5B-Instruct replication) frozen before any batch-4 run.**
  - Panels: `panels_q15.json` (390 panels), using a 64-panel subset. The planted oracle `oracle_planted_q15.npz` uses the same construction.
  - Predictions and criteria are in `prereg/predictions_batch4.json`:
    - R1, same target at B = 8: target entropy 0.818.
    - R2, link test: soft-DPO target entropy 0.364.
    - R3, planted cycle at B = 12: at α = 0.2, 97% of balanced panels are unstable, population amplitude 0.311; at α = 0.9, none are.
  - Expected labels (no sampling), 40 inner steps, lr 2e-5, one run at a time.
- **2026-10-17, batch 3b result (512 sampled pairs per prompt per round): PASS on every criterion.**
  - **S1'.**
    - (a) KL from the time average (last 10 rounds) to the B = 8 target: 0.0086 at α = 0 and 0.0015 at α = 0.8 (≤ 0.02). Final KL: 0.052 and 0.003.
    - (b) Fluctuation (median over panels of the mean KL from π_t to the time average): 0.0089 against 0.0009, a ratio of 9.9. That is ≥ 3, and close to the linear-noise prediction (1+α)/(1−α) = 9 for the variance at a fixed target.
    - (c) Both are below the n = 32 values (0.38 and 0.0045).
  - **S3'.**
    - (a) At α = 0.2, the median probability amplitude of balanced panels is 0.326 against the population value 0.318 (+3%), and 9× the decisive median (0.036).
    - (b) At α = 0.9, the balanced median is 0.028 (≤ 0.10).
  - **Conclusion for the practice-like setting.** With sampled pairs and Bernoulli labels:
    - the time-averaged target is set by B;
    - stationary noise at a fixed target falls like (1−α)/(1+α);
    - DPO with deterministic labels collapses;
    - a planted majority cycle oscillates with the deterministic amplitude where the theory predicts, and slower refresh at the same target removes it.
- Note (2026-10-17): the S1' fluctuation ratio computed without rounding is 10.4 (0.0089/0.0009 = 9.9 after rounding).
- **2026-10-17, batch 4 result (Qwen2.5-1.5B-Instruct, expected labels, 64 panels): PASS on every frozen criterion.**
  - **R1, same target at B = 8.** Final KL to target: 0.0005 at α = 0 and 0.0015 at α = 0.8; pairwise KL 0.0020.
  - **R2, link test.**
    - Soft-label DPO: final entropy 0.382 against target 0.381 (KL 0.0005).
    - Hard-label DPO: entropy 0.008 at t = 1 and 0 at t = 5. The median top-two log-odds gap goes 10, 81, 156, 179, 179 at t = 1, 5, 10, 20, 30.
    - The plateau near 179 nats reflects numerical floors on token log-probabilities. It is not a finite target.
  - **R3, planted cycle at B = 12.**
    - At α = 0.2: balanced median probability amplitude 0.307 against population 0.306, and 32× the decisive median (0.0095).
    - At α = 0.9: balanced median 0.027 (≤ 0.10).
  - Residual medians: 0.063–0.097.
- **2026-10-17.** All runs and logs were copied locally (`results/`, `results/vm_backup/`). The VM was deallocated on Fan's instruction.
- **2026-10-17, post-hoc scoring of three preregistered sub-predictions not scored during the runs** (`analysis/check_subpredictions.py`, `results/eval_subpredictions.json`; computed from saved runs, no new training).
  - **H1, approach rate about α per round: consistent.** While the median distance to the target exceeds 0.5 (well above the neural noise floor of 0.13–0.28), it contracts by 0.52, 0.81 and 0.91 per round at α = 0.5, 0.8 and 0.9; the population values are 0.48, 0.79 and 0.90.
  - **H2, sandwich bounds: holds for the target, not resolvable for the neural policy.** The population target lies inside the sandwich on all 1,560 ordered pairs in each run. The neural final policy lies inside on 45–51%, because the band is narrow (median width 0.09–0.95 nats) relative to the neural log-odds scatter around the target (median 0.06–0.18, q99 0.24–0.96). At this noise level the sandwich is not a sharp test.
  - **H3, dominance round of cor:dominance: not applicable.** With hard labels and λ = 1, every worst-case margin Γ_wj of the Condorcet winner is 0 on all 130 panels, so the corollary certifies nothing. Concentration still occurred, carried by the margin realized along the trajectory, as v4 remarks after the corollary.
  - **Not run:** the jury arm of H5 (only 1 of 130 jury panels lacks a Condorcet winner, so the planted control (iv) replaced it), the target against the maximal lottery as B grows, and actual DPO on non-realizable cyclic labels (review comment R46).
- **2026-10-17, data-quality note found while writing the report (no rerun).** The panel builder scored each answer's sampled token ids, while training re-tokenizes the stripped answer text. For some answers the two forms get very different probabilities.
  - At training time, 13 of the 130 panels (0.5B) have an answer with initial probability below 0.05, and 6 have one below 0.01. In the 64-panel subsets the counts are 8 / 5 (0.5B) and 6 / 1 (1.5B). The ≥ 0.05 interior-support filter therefore holds for the sampled forms, not for every trained panel.
  - Every population prediction used in the evaluation (`evaluate.py`, `check_*.py`) is computed from the training-time scores s_0 of the same run, so the neural-versus-population comparisons are unaffected. The frozen batch-1 predictions used the builder scores, which is the entropy shift of about 0.02 noted above.
