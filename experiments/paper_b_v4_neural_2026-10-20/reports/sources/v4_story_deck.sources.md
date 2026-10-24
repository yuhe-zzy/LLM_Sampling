# Sources: the v4 story deck (2026-10-20)

**How the numbers get onto the slides.** Every number is read by `build_deck.py` (with `build_deck_part2.py`, `slide_*.py`) from `deck_data.json`. `collect_deck_data.py` collects that file from the experiment outputs listed here. The figure ids in FACTS refer to `paper/mirror/overleaf/JMLR-B-v4/meta/FACTS.md`.

**Status.**
- **confirmed**: from an evaluation file or FACTS entry marked supported.
- **single**: one run or seed, or a reading made after the result.
- **inference**: ours.

**Manuscript.** Theorem numbers and section numbers refer to JMLR-B-v4 at Overleaf `46fd577` (pushed 2026-10-19).

| Slide | Claim | Source | Status |
|---|---|---|---|
| 2 | Four theorems; verdicts | v4 contributions paragraph; slides 11–25 below | confirmed (Thm 11 neural test: slide 16) |
| 3–4 | Loop and exact recursion; Hodge split; only the cyclic part feeds back | v4 §3–4, thm:master-recursion; B-T1 | confirmed (theory) |
| 5 | Frontier α² + γ² < 1, fixed-target line γ = (1−α)γ_B, α_loc = (γ_B² − 1)/(γ_B² + 1); median balanced γ_B ≈ 1.97 at B = 12 | thm:local-spectrum, prop:fixed-target; `reviews/2026-10-19-practical-diagnostics/text/sources.md` H6g | confirmed (γ_B derived from α_loc 0.59) |
| 6 | Protocol: panels, exact scores, expected loss, 40 Adam steps at lr 2e-5, frozen predictions | `experiments/2026-10-16-v4-validation/prereg/PREREG.md`; B-E6 | confirmed |
| 7–8 | Setup of the eight run families (model, panels, labels, objective, α/β, rounds) | each family's PREREG.md: `2026-10-16-v4-validation`, `2026-10-17-controls`, `2026-10-18-dpo-inner`, `2026-10-19-x10-x5-x12`, `2026-10-19-x14-steer`, `2026-10-19-sycamore-3b`, `2026-10-19-pooled-onset` | confirmed |
| 9 | Per-round loss minus the round optimum (start and after 40 steps), three families; reconstruction within 2e-5 (v4), 2e-6 (X14), 4e-6 (3B); floors 10⁻³–10⁻² IPO, 10⁻⁴–10⁻³ DPO | `experiments/2026-10-16-v4-validation/results/eval_round_losses.json`; `2026-10-19-x14-steer/results/round_losses_x14.json`; `2026-10-19-sycamore-3b/results/round_losses_3b.json`; `convergence.json` (`series_convergence.py`) | confirmed |
| 10 | Answer probabilities over rounds for three prompts under five settings, trained vs population, 0.5B and 3B | `convergence.json` from runs H1_a08_b16, H3_a1_b1, H6_B12_a08, H6_B12_a02, H5_b4_a09 and D3B_*; prompt rule fixed before viewing | confirmed (one seed) |
| 11 | H1 pairwise KL ≤ 0.0076, B = 2 control 0.207–0.227; 1.5B pair 0.0020; H2 entropy against accumulated sharpness | `experiments/2026-10-16-v4-validation/results/eval_main.json`, `eval_q15.json`; B-E7, B-E10 | confirmed |
| 12 | H3 mass on the Condorcet winner → 1.0, 100% of panels; H4 entropies 0.007 / 0.388 (0.385) / 0.677 (0.679); 3B full refresh mass 1.000 on 100% | `eval_main.json`; B-E8; `2026-10-19-sycamore-3b/results/evaluation_3b.json` | confirmed |
| 13 | Pooled curl identity; onset ∝ (cm)⁻¹ against m⁻³ | prop:pooled, cor:pooled-onset (v4 §6) | confirmed (theory); schematic |
| 14 | Worked example: three Bradley–Terry judges with gap m = 1 in rotation; uniform mixture P = [σ(1) + σ(−2) + σ(1)]/3 = 0.527 per edge; routing by criterion P = σ(1) = 0.731; ratio of edge biases 8.5 | v4 §6, prop:pooled and cor:pooled-onset (`paper/mirror/overleaf/JMLR-B-v4/jmlr_main.tex`, 46fd577); recomputed in `slide_pooled_example.py`; example text supplied by Fan 2026-10-20 | confirmed (recomputed); idealized example |
| 15 | Onset β* = √3/(4(p̄ − ½)): 16.0 (uniform), 5.5 (c = 1/4), 1.9 (c = 1); β = 10 (DPO 0.1) stable under the uniform mixture and rotating at c = 1/4; D4 +0.92 pp; on-policy feedback (part 4) | cor:pooled-onset and the paragraph after it (1/β thresholds 0.063 and 0.18); prop:pooled part 4; D4 in `reviews/2026-10-19-practical-diagnostics/` | confirmed (recomputed) |
| 16 | X15 (Fable 5.1) | see the X15 section below | see below |
| 17 | Prevalence: HelpSteer 0.06 / 1.42 / 3.14%, UltraFeedback 0.55%, jury 1/130, rubric 0/155, LLM judge 3.1%; D4 +0.92 pp | B-E23; `experiments/2026-10-19-x10-x5-x12/results/x12_evaluation.json`; diagnostics D4 | confirmed |
| 18 | H5/H6 balanced oscillating shares (neural against population) | `eval_main.json` H56; B-E9 | confirmed (disclosed measurement amendments) |
| 19 | X1: 9.4% / 31.3% against DPO map 0% / 28% and logit map 78% / 75%; agreement 0.91 | `experiments/2026-10-18-dpo-inner/results/evaluation.json`; B-E19 | confirmed |
| 20 | 3B: conv_rm 0.968 vs 0.968; pol_full 1.00 vs 1.00; conv_pl 0% vs 0%; osc_fast 100% vs 100%; osc_slow 100% vs 100%; 0.5B counterparts; 5 of 5 pass | `experiments/2026-10-19-sycamore-3b/results/evaluation_3b.json`; `reviews/2026-10-19-x14-and-3b/text/sources.md` B1–B11 | confirmed (one seed) |
| 21 | X5: 15 vs 33 and 4 vs 8 rounds; X10 failure | `experiments/2026-10-19-x10-x5-x12/results/evaluation_x10_x5.json`; B-E21, B-E22 | confirmed |
| 22 | E1: 0.638 / 0.300 / 0.061 / 0.007, frozen 0.525 (pred. 0.638 / 0.301 / 0.066 / 0.000 / 0.527); E2 KL to start and entropies | `experiments/2026-10-17-controls/results/eval_controls.json`; B-E14 | confirmed |
| 23 | E4 94→0%, 0→59%; E5 α 0.2→0.91, 0%; E6 0.479 / 0.008 / 0.052 | `eval_controls.json`; B-E16 to B-E18 | confirmed |
| 24 | D1–D5 validation numbers | `reviews/2026-10-19-practical-diagnostics/analysis/*.json` | confirmed; thresholds set on these runs |
| 25 | X14: 7 of 9 arms pass; S1-b and S3-c fail; c24 slow spiral | `experiments/2026-10-19-x14-steer/results/evaluation_x14.json`, `diagnose_c24.json` | confirmed; c24 reading post hoc |
| 26 | Parity plot: 16 predicted levels at 0.5B (H4, H5/H6 balanced amplitudes, E1, n = 512 entropies; largest gap 0.024), 4 at 1.5B (0.004), 5 at 3B (0.000); n = 32 KL 1.07; X9 | `eval_main.json`, `eval_controls.json`, `eval_sampled_all.json`, `eval_q15.json`, `evaluation_3b.json`; B-E10, B-E20 | confirmed; the mix of quantities on one axis is ours |
| 27 | Boundary list | B-E10, B-E15, B-E18, B-E20, B-E21; X14; X11/X12 logs | confirmed |
| 28–29 | Scorecard cards | as listed on each card (FACTS ids or experiment) | confirmed |
| 30 | Placement of experiments in the paper | v4 section map at 46fd577 | inference (proposal) |
| 31 | Uncertainty sorting | ours | inference |

## X15 (Fable 5.1, 2026-10-20)
Scored 2026-10-20 05:00 EDT with Fable's `experiments/2026-10-19-pooled-onset/analysis/evaluate_x15.py` against `prereg/predictions_x15.json` (frozen 2026-10-20 00:00 EDT). Runs fetched from the VM after the batch chain wrote ALLDONE (04:54 EDT); all 11 runs complete (120 rounds). Output: `reviews/2026-10-20-v4-story/x15_fetch/evaluation_x15.json`.

| Claim | Value | Source | Status |
|---|---|---|---|
| Direction (criterion 1) | neural share 0 / 7.8 / 48.4 / 81.3% at c = 0 / 0.25 / 0.5 / 1; nondecreasing; ≤ 5% at c = 0; ≥ 50% at c = 1 | `evaluation_x15.json` criteria.1_direction | confirmed: PASS |
| Level (criterion 2, with 7) | 8 of 11 arms within 0.15 of the population from their own initial scores; c = 1: 0.812 vs 0.938 (frozen 0.953); misses: random 0.906 vs 0, IPO 0.125 vs 0.938, K4 c = 1 0.75 vs 1.0 (4 panels) | `evaluation_x15.json` | confirmed: mixed |
| Onset slopes (criterion 3, population) | −1.28 in c (c = 0.5 → 1), −0.98 in m̃ at c = 1 (−1.28 at c = 0.5) | `evaluation_x15.json` criteria.3_onset_slopes | confirmed: PASS |
| Remedy (criterion 4) | c = 1 under α 0.9, β 1: 0.188 vs population 0.156; drop 0.625; median KL to target 0.0010 | `evaluation_x15.json` | confirmed: PASS |
| Random assignment (criterion 5) | share 0.906 (≤ 0.10 required): FAIL; per-round step KL 0.234 vs 0.00075 at c = 0 (noisier: yes) | `evaluation_x15.json` | confirmed: FAIL |
| Map (criterion 6) | DPO map one-step error below the logit map in 4 of 4 arms | `evaluation_x15.json` | confirmed: PASS |
| Rotation vs jitter (post hoc) | net winding over the last 30 rounds on oscillating panels: c = 1 median 6.3 turns (92% ≥ 1 turn, sign consistency 0.93); random arm 0.05 turns (sign consistency 0.07) | `x15_fetch/diagnose_x15.py`, `diagnose_x15.json` | inference (post hoc) |
| IPO prediction (post hoc) | the frozen IPO prediction and the evaluator used `logit_round` (logit link), while the PREREG text specifies the identity-link master recursion; recomputed with the identity link (`population.trajectory(P − ½, …)`): 12.5% predicted, 12.5% trained, 64 of 64 panels agree | `x15_fetch/ipo_identity_posthoc.json`; `predict_x15.py` line 159 | confirmed (recomputed); correction made after the result |
| Round losses | reconstruction within 1e-7 of the logged first-step loss on the expected-label DPO arms (5.7e-7 IPO; sampled-label arm not reconstructible) | `x15_fetch/round_losses_x15.json` | confirmed |

