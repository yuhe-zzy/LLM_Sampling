# Paper B v4: theory-matched neural experiments (2026-10-16 to 2026-10-20)

Shared by Fan Yao for Yu He on Fan's branch. Prepared by Claude (Opus 5.5). This bundle holds the code, preregistrations,
frozen predictions, numeric summaries and HTML reports of the experiments run for the JMLR v4 draft of Paper B
(Overleaf `JMLR-B-v4` at `46fd577`). Nothing here changes the `main` or `hodge-diagnostics` branches, and no result has
been written into the manuscript beyond what v4 already contains.

## Start here

1. [`reports/v4_story_deck.html`](reports/v4_story_deck.html): the v4 storyline and every experiment that supports it, in one
   self-contained deck (setup, convergence evidence, a worked example of Thm 11, results with X15, scorecard).
2. [`RUNNING.md`](RUNNING.md): how to run and extend these experiments (trainer, Sycamore templates, data, conventions,
   pitfalls).
3. [`reports/x14_steering_and_3b.html`](reports/x14_steering_and_3b.html): the latest text report (X14 steering without P,
   and the Qwen2.5-3B replication on Sycamore).
4. The other text reports below, one per experiment family. Each report's fact sheet is in [`reports/sources/`](reports/sources/).

## The common protocol

- Fixed answer panels: one prompt with K = 4 short answers sampled from the model itself (K = 3 in X15); exact sequence
  log-probabilities; the policy is the softmax over the panel.
- Each outer round minimizes the expected IPO or DPO loss over answer pairs (labels from a reward model, a planted
  preference, or pooled criteria), with 40 fresh Adam steps at lr 2e-5, full fine-tuning, fp32 master weights, one seed.
- The reference is refreshed with weight α (B = β/(1 − α) is the target's sharpness); the sampler mixes the initial and
  current policy with weight λ.
- Before any training, the exact population recursion is run from the model's own initial scores on the same panels and
  labels; its predictions are frozen (hash and timestamp in each `prereg/PREREG.md`), then the model is trained and scored.
- Oscillating means a detrended answer-probability amplitude above 0.0536 over the last 30 rounds.

## Experiments and verdicts

| Folder | What it tests | Model, compute | Verdict | Report |
|---|---|---|---|---|
| [`2026-10-16-v4-validation`](experiments/2026-10-16-v4-validation/) | H1–H6: same target at the same B, entropy plateaus, Condorcet concentration and the link, the stability frontier α² + γ² < 1 on planted cycles; robustness (sampled pairs, 1.5B) | Qwen2.5-0.5B and 1.5B, Azure H100 | Main tests pass; 32 sampled pairs per prompt fail (final KL 1.07) | [text](reports/v4_neural_validation.html), [deck](reports/v4_neural_validation_deck.html) |
| [`2026-10-17-controls`](experiments/2026-10-17-controls/) | E1–E6: judge bias as hidden refresh, popularity penalty, drift controllers, replay, adaptive α, weight averaging | 0.5B, Azure H100 | E1, E2, E3′, E4, E5 pass; E3 (per-panel step budget) and E6 (weight averaging) fail | [text](reports/controls_e1_e6.html) |
| [`2026-10-18-dpo-inner`](experiments/2026-10-18-dpo-inner/) | X1: actual DPO on intransitive labels follows its own map; X9: fewer inner steps | 0.5B, Azure H100 | X1 pass; X9 fail (fresh Adam steps have a fixed size) | [text](reports/dpo_inner_x1_x9.html) |
| [`2026-10-19-x10-x5-x12`](experiments/2026-10-19-x10-x5-x12/) | X5: refresh-then-anchor step limit; X10: lagged reference; X12: LLM-judge census | 0.5B; Qwen2.5-7B judge | X5 pass; X10 fail; X12: 3.1% of panels without a Condorcet winner, below the training trigger | [text](reports/storyline_x5_x10_x12.html), [deck](reports/storyline_x5_x10_x12_deck.html) |
| [`2026-10-19-mixed-criteria`](experiments/2026-10-19-mixed-criteria/) | X11: do rubric contexts or style panels create pooled cycles? | population predictions; judge scoring | No panel predicted to oscillate, so no training (preregistered stop) | [text](reports/mixed_criteria_x11.html) |
| (analysis only) | D1–D5: diagnostics that need no preference matrix | existing runs and datasets | Validated after the fact; thresholds set on these runs | [text](reports/practical_diagnostics_d1_d5.html) |
| [`2026-10-19-x14-steer`](experiments/2026-10-19-x14-steer/) | X14: interventions chosen by the P-free tests avoid collapse, over-sharpening and rotation | 0.5B, Azure H100 | 7 of 9 arms pass; preregistered verdict "not fully supported" (S1-b at the optimizer noise floor; S3-c a slow spiral at B = 24) | [text](reports/x14_steering_and_3b.html) |
| [`2026-10-19-sycamore-3b`](experiments/2026-10-19-sycamore-3b/) | Converge, polarize and oscillate at 3B with the 0.5B settings | Qwen2.5-3B-Instruct, UNC Sycamore, 5 × 1 H100, 6.3 GPU-hours | 5 of 5 criteria pass, plus the directional check | [text](reports/x14_steering_and_3b.html) |
| [`2026-10-19-pooled-onset`](experiments/2026-10-19-pooled-onset/) | X15 (designed and run by Fable 5.1): routing comparisons by criterion makes a trained model rotate (cor:pooled-onset) | 0.5B, Azure H100 | Supported on the DPO loop: rotating share 0 / 8 / 48 / 81% at c = 0 / 0.25 / 0.5 / 1 (population 0 / 8 / 48 / 94%); slower refresh 19%; population onset slope −1.28 in c. Secondary arms: random assignment fails the amplitude rule (label-noise jitter, not rotation); IPO misses its frozen prediction, which used the logit link (identity link: 12.5% predicted and trained); K = 4 has 4 panels | [deck](reports/v4_story_deck.html), slides 13–16 |

## Layout of each experiment folder

- `prereg/`: the preregistration with its timestamped log, and the frozen predictions (`predictions*.json`).
- `analysis/`: prediction, evaluation and diagnosis scripts.
- `vm/`, `src/`, `slurm/`: the training script, batch configurations and Slurm files as run.
- `summaries/`: numeric evaluation outputs. In the local workspace and in the reports this directory is called `results/`;
  it is renamed here because this repository ignores `results/`.
- `figs/` (v4 validation only): the figures and TikZ sources used in the manuscript.
- `common/dpo_map.py`: the population DPO map (Prop. dpo-map), imported by the X1/X9, X11, X12 and X15 scripts.

## Not included

- The answer panels (`data/`): they contain prompts drawn from public datasets (NoveltyBench, WildChat) together with the
  models' raw sampled answers. Per this repository's rules, raw prompt and response text is not published; the panels are
  available from Fan on request. The planted-preference matrices and score files (`*.npz`) are not included either.
- Per-run trajectories (`run.npz`), training logs and VM or Slurm output: kept in the local workspace.
- Model weights: none are saved; every run starts from the public Qwen2.5 Instruct checkpoints.
- Scripts contain absolute paths of Fan's machine and of the VM; they are published as run, not edited for portability.

## Status

As of 2026-10-24 all runs are complete and scored. X15 was scored with its own `evaluate_x15.py` (`summaries/evaluation_x15.json`);
the post hoc checks of 2026-10-20 (winding test, identity-link IPO prediction, round losses) are in its `analysis/*_posthoc.py`
and `summaries/posthoc/`. Results enter the manuscript only after Fan approves them.
