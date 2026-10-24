# Running and extending these experiments

Written 2026-10-24 for Yu He, who will continue from this code. It covers what to run, the conventions that differ from
`scripts/run_cyclic_history.py`, and the pitfalls found so far. The Sycamore setup in
`experiments/2026-10-19-sycamore-3b/` is the most direct starting point.

## 1. The trainer

Use `experiments/2026-10-19-sycamore-3b/src/panel_po_steer.py`. It is a superset of every trainer in this bundle:
- with no extra options it reproduces the v4 trainer `2026-10-16-v4-validation/vm/panel_po.py`;
- the control options of `2026-10-17-controls/vm/panel_po_ctrl.py`;
- the X14 steering options;
- the environment variable `PANEL_CH` (rows per micro-batch; gradients accumulate, so the objective does not change).

One round of the loop, for every panel of K fixed answers:
1. The policy is the softmax of the exact sequence log-probabilities s_t of the K answers.
2. The reference is r_t = (α − ν) s_t + ν s_{t−1} + (1 − α) s_0 (ν = 0 unless `--nu`). The sampler is μ_t = (1 − λ) c + λ π_t,
   with coverage c uniform by default (`--coverage ref` uses π_0).
3. The round objective is the expected IPO or DPO loss over all ordered pairs, weighted by μ_i μ_j, with labels P_ij from
   `--oracle` (`soft` / `hard` reward-model labels, or `npz:<file>` with a P[N, K, K] array). No pairs are sampled unless
   `--sampled n` is given.
4. 40 Adam steps (`--inner 40`) from a fresh optimizer, full fine-tuning, fp32 master weights. **Every run here used
   `--lr 2e-5`; the script default is 1e-5.**

Outputs per run directory: `run.npz` (`s`: rounds+1 × panels × K exact scores; `P`; `loss_first`, `loss_last`; `alphas`,
`betas`; `ids`), `config.json`, `log.txt`, and `steer.json` when a steering option acted. `--rounds 0` only scores the panels,
which gives s_0 for the predictions.

## 2. Environment and compute (as used on Sycamore, 2026-10-19)

- `slurm/setup_env.sbatch`: Python 3.12, torch 2.13.0 (cu130), transformers 5.13.1, numpy 2.5.1, scipy; pandas and pyarrow
  for the panel builder.
- Memory and speed:
  - Qwen2.5-0.5B runs take 18–35 GB of an H100, so at most three run at once on one 94 GB card.
  - Qwen2.5-3B needs `PANEL_CH=32` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. It peaks at 70.8 GiB and takes about
    72 s per round on 64 panels.
- `slurm/train_array.sbatch` runs one line of `slurm/runs.txt` per array task, with `--array=0-4%2` for two GPUs at a time.

## 3. Data

- **Panels.** `src/build_panels.py` builds them from a local copy of the NoveltyBench parquet files: 600 prompts, the
  suffix "Answer with a single short phrase of at most six words", M = 48 samples, K = 4, at most 20 new tokens, seed 0.
  A panel is kept only if every answer's initial probability within the panel is at least 0.05. Each answer is scored with
  Skywork-Reward-Llama-3.1-8B-v0.2 at revision `d4117fbfd81b72f41b96341238baa1e3e90a4ce1` (`slurm/build_panels.sbatch`).
  - Panels hold raw prompts and model answers, so they are not in this public repository. Ask Fan for the exact files used:
    `panels_q05.json` (0.5B, 130 panels), `panels_q15.json` (1.5B), `panels_q3b.json` (3B), and the X15 specialist panels.
- **Training subsets.** H1–H4 use all 130 0.5B panels. Every other 0.5B family uses ids 0–63 (`--ids`, in the batch JSONs).
  The 3B runs use 3B panels 0–63.
- **Planted cycles.** Run `2026-10-16-v4-validation/analysis/make_planted.py <panels.json> <oracle.npz> <pred.json>`. It
  is seeded, and its parameters are in `prereg/make_planted_q3b.json`. Half the panels are balanced (a majority cycle,
  Δ = 0.05) and half decisive (a Condorcet winner, Δ = 0.2).

## 4. Predict, train, score

1. **Score the panels** with `--rounds 0` to get the model's own s_0.
2. **Predict with the exact population recursion and freeze the output** (store a hash and time in a PREREG.md) before
   training. The map depends on the objective:
   - IPO: `population.trajectory(P − ½, s_0, α, β, λ, T)` from `2026-10-16-v4-validation/analysis/population.py`. This is the
     identity link.
   - DPO with Bradley–Terry labels: the same recursion with A = logit P (`population.link(P, 'logit')`).
   - DPO with intransitive labels: `common/dpo_map.py` (`dpo_round`), the weighted Bradley–Terry projection of Prop. dpo-map.
3. **Train.**
4. **Score against the frozen numbers.** A panel oscillates when its detrended answer-probability amplitude over the last 30
   rounds exceeds 0.0536. Examples: `evaluate.py` (v4), `evaluate_controls.py`, `evaluate_x14.py`, `evaluate_x15.py`.
5. **Check that each round was solved.** `2026-10-19-x14-steer/analysis/round_losses_x14.py <runs dir> <out.json>` recomputes,
   for every round, the loss at the first and last inner step and the round's exact optimum. Each round's starting excess
   should fall towards about 10⁻³ when a target exists.

## 5. Conventions that differ from the history engine

- **β is the theory's step size.** Here h is the score difference to the reference.
  - IPO: the loss per pair is (h_i − h_j − β(P_ij − ½))², so a larger β means larger moves.
  - DPO: the loss is the logistic loss on (h_i − h_j)/β, so β = 1/β_DPO. Here β = 10 means the usual DPO coefficient 0.1.
  - In `run_cyclic_history.py`, β is the loss coefficient for both objectives (the IPO target is 1/(2β), DPO uses βΔ), so a
    larger β there means smaller moves. Convert before comparing settings.
- **α and λ** mean the same as `alpha` and `lambda_current` in the history engine: α weights the current policy in the
  reference, and λ mixes the uniform coverage and π_t in the sampler.
- **Panel answers** are short answers sampled from the model itself, each with initial probability ≥ 0.05. On long answers
  or dataset responses the initial distribution is often nearly degenerate, and then nothing can rotate.

## 6. Pitfalls found so far

- **Each round must be solved.** The theory describes the exact round map. If the inner optimization misses the round's
  optimum by about as much as the predicted move, stability contrasts disappear. Check with the round-loss script, and choose
  cells far from the frontier α² + γ² = 1.
- **Sampled labels at a large step.** At α = 0 with a large β, 32 sampled pairs per prompt make the loop jump (v4 batch 3), and
  so does X15's random-assignment arm at 512 pairs. The amplitude rule counts this jitter as oscillation. Separate the two with
  the winding test in `2026-10-19-pooled-onset/analysis/diagnose_x15_posthoc.py`: a sustained rotation winds several times
  in 30 rounds, while jitter has near-zero net winding.
- **Fixed-size Adam steps.** Fresh Adam steps have a roughly fixed size. Near a target they overshoot, which leaves a noise
  floor of about 10⁻³–10⁻² in the round loss. Fewer inner steps therefore do not act as slower refresh (X9).
- **X15's IPO prediction.**
  - The problem: `predict_x15.py` and `evaluate_x15.py` use `logit_round` (the logit link) for the IPO arm, while that arm's
    PREREG specifies the identity link.
  - The correction: with the identity link (`analysis/ipo_identity_posthoc.py`) the prediction is 12.5% oscillating, and
    the trained model also gives 12.5%, on the same 64 of 64 panels.
  - Two readings of this arm: the PREREG log (written by the agent that ran X15) attributes the miss to the inner
    optimization; this post hoc check finds that each round reaches its optimum (`summaries/posthoc/round_losses_x15.json`).
    See both before reusing that arm.
- **Absolute paths.** The scripts are published as run, so they contain absolute paths. Map them as follows:

  | Path in the scripts | In this bundle |
  |---|---|
  | `ROOT`/`EXP` constants (`/Users/...`) + `experiments/<exp>` | `experiments/<exp>` |
  | `reviews/2026-10-18-jmlr-review/theory` | `common/` |
  | each experiment's `results/` | `summaries/` |
  | `/home/azureuser/v4val/...` (VM) and `/work/users/f/a/fanyao/claude/...` (Sycamore) | your own model, panel and output directories |
