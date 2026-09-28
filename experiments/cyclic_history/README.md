# Cyclic LLM history diagnostics

## September 28 calibration update

Start with [the calibrated protocol](CALIBRATED_PROTOCOL.md) and the
[follow-up handoff for Fan and Yu He](../hodge_diagnostics/HANDOFF_TO_FAN_AND_YU_HE_2026-09-28.md).
The new ten-arm, six-prompt mechanism micropilot is **CPU-calibrated but not
approved or submitted for GPU training**. It keeps raw sequence sums, uses
actual IPO/BT operators, and has support, score and local-spectrum gates.
The original six v1 configurations below are retained as historical pilot
definitions, not a validated stabilization design. Their server executions
are separate immutable copies; this code update does not change them.

`run_oracle_feedback_extrapolation.py` is the accurately named new entry point.
The older `lagged_sampling` name is retained only for legacy reproducibility;
neither version implements the manuscript's signed two-sampler loss.

One training seed (`0`). No scheduler submission or GPU allocation is part of
this deployment. New training requires `--approve-calibrated-micropilot` in
addition to `--execute`; legacy training requires `--allow-uncalibrated-legacy`.

## Two experiment entry points

1. `run_cyclic_lagged_reference.py`: geometric reference with one-round history.
2. `run_cyclic_lagged_sampling.py`: one-round feedback extrapolation. For IPO
   this matches the identity-payoff lagged-sampling update under the pair law
   below. For actual DPO this is **BT-optimizer feedback extrapolation**, not
   the entrywise-logit PsiPO operator in the paper.

Both support IPO and DPO through `experiment_plan.json`. The shared ordinary
control uses `run_cyclic_history.py`. The two entry points share implementation
so that loss computation, data, optimization, and diagnostics stay matched.

Package location: `experiments/cyclic_history/` in this repository. Publishing
this code does not modify production scripts, running jobs, or existing results.
All current runners use sequence-sum scores; historical trainers have been
removed. Download/build the dataset using the root README. All default paths
are relative to the repository root.

## Historical six-run pilot

These are historical **pilot defaults**, not a confirmed final
paper grid or a claim that the ordinary control will oscillate. Do not launch
until the user has reviewed the protocol and parameters.

All six: alpha=0.9, lambda_current=0.8, beta_train=1, training seed=0.

| Run ID | Objective | Scheme | nu | kappa |
|---|---|---|---:|---:|
| ipo_baseline_s0 | IPO | ordinary | 0 | 0 |
| ipo_reference_s0 | IPO | lagged reference | 0.45 | 0 |
| ipo_sampling_s0 | IPO | feedback extrapolation | 0 | 0.25 |
| dpo_baseline_s0 | DPO | ordinary | 0 | 0 |
| dpo_reference_s0 | DPO | lagged reference | 0.45 | 0 |
| dpo_sampling_s0 | DPO | BT-feedback extrapolation | 0 | 0.25 |

Experiment 1 compares each reference run with its ordinary control; experiment
2 compares each feedback run with the same ordinary control. Thus six runs,
not eight. No additional alpha/beta/lambda sweep or second seed is scheduled.

Common settings:

- Model: local `Qwen2.5-1.5B`, same initial model and LoRA initialization.
- 500 prompts selected once with support_seed=123; four fixed responses each.
- 100 outer rounds, not 100 optimizer steps.
- 2 pair proposals per prompt per round: 1,000 pairs total for **both** losses.
- One inner epoch, pair batch size 1, gradient accumulation 4: 250 optimizer
  steps per outer round; AdamW LR=1e-5; warmup ratio=0.03 per round.
- AdamW and linear LR schedule reset at each outer round, identically for all
  arms. Weight decay is PyTorch AdamW's default 0.01, betas=(0.9,0.999), eps=1e-8.
- LoRA r=16, alpha=32, dropout=0; q/k/v/o/gate/up/down projections.
- BF16 weights, FP32 log-sum-exp/loss reductions, gradient clipping at 1.
- max_length=1537, scoring batch size 4, one visible allocated GPU required.
- Every-round score snapshots; initial adapter and adapters every 10 rounds.

### Attention backend and finite-gradient checks

The runner explicitly uses SDPA with Flash, Efficient and Math backends allowed,
excluding cuDNN SDPA. A real BF16 Qwen/H100 reproduction produced NaNs in
`ScaledDotProductCudnnAttentionBackward0` for both IPO and DPO, despite finite
forward scores and losses. Disabling gradient checkpointing did not resolve
that reproduction; using eager attention did. This is a backend-specific
training failure, not evidence of sequence-sum mode collapse.

The scoped [PyTorch SDPA context](https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html)
encloses scoring and both training forward/backward. In particular, backward
must stay inside the context because activation checkpointing recomputes the
forward pass. Backend flags are restored on exit; no shared environment or
other running process is changed. `manifest.json` records `attention_policy`.

Sequence sums, loss definitions, BF16, references, sampling and hyperparameters
are unchanged. Non-finite loss, gradient and parameter checks remain enabled.
Do not replace NaNs with zero or disable `error_if_nonfinite` to keep a job alive.
CPU tests check backend selection/restoration and its forward/backward scope;
real pretrained-model GPU validation is still required for a new environment.

`beta_train` is the coefficient in the actual training loss. IPO's pair target
is `1/(2*beta_train)` and its population feedback scales as `1/beta_train`.
DPO uses `BCEWithLogits(beta_train * pair_log_ratio, P_ij)` and its optimal
log-ratio scores are `BT_scores/beta_train`. This is **not** silently identified
with the paper's exponent/update-gain beta.

`lambda_current` is the **current-policy response-mixture weight**, not the
legacy oracle base-sampling weight or the old cyclic margin-weighting knob.
For this pilot, mu=0.2*uniform+0.8*current_panel_policy. No lambda=0 experiment
has been added.

## Preference environment and pair law

This is synthetic **cyclic preference**, not Nemotron/scalar-oracle preference.
It reuses the original four responses and complete preference matrix:

```
P = [[0.5, 1,   1,   0],
     [0,   0.5, 1,   1],
     [0,   0,   0.5, 1],
     [1,   0,   0,   0.5]]
```

Legacy diagonal zeros are changed to 0.5 only on the unused diagonal. Missing
off-diagonal comparisons are rejected, not filled as ties. This matrix is not
a balanced symmetric three-cycle; do not assume a uniform fixed point.

For each prompt, let s_t(i) be the **sum** of response-token log probabilities,
including EOS and excluding prompt/padding. Define:

```
q_t = softmax(s_t)
mu_t = (1-lambda_current)*uniform + lambda_current*q_t
w_t(i,j) = mu_t(i)*mu_t(j) / sum_{a<b} mu_t(a)*mu_t(b),  i<j
```

Pairs are proposed uniformly among the six distinct unordered pairs and given
the nonnegative importance weight `6*w_t(i,j)` **once**. This is an unbiased
estimator of the declared per-prompt pair objective. Proposal RNG is matched
across arms; target weights change as policies diverge. The 500 panels are
training/diagnostic support, not an independent held-out quality evaluation.

## Update definitions

Write s_init for cached initial scores and s_prev for the previous outer round.
Every reference is frozen throughout the inner optimizer steps. Additive
per-prompt normalizing constants cancel from all pairwise losses.

Ordinary reference:

```
r_t = (1-alpha)*s_init + alpha*s_t
```

Experiment 1, lagged reference:

```
r_t = (1-alpha)*s_init + (alpha-nu)*s_t + nu*s_prev
```

With the pilot values these are respectively `(0.1,0.9,0)` and
`(0.1,0.45,0.45)` weights on initial/current/previous **log probabilities**.
No model parameters are interpolated. This is a partial lag, not the maximal
nu=alpha case.

Experiment 2, feedback extrapolation:

1. Calculate the centered population-optimal log-ratio feedback d_t for the
   **actual positive pair loss** with P and mu_t; compute d_prev using mu_prev.
2. Keep the current positive pair distribution and ordinary reference, but
   shift the cached reference by `kappa*(d_t-d_prev)`.
3. Train the usual positive IPO or DPO loss against this effective reference.

```
r_effective = r_t + kappa*(d_t-d_prev)
population_target = center(r_t + (1+kappa)*d_t - kappa*d_prev)
```

This follows by translation equivariance: a pair loss depending only on
`(s_i-s_j)-(r_i-r_j)` has its unconstrained optimum translated by any added
reference score vector. There are **no negative loss weights and no signed
sampling probabilities**. This statement concerns the unconstrained finite
panel optimum, not exact realizability by a shared-parameter LLM or a finite
sample of pairs.

- IPO: `d_t = center((P-0.5) @ mu_t / beta_train)`. Linearity makes the target
  equal the identity-payoff formula evaluated at
  `(1+kappa)*mu_t-kappa*mu_prev`. That signed vector is never used for sampling.
- DPO: solve `min_v sum_{i<j} w_t(i,j)*[softplus(v_i-v_j)-P_ij*(v_i-v_j)]`,
  then use `d_t=center(v)/beta_train`. The four-action solve is CPU-only and
  its maximum gradient residual must be below 1e-10. It does not take logit
  of hard 0/1 labels. **Do not claim this DPO arm directly validates the
  entrywise-logit PsiPO stability boundary.** It tests an actual-DPO extension
  with its own feedback operator, and that distinction belongs in the paper.

At initialization, previous=current=initial. At a time-independent policy,
previous=current and both corrections vanish. The population fixed-point
equations are therefore shared with ordinary training. This does not guarantee
equal neural endpoints, convergence in 100 rounds, or improvement for every
coefficient. The implementation measures finite-inner-training error rather
than assuming it is zero.

## Why a new ordinary control is necessary

The source scripts provide the data conventions, tokenization/LoRA pipeline,
and pair objectives. The diagnostic deliberately aligns all six arms:

- Sequence-sum likelihoods throughout, including IPO, sampling, and scoring.
- Cached **outer-round** reference, not a recomputed within-minibatch policy.
- Explicit response-mixture product pair law, not average-margin weighting.
- BF16 and fail-fast nonfinite checks, never uniform fallback on NaNs.
- The same pair budget and no dropout for both methods.

These are shared protocol changes, not additional ablation axes. Existing
legacy trajectories cannot serve as matched controls for the new history runs.
Only within-objective contrasts differing in nu or kappa isolate the history
intervention. Comparing IPO and DPO at beta_train=1 is not a claim of equal
effective update strength.

## Diagnostics and interpretation

`snapshots/step_XXXX.npz` contains all 500 prompts' sequence log probabilities,
centered logits, panel probabilities, token lengths, entropy, TV, panel mass,
and the reference/feedback/target used for each completed update. Prompt IDs,
texts, responses, matrix and support hash are persisted in `support.json` and
`manifest.json`, allowing the same-prompt trajectory plots across all arms.

`metrics.csv` contains:

- `panel_entropy_mean`: entropy of `softmax(s_t)` on the four candidates.
- `relative_sequence_entropy_mean`: entropy of `softmax(s_t-s_init)`, recorded
  separately; it is **not** panel entropy or token-normalized entropy.
- `tv_mean`: one-round TV change of panel probabilities.
- `panel_log_mass_mean`: log total model mass assigned to the four candidates.
- `operator_residual_rms/max`: centered post-training scores minus this round's
  actual population target; combines sampling/optimization error and model
  representation bias, not distance to a solved fixed point or pure noise.
- Pair counts, optimizer steps, losses, unclipped gradient norms, wall-clock,
  and the DPO CPU solver residual.

The primary comparisons are same-prompt trajectories and late-round change/
variability, together with operator error. Mean entropy alone can hide cycling.
No independent fixed-point solution, WR evaluation, open-generation collapse
evaluation, relaxation control, or parameter sweep is included in these two
prepared experiments. Do not assert stability-boundary validation or full LLM
convergence from these diagnostics alone. If ordinary training is already
stable, report that; select any additional setting transparently and only
after discussion, not by silently changing this pilot.

## Safe previews and validation

From the repository root, these commands are read-only and never train:

```
python experiments/cyclic_history/run_cyclic_lagged_reference.py --run-id ipo_reference_s0 --check-data
python experiments/cyclic_history/run_cyclic_lagged_sampling.py --run-id dpo_sampling_s0 --check-data
python experiments/cyclic_history/run_cyclic_history.py --run-id dpo_baseline_s0
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m unittest discover -s experiments/cyclic_history -p 'test_*.py' -v
```

The tensor tests use synthetic tensors and a tiny random CPU model, not the
real Qwen weights. Plain entry-point execution is preview-only. Actual training
additionally requires `--execute`, a Slurm allocation, and exactly one visible
GPU. Output directories cannot already exist; there is no implicit resume,
overwrite, auto-resubmit, or early termination. Credentials are not in this
package. The optional `slurm/cyclic_history.sh` template does not submit itself.
Override paths using `--model-path`, `--eval-path`, and `--output-root`.

Before any later user-approved submission, inspect the **full** running and
pending queue and verify allocations. Follow the active workspace `AGENTS.md`
and [phased budget](../../docs/SCHEDULING.md); publishing this new protocol
does not increase the budget. Array throttles alone do not enforce the total; do not launch
independent arrays without accounting for existing work and dependencies.
The one-GPU code guard is not an account-wide scheduler cap.
