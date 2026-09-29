# Stage B: six-arm 100-round LLM phenomenon comparison

Completion: all six tasks in array `4605560` reached state 100 and COMPLETED
with exit 0:0, checked at 2026-09-29T03:56Z. The account queue was empty and
allocated GPUs zero. See `../cyclic_history_stage_b100_results_20260929/REPORT.md`
for the downloaded, validated full trajectories and figures. The one-time
`stage-b` result plotting automation was paused after the check and plotting.

User authorization: proceed directly with Stage B; do not require the LLM to
accurately realize all population-theory results. The user also explicitly
approved two matched ordinary baselines so all six arms cover outer states
**0 through 100**. This is 100 outer updates from the initial model, not 100
optimizer steps or continuation from the fixed-target probes.

| Task | Method | Arm | Alpha | Lambda | Beta | Nu | Kappa |
|---|---|---|---:|---:|---:|---:|---:|
| 0 | IPO | ordinary | .9 | .8 | .2 | 0 | 0 |
| 1 | IPO | lagged reference | .9 | .8 | .2 | .45 | 0 |
| 2 | IPO | feedback extrapolation | .9 | .8 | .2 | 0 | .5 |
| 3 | DPO | ordinary | .9 | .8 | .8 | 0 | 0 |
| 4 | DPO | lagged reference | .9 | .8 | .8 | .45 | 0 |
| 5 | DPO | feedback extrapolation | .9 | .8 | .8 | 0 | .5 |

All use seed 0 and the SAME six previously calibrated prompts, response panels,
role-aligned soft cyclic preference matrices and initial Qwen2.5-1.5B. This is
not the transitive Nemotron scalar-reward experiment. Feedback extrapolation
uses the full synthetic P and is not the manuscript's signed two-sampler loss.

Training is unchanged: response sequence sums including EOS; six unordered
pairs/prompt with soft labels and exact weights; 10 inner epochs, 90 optimizer
updates per outer round; AdamW reset each outer round; LR1e-5, warmup .03;
LoRA r16/alpha32/dropout0; BF16 with cuDNN-SDPA excluded. Total 9,000 optimizer
updates per arm. Raw sampler stays `.2*uniform + .8*softmax(sequence_sum)`.

The source is the immutable Git archive at
`f0fe034cc0d8e3bef35b0f5e02806ee81340da76`. No training code was changed.
The derived plan changes only run IDs, paths, iters 30->100 and the corresponding
configuration contract hashes. Support, losses, coefficients, and inner budget
are unchanged. No additional fixed-target experiments or mixed-orientation
Stage C arms are included. Old results and checkpoints are preserved.

## Measurements and interpretation

- Write one metric row and NPZ snapshot for every state 0..100, 101 in total.
- Save initial adapter plus checkpoints every five rounds, including step 100.
- Compare same-prompt phase portraits, amplitude, phase progression, step size
  and relative-sequence entropy across ordinary/reference/feedback.
- Keep target-fit error and fixed-point distance as diagnostics, not a condition
  to stop, approve another run, or declare a neural result invalid.
- Numerical non-finite values, OOM, data mismatch and other execution failures
  remain errors requiring review; they are not silently ignored.
- Interpret attenuation/continued motion as relative empirical observations.
  Do not require exact population convergence or claim a proven neural limit
  cycle or global stabilization. This remains a selected six-prompt, one-seed
  mechanism study, not a representative general-quality evaluation.

## Deployment and resource guard

Submitted array **4605560**, indices **0-5%6**, at
`2026-09-29T01:18:53.872246+00:00` (September 28 local time).
The full account queue was empty at the submission preflight. At
`2026-09-29T01:20:16Z`, all six tasks were RUNNING, with six allocated H100s
total and no other owned queued jobs. All six initial calibration gates passed,
step-0 metrics and finite snapshots were present, and the log scan found no
execution error. These are startup observations, not completion claims.
At `2026-09-29T01:20:52Z`, all six had completed outer step 1; metrics were
contiguous for states 0..1 and all latest snapshots were finite. No execution
errors were found. The account allocation remained six GPUs; see
`first_round_progress.txt` for this read-only follow-up.

Preflight validation: all 46 remote tests passed, shell syntax passed, and all
six derived configurations passed the data/support/calibration contract checks.
The source archive SHA256 is
`4d09a8748c3b4524fe5e14a1f4786474073573d485d49bfc56af18f1b0c757fb`.
Local receipts: `deployment.json`, `submission_intent.json`,
`submission_receipt.json`, `queue_verified.json`, `cpu_predictions.json`, and
`startup_progress.txt`. The immediate post-submit queue receipt is a racing
scheduler snapshot; use the later startup check for the six-RUNNING state.

Remote code/launch root:
`/work/users/y/u/yuhe32/ipo/diagnostics/history_stage_b100_20260929`

Remote output root:
`/work/users/y/u/yuhe32/ipo_runs/cyclic_history_stage_b100_20260929`

Slurm logs: `ipo_runs/slurm_logs/hist_B100_0929-<job>_<task>.out/.err`.
One GPU per task, array throttle 6, six-hour wall-time limit, no automatic
requeue. Estimate about 2.5 hours per arm after start based on Stage A timings,
excluding queue wait. Actual runtime may vary.

The launcher checks full `squeue -a -r` by owner/UID and `scontrol -a show job`.
It submits only if no other owned running, pending or releasing jobs appeared,
and checks the post-submit total generic `gres/gpu` allocation against six.
Independent submissions can still violate the budget; there is no hard
account-wide cap. Never duplicate a submission after an ambiguous response:
inspect `submission_intent.json`, `submission_receipt.json`, and the live queue.

`inspect_stage_b100.py` is read-only and separate from frozen training source.
Pending(Resources) and Pending(JobArrayTaskLimit) are normal scheduler states.
GitHub synchronization was requested later on September 29. The current
publication includes this launch record; original deployed source is unchanged.
