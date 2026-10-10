# Oracle2 real-panel campaign, 2026-10-09

## Latest monitoring check blocked by SSH connectivity

At **2026-10-10T10:00:29Z**, both connection attempts for the 09:58 heartbeat
had timed out in `sock.connect`, before any remote command ran. Current GPU
allocation, training progress and execution errors are **unknown**. This does
not establish an experiment failure. The 08:02 checkpoint snapshot below is
the last successful check, not a current-status claim. The owner was asked to
check UNC VPN and Sycamore connectivity. No task, source or timer was changed;
the existing two-hour monitor remains in place. See
[connection failure record](records/attempt4/connection_failure_20261010T1000.json).

## WR checkpoints through step 60 saved

At **2026-10-10T08:02:07Z**, IPO ordinary (`4773902_0`) reached complete outer
**61/100** and IPO reference (`4773902_1`) reached **62/100**. Both have saved
their **step-10/20/30/40/50/60 adapters** for later open-generation evaluation.
Metrics are contiguous from state 0, with **125 finite numeric snapshots**
in total; initial scores still match and
frozen source/review hashes verify. The shared 800-response baseline remains
complete and hash-valid. Tasks 2..5 remain PENDING(JobArrayTaskLimit), with
TWO H100s allocated across the full account and no unrelated owned work.
No logged OOM/Traceback/nonfinite failure was found. No trained-checkpoint
generation or WR result exists yet; no phase was submitted or changed.
See [current checkpoint milestone](records/attempt4/checkpoint60_status.json),
[earlier step-50 milestone](records/attempt4/checkpoint50_status.json),
[earlier step-40 milestone](records/attempt4/checkpoint40_status.json),
[earlier step-30 milestone](records/attempt4/checkpoint30_status.json),
[earlier step-20 milestone](records/attempt4/checkpoint20_status.json),
and [earlier step-10 milestone](records/attempt4/checkpoint10_status.json).

## Baseline complete; first two training arms running

Checked **2026-10-09T19:55:56Z**: baseline **4773899 COMPLETED, exit 0:0**,
elapsed 37m33s, with all **800/800 responses** and a verified file hash. Its
dominant-token >=.95 fraction is zero; this is an initial-model diagnostic,
not a trained-model generation result.

Training **4773902_0 (IPO ordinary)** and **4773902_1 (IPO reference)** are
RUNNING, both at complete outer state **6/100**. Metrics cover states 0..6
without gaps; all **14 numeric snapshots** are finite with shape 500x4, matched
initial scores, unchanged source/review hashes and no prompt truncation.
Tasks 2..5 are normally PENDING(JobArrayTaskLimit). The full owner/UID queue
and scontrol allocation confirm TWO H100s total and no other owned work.
No logged OOM, Traceback or nonfinite failures. The frozen trainer does not
record GPU-memory peaks; do not reuse candidate-judge peaks as training telemetry.

No checkpoint-generated WR is available yet. All phases and scientific settings
are unchanged; no job was submitted/cancelled/reconfigured during this check.
The existing monitor remains every two hours. See
[aggregate milestone](records/attempt4/training_started_status.json).

## Six-arm submission confirmed

At **2026-10-09T17:55:53Z**, shared baseline job **4773899** was submitted on
one H100 after an empty full-account preflight. At **17:56:02Z**, training
array **4773902_0..5%2** was submitted with **afterok:4773899**, one H100 per
task. Task order: IPO ordinary/reference/feedback, then DPO ordinary/reference/
feedback. Frozen execution source is `23401de58b27ef2f7b1ea645551f37602a3c1bb8`.

All **113 server CPU tests passed without skips** (37 Oracle2 + 76 shared),
shell syntax and six config/review checks passed, and completed scores and
audited panels were hash-verified without rescoring. At **17:56:36Z**, baseline
was RUNNING with 7 complete generated-response lines out of the intended 800,
six training tasks were PENDING(Dependency), and the full account allocated
ONE GPU, with no other owned work. No logged OOM/Traceback/nonfinite error.
This is startup evidence, not baseline completion or successful training.
Generation/WR phases remain unsubmitted until their prerequisites succeed.

The existing `oracle2` heartbeat is ACTIVE every TWO hours following the
completed memory/scoring audit and explicit empirical-scope approval.
See [baseline receipt](records/attempt4/baseline_submission_receipt.json),
[training receipt](records/attempt4/train_submission_receipt.json),
[training dependency and preflight](records/attempt4/train_submission_intent.json),
[validation](records/attempt4/validation_summary.json), and
[startup status](records/attempt4/startup_status.json). Never duplicate these jobs.

## User-approved empirical stability scope (supersedes the hold)

On 2026-10-09, after seeing the 0/1/0 cyclic counts below, the owner explicitly
approved continuing the same six arms even with only one cyclic prompt. The
objective is to inspect relative stabilization trends, not require elimination
of oscillations or establish a replicated cyclic-vs-transitive effect.

[Exact audit review](audit_review_empirical.json) records this limitation and
the observed counts. No panel, split, reward, mixture, temperature, threshold,
seed, arm or training formula changes. Ambiguous panels are not automatically
non-ST. The single cyclic training panel is descriptive; the empty cyclic
evaluation subgroup must remain missing in WR summaries, not be filled with zero.
Retain all prompts and counterexamples. WR trends are not a convergence proof.

New immutable launch root: `ipo/diagnostics/oracle2_real_20261009_v4_empirical`.
Reuse audited private data and the existing output root from `_v3_2gpu` with
the byte-identical `plan_two_gpu.json`; do not rescore or modify the v3 source.
See [reuse/provenance contract](reuse_candidate_audit.json). The new source and
review are bound in the deployment and each training manifest. Local records
are under `oracle2_real_launch_20261009/attempt4`.

Launch sequence: shared initial-model baseline (one H100), then the six-arm
training array `0-5%2` with `afterok` on that recorded baseline job. Queueing
this dependency is the only empty-account exception; unrelated owned work
blocks submission. A failed predecessor is not bypassed. Startup also verifies
the completed 800-response baseline. Generate all saved WR checkpoints only
after training completes, then score the generated banks with two H100s.
Phases never overlap; full account limit FOUR, current phase maximum TWO.
Deployment must pass CPU tests and all six config/data/review checks before
submission. The confirmed submissions above implement this authorization.

The completed scoring/memory audit plus this explicit scope decision allowed
switching the existing `oracle2` heartbeat to every two hours after launch.
Report failures, meaningful milestones and new jobs; unchanged progress is silent.

## Historical scoring completion and initial hold (superseded above)

Checked **2026-10-09T17:42:45Z**: job **4773843 COMPLETED, exit 0:0**, elapsed
18m07s. Both Nemotron and Skywork completed **3200/3200** scores without logged
OOM, Traceback or nonfinite failures. Nemotron peak tensor allocation was
66.33/66.30 GiB across two H100s; Skywork peak was 14.18 GiB on its active GPU.
This validates two-GPU scoring for this candidate bank, not every future input.
The complete owner/UID queue was empty, total allocated GPUs zero.

The **scientific readiness review is on hold**, distinct from successful code
execution. Frozen 0.6/0.4 mixture, temperatures 1, four real responses per prompt:

| Split | Total prompts | Cyclic | Transitive | Ambiguous |
|---|---:|---:|---:|---:|
| Calibration | 100 | 0 | 81 | 19 |
| Training | 500 | 1 | 436 | 63 |
| Evaluation | 200 | 0 | 160 | 40 |

"Cyclic" uses the prespecified robust criterion: one directed triangle with
**all three edges P > 0.52**. Ambiguous panels may include near ties or weaker
cycles; these counts do not assert that every ambiguous panel is acyclic.
The cached-score diagnostic 0.7/0.3 mixture has zero robust cyclic panels in
all three splits, so simply switching to 7/3 is not supported by this audit.
Training-set component saturation (P<.01 or P>.99) is 34.8% for Nemotron and
49.3% for Skywork; these are diagnostics, not proof of the cause of scarcity.
BT solver checks had no failures (maximum training residual <1e-10).

One training cyclic prompt and none in evaluation cannot support the intended
cyclic/noncyclic comparison. **No baseline or six-arm training is approved or
submitted**, no `APPROVE_SIX_ARMS` was created, and no settings were retuned.
Preserve all scores and await user discussion of data/scale diagnostics.
Heartbeat `oracle2` retains its ten-minute schedule because readiness did not
pass; unchanged waiting status remains silent. No task was cancelled or rerun.

Evidence: [candidate audit](records/attempt3/completed_candidate_audit.json),
[score/memory manifest](records/attempt3/completed_score_manifest.json), and
[held readiness review](records/attempt3/completion_review.json).
Actual execution source remains `55e835ec5b9015404a7068c7ebf940af1feb020f`.

## First actual two-GPU scoring milestone

At 2026-10-09T17:22:53Z, **4773843** remained the sole owned RUNNING job,
using two H100s. Nemotron progress reached **250/3200**, with approximately
11 GiB free and 66.3 GiB peak tensor allocation per GPU; no logged OOM,
Traceback or nonfinite scores. Initial inference now works on two GPUs, but
this does not establish full-input memory sufficiency or audit completion.
Skywork and cyclic-group counts are still unavailable. The existing timer
remains at ten minutes; no task was submitted, cancelled or reconfigured.
See [numeric milestone](records/attempt3/first_scoring_status.json).
The raw line count (231) was read before the progress file (250); concurrent
live-file reads are not an atomic snapshot and are not completion evidence.

## Two-GPU submission

**4773843** submitted at **2026-10-09T17:17:54Z**, two H100s, frozen source
`55e835ec5b9015404a7068c7ebf940af1feb020f`. All **105 server CPU tests**
passed without skips, plus actual tokenizers, shell syntax and full data checks.
969 panels were eligible; unchanged 100/500/200 split, 3200 real candidates.
At 17:18:32Z the full owner/UID queue contained only this RUNNING job, total
allocation two GPUs. Score initialization was in progress with no logged error;
GPU memory sufficiency and cyclic counts were not yet established. Six training
arms are still unsubmitted. Never duplicate this audit. See
[submission receipt](records/attempt3/audit_submission_receipt.json),
[validation](records/attempt3/validation_summary.json), and
[post-submit state](records/attempt3/post_submit_status.json).

## Latest authorization: two-GPU audit, four-GPU account ceiling

The user lifted the temporary no-retry hold and authorized starting the repaired
code with TWO GPUs. The new total account limit is FOUR, superseding all older
six-GPU text. New immutable launch/output suffix: `oracle2_real_20261009_v3_2gpu`,
local records: `oracle2_real_launch_20261009/attempt3`.
Use [plan_two_gpu.json](plan_two_gpu.json); only allocation and paths change.
The submission above is the execution of this authorization. Inspect receipts first.

Candidate scoring and WR scoring load Nemotron then Skywork sequentially on
two H100s. Training/generation retain six arms at one H100 each, array `0-5%2`.
Each phase requires an empty full-account queue and checks allocation at startup;
no phases overlap, and unrelated jobs must be included. This is not an installed
administrator account cap. No old job is cancelled or duplicated.

The user-authorized heartbeat `oracle2` checks every 10 minutes for OOM,
Tracebacks, numerical errors and progress, and may debug technical failures.
After BOTH judges complete all candidate scores and the audit passes with actual
cyclic counts, update this same heartbeat to every two hours. Missing/tiny cyclic
groups or saturation issues require discussion, not silent scientific changes.
No-change checks stay quiet. Source/receipt and completed artifacts stay immutable.

## Earlier repair validation and temporary hold (superseded)

The user explicitly requested **code repair only; do not resubmit for now**.
No new GPU job, six-arm training or automatic follow-up was submitted/scheduled.
Future GPU submission requires fresh user authorization.

Corrected source `7d90b1af6532852b5e7535927fc40aa443b490f9` was deployed to
the separate `_v2` root for CPU checks only. All **101 server CPU tests passed
without skips** (25 oracle2 + 76 shared), shell syntax passed, and actual locked
Nemotron/Skywork tokenizers produced valid `[1,61]` / `[1,38]` input tensors
on the probe. Full corrected token-length and raw-provenance checks passed:
969 eligible panels, 31 excluded for policy length, unchanged 100/500/200 split.
This does **not** validate GPU reward inference or establish cyclic-group counts.
At 2026-10-09T17:07:44Z the full owner/UID queue was empty, zero allocated GPUs;
the corrected root had no submission receipts or runs. WR remains 0,10,...,100.
See [CPU-only validation/provenance](records/attempt2/validation_summary.json),
[data manifest](records/attempt2/data_manifest.json), and
[read-only account snapshot](records/attempt2/cpu_only_status.json).

## First-attempt compatibility failure

Audit 4773811 FAILED, exit1:0, at 2026-10-09T16:57:22Z before
writing any reward (0/3200). It loaded Nemotron but failed constructing the
input tensor. Transformers 5.13.1 returns `BatchEncoding` by default from
`apply_chat_template`, not a flat token-ID list. Consequently the initial
judge length check incorrectly counted two dictionary keys; do not treat that
part of the initial preflight as valid. Raw-data and policy-length checks were
unaffected. No training has run and no mixture/cycle result exists yet.

The repair explicitly sets `return_dict=False`, validates a nonempty flat
integer list, and uses that single helper for preparation and scoring. Actual
locked-tokenizer CPU probes reproduced the default and verified the corrected
list/tensor contract (61 Nemotron and 38 Skywork tokens for the public test
conversation). Added regression tests also reject dictionary outputs. The
strict >.52 cycle boundary is now compared directly to avoid subtraction
roundoff. Historical sources/results are not overwritten.

Attempt2 is **CPU-validated, not submitted**, with identical scientific
parameters and separate roots ending in `_v2`, using
[plan_attempt2.json](plan_attempt2.json). The completed checks are recorded
above. The user declined resubmission for now; there is no automatic rerun or
training-chain authorization from the failure itself.

The inspector also now distinguishes terminal records retained by scontrol
from live allocations. A terminal job's historical AllocTRES must not be
reported as current GPU use. The original 16:56 RUNNING snapshot below is
historical; the later failure supersedes it.

## Submitted candidate audit

**Job 4773811**, submitted **2026-10-09T16:55:18Z**, requests three H100s.
Actual frozen source: `975726c30675e9a3df11da1582482f6bd9c07b46`.
At 16:56:02Z it was RUNNING; full owner/UID queue and scontrol showed only
this owned job, generic AllocTRES gpu=3. No other owned pending/running jobs
and no logged execution error at that time. Judge loading had begun; no
inference/cyclic-audit success is inferred from a running scheduler state.

All **96 server CPU tests passed without skips** (20 new + 76 shared engine),
as did shell syntax and raw-data/both-tokenizer checks. Exactly 969/1000
panels passed no-truncation eligibility; 800 were selected before scoring
and split 100/500/200. All 3,200 candidates are real source responses.

The six formal training tasks have **not** been submitted. Review the actual
mixed matrices, group counts, scales and numerical checks before approving
that phase. No follow-up timer or automatic phase chain was created. Never
duplicate the audit job or replace its source. These facts supersede the
earlier preparation-only paragraph below; timestamps are not live guarantees.

Machine-readable receipts, model/data provenance and validation summaries
are under [records](records/). Raw text/reward dumps/weights remain private.

## Authorization and current preparation

User confirmed 6/4 Nemotron/Skywork, fixed real candidates, the six matched
ordinary/reference/feedback arms, 100/500/200 disjoint prompt split and seed0.
**WR cadence is every ten outer updates, including 0 and 100.**
No additional seeds, mixing weights, temperature changes or parameter sweeps.

Read-only server preflight found no owner/UID448057 jobs in the full queue;
anchor90 tasks 4728255_0 and 4728255_1 were COMPLETED, exit0:0. This is a
historical snapshot, not a continuing allocation guarantee.

Pinned Skywork revision: `d4117fbfd81b72f41b96341238baa1e3e90a4ce1`.
Downloaded into a new revision-suffixed model directory; old Qwen/Nemotron
directories and running/completed experiment sources were not modified.
Initial policy-only length inventory found 969/1000 usable panels. Full
raw provenance and both judge-tokenizer checks still run before GPU scoring.

At document creation, no oracle2 GPU task has been submitted. Subsequent
submission records and audit results must explicitly supersede this statement;
never infer a job ID from an authorization or code commit.

## Current locations (two-GPU attempt)

- Launch: `/work/users/y/u/yuhe32/ipo/diagnostics/oracle2_real_20261009_v3_2gpu`
- Frozen source: launch root `/source`
- Private input panels: launch root `/private_data`
- Outputs: `/work/users/y/u/yuhe32/ipo_runs/oracle2_real_20261009_v3_2gpu`
- Local records: `oracle2_real_launch_20261009/attempt3`

The source archive includes this experiment and the unchanged shared history
engine. `deployment.json` binds the actual commit and every deployed source
file. Model lock and plan hashes are carried into data/scoring/training
manifests. Do not replace a deployment in place after a task uses it.

## Commands after reviewed deployment

Use the existing h100env312 Python. Preparation is CPU-only. These commands
are documentation, not a batch that blindly starts every phase:

```bash
ROOT=/work/users/y/u/yuhe32/ipo/diagnostics/oracle2_real_20261009_v3_2gpu
PY=/work/users/y/u/yuhe32/h100env312/bin/python
CODE=$ROOT/source/experiments/oracle2
# Data preparation is performed once by deployment; do not overwrite it.
$PY $CODE/campaigns/oracle2_real_20261009/queue.py --root "$ROOT" --phase audit
# --submit only after preflight and confirming no existing intent/receipt.
```

Inspect candidate score manifest and `private_data/scored/audit.json` before
training. If adequate cyclic/transitive groups and numerics pass review,
record `audit_review.json` with `decision=APPROVE_SIX_ARMS`, the exact
`audit_sha256`, actual group counts and the review rationale. Never create
that approval in advance. If scales saturate or cycles are too scarce, report
the evidence and discuss calibration; do not silently change the 6/4 oracle.

Subsequent supported phases: `baseline`, `train`, `generate`, `wrscore`.
Wait for each preceding phase to finish, inspect results and live account
state, then submit the next. No phase submits another phase. The training
array uses 0..5%2, one H100/task. Candidate/WR scoring tries two H100s.
All submissions require fresh empty-account checks and exclusive intent files.

See the [experiment README](../../README.md) for objectives, grouping, WR
definition, resource schedule and privacy restrictions.
