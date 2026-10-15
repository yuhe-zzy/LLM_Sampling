# Oracle2 real-panel campaign, 2026-10-09

## Monitoring stopped by owner, 2026-10-14

The owner requested stopping this timer. Automation `oracle2` was deleted in
the app, with deletion confirmed. No server job was cancelled, changed or
submitted. This stops automatic checks and phase progression through the timer,
not the experiments already on Sycamore. Do not recreate it without a new user
request. Current remote status remains unknown after the recorded SSH failures.
Earlier statements that the two-hour monitor continues are historical.

## Monitoring connection blocked, 2026-10-14

The18:58Z check made two SSH connection attempts, both timing out before any
remote command. Current allocation, task progress and logs are UNKNOWN, not
inferred from older snapshots. No recovery/WR submission or job change was made.
Last successful inspection05:01:00Z saw only4794738_4/5 RUNNING on twoH100s,
validated banks0/10/20/30/40, partial50 rows548/461, and no new logged error.
Recovery had no output/intent/receipt and WR was unsubmitted at that older check.
The owner was asked to check UNC VPN/Sycamore access. Existing two-hour monitor
continues with read-only inspection; all phase guards remain required.
See [connection failure record](records/attempt5/connection_failure_20261014T1858.json).

## Two generators complete; final original pair running

At2026-10-13T23:00:08Z, IPO feedback4794738_2 and DPO ordinary4794738_3
were COMPLETED0:0. Each has all11 generation banks at states0,10,...,100,
800 responses per bank (8800 per arm), with count/hash/identity and finite
diagnostic checks passed. The original queue automatically started DPO
reference4794738_4 and feedback4794738_5; both have complete state0 and partial
state10 (150/79 rows). They retain24-hour limits. The full owner/UID account
contains only those two running tasks, allocating TWO H100s, no pending jobs.

Original0/1 remain TIMEOUT, with their validated complete/partial banks intact.
Recovery has no output/intent/receipt yet; WR remains unsubmitted and unavailable.
All606 training snapshots remain finite, metrics contiguous, initial scores
matching, baseline800/800 hash-valid, and frozen source/review unchanged. No new
logged OOM/Traceback/nonfinite error. Training/generation memory peaks are not
recorded. No job, science setting, source or result bank was changed.

Wait for4/5 to succeed and full account idle, then follow the existing v5 recovery
guards; do not duplicate completed2/3. The same two-hour monitor continues.
See [validated completion milestone](records/attempt5/two_generators_complete_20261013T2300.json).
All running/pending snapshots in earlier sections are historical.

## Recovery deployed for CPU validation, not submitted

At2026-10-13T06:40:26Z, recovery source
**2d6b5747c251abc44700799be24e3e1e05e9b984** was deployed immutably at
`/work/users/y/u/yuhe32/ipo/diagnostics/oracle2_real_20261009_v5_recovery`.
All129 server CPU tests passed without skips, shell syntax and six reviewed
config checks passed, and real original response identities/seeds/hash bindings
were validated. The byte-identical original plan/lock/review and all old banks
are unchanged. Recovery GPU inference has NOT run.

The contract binds each arm's eight complete banks and state80 prefixes728/561;
remaining responses are1672/1839. New outputs will be written only under
`/work/users/y/u/yuhe32/ipo_runs/oracle2_generation_recovery_20261013_v1`.
No recovery output, submission intent, receipt, or WR job exists yet.
Tasks4794738_2/3 are still RUNNING (state0 rows250 each),4/5 pending, and the
full account allocates2H100s. Recovery preflight correctly rejects this state.
No new logged OOM/Traceback/nonfinite failure was found.

When all four remaining original generators are COMPLETED0:0, recheck the
full owner/UID account is empty and inspect latest intents/receipts. Then use
the frozen v5 `queue.py --root <v5-root> --phase recover --submit` once.
It creates two1-H100 tasks at most2 concurrent,24-hour limits. The subsequent
v5 WR phase uses the two verified recovered arms plus four original arms and
unchanged baseline, still requiring all66 complete banks and an idle account.
Never submit the original six-arm generate phase again. Original running code
and all training remain source23401de. Existing monitoring remains every2hours.
See [deployment and validation status](records/attempt5/recovery_preparation_status.json)
and [bound recovery contract](records/attempt5/generation_recovery.json).

## Missing-response recovery implementation

A tested copy-on-recovery path now preserves completed/partial outputs from
timed-out generation0/1 and generates only missing responses in a separate
directory. No scientific setting or original frozen source is changed.
Recovery submission must wait for successful tasks2..5 and an empty full
account; WR still waits for all66 complete banks. Code publication is not
submission. See [recovery protocol and validation](RECOVERY_20261013.md).

## Policy result inspection, 2026-10-13

Generation tasks4794738_0/1 are now **TIMEOUT**, confirmed06:15:34Z UTC.
Both retain complete states0..70, and partial80 (728/561 rows). Tasks2/3
started under24-hour limits;4/5 wait normally, full account2H100s. WR remains
unsubmitted. Stale RUNNING manifests do not override sacct TIMEOUT. No jobs
were changed or retried, and no completed/partial outputs were overwritten.
See [timeout evidence](records/attempt4/generate_timeout_and_policy_results_20261013.json).

All six completed training trajectories are now available as matched policy
plots, all500-prompt diagnostics and text-free numeric results. Generated WR
is not available yet. The only cyclic training prompt416 starts with almost
all panel probability on a two-token candidate; its flat raw policy is not
evidence of history-induced stabilization. See the full definitions, fixed
prompt-selection rule, counterexamples and reproduction instructions in
[policy results](policy_results_20261013/README.md).

## Generation time-limit risk and partial extension

At **2026-10-12T22:19:21Z**, generation tasks `4794738_0/1` had run for
about four hours. IPO ordinary had complete banks at states **0/10/20** and
91/800 rows at30; IPO reference had complete **0/10** and775/800 rows at20.
All completed banks passed SHA256,800-row,unique-ID,prompt-count and finite
diagnostic checks. The shared baseline remains hash-valid and all606 training
snapshots remain finite. No logged OOM/Traceback/nonfinite failures were found.
Tasks2..5 were normally pending; the full account used two H100s with no
unrelated owned work. WRscore is still unsubmitted.

The observed throughput suggests that generating all11 banks per arm may
exceed the original12-hour request. This is a runtime risk, not an assertion
of inevitable timeout: response lengths can change between checkpoints.
After fresh full-account/source checks, an in-place scheduler update requested
**24 hours for the existing array**, with no cancellation, restart or new job.
At **22:21:28Z**, the update returned code1 and
`4794738_0-1: Access/permission denied`. Post-update scontrol and sacct showed
**partial success**, which must not be inferred from the exit code alone:

| Tasks | State | Actual time limit after update |
| --- | --- | --- |
| 4794738_0,4794738_1 | RUNNING | 12 hours, unchanged; update denied |
| 4794738_2..5 | PENDING | 24 hours, successfully extended |

`TZ=UTC scontrol` confirms the running tasks' scheduled end is
**2026-10-13T06:14:50Z**. Administrator assistance is needed to extend these
running tasks in place; the owner has been notified. Do not repeat the denied
operation or bypass permissions. The pending tasks retain the same IDs and
array throttle2; allocated GPUs remain2 and the account ceiling remains4.
The partition maximum is five days, but that does not grant this non-admin
account permission to extend an already-running job.

This is a **scheduler-only override**. Frozen source23401de, plan, review,
models, data, sampling parameters and original submission receipts are
unchanged. There is no replacement launch or resubmission. Preserve complete
and partial generation outputs if a later timeout occurs; diagnose and test
any recovery without overwriting or regenerating completed banks. The existing
two-hour monitor now includes this risk and partial result. Only all66
completed and validated banks plus an idle account can unlock WRscore.

See [time-limit risk and verified partial update](records/attempt4/generate_timelimit_risk_20261012T2221.json).

## All training complete; open generation started

At **2026-10-12T18:10:50Z**, all six `4773902_0..5` tasks were verified
**COMPLETED, exit 0:0**, with all outer states **0..100**. All **606 numeric
snapshots** are finite, metrics are contiguous, initial scores match, and
source/review hashes are unchanged. The shared baseline remains hash-valid,
800/800 responses. No logged OOM/Traceback/nonfinite failures were found.
The full owner/UID account queue was empty with zero allocated GPUs.

After verifying all 66 required state adapters and the absence of earlier
generation outputs/intents/receipts, the frozen `queue.py` repeated its
source/config/review, phase-completion, and empty-account guards and submitted
**generation array `4794738_0..5%2` at 2026-10-12T18:14:50Z**. Each task
requests one H100, 80 GB host memory, and a 12-hour limit. Task order remains
IPO ordinary/reference/feedback, then DPO ordinary/reference/feedback.
**Never duplicate this submission.** No training or candidate scoring was rerun.

At **18:16:56Z**, tasks0/1 were RUNNING and tasks2..5 were normally pending
with `JobArrayTaskLimit`. Full-account allocation was **TWO H100s**, with no
unrelated owned work. The two running arms had written **14/800** and
**13/800** step-0 responses respectively; these are partial banks, not completed
checkpoints or WR results. No logged execution errors were found. The initial
18:15:29Z snapshot had zero written rows, so actual generation has progressed.

Each arm will generate 800 responses (200 evaluation prompts, four draws) at
each of 11 states: 0,10,...,100. WR scoring remains **unsubmitted**. Only after
the baseline and all 66 arm/state banks complete with validated counts/hashes
and the full account is idle may the approved two-GPU sequential judge phase
start. The existing two-hour monitor now records this generation phase.
The four-GPU account ceiling still applies; the array throttle is not a hard
account-wide cap. Training/generation peak GPU-memory telemetry is not recorded
by the frozen executables; earlier candidate-judge peaks must not substitute.

Execution source remains **`23401de58b27ef2f7b1ea645551f37602a3c1bb8`** in the
immutable `_v4_empirical` launch, reusing the same v3 private data/output root.
This update changes documentation and aggregate records only, not code or
scientific settings. Completion does not establish convergence or a history-arm
benefit; those questions require trajectory and subsequent WR analysis.

Records: [training completion](records/attempt4/training_complete_status_20261012.json),
[generation intent](records/attempt4/generate_submission_intent.json),
[generation receipt](records/attempt4/generate_submission_receipt.json), and
[generation startup](records/attempt4/generate_started_status_20261012.json).

## DPO history arms saved step-90 checkpoints

At **2026-10-12T09:30:31Z**, DPO reference (`4773902_4`) and feedback
(`4773902_5`) reached complete outer states **92/100** and **91/100**;
both saved adapters at 10,20,...,90. The other four arms remain COMPLETED,
exit 0:0, at state100. Only tasks4/5 are in the full owner/UID queue,
allocating **TWO H100s total**, with no unrelated owned work. All **589
snapshots** are finite and metrics are contiguous from state0; initial scores
match and frozen source/review hashes remain unchanged. The shared 800/800
baseline is still hash-valid. No logged OOM/Traceback/nonfinite failures.

No resource action, code or scientific change was made. Generation and WR
remain unsubmitted until all six training arms succeed and the full account
is idle. The two-hour monitor continues. The frozen trainer has no GPU-memory
peak telemetry; candidate-judge peaks do not describe training memory.
See [validated DPO step-90 milestone](records/attempt4/dpo_checkpoint90_status_20261012T0930.json).

## DPO reference saved its step-80 checkpoint

At **2026-10-12T06:56:10Z**, DPO reference (`4773902_4`) reached complete
outer state **80/100**, with adapters saved through80. DPO feedback
(`4773902_5`) reached **79/100**, with adapters through70; its step80 is not
yet complete. The other four arms remain COMPLETED, exit0:0, at state100.
Only tasks4/5 are in the full owner/UID queue, allocating **TWO H100s total**,
with no unrelated owned work. All **565 snapshots** are finite, metrics are
contiguous from state0, initial scores match, and frozen source/review hashes
remain unchanged. The shared 800/800 baseline remains hash-valid.
No logged OOM/Traceback/nonfinite failures were found.

No resource, code or scientific change was made. Generation and WR remain
unsubmitted until all six training arms succeed and the full account is idle.
The two-hour monitor continues. Training GPU-memory peaks are not recorded
by the frozen trainer and cannot be inferred from candidate-judge telemetry.
See [validated reference step-80 milestone](records/attempt4/dpo_reference_checkpoint80_status_20261012T0656.json).

## DPO history arms saved step-70 checkpoints

At **2026-10-12T04:59:33Z**, DPO reference (`4773902_4`) and feedback
(`4773902_5`) reached complete outer states **71/100** and **70/100**;
both saved adapters at 10,20,...,70. The other four arms remain COMPLETED,
exit 0:0, at state100. Only tasks4/5 are in the full owner/UID queue,
allocating **TWO H100s total**, with no unrelated owned work. All **547
snapshots** are finite and metrics are contiguous from state0; initial scores
match and frozen source/review hashes remain unchanged. The shared 800/800
baseline is still hash-valid. No logged OOM/Traceback/nonfinite failures.

No resource action, code or scientific change was made. Generation and WR
remain unsubmitted until all six training arms succeed and the full account
is idle. The two-hour monitor continues. The frozen trainer has no GPU-memory
peak telemetry; candidate-judge peaks do not describe training memory.
See [validated DPO step-70 milestone](records/attempt4/dpo_checkpoint70_status_20261012T0459.json).

## DPO history arms saved step-60 checkpoints

At **2026-10-12T03:01:46Z**, DPO reference (`4773902_4`) and feedback
(`4773902_5`) reached complete outer states **62/100** and **61/100**;
both saved adapters at 10,20,...,60. The other four arms remain COMPLETED,
exit 0:0, at state100. Only tasks4/5 are in the full owner/UID queue,
allocating **TWO H100s total**, with no unrelated owned work. All **529
snapshots** are finite and metrics are contiguous from state0; initial scores
match and frozen source/review hashes remain unchanged. The shared800/800
baseline is still hash-valid. No logged OOM/Traceback/nonfinite failures.

No resource action, code or scientific change was made. Generation and WR
remain unsubmitted until all six training arms succeed and the full account
is idle. The two-hour monitor continues. The frozen trainer has no GPU-memory
peak telemetry; candidate-judge peaks do not describe training memory.
See [validated DPO step-60 milestone](records/attempt4/dpo_checkpoint60_status_20261012T0301.json).

## Connection recovered; four of six training arms complete

At **2026-10-12T01:05:30Z** (October 11, 20:05 America/Chicago), SSH access
recovered. The full owner/UID queue, scontrol allocations and sacct agree:

| Task | Arm | State | Latest complete outer state |
| --- | --- | --- | --- |
| 4773902_0 | IPO ordinary | COMPLETED, exit 0:0 | 100 |
| 4773902_1 | IPO reference | COMPLETED, exit 0:0 | 100 |
| 4773902_2 | IPO feedback | COMPLETED, exit 0:0 | 100 |
| 4773902_3 | DPO ordinary | COMPLETED, exit 0:0 | 100 |
| 4773902_4 | DPO reference | RUNNING | 53 |
| 4773902_5 | DPO feedback | RUNNING | 52 |

The only owned live jobs are tasks 4 and 5, allocating **TWO H100s total**;
no unrelated owned running or pending work was found. All **511 snapshots**
through the latest complete states are finite, with contiguous metrics,
matching initial scores, unchanged source/review hashes and no prompt
truncation. All four completed arms saved adapters at 10,20,...,100; the two
running arms saved them through 50. No logged OOM/Traceback/nonfinite failures.
Training GPU-memory peaks are not recorded by the frozen trainer; candidate
judge peaks are not training telemetry.

The shared **800/800 baseline** remains complete and hash-valid. Trained
checkpoint generation and WR scoring have **not** been submitted: wait until
all six training arms complete successfully and the full account is idle.
No submission, cancellation, rerun, concurrency or scientific change was made.
The existing two-hour monitor continues. Execution source remains
`23401de58b27ef2f7b1ea645551f37602a3c1bb8`, not this status-documentation commit.
See [validated recovery and completion milestone](records/attempt4/four_arms_complete_status_20261012T0105.json).
This status alone is not evidence for convergence or a history-term benefit.

## Historical SSH connectivity interruption (resolved above)

At **2026-10-10T10:00:29Z**, both connection attempts for the 09:58 heartbeat
had timed out in `sock.connect`, before any remote command ran. GPU allocation,
training progress and execution errors were **unknown during the interruption**.
This did not establish an experiment failure. The 08:02 checkpoint snapshot below
was then the last successful check. The owner was asked to
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
