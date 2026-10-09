# Experiment collaboration

## Synchronize changes to GitHub

The owner requested on 2026-09-29 that every experiment modification be
synchronized to GitHub. Fetch and integrate collaborator updates first, keep
edits scoped, run relevant checks, commit and push, then verify the remote
commit. Never force-push or discard collaborators' changes. The current cyclic
campaign is shared on `hodge-diagnostics`; do not silently rewrite `main`.
If remote access fails, retain the local work and report that it is not synced.

Include code, configs, tests, reproducibility instructions, dated execution
status and shareable numeric summaries/figures. Exclude credentials, private
raw prompt/response text, model weights, adapters, raw dumps, caches and logs.
Do not use `git add .` without reviewing the complete file list. Do not replace
an immutable campaign source with current HEAD or claim that a documentation
update was used in an earlier run. Publishing code does not authorize GPU jobs.

## GPU and campaign safety

Oracle2 checkpoint milestone 2026-10-09T21:55:27Z: train4773902_0/1 remain
RUNNING, both complete outer15/100, step-10 adapters saved. All states0..15
metrics contiguous, 32 finite snapshots total, matching initial scores and
unchanged review/source. Baseline800/800 remains hash-valid. _2..5 normal
PENDING(JobArrayTaskLimit); full account TWO H100s, no unrelated owned work
or logged OOM/Traceback/nonfinite. No generated-checkpoint WR yet; no tasks
changed. Existing two-hour monitor continues. See records/attempt4/checkpoint10_status.json.

Oracle2 milestone 2026-10-09T19:55:56Z: baseline4773899 COMPLETED 0:0,
800/800 generated responses, count/hash verified. Train4773902_0 IPO ordinary
and _1 IPO reference RUNNING, both complete outer6/100; metrics0..6 and all
14 finite numeric snapshots validated, identical initial scores. _2..5 remain
PENDING(JobArrayTaskLimit), normal. Full owner/UID queue: TWO H100s allocated,
no other owned work, no logged OOM/Traceback/nonfinite. No checkpoint WR yet.
No resource actions or code changes; timer remains two hours. See campaign
records/attempt4/training_started_status.json. Do not duplicate baseline/train.

Oracle2 empirical submission confirmed 2026-10-09T17:56Z: baseline4773899
(one H100) and train4773902_0..5%2 (one H100/task, afterok:4773899), source
23401de58b27ef2f7b1ea645551f37602a3c1bb8. All 113 server CPU tests passed
without skips plus six reviewed config checks and score/support hash reuse.
At17:56:36Z baseline RUNNING, 7/800 response rows, training PENDING(Dependency),
full account ONE GPU and no other owned work, no logged errors. No training
completion claim. Receipts under records/attempt4 and local attempt4; never
duplicate. Existing oracle2 timer now every TWO HOURS; generate/WR not yet
submitted and must wait for complete successful prerequisite phases.

Latest Oracle2 authorization 2026-10-09 supersedes the sparse-cycle HOLD below:
the owner accepts only one robust cyclic training prompt and approves the same
six arms to inspect empirical stability trends, not eliminate oscillations or
prove convergence. Use audit_review_empirical.json with exact audit hash and
counts; no scientific settings change, no ambiguous-to-non-ST relabeling.
Create immutable `_v4_empirical` launch using the unchanged plan_two_gpu.json
and reused v3 data/output paths; never rescore completed audit4773843 or edit
v3 source. Submit one-GPU baseline then six one-GPU arms 0-5%2 with afterok
of that baseline. This baseline-only dependency is the empty-queue exception;
unrelated owned jobs still block submission. No overlapping phases, FOUR-GPU
full-account maximum. Preserve intent/receipt; never blindly duplicate phases.
After reviewed successful audit and launch, update existing oracle2 monitor to
two hours. Missing cyclic eval WR stays missing. See campaign README for scope.

Oracle2 authorization, 2026-10-09: the owner confirmed fixed real-candidate
training plus open-generation WR, .6 Nemotron/.4 Skywork, six IPO/DPO
ordinary/reference/feedback arms, seed0, 100 outer updates, and disjoint
100 calibration/500 train/200 evaluation prompts with four candidates each.
WR must use outer states 0,10,...,100. Code and status live in
`experiments/oracle2/campaigns/oracle2_real_20261009`. Do not duplicate phases;
read deployment/intent/receipt and live owner/UID queue first. Audit the true
mixed matrices and judge interfaces before training, report too few cycles
or saturated scales rather than silently changing temperatures or mixture.
Oracle2 latest 2026-10-09T17:42:45Z: **4773843 COMPLETED 0:0**, 18m07s,
both judges 3200/3200, no execution errors; two-H100 memory check passed for
the candidate bank (Nemotron peak 66.3 GiB per GPU). Full queue empty, zero
allocated GPUs. Readiness review HOLD: robust cyclic counts calib/train/eval
0/1/0, transitive 81/436/160, ambiguous 19/63/40. Offline .7 mixture also
has zero robust cycles. Do not start any follow-on GPU phase or create a
training approval; await user discussion. No rerun or scientific retuning.
Heartbeat oracle2 remains 10-minute, unchanged hold silent. See campaign
records/attempt3/completion_review.json. Source remains frozen at 55e835e.

Historical Oracle2 check at 2026-10-09T17:22:53Z: **4773843** is sole owned RUNNING
job, total two H100s. Nemotron progress reached 250/3200 with about 11 GiB
free and 66.3 GiB peak allocated per GPU, no logged errors. This is initial
inference success, NOT complete candidate/cycle audit; Skywork still pending.
Keep the 10-minute heartbeat, do not start training or duplicate scoring.
See campaign records/attempt3/first_scoring_status.json; live line/progress
reads are not simultaneous. No GPU/scheduling changes were made in this check.

Oracle2 latest submission: **4773843**, two H100s, 2026-10-09T17:17:54Z,
frozen source `55e835ec5b9015404a7068c7ebf940af1feb020f`, `_v3_2gpu` roots.
105 server CPU tests passed without skips and data/tokenizer checks passed.
At 17:18:32Z it was RUNNING, sole owned job, total allocation two GPUs;
judge scores were initializing with no logged errors. This is not a completed
GPU memory/cycle audit. Six training arms are unsubmitted. Never duplicate;
see campaign records/attempt3 receipts and current live state before actions.

Latest user authorization 2026-10-09 supersedes older six-GPU limits and the
no-retry hold below: FOUR GPUs maximum across the entire account. Try TWO
H100s for the repaired Oracle2 audit in new immutable `_v3_2gpu` roots using
`plan_two_gpu.json`. Six one-GPU training/generation tasks use 0-5%2, and
scoring loads the two judges sequentially. Phases never overlap; full-account
empty-queue preflight and startup allocation guards are mandatory. Preserve
old sources/results; check new intent/receipt before any retry. Technical
debugging/repair is authorized, not altered science or duplicate completed work.
Heartbeat `oracle2`: 10-minute checks until both judges finish all candidate
scores and cyclic-count audit passes, then update it to two-hour checks.
Notify errors/milestones only; absent/tiny cyclic groups need discussion.
No extra seed/arm and no cancellation of unrelated jobs. Hard scheduler cap
is not installed; guards cannot prevent independent external submissions.

Oracle2 candidate audit has been submitted as **4773811**, three H100s,
frozen source `975726c30675e9a3df11da1582482f6bd9c07b46`; 96 server CPU tests
passed without skips. At 2026-10-09T16:56:02Z it was RUNNING and was the only
owned job (three allocated GPUs). Judge scores were still initializing, so
this is not inference/audit completion. Six formal training arms have NOT
been submitted; inspect the actual audit before creating its review record.
Never duplicate the audit or silently retune temperatures. No automatic
phase chaining/follow-up was created. See the campaign's receipts/status.

Later 2026-10-09 update supersedes the running snapshot: 4773811 FAILED at
16:57:22Z before scoring any candidate. Transformers 5 chat templates returned
BatchEncoding instead of token IDs; the original judge-length check also
counted dict keys and is invalid. The repair explicitly requests/validates flat
IDs for both preprocessing/scoring. Attempt2 uses separate `_v2` roots and
unchanged experiment parameters. The user then explicitly requested code
repair only, NO RESUBMISSION for now. Corrected frozen source
`7d90b1af6532852b5e7535927fc40aa443b490f9` passed all 101 server CPU tests
without skips, both real-tokenizer probes and full corrected length/provenance
checks (969 eligible, 100/500/200 selected). No post-repair GPU inference has
been tested. No retry or training is submitted, and no follow-up is scheduled.
Any new GPU submission requires fresh user authorization. Full owner/UID
queue at 2026-10-09T17:07:44Z was empty, zero allocated GPUs. Preserve the
failed source and first-attempt records; see the campaign's attempt2 records.

At most FOUR allocated GPUs total across all owner programs, not four per array.
Before resource actions inspect both running and pending work using the full
`squeue -a -r` owner/UID listing and `scontrol -a show job`. On Sycamore the owner
is yuhe32, UID448057; `squeue -u` has returned false empty results. Count generic
AllocTRES `gres/gpu` once. There is no installed account-wide hard cap.

The dated status in `experiments/cyclic_history/campaigns/README.md` is a
historical snapshot, not a current allocation guarantee. A/B, fixed-target
probes and Stage C have already been submitted; never duplicate their IDs or
restore old dependencies. Keep oracle tasks serial. Preserve results and
checkpoints. New submissions, reruns or cancellation need user authorization;
the two-week roadmap is a proposal, not such authorization.
