# Experiment collaboration

## Conversation style

Owner request, 2026-10-19: use ASD-STE100 for future conversational answers.
Use short sentences, consistent technical terms, explicit subjects and clear
separation of verified facts, dated status, plans and interpretation limits.
Continue in Simplified Chinese unless the owner requests another language.
ASD-STE100 defines controlled English: apply its clarity principles to Chinese
without claiming formal Chinese compliance or an unperformed compliance audit.
Explain technical terms when first used. Keep necessary scientific caveats.

## Agent-native project context

The owner enrolled this project in agent-native-context on 2026-10-13.
This Git checkout is the ONE canonical context root. The outer local workspace
contains private working artifacts and points here; do not duplicate memory.
At session start read `handoff/current.md`, `doc/INDEX.md`, and the newest
`daily-notes/YYYY-MM-DD.md`. Follow the linked campaign evidence as needed.
During substantive work append dated findings/decisions to the daily note,
update current docs, and leave a concise handoff (normally under 80 lines).
Use America/Chicago dates/times with timezone labels. Avoid notes for unchanged
routine polling. Distinguish requirements, verified evidence, historical
reports, known issues and proposals. Never store hidden reasoning, credentials,
raw private text or whole tool transcripts in context files. Keep existing
`docs/` reference documents and immutable campaign paths; `doc/` indexes them.
Publish reviewed results via `docs/RESULTS_INDEX.md`, not the whole workspace.
Always recheck live scheduler state before resource operations. Current budget
is FOUR account GPUs, with Oracle2 at two concurrent one-GPU generators or one
two-GPU judge job, nonoverlapping phases; older six-GPU notes are historical.

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

Latest owner override2026-10-21: retain ONLY NEW Oracle1/Qwen3 IPO ordinary,
array index0, including its training/generation and shared initial baseline.
Owner explicitly confirms index0 and cancels further work on indices1..5.
Do not auto-advance either old six-arm launcher or retry canceled-scope arms.
Preserve all existing outputs/checkpoints/receipts; cancellation of future work
does not relabel earlier COMPLETED/FAILED tasks as CANCELLED. Inspect current
queue before scoped cancellation; never cancel the parent array or unrelated work.
See current handoff/campaign for verified status. Owner subsequently explicitly
approves finishing first-arm evaluation and WR, through100. Use the tested
immutable retained.py route, not old failed-accounting guards; no retraining.
Same5h monitor follows index0 only. Annotation review remains required.

Historical owner override2026-10-19: interim six-arm Oracle1 WR curves0,10,...80
may use TWO extra H100s concurrently with training, full account at most FOUR.
See Oracle1 campaign INTERIM_WR_80.md for the bound new operational extension.
Preserve immutable base source0e7793f, checkpoints, settings and receipts.
Legacy array4859197 pending workers reject any sidecar, including pending jobs;
first sidecar must wait until all have passed startup/reload or completed.
New extension reserves training2GPU and evaluation2GPU lanes with full account,
pending/releasing resources and receipt checks; no unrelated work may coexist.
Do not use the old advance launcher while extension lanes are active. No job
cancellation/requeue or oldOracle2 work. This supersedes blanket nonoverlap only
for the explicitly approved new Oracle1 interim evaluation, not historical jobs.

Oracle1 connection recovered and continuation verified2026-10-19T22:41:45Z:
at22:32:35Z sequence0010 array4852338_0..5 and all52 receipt tasks COMPLETED0:0
at82/80/80/82/80/80. All490 states finite/contiguous/hash-valid, shared initial
scores/support, frozen contract and latest losses/gradients valid; no errors.
Last pair verified reload70 then ten further updates. Full account empty.
Frozen dry-run and fresh guarded submit created4859197_0..5%2 at22:38:18Z,
sequence0011, up to10 updates toward92/90/90/92/90/90, oneH100/task,two concurrent,
24h/18h soft stop. Latest0/1 RUNNING,2..5 PENDING(JobArrayTaskLimit), full account
2H100, no unrelated/releasing work or errors. Fresh RESUME_VERIFIED IPOordinary82/
IPOreference80 and new RUNNING manifests checked; no new outer update complete
or new-round memory peak yet. Source0e7793f/science/same5h monitor unchanged;
no downstream or oldOracle2 action. Exact0011 intent/receipt and completion/
startup audits in Oracle1 campaign records. Never duplicate0011. The18:00 SSH
blocker below is resolved for this check; its underlying cause is not established.

Oracle1 monitoring blocked2026-10-19T18:00:06Z: two SSH attempts timed out
in TCP connect before authentication or any remote command. Current queue/GPU
allocation, training progress, errors and downstream state UNKNOWN. Do not treat
07:22:24Z below as live or infer completion/failure from elapsed time. No job,
remote/source/science or timer changes. Same5h monitor; next check read-only
after owner checks UNC VPN/Sycamore reachability. No new segment without fresh
successful receipt/account/checkpoint validation. See Oracle1 campaign
records/connection_failure_20261019T1800.json. This is a connection blocker,
not an established experiment failure; all earlier snapshots are historical.

Oracle1 progress2026-10-19T07:22:24Z: sequence0010 array4852338_0/1
COMPLETED0:0 at82/80;2/3 RUNNING at75/77,4/5 PENDING(JobArrayTaskLimit),
retaining70/70. All460 states finite/contiguous/hash-valid, initial scores/support
match, frozen contract and latest loss/gradients valid. Active pair verifies
reload IPOfeedback70/DPOordinary72 plus five updates each. Full account2H100,
no unrelated/releasing work or logged errors. Only0001..0010 intent/receipt
pairs; no new submission, source/science/downstream/oldOracle2 action. Same5h
monitor. See Oracle1 campaign records/segment10_progress_20261019T0722.json.
Next advance still requires all current receipt tasks successful and full account
empty. Never duplicate0010 or infer100-round completion. Older checks follow.

Oracle1 continuation2026-10-19T02:09:05Z: sequence0009 array4849728_0..5
and all46 earlier receipt tasks COMPLETED0:0 at72/70/70/72/70/70. All430
states finite/contiguous/hash-valid, shared initial scores/support, frozen
contract and latest losses/gradients valid; no logged execution errors. Fresh
empty-account preflight/dry-run passed; guarded submit created4852338_0..5%2
at02:06:07Z, sequence0010, up to10 further updates toward82/80/80/82/80/80.
At02:09:05Z0/1 RUNNING with fresh RESUME_VERIFIED72/70,2..5 pending normally;
full account2H100, no unrelated/releasing work. No new outer complete yet.
Exact0010 intent/receipt preserved; never repeat. Source0e7793f and science
unchanged, same5h monitor, no downstream/oldOracle2 action. See Oracle1 campaign
records/segment9_complete_20261019T0202.json and segment10_startup_20261019T0209.json.
Only successful prior receipts and empty full account permit the next advance.
This is continuation, not completion of100 rounds. Older checks follow.

Oracle1 progress2026-10-18T20:47:29Z: sequence0009 array4849728_0/1
COMPLETED0:0 at72/70;2/3 RUNNING at64/66,4/5 PENDING(JobArrayTaskLimit),
retaining60/60. All398 states finite/contiguous/hash-valid, initial scores/support
match, frozen contract and latest loss/gradients valid. Active pair verifies
reload IPOfeedback60/DPOordinary62 plus four updates each. Full account2H100,
no unrelated/releasing work or logged errors. Only0001..0009 intent/receipt
pairs; no new submission, source/science/downstream/oldOracle2 action. Same5h
monitor. See Oracle1 campaign records/segment9_progress_20261018T2047.json.
Next advance still requires all current receipt tasks successful and full account
empty. Never duplicate0009 or infer100-round completion. Older checks follow.

Oracle1 connection recovered2026-10-18T15:53:17Z: sequence0008 array4846960
and all40 prior receipt tasks COMPLETED0:0 at62/60/60/62/60/60. All370 states
finite/contiguous/hash-valid, initial scores/support match, contract unchanged,
latest loss/gradients finite, no errors. DPOreference/feedback reload50 then ten
updates verified. Full account empty. Frozen dry-run and fresh guarded submit
created4849728_0..5%2, sequence0009, at15:56:47Z toward72/70/70/72/70/70,
at most10 updates,1H100/task,two concurrent,24h/18h soft stop. At16:00:12Z0/1
RUNNING,2..5 PENDING(JobArrayTaskLimit), full account2H100, no unrelated/releasing
work or errors. Fresh RESUME_VERIFIED IPOordinary62/IPOreference60 and new
RUNNING manifests checked; no new outer complete yet. Exact0009 intent/receipt
saved; never duplicate0009 or earlier. Same5h monitor/source/science, no
downstream or oldOracle2 action. See Oracle1 campaign records/
segment8_complete_20261018T1553.json and segment9_startup_20261018T1600.json.
The10:48Z outage and older scheduler snapshots below are historical.

Oracle1 monitoring blocked2026-10-18T10:48:36Z: two SSH TCP timeouts before
authentication/remote commands. Current queue/GPU allocation, training steps,
logs and downstream state UNKNOWN; not a confirmed experiment failure. Last
successful05:49:29Z snapshot below is historical. No submission, cancellation,
remote/source/science or timer change. Owner asked to check UNC VPN/Sycamore;
next5h check retries read-only and reads latest intents/receipts before actions.
Do not infer segment completion from elapsed time or submit without fresh
successful-predecessor/full-empty-account evidence. See Oracle1 campaign
records/connection_failure_20261018T1048.json.

Oracle1 progress2026-10-18T05:49:29Z: sequence0008 array4846960_0/1 now
COMPLETED0:0 at62/60;2/3 RUNNING at54/56,4/5 PENDING(JobArrayTaskLimit),
retaining50/50. All338 committed states finite/contiguous/hash-valid, shared
initial scores/support, frozen contract and latest loss/gradient checks pass.
Active pair RESUME_VERIFIED IPOfeedback50/DPOordinary52 plus four new updates
each checked. Full account2H100, no unrelated/releasing work or logged errors.
No new submission, scientific/source/oldOracle2 action or downstream phase.
Same5h monitor. See campaign records/segment8_progress_20261018T0549.json.
Wait for all current receipt tasks successful and full account empty before
the next frozen advance. Never duplicate0008 or infer100-round completion.

Oracle1 connection recovered2026-10-18T00:51:42Z: sequence0007 array4839924
and all prior receipt tasks COMPLETED0:0. Six arms at52/50/50/52/50/50,
310 finite/contiguous/hash-valid states, matching initial scores/support and
finite latest losses/gradients; contract/source unchanged, no logged errors.
DPOreference/feedback reload40 then ten more updates checked. Full account
empty. Frozen dry-run then guarded submit created4846960_0..5%2, sequence0008,
at00:55:40Z, targets62/60/60/62/60/60, oneH100/task,two concurrent,24h/18hsoft
stop. At01:00:06Z0/1 RUNNING,2..5 PENDING(JobArrayTaskLimit), full account2H100,
no unrelated/releasing work or errors. Fresh RESUME_VERIFIED IPOordinary52 and
IPOreference50; new manifests RUNNING, no new outer complete yet. Exact0008
intent/receipt saved; never duplicate0008 or earlier. See campaign records/
segment7_complete_20261018T0051.json and segment8_startup_20261018T0100.json.
Same5h monitor, no scientific/source/downstream/oldOracle2 changes. The outage
and older scheduler snapshots below are historical, not current allocation.

Oracle1 monitoring blocked2026-10-17T19:50:09Z: two SSH TCP connections timed
out before any remote command. Current queue/GPU allocation, training steps,
logs and downstream state UNKNOWN; not evidence of experiment failure. Last
successful check05:55:22Z below is historical. No submission, cancellation,
remote/source/science or timer change. Ask owner to check UNC VPN/Sycamore;
next5h check retries read-only and reads latest intents/receipts before actions.
See Oracle1 campaign records/connection_failure_20261017T1950.json.

Oracle1 sequence0007 submitted2026-10-17T05:52:03Z:4839924_0..5%2, same
source0e7793f/contract. At05:49:18Z all0001..0006 tasksCOMPLETED0:0, arms
at42/40/40/42/40/40;250 finite/contiguous/hash-valid states, shared initial
scores/support and finite latest loss/gradients. Full account empty, no errors.
DPOreference/feedback reload30 and ten further updates verified. Frozen dry-run
then fresh guarded advance resumes up to10 updates toward52/50/50/52/50/50.
At05:55:22Z0/1 RUNNING,2..5 PENDING(JobArrayTaskLimit), account2H100, no
unrelated work/errors. Fresh RESUME_VERIFIED IPOordinary42/IPOreference40,
both new manifests RUNNING; no new round complete yet. Exact0007 intent/receipt in
campaign records and local private launch dir; never duplicate0007 or earlier.
See records/segment6_complete_20261017T0549.json and
records/segment7_startup_20261017T0555.json. Same5h monitor; no source,
science, downstream or oldOracle2 change. Read latest records first.

Oracle1 progress2026-10-16T04:37:02Z: sequence0004 array4814410_0/1 now
COMPLETED0:0 at22/20;2/3 RUNNING at15/17;4/5 PENDING(JobArrayTaskLimit),
retaining previous complete10/10. Full account2H100, no unrelated/releasing
work or logged execution errors. All100 saved states finite/contiguous/hash-valid,
initial scores/support match, frozen contract unchanged. New logs confirm
RESUME_VERIFIED IPOfeedback10/DPOordinary12 and five further updates each.
Only0001..0004 intents/receipts; no new submission or scheduler/source changes.
Same5h monitor. Next advance still waits for all receipts successful and empty
account. See records/segment4_progress_20261016T0437.json in Oracle1 campaign.
The startup snapshot below is historical; never duplicate0004.

Oracle1 sequence0004 submitted2026-10-15T23:32:42Z:4814410_0..5%2, same
source0e7793f/contract. At23:31:17Z all0001..0003 tasksCOMPLETED0:0, six arms
at12/10/10/12/10/10 with70 finite/contiguous/hash-valid states and shared
initial scores/support; account empty. Frozen dry-run then guarded submit
resumes at most10 updates, targets22/20/20/22/20/20. At23:35:26Z0/1 RUNNING,
2..5 normalPENDING(JobArrayTaskLimit), account2H100, no unrelated work/errors.
Logs confirm RESUME_VERIFIED IPOordinary12/IPOreference10, both training;
no new round complete yet. Pending arms have not reloaded in this allocation.
Read latest records0004 before any action; never duplicate. Same5h monitor,
no downstream stage or oldOracle2 action; no source/science changes.

Oracle1 progress2026-10-15T18:32:21Z: all six arms started.4811252_0..3
COMPLETED0:0 for this10-update segment at12/10/10/12;4/5 RUNNING at4/4.
Full account2H100, no pending/unrelated work or logged execution errors.
All58 committed states finite, contiguous and hash-valid; initial scores and
token support match, deployment/data/model contract unchanged. Only0001..0003
intents/receipts exist. No new job/scheduler action; wait for4/5 success and idle
account before advancing. Same5h timer; no baseline/generation/WR/labels yet.
Do not confuse segment completion with100 updates or the older snapshots below.

Oracle1 full-model resume PASSED2026-10-15T10:25:58Z:4811197_0/3 both
COMPLETED0:0 atstep2, RESUME_VERIFIED, finite loss/gradients/states, valid
contiguous checkpoint hashes and shared initial support. Source0e7793f unchanged.
After empty-account preflight/dry-run, all six4811252_0..5%2 submitted10:27:10Z
as sequence0003, up to10 outer updates per arm, oneH100/task,24h/18h soft stop.
At10:30:56Z0/1 RUNNING,2..5 normalPENDING(JobArrayTaskLimit), full account2H100,
no unrelated work/errors. Both ordinary arms continue from2; others start0.
IPOordinary RESUME_VERIFIED at2; IPOreference finite0/initial support verified.
All six submitted does NOT mean all six running. SAME monitor nowACTIVE every
5HOURS, tool/stored schedule verified. Do not repeat startup/resume probes or
duplicate0003. Future advance needs all receiptsCOMPLETED0:0 and account idle.
The earlier running/pending-resume snapshots below are historical.

Oracle1 resume submitted2026-10-15T09:51:04Z:4811197_0,3%2, sequence0002,
same frozen source0e7793f and contract. First4811149_0/3 bothCOMPLETED0:0
atstep1;09:49:45Z hashes/finite states/initial support verified, full account
empty. Peak allocated11.90GiB, reserved28.59GiB. Dry-run then guarded advance
submitted exactly one further update tostep2. At09:53:56Z bothRUNNING,
full account2H100, no other owned work/logged error. Both actual4B reloads
emit RESUME_VERIFIED atouter1 with matching scores/support; continuedstep2
is NOT yet complete. Keep5min timer. Verify finite completedstep2
for both before all-six advancement and5h cadence. Four other arms unsubmitted.
Read records0002 and live state; never duplicate. No science/source changes.

Oracle1 Qwen3 submitted2026-10-15T09:19:12Z:4811149_0,3%2, oneH100/task,
one real outer update for IPO/DPO ordinary. Source0e7793f5fdd6afdf2fcf80c4ff40649e8926f5a2
frozen/pushed/verified. Final23 serverCPU tests no skips, shell/contract checks
pass. At09:25:22Z both RUNNING, account2H100, no other work/errors, both finite
step0 committed with matching initial scores/support, first update ongoing.
GPU training/reload not validated merely by CPU tests. Five-minute
heartbeat oracle1-qwen3-training-and-resume-checks ACTIVE; update SAME timer to
5h only after both startup and cross-allocation resume/update pass. Read campaign
records/intent/receipt before any advance. Current v1 root is documented below.

Latest owner authorization2026-10-15 supersedes old Oracle2 continuation:
set Oracle2 aside, use Oracle1 Nemotron ONLY for BOTH training preferences and
WR in a NEW Qwen3-4B six-arm experiment. Five traits/Qwen3-32B rater, seed0,
same100/500/200 panels, states0..100, generation every10. See
experiments/oracle1_semantic/README.md and campaigns/oracle1_qwen3_20261015.
FOUR full-account GPUs maximum; start2 concurrent1GPU tasks, judge2GPU,
nonoverlapping phases, empty account before submission. Preserve old Oracle2
results/jobs; do not recover or score them. New5-minute monitor is authorized,
switch SAME timer to5hours only after IPO/DPO actual training and full-model
resume/continued update verified. Per-outer atomic checkpoints must retain
adapter, s0/st/st-1 and RNG. Never duplicate intents/receipts or blind-retry
FAILED/TIMEOUT. User authorizes tested technical fixes, not science changes.
Human review of the200-response style pilot precedes full annotation.
DPO feedback is exactly null under a single BT oracle; retain as null control.
Source must be tested/pushed/frozen before GPU submission. Preparation is not
GPU validation. The dated Oracle2 instructions below are historical only.

Oracle2 recovery prepared2026-10-13T06:40:26Z: frozen source2d6b5747c251abc44700799be24e3e1e05e9b984
is deployed at ipo/diagnostics/oracle2_real_20261009_v5_recovery. All129 server
CPU tests passed without skips, plus six config checks and real response-prefix,
seed, hash and adapter validation. This is NOT GPU inference or submission.
Original training/generators remain immutable source23401de. Recovery contract
hash89c1397acd414d3f7badddb8d6a885f74b8ae33a2b894c33d2166dc95e8309fd
reuses eight complete banks per failed arm plus state80 rows728/561; only
1672/1839 responses remain. Original banks remain untouched; new recovery
output ipo_runs/oracle2_generation_recovery_20261013_v1 does NOT exist yet.
No recovery intent/receipt or WR job. Tasks4794738_2/3 RUNNING (step0 rows250),
4/5 PENDING(JobArrayTaskLimit), full accountTWO H100s. The recovery phase guard
was verified BLOCKED while predecessors remain incomplete. Once original2..5
COMPLETED0:0 and full account idle, recheck latest intents/receipts and use v5
frozen queue.py --phase recover once, oneH100/task,two concurrent,24h. Do not
resubmit original generate. Later v5 WRscore routes two recovered arms plus
four originals; all66 banks, successful jobs and idle account are still required.
Do not implement incremental WR based only on the user's explanatory question.
Records: campaign records/attempt5 and RECOVERY_20261013.md. Monitor stays2hours.

Oracle2 result inspection2026-10-13T06:15:34Z: six TRAINING arms remain
COMPLETE0..100; policy figures/numeric summaries published in campaign
policy_results_20261013. Generation4794738_0/1 now TIMEOUT after12:00:26,
not successful despite sacct exit0:0 and stale RUNNING file manifests.
Each has validated complete banks0..70; partial80 rows728/561 preserved.
Tasks2/3 RUNNING with24-hour limits,4/5 PENDING(JobArrayTaskLimit), full
accountTWO H100s. No OOM/Traceback/nonfinite found; WR still unsubmitted.
No retries, cancellations, source/science changes or overwritten banks.
Do not blindly restart generators (existing output dirs are rejected).
Any recovery must be tested, preserve/reuse finished and partial banks,
use new immutable source, sync GitHub and obey existing full-account guards.
The sole cyclic prompt416 initially assigns near-unit panel probability to
a2-token candidate; flat raw policy is not evidence of history stabilization.
See records/attempt4/generate_timeout_and_policy_results_20261013.json.

Oracle2 generation time-limit warning 2026-10-12T22:21:28Z: array4794738
tasks0/1 remain RUNNING with12-hour limits; tasks2..5 are PENDING and NOW
have24-hour limits. An in-place array update to1-00:00:00 returned code1:
pending tasks changed, but `4794738_0-1: Access/permission denied` for the
running tasks. This was partial success, not an all-or-nothing failure.
UTC scontrol verified0/1 EndTime2026-10-13T06:14:50Z. Current throughput
suggests overrun risk, not certain timeout; owner notified that administrator
help is needed to extend running jobs. Do not repeat the denied operation.
No cancellation/requeue/restart, GPU/concurrency or scientific changes.
Full account remains TWO H100s, throttle2, no unrelated owned work. At22:19:21Z,
ordinary generated complete states0/10/20 plus91/800 rows at30; reference
complete0/10 plus775/800 at20. Complete banks passed count/hash/finite-field
checks, baseline hash valid, all606 training snapshots finite; no logged
execution errors. WRscore remains unsubmitted. Preserve source23401de and
original12-hour submission receipt; this is a scheduler override only.
Campaign record: records/attempt4/generate_timelimit_risk_20261012T2221.json.
Existing two-hour monitor includes this warning; no new timer or experiment.

Oracle2 generation submitted 2026-10-12T18:14:50Z: **4794738_0..5%2**,
one H100/task, frozen source23401de58b27ef2f7b1ea645551f37602a3c1bb8.
All train4773902_0..5 COMPLETED0:0 at100; 606 finite snapshots, contiguous
metrics, matching initial scores, unchanged review and hash-valid800 baseline
verified at18:10:50Z. Empty full-account preflight and all66 required adapter
checks passed. Generation intent/receipt saved in campaign records/attempt4;
NEVER duplicate this array. At18:16:56Z tasks0/1 RUNNING (IPO ordinary/reference),
step0 partial rows14/13; tasks2..5 normal PENDING(JobArrayTaskLimit). Full account
TWO H100s, no unrelated owned work or logged OOM/Traceback/nonfinite. No complete
checkpoint generation bank or WR result yet. WRscore remains unsubmitted until
baseline and all66 generated banks pass complete count/hash checks and the full
account is empty. Existing oracle2 monitor updated to generation phase, still
every two hours. FOUR-GPU account ceiling and nonoverlapping phases remain;
no hard account cap exists. See training_complete_status_20261012.json and
generate_started_status_20261012.json under campaign records/attempt4.
No scientific or executable source changes.

Oracle2 checkpoint milestone 2026-10-12T09:30:31Z: DPO reference4773902_4
and feedback4773902_5 RUNNING at complete outer92/91, both saved step90.
Tasks0..3 remain COMPLETED0:0 at100. Full account TWO H100s, no unrelated
owned work; all 589 snapshots finite, contiguous metrics, matching initial
scores, unchanged source/review and hash-valid800/800 baseline. No logged
OOM/Traceback/nonfinite. No resource or scientific changes; generate/WR wait
for all six completed training arms and empty account. Two-hour timer unchanged.
See campaign records/attempt4/dpo_checkpoint90_status_20261012T0930.json.

Oracle2 checkpoint milestone 2026-10-12T06:56:10Z: DPO reference4773902_4
and feedback4773902_5 RUNNING at complete outer80/79; saved adapters through
80/70 respectively. Tasks0..3 remain COMPLETED0:0 at100. Full account TWO
H100s, no unrelated owned work; 565 finite snapshots, contiguous metrics,
matching initial scores, unchanged source/review and hash-valid800/800 baseline.
No logged OOM/Traceback/nonfinite. No resource or scientific changes;
generate/WR still await all six training completions and an empty account.
Two-hour timer unchanged. See campaign records/attempt4/
dpo_reference_checkpoint80_status_20261012T0656.json.

Oracle2 checkpoint milestone 2026-10-12T04:59:33Z: DPO reference4773902_4
and feedback4773902_5 RUNNING at complete outer71/70, both saved step70.
Tasks0..3 remain COMPLETED0:0 at100. Full account TWO H100s, no unrelated
owned work; all 547 snapshots finite, contiguous metrics, matching initial
scores, unchanged source/review and hash-valid800/800 baseline. No logged
OOM/Traceback/nonfinite. No resource or scientific changes; generate/WR wait
for all six completed training arms and empty account. Two-hour timer unchanged.
See campaign records/attempt4/dpo_checkpoint70_status_20261012T0459.json.

Oracle2 checkpoint milestone 2026-10-12T03:01:46Z: DPO reference4773902_4
and feedback4773902_5 RUNNING at complete outer62/61, both saved step60.
Tasks0..3 remain COMPLETED0:0 at100. Full account TWO H100s, no unrelated
owned work; all 529 snapshots finite, contiguous metrics, matching initial
scores, unchanged source/review and hash-valid800/800 baseline. No logged
OOM/Traceback/nonfinite. No resource or scientific changes; generate/WR wait
for all six completed training arms and empty account. Two-hour timer unchanged.
See campaign records/attempt4/dpo_checkpoint60_status_20261012T0301.json.

Oracle2 connection recovered 2026-10-12T01:05:30Z: train4773902_0..3
COMPLETED exit0:0, all states0..100 (IPO ordinary/reference/feedback and DPO
ordinary). DPO reference _4 and feedback _5 RUNNING at complete outer53/52,
both with adapters through50; completed arms have adapters through100.
Full owner/UID queue contains only _4/_5, TWO H100s total. All 511 snapshots
finite, metrics contiguous, initial scores matching, source/review unchanged;
baseline800/800 hash-valid, no logged OOM/Traceback/nonfinite. Frozen trainer
does not record GPU-memory peaks. No new phase or resource action taken.
Wait for ALL six training arms to finish and full account idle before generate;
checkpoint generation/WR remain unsubmitted. Two-hour monitor unchanged.
See campaign records/attempt4/four_arms_complete_status_20261012T0105.json.
The SSH blocker and earlier running snapshots below are historical.

Oracle2 monitoring blocked 2026-10-10T10:00:29Z: two SSH connection attempts
timed out before any remote command. Current GPU allocation, progress and
logs are UNKNOWN; do not treat the 08:02 checkpoint snapshot below as live.
No experiment or timer changed. Ask owner to check UNC VPN/Sycamore access;
retry read-only on the next check. Campaign record:
records/attempt4/connection_failure_20261010T1000.json.

Oracle2 checkpoint milestone 2026-10-10T08:02:07Z: train4773902_0/1 remain
RUNNING, ordinary complete outer61/100 and reference62/100; step-10/20/30/40/50/60
adapters saved. Metrics contiguous from state0, 125 finite snapshots, matching initial scores and
unchanged review/source. Baseline800/800 remains hash-valid. _2..5 normal
PENDING(JobArrayTaskLimit); full account TWO H100s, no unrelated owned work
or logged OOM/Traceback/nonfinite. No generated-checkpoint WR yet; no tasks
changed. Existing two-hour monitor continues. See records/attempt4/checkpoint60_status.json.

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
