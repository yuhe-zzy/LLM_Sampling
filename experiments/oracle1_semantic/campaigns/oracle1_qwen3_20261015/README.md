# Oracle1 Qwen3 campaign, 2026-10-15

See [protocol and continuation](../../README.md). Owner approved implementation,
submission and early5-minute checks, changing the same monitor to5-hour checks
only after real IPO/DPO training AND checkpoint reload/continued training work.
No new Oracle2 work. Full-account FOUR-GPU ceiling.

## Current scope: retain IPO ordinary only

October21 owner reports VPN reconnected, asks whether the first arm finished,
and explicitly confirms retaining IPO ordinary/index0 training and generation
while canceling remaining arms. This supersedes the historical six-arm plan.
All existing results/checkpoints and shared new-Qwen3 baseline remain preserved.
Same5h monitor now follows index0 only, with no six-arm advance or automatic
retry. Do not restart indices1..5, oldOracle2 or unrelated tasks. The owner has
now explicitly authorized finishing the retained arm's evaluation and win rate.
Use the [single-arm immutable route](IPO_ONLY_EVALUATION.md) for0..100, not the
old six-arm failed-receipt guards. No new job at the preparation checkpoint.

## Requested interim WR

Historical October19 authorization: curves for all six arms at0,10,...80 and TWO
additional H100s, with FOUR maximum across the entire account. This supersedes
the earlier one-extra-card limit and permits training/evaluation overlap through
explicit cooperative guards. Saved-checkpoint inference needs no retraining.
See [interim execution and reuse](INTERIM_WR_80.md). Original frozen source and
scientific settings stay unchanged. Existing legacy pending workers reject any
sidecar; wait until all have started and verified reload (or completed), without
cancel/requeue/source edits. New immutable extension uses separate bounded
training/evaluation lanes; never bypass guards or directly sbatch jobs.

## Submitted and running

October21 connectivity recovered on the owner's request. Full owner/UID queue
is empty. Native parent accounting finds training4863156_0/1 COMPLETED0:0,
2..5 FAILED1:0; generation4861879_0/1 COMPLETED0:0,2..5 FAILED1:0. No active
task required scancel; user withdrawal concerns future work, not relabeling the
already terminal jobs. Task0 training manifest is COMPLETE at100. Generation
is not complete despite successful segment exit:0..70 finished, partial80 remains.
No WR result. Training task2 failed in startup on Ambiguous accounting record;
generation task2 was blocked by the prior training failure. No source hot-fix,
retry or new job. Same5h monitor narrowed to index0 only. Older snapshots follow.

Artifact auditOctober21T08:59:59Z:568 states at100/100/90/92/90/90 pass checkpoint
continuity/hashes, finite saved arrays and initial-score/support checks. Retained
task0 has101 valid states. All14,100 existing responses validate with canonical
seed/input/row hashes and finite diagnostics, zero empty; baseline800 plus16
complete checkpoint banks, partial80 rows249/251 for ordinary/reference.
Retained IPOordinary:6,649/7,200 interim responses,551 missing at80. Complete
banks equal journals and retain original adapter hashes. Generator peak8.05GiB;
cap/repetition diagnostics retained without retuning. No WR receipt/results.
Fresh full account09:01:33Z empty, zero GPUs. [Scope and terminal audit](records/retain_ipo_ordinary_20261021.json).

HistoricalOctober21T07:48:09Z: SSH still times out before remote commands. Read-only
local checks confirm Cisco VPN adapter Disabled and client state Disconnected.
Owner must reconnect UNC VPN. Remote queue/GPU/progress/errors/WR remain UNKNOWN;
local evidence does not establish remote cluster health. No network/job/source/
science/timer changes. Same5h monitor ACTIVE; next frozen EXT/ledger/full-account
check remains mandatory, never duplicate4863156/4861879.
[VPN evidence](records/vpn_disconnected_20261021T0748.json). Older snapshots follow.

Monitoring blockedOctober20T16:45:10Z: two SSH TCP timeouts before authentication
or any remote command. Current queue/GPU/progress/errors/WR state UNKNOWN.
No new job, remote/source/science or timer action. SAME5h monitor stays ACTIVE.
Owner should check UNC VPN/Sycamore access; next check reads frozen EXT and both
ledgers first. Never duplicate4863156/4861879 or infer completion from elapsed time.
See [connection record](records/connection_failure_20261020T1645.json).
Successful snapshots below are historical, not live allocation evidence.

LatestOctober20T12:00:36Z: **4863156_0..5%2**, EXTsequence0003 training, is active.
All58 original receipt tasks including4859197 completed successfully at
92/90/90/92/90/90;11:49:14Z validates550 contiguous/hash/finite states and initial
scores/support. Training-lane dry-run passed11:52:00Z; guarded submit11:54:06Z
targets100 per arm with at most10 updates, original frozen trainer unchanged.
OneH100/task,two concurrent,24h/18h soft stop. Tasks0/1 verify reload92/90 and
matching current-allocation manifests;2..5 pending normally, no new update claimed.
Generator4861879 remains active:4/54 complete checkpoint banks (IPOordinary/
reference0/10), step20 partial359/360, baseline800 complete;4719 total valid rows.
Bank/journal/seed/hash/finite checks pass, zero empty, no logged errors.
Generator peak allocated8.05GiB each. Full account4H100, potential4, no unrelated
work; full queue/native-parent accounting pass. Same5h monitor; no WR yet.
[Audit](records/interim_training_final_segment_startup_20261020T1200.json),
exact [intent](records/interim_v2_0003_intent.json) and [receipt](records/interim_v2_0003_receipt.json).
Never duplicate either active array or use BASE advance before interim handback.
Earlier dated startup/status snapshots below are historical.

LatestOctober20T06:44:36Z: baseline **4861201** COMPLETED0:0;800/800 canonical
bank/journal/seed/hash/finite checks pass, zero empty responses,398 cap hits
(49.75%), mean196.7925tokens, allocated peak7.85GiB. No quality or WR claim.
Frozen dry-run passed06:41:03Z; guarded EXTsequence0002 submitted
**4861879_0..5%2** at06:41:42Z, oneH100/task,24h/18h soft stop. Tasks0/1 RUNNING
with17 valid step0 responses each, zero empty, four cap hits;2..5 normal pending.
Both startup guards passed, no logged OOM/Traceback/nonfinite. Memory progress
not yet emitted for these generators. Full account4H100, reserved potential4,
no unrelated/releasing work. Training06:43:56Z:92/90/90/92/84/83,537 valid states.
Same monitor restored5hours after actual baseline AND generator validation.
No WR submitted. Never duplicate. [Audit](records/interim80_generation_startup_20261020T0644.json),
exact [intent](records/interim_v2_0002_intent.json) and [receipt](records/interim_v2_0002_receipt.json).
Both lanes remain active; wait for lane-specific successful receipts before advance.
Earlier dated startup/status snapshots below are historical.

NEW baseline **4861201**, EXTsequence0001, submittedOctober20T05:28:21Z after
successful dry-run and fresh guarded checks. OneH100,24h/18h soft stop. At05:33:52Z,
62/800 real responses pass canonical identity/seed/hash and finite checks; no
empty responses or logged errors,20 cap hits, allocated peak7.75GiB at50 rows.
Full account3H100 (two original trainers plus baseline), reserved potential4,
no unrelated/pending/releasing work. This is initial inference, not WR/completion.
Exact [intent](records/interim_v2_0001_intent.json), [receipt](records/interim_v2_0001_receipt.json)
and [startup audit](records/interim80_baseline_startup_20261020T0533.json). Never duplicate.

Previous duplicate Slurm accounting cleared after task5 started05:23:18Z.
Unchanged v2 and raw parent accounting pass, without deduplication/source/guard
or scheduler changes. Both last legacy workers verified reload80 and matching
manifests before baseline joined. Training05:30:41Z:92/90/90/92/80/80,530 valid
states,0..3 segment-complete,4/5 running, no errors. [Earlier diagnosis](records/accounting_overlap_20261020T0517.json)
is historical. Same5min monitor until baseline AND first checkpoint generator
write healthy real responses. No checkpoint generator or WR submission yet.
The following launch/status snapshots are historical.

Connection recovered2026-10-19T22:32:35Z on the owner's retry request. Sequence0010
array4852338_0..5 and all52 receipt tasks COMPLETED0:0. Complete steps82/80/80/
82/80/80;490 states finite/contiguous/hash-valid, initial scores/support match,
contract unchanged and latest losses/gradients finite. No logged errors. Last
pair verified reload70 and ten further updates. Full account empty, including no
releasing allocations. Latest-round peaks at most11.90GiB allocated/39.36GiB
reserved, not lifetime device telemetry. See [completion audit](records/segment10_complete_20261019T2232.json).
The earlier [connection failure](records/connection_failure_20261019T1800.json) is no longer blocking this check.

Frozen dry-run passed22:36:30Z; fresh guarded submit created **4859197_0..5%2**,
sequence0011, at22:38:18Z, up to10 updates toward92/90/90/92/90/90. OneH100/task,
two concurrent,24h/18h soft stop. At22:41:45Z0/1 RUNNING,2..5 normal
PENDING(JobArrayTaskLimit), full account2H100, no unrelated/releasing work or
errors. Fresh RESUME_VERIFIED IPOordinary82/IPOreference80 and new RUNNING
manifests checked. No new outer update complete or new-round memory peak yet;
all490 states remain valid. Exact [intent](records/0011_intent.json),
[receipt](records/0011_receipt.json) and [startup audit](records/segment11_startup_20261019T2241.json)
preserved. Same5h monitor and frozen source0e7793f, no scientific or oldOracle2
changes. Baseline/generation/WR/labels unsubmitted. Never duplicate0011.
Older execution snapshots follow.

Verified2026-10-19T02:02:34Z: sequence0009 array4849728_0..5 and all46 prior
receipt tasks COMPLETED0:0 at72/70/70/72/70/70. All430 committed states finite,
contiguous/hash-valid, initial scores/support match, frozen contract unchanged
and latest losses/gradients finite. No logged errors. DPOreference/feedback
reload60 then ten more updates verified. Latest-round maxima11.90GiB allocated/
35.41GiB reserved, not lifetime telemetry. Fresh02:03:16Z preflight confirmed
full account empty; the earlier inspection queue preceded the final completion.
See [completion audit](records/segment9_complete_20261019T0202.json).

Frozen dry-run then fresh guarded submit created **4852338_0..5%2** at02:06:07Z,
sequence0010, up to10 updates toward82/80/80/82/80/80. OneH100/task,two concurrent,
24h/18h soft stop. Exact [intent](records/0010_intent.json) and
[receipt](records/0010_receipt.json) preserved; never duplicate0010 or earlier.
Latest07:22:24Z0/1 COMPLETED0:0 at82/80;2/3 RUNNING at75/77;4/5 normal
PENDING(JobArrayTaskLimit), retaining70/70. All460 states finite/contiguous/
hash-valid, shared initial scores/support, frozen contract and latest finite
loss/gradients checked. Active pair verifies IPOfeedback70/DPOordinary72 reload
and five further updates each. Full account2H100, no unrelated/releasing work
or logged errors. Latest active-round maxima11.90GiB allocated/39.36GiB reserved,
not lifetime telemetry. No new submission this check; pending manifests describe
their prior segment. See [progress audit](records/segment10_progress_20261019T0722.json)
and [startup audit](records/segment10_startup_20261019T0209.json).
Same5h monitor, source0e7793f and science unchanged. No baseline/generation/WR/
annotation or oldOracle2 action. Older execution snapshots below are historical.

Connection recovered2026-10-18T00:51:42Z: sequence0007 array4839924_0..5
allCOMPLETED0:0 at52/50/50/52/50/50; all34 previous receipt tasks successful.
All310 committed states finite/contiguous/hash-valid, initial scores/support
match, contract unchanged and latest losses/gradients finite. No logged errors.
DPOreference/feedback reload40 then ten updates checked. Full account empty.
Latest-round maxima11.90GiB allocated/24.32GiB reserved, not lifetime telemetry.
See [completion audit](records/segment7_complete_20261018T0051.json).
The earlier [connection outage](records/connection_failure_20261017T1950.json)
was not a confirmed experiment failure and is no longer blocking this check.

Frozen dry-run then guarded submit created **4846960_0..5%2** at00:55:40Z,
sequence0008, at most10 further updates to62/60/60/62/60/60. OneH100/task,
two concurrent,24h allocation/18h soft stop. Latest05:49:29Z0/1 COMPLETED0:0
at62/60;2/3 RUNNING at54/56,4/5 PENDING(JobArrayTaskLimit), retaining50/50.
Full account2H100, no unrelated/releasing work or errors. All338 committed states
finite/contiguous/hash-valid, shared initial scores/support, unchanged contract,
latest losses/gradients finite. Active IPOfeedback/DPOordinary reload50/52 and
four further updates each verified. Latest active-round peaks at most11.90GiB
allocated/29.52GiB reserved, not lifetime telemetry. No new submission. See
[progress audit](records/segment8_progress_20261018T0549.json) and
[startup audit](records/segment8_startup_20261018T0100.json).
Exact [intent](records/0008_intent.json) and [receipt](records/0008_receipt.json)
preserved; never duplicate0008 or earlier. Same5h monitor, source0e7793f and
science unchanged. Baseline/generation/WR/labels still unsubmitted; no oldOracle2
action. Prior completion/launch snapshots below are historical.

Sequence0006 completion verified2026-10-17T05:49:18Z:4836621_0..5 all
COMPLETED0:0 at42/40/40/42/40/40; all earlier receipts successful. All250
committed states finite/contiguous/hash-valid, initial scores/support match,
contract unchanged, latest losses/gradients finite, no logged execution errors.
Full account empty. DPOreference/feedback reload30 then ten updates checked.
Latest-round peaks at most11.90GiB allocated/36.35GiB reserved, not lifetime
device telemetry. See [completion audit](records/segment6_complete_20261017T0549.json).
Frozen dry-run then guarded submit created **4839924_0..5%2** at05:52:03Z,
sequence0007, at most10 further updates to targets52/50/50/52/50/50. OneH100/task,
two concurrent,24h allocation/18h soft stop. At05:55:22Z0/1 RUNNING,2..5
PENDING(JobArrayTaskLimit), account2H100, no unrelated work/errors. Fresh logs
confirm RESUME_VERIFIED IPOordinary42/IPOreference40, both new manifests RUNNING;
no new round complete or new-round memory peak yet. See
[startup audit](records/segment7_startup_20261017T0555.json).
Exact [intent](records/0007_intent.json) and [receipt](records/0007_receipt.json)
preserved; never duplicate0007. Same5h monitor, no source/science changes,
no baseline/generation/WR/labels or oldOracle2 action.

The earlier segment0006 progress and launch snapshots below are historical:

Sequence0005 completion verified2026-10-16T19:42:55Z:4819533_0..5 all
COMPLETED0:0 at32/30/30/32/30/30; all earlier receipts successful. All190
committed states finite/contiguous/hash-valid, initial scores/support match,
contract unchanged, latest losses/gradients finite, no logged execution errors.
Full account empty. DPOreference/feedback reload20 then ten updates checked.
See [completion audit](records/segment5_complete_20261016T1942.json).
Frozen dry-run then guarded submit created **4836621_0..5%2** at19:44:50Z,
sequence0006, at most10 further updates to targets42/40/40/42/40/40. OneH100/task,
two concurrent,24h allocation/18h soft stop. Latest2026-10-17T00:37:34Z0/1
COMPLETED0:0 at42/40;2/3 RUNNING at34/36;4/5 PENDING(JobArrayTaskLimit),
retaining30/30. Account2H100, no unrelated/releasing work/errors. All218 states
finite/contiguous/hash-valid, initial scores/support and frozen contract pass.
Logs confirm RESUME_VERIFIED IPOfeedback30/DPOordinary32 and four new updates
each. Active arms' latest-round peaks at most11.90GiB allocated/36.34GiB reserved,
not lifetime device telemetry. No new job/scheduler action. See
[current progress](records/segment6_progress_20261017T0037.json) and
[startup audit](records/segment6_startup_20261016T1947.json).
Exact [intent](records/0006_intent.json) and [receipt](records/0006_receipt.json)
preserved; never duplicate0006. Same5h monitor, no source/science changes,
no baseline/generation/WR/labels or oldOracle2 action.

The earlier segment0005 snapshots below are historical:

Sequence0004 completion verified2026-10-16T09:35:06Z:4814410_0..5 all
COMPLETED0:0 at22/20/20/22/20/20; all earlier receipts successful. All130
committed states finite/contiguous/hash-valid, initial scores/support match,
contract unchanged, latest losses/gradients finite, no logged execution errors.
Full account empty. See [completion audit](records/segment4_complete_20261016T0935.json).
Frozen dry-run then guarded submit created **4819533_0..5%2** at09:37:17Z,
sequence0005, at most10 further updates to targets32/30/30/32/30/30. OneH100/task,
two concurrent,24h allocation/18h soft stop. Latest14:30:00Z0/1 COMPLETED0:0
at32/30;2/3 RUNNING at24/26;4/5 PENDING(JobArrayTaskLimit), retaining20/20.
Full account2H100, no unrelated/releasing work or logged errors. All158 saved
states finite/contiguous/hash-valid; initial scores/support and contract pass.
Fresh logs confirm RESUME_VERIFIED IPOfeedback20/DPOordinary22, four subsequent
updates each. Active arms' latest-round peaks at most11.90GiB allocated and
25.73GiB reserved, not lifetime memory telemetry. No new submission/scheduler
action. See [progress audit](records/segment5_progress_20261016T1430.json) and
[startup audit](records/segment5_startup_20261016T0941.json).
Exact [intent](records/0005_intent.json) and [receipt](records/0005_receipt.json)
preserved; never duplicate0005. Same5h monitor, no source/science changes,
no baseline/generation/WR/labels or oldOracle2 action.

The earlier segment0004 progress and launch snapshots are historical:

Latest check2026-10-16T04:37:02Z: sequence0004 array4814410_0/1 COMPLETED0:0
at22/20;2/3 RUNNING at15/17;4/5 PENDING(JobArrayTaskLimit), retaining previous
complete10/10. All100 committed states finite/contiguous/hash-valid; initial
scores/support and frozen contract pass. Full account2H100, no unrelated or
releasing work, no loggedOOM/Traceback/nonfinite. Fresh logs confirm successful
IPOfeedback10/DPOordinary12 reloads and five further updates each. Latest-round
maximum allocated11.90GiB/reserved30.38GiB, not lifetime device-wide telemetry.
Only0001..0004 intent/receipt pairs exist. No new job/scheduler/source change;
next advance waits for all receipt tasks successful and full account idle.
Same5h monitor; no baseline/generation/WR/annotation submission. See
[progress audit](records/segment4_progress_20261016T0437.json). Never repeat0004.

The sequence0004 launch snapshot below is historical:

Sequence0003 completed2026-10-15T23:31:17Z check:4811252_0..5 allCOMPLETED0:0
at12/10/10/12/10/10, respectively. All70 committed states finite/contiguous/
hash-valid with matching initial scores/token support and unchanged contract.
No logged execution errors; full account empty. This is not100-round completion.
Fresh frozen queue dry-run and guarded submit created **4814410_0..5%2** at
23:32:42Z, sequence0004, oneH100/task, at most2 concurrent,24h/18h soft stop.
Resume existing states for up to10 further updates; targets22/20/20/22/20/20.
At23:35:26Z0/1 RUNNING,2..5 PENDING(JobArrayTaskLimit), full account2H100 and
no unrelated work/logged errors. RESUME_VERIFIED confirms IPOordinary12 and
IPOreference10 loaded scores/support; both training, no new round complete yet.
See [startup audit](records/segment4_startup_20261015T2335.json).
Exact [intent](records/0004_intent.json)
and [receipt](records/0004_receipt.json) preserved; never duplicate0004.
See [completed predecessor audit](records/segment3_complete_20261015T2331.json).
Same5-hour timer; no baseline/generation/WR/annotation submission.

The earlier checks below are historical:

Check2026-10-15T18:32:21Z: all six arms have now started; four finished
their first all-arm SEGMENT, not all100 updates. Tasks4811252_0..3 COMPLETED0:0
at IPOordinary12, IPOreference10, IPOfeedback10, DPOordinary12. DPOreference4
and DPOfeedback5 RUNNING at complete outer4/4, toward10. Full owner/UID queue
has only these two one-H100 tasks, no pending/unrelated work. All58 committed
states finite/contiguous/hash-valid, shared initial scores/token support match,
frozen deployment/data/model contract unchanged. No logged execution errors;
latest round peaks approximately11.90GiB allocated, up to28.60GiB reserved.
Only0001..0003 intents/receipts exist. No new job or scheduler action taken.
Next segment waits for both remaining tasks to succeed and full account idle.
Same5-hour monitor continues; baseline/generation/WR/annotation not submitted.
See [checked progress](records/segment_progress_20261015T1832.json).

The startup and submission snapshot below is historical:

2026-10-15T10:25:58Z: **full-model checkpoint resume PASSED** for IPO and DPO.
First4811149_0/3 and resumed4811197_0/3 allCOMPLETED0:0. Both actual4B adapters
reproduce saved scores/support, emit RESUME_VERIFIED, and finishstep2 with finite
loss/gradients and contiguous hash-valid states0..2. Resume peak allocation
11.89GiB and reservation21.35GiB perGPU; no loggedOOM/Traceback/nonfinite.

Fresh full-account empty preflight and guarded dry-run passed, then submitted
**4811252_0..5%2** at10:27:10Z, sequence0003, up to10 outer updates per arm.
One H100/task, at most two concurrent,24h allocation/18h soft stop. Ordinary
arms continue fromstep2 toward12, others start0 toward10, not restarting probes.
At10:30:56Z tasks0/1 RUNNING,2..5 PENDING(JobArrayTaskLimit), full account2H100,
no unrelated work/errors. All six are SUBMITTED, not all running simultaneously.
IPOordinary logs RESUME_VERIFIED at2; IPOreference committed finite0 with matching
initial scores/support. Both are training; no later completed round inferred.
Source/plan/contract/scientific settings unchanged.

Same heartbeat `oracle1-qwen3-training-and-resume-checks` updated to5HOURS;
tool and stored schedule verified after actual startup/resume completion.
Read [records0003](records/0003_receipt.json) and
[resume-pass/status](records/resume_passed_six_arms_20261015.json) before actions.
Only all successful previous receipts and empty full account permit next segment.
Baseline/generation/WR/annotation remain unsubmitted. No oldOracle2 action.

## Historical startup

2026-10-15T09:49:45Z: first tasks4811149_0/3 bothCOMPLETED0:0, each has complete
states0..1 and finite training loss/gradients. Hashes/initial scores/support pass.
Elapsed22m30s/22m32s including loading/scoring; each GPU peak allocated11.90GiB,
reserved28.59GiB. No logged OOM/Traceback/nonfinite; full account empty.
After a fresh guarded dry-run, **4811197_0,4811197_3%2** submitted09:51:04Z,
sequence0002, to reloadstep1 and perform one more real update tostep2.
At09:53:56Z bothRUNNING, full account2H100 and no unrelated work/errors.
Both log RESUME_VERIFIED atouter1: actual4B adapters reproduce saved sequence
sums/support. Continuedstep2 is in progress, NOT completed full resume validation.
Remaining four arms still unsubmitted. Verify both reload logs and completed
step2/sacct before advancing all six; retain the five-minute timer meanwhile.
Source/contract/scientific settings are unchanged; see records0002 and
[resume status](records/resume_started_20261015.json). Do not duplicate.

The following initial submission snapshot is historical:

2026-10-15T09:19:12Z: **4811149_0,4811149_3%2** submitted once, actual frozen
source `0e7793f5fdd6afdf2fcf80c4ff40649e8926f5a2`, one H100/task. Final server
23 CPU tests passed without skips, shell syntax and full contract/six configs
passed. Fresh complete-account preflight empty. At09:25:22Z both RUNNING,
full account TWO H100s, no unrelated work or logged execution errors. Both
completed initial scoring and committed finite step0; initial scores/support
match across arms. First outer updates are in progress; no completed update
or full-model GPU resume success claimed yet.

Five-minute heartbeat `oracle1-qwen3-training-and-resume-checks` ACTIVE and
verified in app. Change SAME timer to5hours only after both actual objective
startup and next-allocation resume pass. Intent/receipt/deployment and aggregate
startup status are in [records](records/). Never duplicate this submission.
Remaining four arms are approved but not yet submitted; receipt-guarded advance
unlocks them only after the initial IPO/DPO startup and resume phases succeed.

## Earlier preparation evidence

At approximately03:42 CDT, SSH reconnected and full owner/UID account inspection
found no queued/running/releasing jobs and zero allocated GPUs. This is a dated
snapshot, not a reservation. Old final Oracle2 generators report COMPLETED in
sacct; their output completeness was NOT audited or used for this new campaign.

Models downloaded on the login node without GPU allocation and pinned:

| Role | Model | Revision |
|---|---|---|
| Policy | Qwen/Qwen3-4B | 1cfa9a7208912126459214e8b04321603b3df60c |
| Five-trait rater | Qwen/Qwen3-32B | 9216db5781bf21249d130ec9da846c4624c16137 |
| Oracle1 | nvidia/Llama-3.1-Nemotron-70B-Reward-HF | existing audited lock, unchanged |

Model lock fingerprints small configuration/tokenizer files and weight metadata;
it is not a newly computed full SHA256 of every weight shard. Models are BF16,
no quantization. Memory sufficiency is not established just by downloading.

New roots:
- Launch: `/work/users/y/u/yuhe32/ipo/diagnostics/oracle1_qwen3_20261015_v1`
- Output: `/work/users/y/u/yuhe32/ipo_runs/oracle1_qwen3_20261015_v1`
- Local private receipts: `oracle1_qwen3_launch_20261015` in the outer workspace.

The real-data audit rebuilt pure Oracle1 preferences from all800 original
panels, preserving100/500/200 and4 candidates. Full Qwen3 chat+answer+EOS length
checks pass at inherited1537 cap; no truncation or dropped data. Evaluation
prompts plus256 generated tokens also pass. No repeated candidate judge job.

The first approved scheduler action will be just two actual first-round tasks:
IPO ordinary(index0), DPO ordinary(index3), one H100 each. Receipt/publication
records below determine whether this has been submitted; preparation alone
does not mean inference or training passed. First resume allocation likewise
contains0/3 for one more outer update, then all six arms use at most2 concurrency.

Do not edit plan/lock/data/source after deployment. Technical fixes must be tested,
synced to GitHub and deployed as a new immutable version with explicit provenance.
Never treat a failed scheduler task's exit0 or stale file manifest as completion.
