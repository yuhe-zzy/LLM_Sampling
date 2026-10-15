# Oracle1 Qwen3 campaign, 2026-10-15

See [protocol and continuation](../../README.md). Owner approved implementation,
submission and early5-minute checks, changing the same monitor to5-hour checks
only after real IPO/DPO training AND checkpoint reload/continued training work.
No new Oracle2 work. Full-account FOUR-GPU ceiling.

## Submitted and running

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
