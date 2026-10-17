# Current experiment status

## Active authorization: Oracle1 / Qwen3, 2026-10-15

Verified2026-10-15T10:25:58Z: both first4811149_0/3 and resume4811197_0/3
COMPLETED0:0. Actual Qwen3-4B IPO/DPO reloads reproduce saved scores/support,
emit RESUME_VERIFIED and complete another update tostep2 with finite loss and
gradients. Contiguous checkpoint hashes/states and shared initial scores pass.
Resume peak allocated11.89GiB/reserved21.35GiB; no logged execution errors.
Full-model startup AND cross-process continued-training gate has passed.

At2026-10-16T19:42:55Z sequence0005 array4819533_0..5 allCOMPLETED0:0,
as are all prior receipt tasks. Complete outer32/30/30/32/30/30 (IPOordinary/
reference/feedback, then DPOordinary/reference/feedback). All190 saved states
finite/contiguous/hash-valid, initial scores/token support match, frozen contract
unchanged and latest losses/gradients finite. No logged execution errors.
DPOreference/feedback reload20 and ten subsequent updates independently checked.
Latest-round maximum allocated11.90GiB/reserved25.73GiB, not lifetime telemetry.
Full owner/UID account empty, including no releasing allocations; one terminal
scontrol record was correctly excluded from live GPU totals.
Frozen dry-run and fresh guarded submit created **4836621_0..5%2** at19:44:50Z,
sequence0006, oneH100/task, two concurrent,24h/18h soft stop, up to10 further
updates toward42/40/40/42/40/40. Latest2026-10-17T00:37:34Z0/1 COMPLETED0:0
at42/40;2/3 RUNNING at34/36;4/5 PENDING(JobArrayTaskLimit), retaining30/30.
Full account2H100, no unrelated/releasing work/errors. All218 saved states
finite/contiguous/hash-valid; initial scores/support and frozen contract pass.
Logs confirm RESUME_VERIFIED IPOfeedback30/DPOordinary32 and four new updates
each. Active arms' latest-round peaks at most11.90GiB allocated/36.34GiB
reserved, not lifetime telemetry. Only0001..0006 intent/receipt pairs;
no new submission or scheduler action. See [current progress](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/segment6_progress_20261017T0037.json),
[completed segment5](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/segment5_complete_20261016T1942.json)
and [new receipt](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/0006_receipt.json).
This is checkpoint continuation, not restart from0; never duplicate0006.
Segment completion does NOT mean the100-round experiment is finished.
Source remains0e7793f5fdd6afdf2fcf80c4ff40649e8926f5a2;23 serverCPU tests passed
without skips before freeze. No code/scientific change. The SAME heartbeat
oracle1-qwen3-training-and-resume-checks nowACTIVE every5HOURS, tool and stored
configuration verified. Baseline/generation/WR/annotation not submitted.

Owner now explicitly authorizes a NEW campaign with Oracle1 for training AND WR,
Qwen3-4B policy and five-trait Qwen3-32B measurement; implementation and submission
approved. Old Oracle2 recovery/WR are set aside, not cancelled or resumed.
See [new protocol](../experiments/oracle1_semantic/README.md) and
[campaign](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/README.md).
Every outer update commits a complete restart state, with10-update/18h soft-stop
segments and24h allocation. Start IPO/DPO ordinary one update each, then verify
real reload and one further update before all six arms. FOUR accountGPUs max,
two1GPU training/generation tasks, one2GPU judge, one1GPU rater, no overlap.
New monitor authorized5min, change same timer to5h only after startup/resume
validation. Human pilot review still gates bulk annotation. No cycle-count gate.

At03:42 CDT account empty after SSH reconnect. Qwen3 model revisions downloaded
and pinned, all800 exact panels pass new tokenizer audit without truncation.
Earlier22 CPU tests passed, superseded by final23 above; includes actual tiny
Qwen3/LoRA save-load-train equivalence. Full-size startup/reload/continued update
have now separately passed as recorded above; future jobs still require checks.
The status and future-action instructions below concern historical Oracle2 only.

Owner update2026-10-14 23:31 CDT: `oracle2` timer deleted in the app at the
owner's request; deletion confirmed. No server experiment was cancelled or
modified. No automatic checks or stage submissions will continue through this
timer. Do not recreate it without a new request. Remote state remains unknown.

Updated 2026-10-14, America/Chicago. At14:00 CDT, two SSH attempts timed out
before remote commands. Current GPU allocation, progress and logs are UNKNOWN.
No recovery/WR submission or remote task change was made. Owner was asked to
check UNC VPN/Sycamore access; monitoring has since been stopped above. See the
[connection failure record](../experiments/oracle2/campaigns/oracle2_real_20261009/records/attempt5/connection_failure_20261014T1858.json).
Last successful scheduler evidence is00:01:00 CDT (05:01:00 UTC), not live.

## Oracle2: verified evidence

- Candidate audit4773843 and baseline4773899 completed. Each judge scored
  3200 candidates; baseline has800 responses. Training4773902_0..5 completed
  outer states0..100, with606 validated finite snapshots. Actual frozen source:
  `23401de58b27ef2f7b1ea645551f37602a3c1bb8`.
- Generation4794738_0/1 timed out at12hours. Complete banks0..70 plus partial80
  rows728/561 are preserved. Stale RUNNING file manifests do not override sacct.
- Generation2/3 (IPO feedback/DPO ordinary) COMPLETED0:0, all11 banks0..100
  hash/count/identity validated,800 rows each. Original queue started4/5
  (DPO reference/feedback), running on twoH100s with complete0/10/20/30/40,
  partial50 rows548/461 at the last successful check. No pending or unrelated owned work; no new
  logged execution error. All606 training snapshots remain finite.
- At the last successful check, no generated-response WR existed. WR checkpoints are0,10,...100, but the
  implementation scores in a later batch after generation, not live in training.
- Missing-only recovery was tested and CPU-deployed, not submitted or GPU-tested
  at the last successful check; re-read remote intents/receipts after reconnection.
  Source `2d6b5747c251abc44700799be24e3e1e05e9b984`, frozen launch
  `oracle2_real_20261009_v5_recovery`; original v4 remains unchanged.

See [policy results](../experiments/oracle2/campaigns/oracle2_real_20261009/policy_results_20261013/README.md)
and [recovery protocol](../experiments/oracle2/campaigns/oracle2_real_20261009/RECOVERY_20261013.md).
The private raw status transcript stays outside Git. This document records only
reviewed aggregate facts.

## Preconditions for future requested continuation

Only after original tasks2..5 are COMPLETED0:0, all their banks validate, and
the FULL owner/UID account is empty: inspect latest v4/v5 intents and receipts,
verify frozen source/contract, then use v5 queue.py phase recover ONCE. Two
one-H100 tasks, at most2 concurrent,24hours. Preserve every original bank.
After successful recovery and all66 banks plus baseline validate, and the
account is empty, submit v5 WRscore (twoGPUs, sequential judges).

The oracle2 two-hour monitor is deleted at the owner's request. No automatic
continuation is scheduled. Account ceiling4 overrides historical6.
Throttle2 is not a scheduler-wide hard cap.
Do not overlap phases, cancel unrelated jobs, bypass guards, duplicate jobs or
turn an explanation request into authorization for incremental scoring.

## Other batches and publication limitations

A/probes/B/C, Reference90 and the trend sweep have published results; the sweep
has50 logical arms including6 reused Stage B arms, not50 new replications.
Anchor90 jobs4728255_0/1 were reported COMPLETED0:0 in the October9 preflight,
but a dedicated validated trajectory/figure export is still absent locally.
Do not present its older pending launch record as current, or invent plots.
Oracle1 has a dated September11 snapshot, not a certified final16-cell grid.
Historical sampling has a numerical-failure caveat and token-average panels;
it is not current sequence-sum evidence. See the atlas for exact coverage.

## Latest user requirements

2026-10-13: enroll this project in agent-native-context and publish shareable
results on GitHub. Use this checkout as canonical context root. Publish reviewed
numeric evidence/plots, preserve failures and missing cells, exclude raw private
prompt/response text, weights/adapters, credentials and bulk raw archives.
This request does not change experimental science or authorize extra GPU work.
