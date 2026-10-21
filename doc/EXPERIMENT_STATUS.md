# Current experiment status

## Active authorization: Oracle1 / Qwen3, 2026-10-15

Latest connection diagnosis2026-10-21T07:48:09Z: SSH TCP timeout before
authentication/remote commands. Read-only local checks confirm Cisco VPN adapter
Disabled and Cisco Secure Client state Disconnected. Owner must reconnect UNC
VPN; this does not independently establish cluster health or the sole cause.
No network setting, job, remote/source/science or timer change. Same5h monitor
verified ACTIVE. Current experiment/account state remains UNKNOWN. Next check
must read frozen EXT, both ledgers and full account before any advancement.
[VPN evidence](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/vpn_disconnected_20261021T0748.json).
The following dated connection and successful snapshots are historical.

Monitoring blocked2026-10-20T16:45:10Z: two SSH TCP connection attempts timed
out before authentication or remote commands. Current scheduler/GPU allocation,
training steps, generation progress, logs and WR status UNKNOWN. No experiment
failure is established. No new submission, cancellation, remote/source/science
or timer change; SAME monitor verified ACTIVE every5hours. Owner should check
UNC VPN/Sycamore reachability. Next check retries read-only frozen EXT inspection
and both ledgers; no advancement without fresh lane-success/full-account checks.
Never duplicate4863156/4861879. [Connection record](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/connection_failure_20261020T1645.json).
All successful progress and allocation snapshots below are historical.

Latest milestone2026-10-20T12:00:36Z: final-target training array
**4863156_0..5%2** is active, EXTsequence0003. Predecessor4859197 and all58
original receipt tasks COMPLETED0:0 at92/90/90/92/90/90;11:49:14Z inspector
validates550 contiguous/hash/finite states and shared initial scores/support.
Frozen dry-run passed11:52:00Z; fresh guarded submit11:54:06Z resumes at most10
updates toward100 per arm, oneH100/task,two concurrent,24h/18h soft stop.
Tasks0/1 verify reload92/90 and matching current-allocation RUNNING manifests;
2..5 normal array-limit pending. No new completed update or training100 claimed.
The extension executes original frozen train.py; no source/science change.

Baseline **4861201** remains COMPLETED0:0 with800 validated responses.
Generator **4861879** has4/54 complete checkpoint banks: IPOordinary/reference
states0/10, each800. Their step20 journals have359/360 valid responses.
Total4719/44000 with baseline; canonical identity/seed/hash/finite checks pass,
complete bank/journal equality and bound adapters pass. Zero empty responses;
cap/repetition diagnostics retained in the record. Generator peak allocated
8,642,043,904bytes each (8.05GiB); progress at350 rows lags live journal reads.
No OOM/Traceback/nonfinite. Full account4H100, potential4, no unrelated/releasing
work; native parent sacct --array and full squeue/scontrol checks pass. Same5h
monitor unchanged. No WR submitted or complete. See
[startup audit and exact receipt links](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/interim_training_final_segment_startup_20261020T1200.json).
Never duplicate either array. Both lanes active; advance only after that lane's
successful receipts and all guards pass. No original BASE advance while EXT active.

The05:17 accounting overlap cleared after the last legacy task started05:23:18Z.
Raw parent accounting and unchanged v2 inspector pass; no deduplication, source,
guard or scheduler modification. Both active workers verified reload80 and wrote
matching current-allocation manifests before evaluation joined. Training audit
06:43:56Z was92/90/90/92/84/83,537 valid states,0..3 segment-complete,4/5 running,
no errors. [Earlier diagnosis](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/accounting_overlap_20261020T0517.json)
and [generation startup](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/interim80_generation_startup_20261020T0644.json) are historical.

Latest request2026-10-19: interim six-arm Oracle1 WR curves at0,10,...80 and
TWO additional H100s, provided full-account concurrent use never exceeds FOUR.
This explicitly supersedes the one-extraGPU limit and allows checkpoint evaluation
alongside training; no retraining/science change. New bounded cooperative launcher
preserves base source0e7793f and original identity, reuses canonical response/score
journals and labels interim completion separately. Existing legacy pending workers
still reject sidecars, so all must start and pass reload or complete before first
sidecar submission. No cancellation/requeue/guard bypass. See
[interim implementation/status](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/INTERIM_WR_80.md).
New extensionv2 sourcecaf5ac2 frozen00:15:51Z;43 server CPU tests no skips and
native expanded parent-array accounting passed. Initial dry-run correctly blocked
on legacy pending startup; this historical wait has now cleared as recorded above.
The temporary5minute startup checks ended after baseline AND generator validation.
At2026-10-20T00:18:18Z six steps86/84/80/82/80/80;498 valid states; original0/1
RUNNING,2..5 pending, full account2H100, no logged errors. See
[prepared/deployment evidence](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/interim80_prepared_20261020.json).

Connection recovered on the owner's retry request, verified2026-10-19T22:32:35Z
(17:32 CDT). Sequence0010 array4852338_0..5 and all52 receipt tasks COMPLETED0:0.
Complete steps82/80/80/82/80/80;490 states finite/contiguous/hash-valid, matching
initial scores/support and unchanged contract. Latest losses/gradients finite,
no logged errors. Last pair reloaded70 and completed ten further updates.
Latest-round peaks at most11.90GiB allocated/39.36GiB reserved, not lifetime
device telemetry. Full account empty, including no releasing allocations.
See [completion audit](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/segment10_complete_20261019T2232.json).
The earlier [connection failure](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/connection_failure_20261019T1800.json) is resolved for this check, not evidence of a training failure.

Frozen dry-run passed22:36:30Z, then guarded submit created **4859197_0..5%2**,
sequence0011, at22:38:18Z, up to10 more updates toward92/90/90/92/90/90.
OneH100/task,two concurrent,24h/18h soft stop. At22:41:45Z0/1 RUNNING,2..5
PENDING(JobArrayTaskLimit), full account2H100, no unrelated/releasing work or
errors. Fresh RESUME_VERIFIED IPOordinary82/IPOreference80 and new RUNNING
manifests checked. No new outer update complete or new-round memory peak yet;
all490 states remain valid. Exact
[intent](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/0011_intent.json),
[receipt](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/0011_receipt.json) and
[startup audit](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/segment11_startup_20261019T2241.json)
preserved. Same5h monitor, unchanged source/science; no baseline/generation/WR/
labels or oldOracle2 action. Never duplicate0011.
Older execution snapshots below are historical.

Verified2026-10-19T02:02:34Z (October18 21:02 CDT): sequence0009 array
4849728_0..5 and all46 prior receipt tasks COMPLETED0:0. Six arms now at
72/70/70/72/70/70,430 committed states finite/contiguous/hash-valid; shared
initial scores/support, frozen contract and latest finite losses/gradients valid.
No logged errors. DPOreference/feedback reload60 then ten more updates verified.
Latest-round maxima11.90GiB allocated/35.41GiB reserved, not lifetime telemetry.
Earlier queue reading preceded the final completion; a fresh02:03:16Z preflight
confirmed the full account empty, excluding one terminal scontrol record.
See [completion audit](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/segment9_complete_20261019T0202.json).

Frozen dry-run then fresh guarded submit created **4852338_0..5%2** at02:06:07Z,
sequence0010, up to10 updates toward82/80/80/82/80/80. OneH100/task,two concurrent,
24h/18h soft stop. Exact
[intent](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/0010_intent.json)
and [receipt](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/0010_receipt.json)
preserved. Latest07:22:24Z0/1 COMPLETED0:0 at82/80,2/3 RUNNING at75/77,
4/5 PENDING(JobArrayTaskLimit), retaining70/70. Full account2H100, no unrelated/
releasing work or errors. All460 saved states finite/contiguous/hash-valid,
shared initial scores/support and frozen contract unchanged; latest loss/gradients
finite. Active pair verifies IPOfeedback70/DPOordinary72 reload plus five updates
each. Latest active-round peaks at most11.90GiB allocated/39.36GiB reserved,
not lifetime device-wide telemetry. No new submission this check. Pending
manifests describe their preceding segment, not current success. See
[progress audit](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/segment10_progress_20261019T0722.json) and
[startup audit](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/segment10_startup_20261019T0209.json).
Same5h monitor/source/science; no downstream or oldOracle2 action. Never
duplicate0010 or earlier. Older execution snapshots below are historical.

Verified2026-10-15T10:25:58Z: both first4811149_0/3 and resume4811197_0/3
COMPLETED0:0. Actual Qwen3-4B IPO/DPO reloads reproduce saved scores/support,
emit RESUME_VERIFIED and complete another update tostep2 with finite loss and
gradients. Contiguous checkpoint hashes/states and shared initial scores pass.
Resume peak allocated11.89GiB/reserved21.35GiB; no logged execution errors.
Full-model startup AND cross-process continued-training gate has passed.

At2026-10-18T00:51:42Z sequence0007 array4839924_0..5 allCOMPLETED0:0,
as are all34 prior receipt tasks. Complete outer52/50/50/52/50/50 (IPOordinary/
reference/feedback, then DPOordinary/reference/feedback). All310 saved states
finite/contiguous/hash-valid, initial scores/token support match, frozen contract
unchanged and latest losses/gradients finite. No logged execution errors.
DPOreference/feedback reload40 and ten subsequent updates independently checked.
Latest-round maximum allocated11.90GiB/reserved24.32GiB, not lifetime telemetry.
Full owner/UID account empty, including no releasing allocations.
Frozen dry-run and fresh guarded submit created **4846960_0..5%2** at00:55:40Z,
sequence0008, oneH100/task, two concurrent,24h/18h soft stop, up to10 further
updates toward62/60/60/62/60/60. Latest05:49:29Z0/1 COMPLETED0:0 at62/60,
2/3 RUNNING at54/56,4/5 PENDING(JobArrayTaskLimit), retaining50/50. Full account
2H100, no unrelated/releasing work or errors. All338 saved states finite,
contiguous/hash-valid, shared initial scores/support and unchanged contract;
latest losses/gradients finite. Active IPOfeedback/DPOordinary logs verify
reload50/52 plus four new updates each. Active-arm latest-round peaks at most
11.90GiB allocated/29.52GiB reserved, not lifetime telemetry. No new submission.
See [progress audit](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/segment8_progress_20261018T0549.json) and
[startup audit](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/segment8_startup_20261018T0100.json).
Only0001..0008 intent/receipt pairs. See
[completed segment7](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/segment7_complete_20261018T0051.json)
and [new receipt](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/0008_receipt.json).
This is checkpoint continuation, not restart from0; never duplicate0008.
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
