# Current experiment status

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
