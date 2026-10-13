# Current experiment status

Updated 2026-10-13, America/Chicago. Scheduler evidence below was captured at
18:00:08 CDT (23:00:08 UTC); it is not a live dashboard.

## Oracle2: verified evidence

- Candidate audit4773843 and baseline4773899 completed. Each judge scored
  3200 candidates; baseline has800 responses. Training4773902_0..5 completed
  outer states0..100, with606 validated finite snapshots. Actual frozen source:
  `23401de58b27ef2f7b1ea645551f37602a3c1bb8`.
- Generation4794738_0/1 timed out at12hours. Complete banks0..70 plus partial80
  rows728/561 are preserved. Stale RUNNING file manifests do not override sacct.
- Generation2/3 (IPO feedback/DPO ordinary) COMPLETED0:0, all11 banks0..100
  hash/count/identity validated,800 rows each. Original queue started4/5
  (DPO reference/feedback), running on twoH100s with complete0, partial10
  rows150/79 at the dated check. No pending or unrelated owned work; no new
  logged execution error. All606 training snapshots remain finite.
- No generated-response WR exists. WR checkpoints are0,10,...100, but the
  implementation scores in a later batch after generation, not live in training.
- Missing-only recovery is tested and CPU-deployed, NOT submitted or GPU-tested.
  Source `2d6b5747c251abc44700799be24e3e1e05e9b984`, frozen launch
  `oracle2_real_20261009_v5_recovery`; original v4 remains unchanged.

See [policy results](../experiments/oracle2/campaigns/oracle2_real_20261009/policy_results_20261013/README.md)
and [recovery protocol](../experiments/oracle2/campaigns/oracle2_real_20261009/RECOVERY_20261013.md).
The private raw status transcript stays outside Git. This document records only
reviewed aggregate facts.

## Authorized next action, not yet performed

Only after original tasks2..5 are COMPLETED0:0, all their banks validate, and
the FULL owner/UID account is empty: inspect latest v4/v5 intents and receipts,
verify frozen source/contract, then use v5 queue.py phase recover ONCE. Two
one-H100 tasks, at most2 concurrent,24hours. Preserve every original bank.
After successful recovery and all66 banks plus baseline validate, and the
account is empty, submit v5 WRscore (twoGPUs, sequential judges).

The existing oracle2 two-hour monitor continues; no extra timer. It stays quiet
on normal progress and reports meaningful changes/failure/action needs. Account
ceiling4 overrides historical6. Throttle2 is not a scheduler-wide hard cap.
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
