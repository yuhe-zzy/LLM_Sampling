# Oracle2 real-panel campaign, 2026-10-09

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

## Locations

- Launch: `/work/users/y/u/yuhe32/ipo/diagnostics/oracle2_real_20261009`
- Frozen source: launch root `/source`
- Private input panels: launch root `/private_data`
- Outputs: `/work/users/y/u/yuhe32/ipo_runs/oracle2_real_20261009`
- Local records: `oracle2_real_launch_20261009`

The source archive includes this experiment and the unchanged shared history
engine. `deployment.json` binds the actual commit and every deployed source
file. Model lock and plan hashes are carried into data/scoring/training
manifests. Do not replace a deployment in place after a task uses it.

## Commands after reviewed deployment

Use the existing h100env312 Python. Preparation is CPU-only. These commands
are documentation, not a batch that blindly starts every phase:

```bash
ROOT=/work/users/y/u/yuhe32/ipo/diagnostics/oracle2_real_20261009
PY=/work/users/y/u/yuhe32/h100env312/bin/python
CODE=$ROOT/source/experiments/oracle2
$PY $CODE/prepare_data.py --plan $ROOT/plan.json
$PY $CODE/campaigns/oracle2_real_20261009/queue.py --phase audit --submit
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
array uses 0..5%6, one H100/task. Candidate/WR scoring uses three H100s only.
All submissions require fresh empty-account checks and exclusive intent files.

See the [experiment README](../../README.md) for objectives, grouping, WR
definition, resource schedule and privacy restrictions.
