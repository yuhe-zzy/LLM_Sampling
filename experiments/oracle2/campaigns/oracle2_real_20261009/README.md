# Oracle2 real-panel campaign, 2026-10-09

## Compatibility failure and repair preparation

**Latest:** audit 4773811 FAILED, exit1:0, at 2026-10-09T16:57:22Z before
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

Attempt2 is **prepared, not submitted**, with identical scientific parameters
and separate roots ending in `_v2`, using [plan_attempt2.json](plan_attempt2.json).
Rerun full length/provenance checks there before any retry. User confirmation
for resubmitting the failed audit has been requested; there is no automatic
rerun or training-chain authorization from the failure itself.

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
ROOT=/work/users/y/u/yuhe32/ipo/diagnostics/oracle2_real_20261009_v2
PY=/work/users/y/u/yuhe32/h100env312/bin/python
CODE=$ROOT/source/experiments/oracle2
$PY $CODE/prepare_data.py --plan $ROOT/plan.json
$PY $CODE/campaigns/oracle2_real_20261009/queue.py --root "$ROOT" --phase audit
# --submit only after explicit approval to retry the failed audit.
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
