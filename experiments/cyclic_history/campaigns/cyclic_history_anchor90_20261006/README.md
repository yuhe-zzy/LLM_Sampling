# Initial-anchor 90% control: October 6, 2026

The user approved two additional cyclic LLM runs, one IPO and one DPO:
`r_t = .9*s_0 + .1*s_t`, or alpha=.1, nu=0, kappa=0. This is NOT the previous
reference90 campaign, whose weights were `(0,.1,.9)`, nor a fully frozen
reference. Policy training and adaptive sampling continue normally.

| Task | Objective | alpha | lambda_current | beta_train | nu | kappa |
|---|---|---|---|---|---|---|
| 0 | IPO | .1 | .8 | .2 | 0 | 0 |
| 1 | DPO | .1 | .8 | .8 | 0 | 0 |

Seed0, Qwen2.5-1.5B, sequence-sum, the same calibrated four-response panels
for prompts 54/251/612/737/867/945, and 100 outer updates (states 0..100).
Ten inner epochs per outer update, learning rate 1e-5, and all other training
and support settings match completed Stage B ordinary tasks 4605560_0/3.
Only reference alpha changes experimentally; no new ordinary baseline is
needed. Existing results, sources and model checkpoints remain untouched.

Use the existing empirical partial-refresh protocol introduced for the trend
sweep. Support identity, preference calibration, fresh GPU initial-score checks,
and finite-gradient/score checks remain mandatory. Do not impose a theoretical
fixed-point/stability acceptance gate or recompute the calibrated preference P
for alpha=.1. No training algorithm changes are required.

## Execution

Submitted as **4728255_0..1%2** on October 6 at 09:52:50 America/Chicago
(14:52:50 UTC). Task 0 is IPO; task 1 is DPO. At 09:53:25 America/Chicago,
both tasks were **PENDING(Priority)**, with zero allocated GPUs and no other
owned running or pending work. This is normal scheduling, not failure.
No fresh GPU initial-score check has run while pending; it is mandatory at
each task's startup. These are dated snapshots, not current-state guarantees.

All **82 server CPU tests passed with no skips** (76 existing plus six new),
both config/support audits passed, and the full account was empty immediately
before submission. Frozen source commit:
`6d5510875eee5127b7307fe7a85cab0b6b3dc0f3`. The core training, history math
and calibration code are unchanged from the trend sweep's 4195658 source.
The new commit adds this campaign's configuration, launch checks and records.

See `deployment.json`, `submission_intent.json`, `submission_receipt.json`,
`queue_verified.json` and `startup_check.json`. Never duplicate the submission
or retry after an intent without reconciling scheduler state. No automatic
follow-up is scheduled. Later documentation commits do not change this frozen
training source or the already-submitted jobs.

- Remote launch: `/work/users/y/u/yuhe32/ipo/diagnostics/history_anchor90_20261006`
- Output: `/work/users/y/u/yuhe32/ipo_runs/cyclic_history_anchor90_20261006`
- Local records: `cyclic_history_anchor90_launch_20261006`
- One H100/task; array `0-1%2`, full-account ceiling six GPUs.
- Submission requires the full owner/UID queue to be empty, checking pending
  as well as running work through `squeue -a -r` and `scontrol -a show job`.
- Deployment records the exact source commit and SHA256 of every source/config
  file. Tests and support checks must pass before submission. Fresh score checks
  run only when each task obtains its GPU, not at CPU packaging time.

`build_plan.py` deterministically derives the two-arm plan from Stage B.
`test_plan.py` checks matched parameters and that the changed reference leaves
the sampler and preference feedback identical for the same input policy.
`queue_anchor90.py` validates immutable packaging, audits support, runs CPU
regressions, and uses exclusive intent/receipt creation for at-most-once launch.
`inspect_anchor90.py` reads account allocation, Slurm state, metrics, completed
numeric snapshots and errors. It never changes scheduler state.

For analysis, compare all six prompts against the completed alpha=.9 ordinary
arms using identical colors/axes. Distinguish probability excursion size from
adjacent-step jitter, and raw panel entropy from relative-sequence entropy.
One seed and six calibrated prompts do not establish universal convergence or
absence of token repetition in open generation.
