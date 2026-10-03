# Calibrated cyclic history Stage A launch - 2026-09-28

## Authorization and scheduler state

User approved queueing the new scheme after the currently running six tasks
finish, preserving all running work and limiting the account to six GPUs.
Only Stage A is submitted; Stage B/C require review of Stage A evidence.

- New array: **4603433**, submitted 2026-09-28T05:44:14Z.
- Dependency: **afterok:4593149**, covering the entire old six-task array.
- Array: `0-3%4`; one H100 per task, at most four new GPUs simultaneously.
- Verified 2026-09-28T05:44:35Z: old six tasks RUNNING, total allocated GPUs 6;
  all four new tasks PENDING(Dependency), allocated GPUs 0; no other owned jobs.
- Old tasks were not cancelled, requeued, or modified. New tasks do not hold
  GPU allocations while waiting. This is a live-state snapshot, not a hard
  account-wide scheduler cap against independent future submissions.
- Afterok means successful completion of all predecessor tasks. A failure
  blocks Stage A pending explicit review; do not remove dependencies blindly.

## Task mapping

All tasks: alpha=.9, lambda_current=.8, seed0, six frozen calibrated prompts,
30 outer rounds, complete soft-label pair enumeration, sequence sums.

| Index | Run | beta_train |
|---|---|---:|
| 0 | ipo_ordinary_calibrated_s0 | .2 |
| 1 | ipo_stable_calibrated_s0 | .4 |
| 2 | dpo_ordinary_calibrated_s0 | .8 |
| 3 | dpo_stable_calibrated_s0 | 1.6 |

## Frozen deployment and outputs

Commit: `f0fe034cc0d8e3bef35b0f5e02806ee81340da76`, already on GitHub
`hodge-diagnostics`. The source archive SHA256 is
`4d09a8748c3b4524fe5e14a1f4786474073573d485d49bfc56af18f1b0c757fb`.

Remote deployment:
`/work/users/y/u/yuhe32/ipo/diagnostics/history_calibrated_20260928_stage_a`

Results:
`/work/users/y/u/yuhe32/ipo_runs/cyclic_history_calibrated_20260928/cyclic_history_calibrated/<run_id>`

Logs:
`/work/users/y/u/yuhe32/ipo_runs/slurm_logs/hist_cal_A_0928-4603433_<index>.out/.err`

The deployment hashes all source files and both launch utilities. Each task
verifies the frozen source before running. Training then verifies fresh model
initial scores, token lengths, support hashes, and population prediction gates
before any optimizer update. CPU source/data/calibration checks passed for all
four Stage A arms. These checks are not GPU training results.

## Receipts and duplicate protection

- `deployment.json`: hashes, source commit, frozen run mapping.
- `submission_intent.json`: pre-submit full-account queue and sbatch arguments.
- `submission_receipt.json`: successful scheduler receipt.
- `queue_verified.json`: post-submit dependencies, allocations, throttles.
- `queue_stage_a.py`: refuses a repeated submission attempt, unexpected owned
  work, source drift, existing output directory, or predecessor failure.

Do not rerun the submit command. Existing/new result directories and old
checkpoints are preserved. Recheck all owned running and pending tasks before
future submissions or concurrency changes; generic GPU AllocTRES is counted
once, not added again to the typed H100 breakdown.
