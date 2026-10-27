# Fixed-target inner-fit diagnostic launch

Approved and submitted on 2026-09-28. Job array: **4605171**, tasks 0-3%4,
one H100 allocation per task, at most four concurrent tasks. The account was
empty immediately before submission. This is not an account-wide scheduler cap.

| Task | Source arm | Alpha | Lambda | Beta |
|---|---|---:|---:|---:|
| 0 | IPO ordinary | .9 | .8 | .2 |
| 1 | IPO stable control | .9 | .8 | .4 |
| 2 | DPO ordinary | .9 | .8 | .8 |
| 3 | DPO stable control | .9 | .8 | 1.6 |

All runs use seed 0 and the same six calibrated prompts. Each loads its own
completed Stage A adapter at step 20. It freezes the source round 20-to-21
sampler, reference, pair weights and target, retaining the original IPO/DPO
pair loss. There are **zero outer updates** and no Stage B/C submission.

Six identical 10-epoch blocks are trained consecutively against that fixed
target. Each block resets AdamW and uses the unchanged 90-step LR schedule and
round-20 pair order. Budgets 10/30/60 share a common training prefix. Measurements
are written after every block, adapters after cumulative epochs 10/30/60.

Checks before training:
- All four parent tasks 4603433 completed with exit 0:0.
- Original support, checkpoint files and core source hashes are validated.
- Frozen targets and all intermediate outer-state fields reproduce saved data.
- Runtime package versions match the parent run.
- Fresh checkpoint scores and token lengths match the source before optimization.
- The frozen state is checked for mutation after every training block.

Validation: local test discovery ran 51 tests, with 11 unavailable Torch tests
skipped. On the server, 5 new fixed-target tests, 23 history math tests and 9
Torch CPU tests all passed. An initial remote history test attempt lacked the
test-only experiment_plan.json fixture; after copying that unchanged fixture,
all 23 tests passed. This did not change any training source or configuration.

Server deployment:
`/work/users/y/u/yuhe32/ipo/diagnostics/history_fixed_target_20260928`

Server results:
`/work/users/y/u/yuhe32/ipo_runs/cyclic_fixed_target_20260928/<source_run>_fixed_t20`

Slurm logs:
`/work/users/y/u/yuhe32/ipo_runs/slurm_logs/hist_fixed_target_0928-4605171_<task>.out/.err`

Local `deployment.json`, `submission_intent.json`, `submission_receipt.json`
and `queue_verified.json` record deployment hashes and scheduler responses.
The inspector is read-only and separately stored; it is not part of the frozen
training source. Pending(Resources) is normal and is not an experiment failure.

Do not duplicate this submission, restart completed probes, overwrite Stage A,
or infer authorization for Stage B/C. New results should first be compared at
matched training budgets. Git publication was later requested on September 29;
see the [campaign index](../README.md) for current status and synchronization.

## Startup verification

At 2026-09-28T19:30:22Z all four tasks were RUNNING, totaling four allocated
GPUs across the account with no other owned jobs. All four checkpoint score
replay errors were exactly zero. Tasks 0/1/2 had completed their first block
(10 epochs); task 3 started about one minute later and was in its first block.
No Traceback/OOM/non-finite snapshot was found. See `first_progress.txt`.

The first trained block is not a bitwise replay of the original outer step 21:
maximum raw score differences for tasks 0/1/2 were .291/.172/.182 nats. Original
training RNG states are not saved; this probe initializes its seed to zero and
may also have numerical execution differences. The cause of that discrepancy
has not been isolated. The budget contrast must use the NEW probe's shared
10/30/60-epoch prefix, not treat the original step-21 values as its 10-epoch arm.
Both initial-score equality and the discrepancy are explicitly recorded.

## Completion check

At 2026-09-29T01:06:55Z, all four tasks were COMPLETED with exit 0:0 and six
blocks written. Elapsed times were 8:39, 8:44, 8:52, and 8:51. The full account
queue was empty and allocated GPU count was zero. No Traceback/OOM or non-finite
snapshot was detected. Raw files are downloaded under
`../cyclic_fixed_target_results_20260928/raw/`; their hashes and independently
verified metrics are saved alongside them.

Normalized target error (initial fixed-target distance = 1):

| Arm | 10 epochs | 30 epochs | 60 epochs |
|---|---:|---:|---:|
| IPO ordinary | .714 | 1.181 | 1.188 |
| IPO stable | 1.293 | 1.310 | 1.463 |
| DPO ordinary | .707 | 1.094 | 1.166 |
| DPO stable | 1.379 | 1.473 | 1.407 |

Increasing the budget under the repeated-reset training procedure did not
solve target fitting. Endpoint expected pairwise loss at 60 epochs was also
higher than before training in all four arms. For both ordinary arms, all six
prompts had larger target error at 60 than at 10 epochs. This does not isolate
learning rate, optimizer reset, minibatch effects, precision or representation
capacity, and does not establish token collapse. Stage B/C were unsubmitted
at this historical check; the later campaign index records B completion and
C submission under subsequent user authorization.
