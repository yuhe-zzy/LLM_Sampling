# GPU budgets and phased scheduling

This repository update does not launch or modify any job.

## Preflight

Before submitting, releasing, requeuing, or changing concurrency, inspect both
running and pending jobs for the account. Include unrelated experiments,
interactive jobs, and jobs still completing/releasing GPUs.

On the original host, user-filtered queue queries have sometimes returned an
empty result despite active jobs. Cross-check the full expanded queue
(`squeue -a -r`) by displayed owner/UID, then verify selected allocations with
`scontrol show job`. Count generic AllocTRES `gres/gpu` once; do not add the
typed `gres/gpu:h100` breakdown again.

## Phase order

1. Preserve already-approved running work. An existing 3-GPU oracle plus
   one 1-GPU sampling task can be a transitional four-GPU allocation.
2. After the preserved oracle ends, sampling only: at most two concurrent
   one-GPU jobs and at most **two allocated GPUs across all account programs**.
3. After the entire sampling phase ends, remaining oracle jobs run serially:
   one three-GPU task at a time.

The oracle phase needs three GPUs. Do not describe the sampling two-GPU cap
as an absolute cap applying to oracle training.

## Array templates are not account-wide caps

- Sampling templates use `%2`; command-line `--array` overrides must retain it.
- Oracle templates use `%1`.
- Separate arrays have independent throttles. Two sampling arrays at `%2`
  can allocate four GPUs. Chain them or otherwise account for the total.
- Do not reserve GPUs just to wait in a shell loop for another job.
- No account-wide association cap is installed by these scripts. A hard
  account cap would require scheduler-administrator support.

After live preflight, an approved new sampling array can be followed by an
oracle array using a dependency on the **whole** sampling array:

```bash
sampling_id=$(sbatch --parsable --array=0-1%2 slurm/nonoracle.sh configs/nonoracle_transitive.json)
sbatch --dependency=afterany:"$sampling_id" --array=0-15%1 slurm/oracle.sh configs/oracle.json
```

This example assumes no conflicting current/pending work and that all
specified configurations are approved and still needed. It is not safe to
execute blindly on a busy account. `afterany` waits for termination, not
experimental success. Use `afterok` when successful completion is required,
and explicitly inspect failed dependencies.

If a preserved oracle **task** must finish before sampling, make sampling
depend on that individual task. Do not make sampling depend on an entire
oracle array that itself has remaining work waiting on sampling; that creates
a dependency cycle.

## Failures and resumption

Do not treat PENDING(Resources), PENDING(JobArrayTaskLimit), or
PENDING(Dependency) as failed experiments. Inspect OOM, Traceback, TIMEOUT,
FAILED and metrics. Report failures without automatic restarts. Output
directories and checkpoints are preserved; the launcher refuses overwrite.

Changing the scheduler template or publishing source does not alter already
running Python processes. Do not cancel/requeue work or change dependencies
merely to align it with this release without explicit authorization.
