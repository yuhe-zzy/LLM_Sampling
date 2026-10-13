# Current handoff

Updated2026-10-13, America/Chicago. Read [context index](../doc/INDEX.md).

## Latest request

Enroll this project in agent-native-context and put reviewed shareable results
on GitHub. The canonical context root is this Git checkout (`github_release`
in the owner's local workspace), not the outer private working directory.

## Current work

Context and results reconciliation are locally complete and validated. Six
publication tests pass; inventory contains435 PNG/PDF/CSV artifacts. Check the
daily note for final remote verification; do not mistake preparation for push.
Start with [results atlas](../docs/RESULTS_INDEX.md) and
[coverage/exclusions](../docs/RESULT_PUBLICATION_20261013.md).

## Experiment state (dated, not live)

At2026-10-13 10:00:34 CDT all six Oracle2 training arms were complete0..100.
Generation4794738_2/3 ran on twoH100s, through complete50/partial60;4/5 queued.
Original0/1 remain TIMEOUT with complete0..70 and partial80 preserved. No WR.
Recovery source2d6b574 is CPU-validated/deployed but not GPU-tested/submitted.
Original training/generation source23401de remains frozen.

## Next operational action

Existing two-hour monitor: re-read live full account and latest intent/receipt.
Wait for original2..5 successful/validated and full account empty. Then v5
frozen queue.py recover once, two1GPU tasks. Only after successful recovery,
all66 banks and baseline validate, and empty account, use v5 WRscore.
See [status and constraints](../doc/EXPERIMENT_STATUS.md) and
[recovery instructions](../experiments/oracle2/campaigns/oracle2_real_20261009/RECOVERY_20261013.md).
Never duplicate training, baseline, candidate audit, original generation or
existing recovery submissions. Do not overlap phases or exceed4 accountGPUs.

## Publication follow-ups, not new experiment authorization

Anchor90 trajectories still need a dedicated validated export; completed Slurm
status alone is not a published figure. Oracle1 final missing/partial grid
coverage remains unreconciled. Oracle2 WR is genuinely absent, not0.
Keep historical token-average panels separate from sequence-sum. Preserve
known DPO NaN tails as invalid, not uniform-policy convergence.
