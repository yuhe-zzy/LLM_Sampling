# Current handoff

Updated2026-10-15, America/Chicago. Read [context index](../doc/INDEX.md).

## Latest operational request: stop the timer

Owner requested stopping this timer on2026-10-14. Automation `oracle2`
(Oracle2 runtime/memory checks) was deleted in the app; deletion confirmed.
No future automatic checks or phase submissions from that timer. No server
job was cancelled, modified or submitted. Do not recreate monitoring without
a new user request. Latest remote state remains unknown after SSH timeouts.

## Latest request: actual generated-output semantic dynamics

Owner corrects the endpoint: test whether actual model outputs oscillate in
shared/global dimensions, not just fixed-response probabilities within prompts.
Open generation plus frozen multidimensional feature measurements is the main
evidence; fixed-panel diagnostics are auxiliary. Each response can have a score
on every trait, not one exclusive label. Compare prompt-balanced feature means,
distributions and fixed-plane trajectories with sampling/annotation uncertainty.
Distinguish drift/jitter from repeated swings; do not presume a positive result.
Existing plan has200 eval prompts x4 responses at11 checkpoints0,10,...,100.
Validated WR generation banks could be reused; current completion is unknown.
Do not invent101 generated states from101 numeric fixed-panel snapshots.
No annotation model/budget, new experiment, code/config change or jobs approved.
See [definitions and proposal](../doc/SCIENTIFIC_CONTEXT.md) and today's note.

Fan's branch at7e152a2 was read, not merged or executed. His S0-S3 fits shared
preference structure; S4 regresses saved fixed-candidate logits, not generated
features. These are different estimands; explain any proposed connection.
Start with [plan and clarifications](../experiments/hodge_diagnostics/plans/latent-estimation-2026-10-13/README.md).
HelpSteer is our existing dataset; its labels apply to the original answers,
not new generated texts. Style measurements and Oracle2 WR have separate roles.
No aligned latent signal computed. Original-slot mean Hodge is not a semantic
average; [results](../experiments/oracle2/campaigns/oracle2_real_20261009/average_hodge_20261013/README.md)
remain valid only under their declared unaligned candidate-slot definition.

## Publication state

Context and reviewed results were pushed to origin/hodge-diagnostics as
2a25ed505978e4ab8acde0c475137db0656988ac; remote SHA verified2026-10-13.
Six publication tests pass; inventory contains435 PNG/PDF/CSV artifacts.
This covers reviewed local artifacts, not missing WR or unexported server data.
Start with [results atlas](../docs/RESULTS_INDEX.md) and
[coverage/exclusions](../docs/RESULT_PUBLICATION_20261013.md).

## Experiment state (dated, not live)

At2026-10-14 14:00 CDT two SSH attempts timed out before remote commands.
Current allocation/progress/logs UNKNOWN; owner asked to check VPN/access.
Last successful check00:01 CDT: all training complete0..100; generation2/3
COMPLETED with11 validated banks;4/5 RUNNING on twoH100s, complete0..40 every10,
partial50 rows548/461, no other owned work or new logged error. Original0/1
TIMEOUT with preserved banks. Recovery source2d6b574 was CPU-deployed but
not GPU-tested/submitted; no WR at that snapshot. No remote change this turn.
Original training/generation source23401de remains frozen.

## Next operational action

Timer stopped by owner. For a future requested inspection, re-read full account
and intents/receipts; do not assume the historical scheduler state is current.
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
