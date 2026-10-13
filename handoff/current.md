# Current handoff

Updated2026-10-13, America/Chicago. Read [context index](../doc/INDEX.md).

## Fan's aligned-latent analysis plan (shared2026-10-13)

Fan requested publishing his Claude-drafted S0–S5 plan so Yu He can run it on
the exact existing500-panel bank. Start with the
[plan bundle](../experiments/hodge_diagnostics/plans/latent-estimation-2026-10-13/README.md)
and execution clarifications (take precedence over original draft shorthand).
Both features requested: frozen embeddings/PCA8,16 first (no annotation fee),
new judged continuous style attributes second (2000 responses; confirm judge
channel/budget). Pure latent fit only this round; identity/logit, each oracle
and mixture, rank/fit, held-out signal and500-draw permutation/flip nulls.
No aligned signal has yet been computed. Keep WR/existing runs untouched;
no new training, rho-scan, oracle construction or extra GPU allocation.

## Latest inspection request

Latest clarification: mean P is the elementwise equal average of500 training
matrices in original candidate slots, with no semantic alignment, policy weights
or step averaging. Verified sum P12=281.2262593683416 and mean=.5624525187366831.
Do not present it as an already aligned global semantic preference matrix.

Follow-up asks for concrete shared-direction measurement standards using the
existing data. Proposed six features (five HelpSteer attributes plus log length),
mean expected-feature trajectories, excursion/backtracking/late-motion metrics,
and optional static-preference-selected 2D plane with250/250 prompt validation.
All remain proposals, not executed or approved new experiments. Attributes still
need identity-checked joining from raw source; six-arm numeric lengths/trajectories
are confirmed locally. The split holds out axis selection, not model training.
See [operational proposal](../doc/SCIENTIFIC_CONTEXT.md#proposed-operational-protocol-not-yet-executed).

User wants shared semantic/output-direction oscillations, not only prompt-level
cycles; first compute average Hodge decomposition. All500 training panels now
computed with existing core: identity norm(mean C)=.022619 vs mean(norm C)=.221816,
aggregate cyclic energy share .357286%. Original candidate indices are not
semantic axes. Logit averaging orders reported separately. Seven focused CPU
unittests passed; no full-suite/GPU test or scheduler action. See
[average report and proposed latent analysis](../experiments/oracle2/campaigns/oracle2_real_20261009/average_hodge_20261013/README.md).
Shared-feature trajectory/latent-plane analysis is a proposal, not yet executed.
Analysis commit657af2ac09f95bb39c918cc3e9e487053c01b9bf was pushed to
origin/hodge-diagnostics and exact remote SHA verified2026-10-13.

## Completed publication request

Enroll this project in agent-native-context and put reviewed shareable results
on GitHub. The canonical context root is this Git checkout (`github_release`
in the owner's local workspace), not the outer private working directory.

## Current work

Context and reviewed results were pushed to origin/hodge-diagnostics as
2a25ed505978e4ab8acde0c475137db0656988ac; remote SHA verified2026-10-13.
Six publication tests pass; inventory contains435 PNG/PDF/CSV artifacts.
This covers reviewed local artifacts, not missing WR or unexported server data.
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
