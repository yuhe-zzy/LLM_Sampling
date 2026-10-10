# Calibration follow-up for Fan and Yu He - 2026-09-28

Status: code and CPU numerical checks, not new neural results or permission to
run jobs. Builds on branch `hodge-diagnostics` at 27c7895. Fan's original reports
and handoff files remain unchanged. The existing six GPU runs are untouched.

## Main correction to the September 26 interpretation

I agree that cyclic energy alone is not realized feedback gain, that measured
reference scores matter, and that actual DPO differs from entrywise-logit PsiPO.
However, `analysis/pilot_frontier.py` originally evaluated IPO only. Its .863
index does not establish that both IPO and DPO ordinary baselines are stable.

For the existing hard tournament at alpha=.9/lambda=.8/beta_train=1 and a flat
reference, the actual IPO radius is .929132 (squared .863287), whereas actual
DPO is 1.288721. With the illustrative 5-nat-spaced reference, the radii are
.900000 and 1.025218 respectively. Finite differences of the training target
confirm these values. The script now prints both actual operators explicitly.

Hard labels do not imply an infinite actual DPO optimum on this strongly
connected cyclic win graph. They do make elementwise logit(P) undefined. The
new tests also reproduce the balanced soft-cycle counterexample: actual DPO
squared radius .936 versus logit-proxy 1.01309 at alpha=.84/lambda=.8/beta=1.

## Implemented changes

- `cyclic_history/population_calibration.py`: actual IPO/BT fixed points,
  implicit BT derivatives, ordinary and augmented-history Jacobians, exact
  trajectories. Failed roots are unresolved, never classified as stable.
- `prepare_calibrated_plan.py`: measured-score selection, explicit balanced
  soft cycles, fixed role assignments, matched reversed-orientation controls,
  per-objective beta calibration, provenance and immutable plan outputs.
- `calibrated_protocol.py`: support/configuration/initial-score/length checks
  and fail-closed local prediction gates, before any training update.
- `run_oracle_feedback_extrapolation.py`: accurate name for the existing
  full-P assisted positive-loss feedback implementation. The signed
  two-sampler loss remains a separate author decision, not a completed test.
- Full-pair soft-label mode, fixed-point residuals, and per-prompt cyclic-mode
  amplitude/phase, while preserving sequence sums and legacy reproduction.
- The Hodge grid no longer skips hard scalar-score IPO. Population output
  labels identity IPO and entrywise-logit PsiPO separately. The existing public
  dataset tables were NOT rerun or retroactively relabeled.
- Hodge tests are now included in GitHub CI. The BF16/cuDNN SDPA repair and its
  CPU tests are included; this numerical kernel issue is distinct from the
  instability mechanism and the older token-average generation collapse.

## Calibration outcome and exact proposed experiments

Only 8/500 existing panels have every initial raw response probability >=.005.
Six pass ordinary radius>=1.03 and both intervention radii<=.98 for BOTH losses
and BOTH cycle orientations. Do not present this as a representative 500-panel
calibration or a final manuscript experiment.

Common alpha=.9, lambda_current=.8, seed0; balanced .8/.2 four-cycle with opposite
pairs at .5. IPO beta_train=.2; DPO beta_train=.8. The planned contrasts are:

1. **Stage A, four runs:** ordinary unstable and ordinary stable control for each
   loss. Stable controls double beta_train to .4 and 1.6.
2. **Stage B, four runs, conditional on Stage A:** reference nu=.45 and
   oracle-feedback kappa=.5 at the SAME beta as the unstable ordinary control.
3. **Stage C, two runs:** mixed orientation ordinary, identical text/support,
   scores, norm, and zero directional component. Shared LoRA coherence is not
   guaranteed merely by assigned labels or response ranks.

The ten-row plan is prepared, not approved for blanket execution. Each uses
six distinct prompts, all six unordered pairs, 10 inner epochs, 90 optimizer
updates/outer round, 30 outer rounds, one GPU, one seed. No GPU wall-time estimate
has yet been measured for this new workload. Follow the supplied workspace
`AGENTS.md` phase rules and inspect existing/unrelated allocations before
submission; this release does not expand prior run-specific GPU approvals.
No scheduler-wide hard cap exists.

| Objective | Ordinary radius | Reference radius | Feedback radius |
|---|---|---|---|
| IPO, measured reference | 1.047--1.078 | .956--.962 | .920--.955 |
| Actual DPO, measured reference | 1.050--1.079 | .957--.962 | .923--.957 |

All selected interventions also recover numerically from the actual initial
scores in a 200-step exact-population check (fixed-point distance <.001);
ordinary remains >2 away. This does not prove global convergence, periodicity,
neural stabilization, or held-out quality preservation.

Full setup, CLI commands, intervention definitions, scope restrictions and
the checked-in numeric plan are in
[CALIBRATED_PROTOCOL.md](../cyclic_history/CALIBRATED_PROTOCOL.md). Exact replay
from that plan needs no GPU or raw dataset. A larger minimum support threshold
blocks this pool instead of silently relaxing selection or duplicating data.

## Decisions still needed

- Approve only Stage A initially, or first assemble a broader set of suitable
  panels. The six-prompt diagnostic is not intended to complete the paper's
  full neural evidence by itself.
- Decide whether the manuscript will study this accurately named full-P
  assisted intervention, a directly implemented two-sampler estimator, or both
  with separate operator definitions and predictions.
- If the exact-population instability is absent in neural training, measure
  inner-update error and accessible modes before adding seeds or a sweep.
- Choose claim-appropriate replication after an informative one-seed pilot.
  Five seeds are not imposed by this implementation.

## Verification performed for this release

- 46 cyclic/history tests passed in an isolated server source directory with
  `CUDA_VISIBLE_DEVICES` empty, including tiny CPU LoRA training for both losses,
  finite-difference actual-operator checks, prediction gates, and stage-launcher
  dry runs using `/bin/echo` instead of Python/model execution.
- 30 original release, sequence-sum, data, and plot tests passed on server CPU.
- 23 Hodge tests passed locally; the only warnings were existing Matplotlib /
  pyparsing deprecations. No packages were installed into the active server env.
- All ten frozen configurations passed read-only checks against the actual
  server dataset, including original/transformed support hashes and local
  predictions. Fresh-model score verification remains a required GPU preflight,
  not something this CPU dataset check can establish.
- The complete 200-step numeric replay passed for all ten arms. Requiring at
  least 32 panels correctly emitted `BLOCKED_INSUFFICIENT_SUPPORT` and no plan.
- Shell syntax and staged launch selection were checked; no `sbatch` was called.

No new-protocol GPU training, neural oscillation, stabilization, or wall time
has been measured. The CPU artifacts must retain those qualifications.

No current job was cancelled, changed, or resubmitted; no new GPU run started.
