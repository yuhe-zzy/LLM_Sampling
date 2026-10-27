# Current handoff

Updated2026-10-26, America/Chicago. Read [context index](../doc/INDEX.md).

## Latest request: retain IPO ordinary only

Owner reports VPN reconnected and explicitly confirms "first group" means
NEW Oracle1/Qwen3 IPO ordinary, array index0. Retain its training/generation
and shared new-Qwen3 baseline; cancel further work on the other five arms.
This supersedes six-arm automatic advancement and the old no-cancellation rule
for indices1..5 only. Preserve all results/checkpoints/receipts; no oldOracle2
or unrelated-job actions. Same monitor is now PAUSED after completed delivery.
Owner now explicitly approves finishing first-arm evaluation and win rate.
Use [single-arm0..100 route](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/IPO_ONLY_EVALUATION.md).
No retraining/other-arm restoration. Generation4878947 and WR4889849 completed.
Do not call BASE or EXT six-arm --advance/--submit or bypass annotation review.

## Verified execution

October21T22:33:17Z:4878947 and4889849 both COMPLETED0:0, all11 banks800 plus
baseline800 and9600 scores hash/finite valid; no empty responses/logged errors.
Source621b7d7; ROOT `ipo/diagnostics/oracle1_qwen3_ipo_only_20261021_v1` unchanged.
Recomputed2200 prompt rows/11 summaries agree to5.56e-17. No writer locks or
unpaired intent; full account empty22:33:04Z,0 allocated/potential GPUs.
Frozen plot exports downloaded/hash-checked; PNG and rendered PDF inspected.
[WR figures/CSV/report](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/ipo_only_wr_results_20261021/README.md)
delivered; SAME monitor confirmed PAUSED. No further automatic work is authorized.
WR47.569% at0 to47.536% at100; paired change-0.033points, prompt SE1.163points.
No clear gain, not proof of equivalence/stability; roughly half hit cap256.
[Completion audit](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/ipo_only_wr_complete_20261021.json).

Historical09:01Z: all old jobs terminal, no scancel needed. Training4863156 and
generation4861879 indices0/1 COMPLETED0:0,2..5 FAILED1:0; preserve these outcomes.
Task0 training100 complete,101 valid states. At08:59Z all568 states and14,100
existing responses passed continuity/hash/finite/initial-support/seed checks.
[Audit](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/retain_ipo_ordinary_20261021.json).
Withdrawn tasks failed on ambiguous accounting or prior-failure guards; no retry.
Old EXT inspect correctly blocks. Never repeat full568-state hashes needlessly.

[Campaign/status](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/README.md),
[interim guide](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/INTERIM_WR_80.md).
BASE `/work/users/y/u/yuhe32/ipo/diagnostics/oracle1_qwen3_20261015_v1`;
OUTPUT `/work/users/y/u/yuhe32/ipo_runs/oracle1_qwen3_20261015_v1`.
Frozen train source0e7793f5fdd6afdf2fcf80c4ff40649e8926f5a2, unchanged.
EXT `/work/users/y/u/yuhe32/ipo/diagnostics/oracle1_qwen3_interim80_20261020_v2`;
sourcecaf5ac2b68d0e4d3778260428eedbeb3441a6658, unchanged.
Never duplicate old jobs; v1 unused. Preserve canonical journals/partial banks/locks.
No administrator GPU hard cap exists; maximum FOUR account GPUs, with full
squeue -a -r, scontrol and native-parent accounting before any resource action.

## Scientific scope

Qwen3-4B BF16 non-thinking, Nemotron Oracle1 training preferences and WR,
500train/200eval, four answers, IPO ordinary seed0 alpha.9 lambda.8 beta.2.
Sequence-sum, states0..100, generation temperature.8/top_p.95/top_k0/cap256.
WR uses generated answers,4x4 comparisons then prompt averaging; prompt SE is
not training-seed uncertainty. Training completion is not generation or WR
completion. Do not use fixed-panel probabilities/zeros as WR. Preserve cap
diagnostics. Five-trait200-response pilot needs human review before bulk labels.
Fixed-discovery PCA is descriptive, not preference-cycle evidence.

## Separate collaborator discussion

Owner asks whether Fan's design supports the uploaded theory, not utility/WR.
Reviewed submission15 PDF, especially theory assumptions and Sections9-10;
finite-panel controlled validation is appropriate, not a proof certification.
Check inner-fit tails, local-boundary versus late amplitude, pre/post-hoc
predictions, pooling-claim coverage, and Section9.2's alpha-only rate claim.
Fan is now3542986: dpo_map.py published and X15 complete; old missing-solver/
pending claims superseded. X15 IPO logit-map mistake corrected post hoc, not
an original prediction success. See doc/SCIENTIFIC_CONTEXT.md. No experiments,
jobs or timers changed; ASD-STE100-style Chinese replies continue.

## Existing results

[Results atlas](../docs/RESULTS_INDEX.md), [publication coverage](../docs/RESULT_PUBLICATION_20261013.md).
OldOracle2 monitor was deletedOctober14; never recreate or resume old work.
Preserve frozen old sources23401de and2d6b574. No old WR completion is inferred.
Fan's plan remains a proposal, not executed. Original-slot average Hodge is not
a semantic average; anchor90 export and oldOracle1 final grid remain unresolved.
