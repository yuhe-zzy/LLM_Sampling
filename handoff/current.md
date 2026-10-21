# Current handoff

Updated2026-10-21, America/Chicago. Read [context index](../doc/INDEX.md).

## Latest request: retain IPO ordinary only

Owner reports VPN reconnected and explicitly confirms "first group" means
NEW Oracle1/Qwen3 IPO ordinary, array index0. Retain its training/generation
and shared new-Qwen3 baseline; cancel further work on the other five arms.
This supersedes six-arm automatic advancement and the old no-cancellation rule
for indices1..5 only. Preserve all results/checkpoints/receipts; no oldOracle2
or unrelated-job actions. Same5h monitor now follows index0 only.
Owner now explicitly approves finishing first-arm evaluation and win rate.
Use [single-arm0..100 route](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/IPO_ONLY_EVALUATION.md).
No retraining/other-arm restoration; no new job at preparation checkpoint.
Do not call BASE or EXT six-arm --advance/--submit or bypass annotation review.

## Verified execution

SSH works again. October21T09:01:33Z full owner/UID queue is empty, zero GPUs. Native parent
sacct --array reports training4863156_0/1 COMPLETED0:0, indices2..5 FAILED1:0;
generation4861879_0/1 COMPLETED0:0, indices2..5 FAILED1:0. No scancel was needed:
these were already terminal, not newly canceled scheduler tasks.
Task0 training manifest is COMPLETE at100, actual job4863157. Generation is
incomplete despite successful18h segment exit: complete0..70,80 has249/800,
551 still missing for interim0..80. WR never submitted. At08:59:59Z all568 saved
states pass hashes/continuity/finite/initial-score/support checks; task0 has101.
All14,100 existing responses validate, zero empty. [Audit](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/retain_ipo_ordinary_20261021.json).
Training task2 failed at startup with Ambiguous accounting record; generation
task2 was blocked by the prior training failure. Do not silently retry them.
Frozen EXT --inspect correctly stops on failed receipts; read-only diagnostic
validation does not alter the frozen execution/submission guards.
Private helper `oracle1_qwen3_launch_20261015/validate_interim_generation_readonly.py`
supports `--diagnose-terminal-failures` only with empty queue/scontrol and terminal
accounting. Do not repeat full hashes on unchanged terminal outputs every check.

[Campaign/status](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/README.md),
[interim guide](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/INTERIM_WR_80.md).
BASE `/work/users/y/u/yuhe32/ipo/diagnostics/oracle1_qwen3_20261015_v1`;
OUTPUT `/work/users/y/u/yuhe32/ipo_runs/oracle1_qwen3_20261015_v1`.
Frozen train source0e7793f5fdd6afdf2fcf80c4ff40649e8926f5a2, unchanged.
EXT `/work/users/y/u/yuhe32/ipo/diagnostics/oracle1_qwen3_interim80_20261020_v2`;
sourcecaf5ac2b68d0e4d3778260428eedbeb3441a6658, unchanged.
Never duplicate baseline4861201, generation4861879 or training4863156.
v1 remains unused. Preserve canonical journals, all partial banks and locks.
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

Latest discussion explains Fan's experimental SETUP and limitations, not results.
Rechecked builder, actual trainer, population map and main/3B preregistrations
at unchanged1ea6a64. Fixed panels, expected pair losses,40 full-data Adam updates
and model-specific support selection form a mechanism test, not held-out
open-generation validation or a matched LoRA/full-finetuning ablation.
3B five-arm core is all IPO; gamma is derived, not independently swept.
Distinguish intended restrictions from tokenization-floor mismatch, post-result
metric revisions, conditional target recalculation and missing numeric/solver
artifacts. See doc/SCIENTIFIC_CONTEXT.md. ASD-STE100-style Chinese replies apply.

## Existing results

[Results atlas](../docs/RESULTS_INDEX.md), [publication coverage](../docs/RESULT_PUBLICATION_20261013.md).
OldOracle2 monitor was deletedOctober14; never recreate or resume old work.
Preserve frozen old sources23401de and2d6b574. No old WR completion is inferred.
Fan's plan remains a proposal, not executed. Original-slot average Hodge is not
a semantic average; anchor90 export and oldOracle1 final grid remain unresolved.
