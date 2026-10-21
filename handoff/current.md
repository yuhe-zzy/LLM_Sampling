# Current handoff

Updated2026-10-21, America/Chicago. Read [context index](../doc/INDEX.md).
Latest discussion: explain Fan's experimental SETUP and limitations, not results.
Rechecked builder, actual trainer, population map and main/3B preregistrations
at unchanged1ea6a64. Fixed panels, expected pair losses,40 full-data Adam updates
and model-specific support selection form a controlled mechanism test, not
held-out open-generation validation or a matched LoRA/full-finetuning ablation.
3B five-arm core is all IPO; gamma is derived, not independently swept.
Separate intended restrictions from confirmed tokenization-floor mismatch,
post-result metric revisions, conditional target recalculation and missing
numeric/solver artifacts. See doc/SCIENTIFIC_CONTEXT.md. No branch merge,
experiment, job or timer changes. ASD-STE100-style replies apply.

## Latest authorization: launch Oracle1 / Qwen3

Owner explicitly switches BOTH training preferences and WR to Oracle1 Nemotron,
sets Oracle2 aside, approves writing code and submitting the new experiment.
Use Qwen3-4B non-thinking policy, frozen Qwen3-32B five-trait rater; seed0 six
IPO/DPO ordinary/reference/feedback arms, same100/500/200 panels, states0..100.
FOUR full-accountGPUs maximum. Full queue/AllocTRES/sacct preflight and receipt
guards mandatory. No new old-Oracle2 recovery or WR, no cancellations.

October19 owner requests interim WR curves0,10,...80 and explicitly permits TWO
extra H100s alongside training, totalFOUR. New cooperative extension reserves
training2/evaluation2; old queued workers must pass startup before sidecar joins.
Never hot-edit base source or bypass guard. [Interim guide](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/INTERIM_WR_80.md).

SAME monitor restored5hours after baseline completion and first checkpoint
generators actually wrote valid responses onOctober20T06:44Z.
Every outer boundary saves adapter+s0/st/st-1+RNG+counts+metrics with atomic
commit/hashes. At most10 updates/segment,18h soft stop,24h allocation.
No blind retries after FAILED/TIMEOUT/ambiguous intent.
Generation/scoring/annotation reuse atomic per-response journals. Human review
of the200-response annotation pilot gates full bulk labels, never fabricated.

## Current execution

October21T07:48:09Z: SSH still times out before authentication/remote commands.
NEW local evidence: Cisco VPN adapter Disabled and vpncli state Disconnected.
Owner must reconnect UNC VPN; remote cluster health/cause remains unverified.
Current queue/GPU/progress/errors UNKNOWN. No network/job/source/science/timer
change; SAME5h monitor ACTIVE. Next check: frozen EXT inspect and both ledgers;
never duplicate4863156/4861879 or advance without fresh full-account guards.
[VPN evidence](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/vpn_disconnected_20261021T0748.json),
[earlier connection record](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/connection_failure_20261020T1645.json).
The successful execution snapshots below are historical, not live state.

[Protocol/code](../experiments/oracle1_semantic/README.md), [campaign](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/README.md).
Launch `/work/users/y/u/yuhe32/ipo/diagnostics/oracle1_qwen3_20261015_v1`;
output `/work/users/y/u/yuhe32/ipo_runs/oracle1_qwen3_20261015_v1`.
Frozen source0e7793f5fdd6afdf2fcf80c4ff40649e8926f5a2 pushed and verified.
October15 startup4811149_0/3 and resume4811197_0/3 allCOMPLETED0:0: actual4B
reload/continuedstep2 gate passed, finite metrics, hashes/shared support valid.
AtOctober20T11:49:14Z all58 original receipt tasks, including4859197_0..5,
COMPLETED0:0 at92/90/90/92/90/90;550 states pass continuity/hash/finite/initial
score/support checks. Frozen EXT dry-run passed; **4863156_0..5%2** submitted
11:54:06Z as EXTsequence0003 toward100 for all arms, at most10 updates each.
At12:00:36Z0/1 RUNNING with verified reload92/90 and matching new manifests;
2..5 pending normally. No completed new update claimed. Original trainer unchanged.
Baseline **4861201** remains complete800/800. Generator **4861879_0..5%2**
has4/54 complete checkpoint banks (IPOordinary/reference0/10), plus359/360
valid responses at20;4719 total with baseline. Zero empty, no logged errors;
generator peak allocated8.05GiB each. Full account4H100, potential4, no unrelated work.
[Evidence/receipts](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/records/interim_training_final_segment_startup_20261020T1200.json). Never duplicate either array.

New EXT `/work/users/y/u/yuhe32/ipo/diagnostics/oracle1_qwen3_interim80_20261020_v2`;
sourcecaf5ac2b68d0e4d3778260428eedbeb3441a6658 pushed/verified,43 server CPU tests
pass without skips. FrozenOctober20T00:15:51Z, binding hash
e56f796f20a6e648a298947ba97bf00a6bdb0f36a7982105bbaf35c0962e0b74.
Base identity unchanged,54 required checkpoint commits bound and validated.
v1 preserved/unused after pending-array accounting issue; query parent IDs with
sacct --array, NOT explicit child filters. Do not retry/deploy from v1.
Baseline complete; checkpoint generators active, no WR submitted or complete.
Same5h monitor unchanged. Generated banks are not WR; no scoring or plot yet.
Inspect both ledgers. Use EXT interim.py --advance --lane evaluation or training,
dry-run then --submit once only when that lane is idle/successful and full account
known/within4. Evaluation joins only after legacy pending tasks have passed reload
or completed. Training lane executes original frozen train.py and resumes to100.
Do NOT use original advance while EXT lanes active. Journals shared, subset flags
separate, no rerun of completed answers/scores. Plot actual0..80 tables via --plot.
Hand back to original pipeline only after interim delivery, training100, both EXT
lanes successful/finished and account empty; then reuse0..80 and fill90/100.

## Interpretation

Primary endpoint is generated-output motion in five simultaneous measured
traits, not fixed-response probabilities alone. WR generation every10states,
scored later in a batch, not live WR during training. New model has NEW initial
baseline; do not reuse1.5B responses. Preserve256-token cap diagnostics.
Raw trait curves and discovery-frozen descriptive PCA are primary views.
Single-BT exact logit Hodge cyclic part is zero. DPO exact feedback is also
zero, so its requested feedback arm is a NULL control, not an expected benefit.
No selecting latent axes to maximize visible oscillations; one seed, prompt SE
not seed/rater uncertainty, no convergence or non-BT mechanism claim.

## Existing results / old campaigns

Old Oracle2 timer was deleted at owner's request onOctober14; do not recreate
it. This new Oracle1 monitoring request is separate. Historical oldtraining is
complete, originalgeneration0/1 timed out with outputs preserved;2..5 now
reportCOMPLETED in scheduler, but latest full-bank audit is not performed here.
No old WR is inferred. Frozen old sources23401de and2d6b574 stay untouched.

[Results atlas](../docs/RESULTS_INDEX.md) and [publication coverage](../docs/RESULT_PUBLICATION_20261013.md).
Fan's plan remains a separate collaborator proposal, not executed. Original-slot
average Hodge is not a semantic average. Anchor90 still needs dedicated export;
old Oracle1 final grid remains unreconciled. No hidden/private text in GitHub.
