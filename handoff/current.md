# Current handoff

Updated2026-10-19, America/Chicago. Read [context index](../doc/INDEX.md).

## Latest authorization: launch Oracle1 / Qwen3

Owner explicitly switches BOTH training preferences and WR to Oracle1 Nemotron,
sets Oracle2 aside, approves writing code and submitting the new experiment.
Use Qwen3-4B non-thinking policy, frozen Qwen3-32B five-trait rater; seed0 six
IPO/DPO ordinary/reference/feedback arms, same100/500/200 panels, states0..100.
FOUR full-accountGPUs maximum, two1GPU training/generation tasks, one2GPU WR
judge, one1GPU rater, nonoverlapping. Full queue/AllocTRES/sacct preflight and
receipt guards mandatory. No new old-Oracle2 recovery or WR, no cancellations.

Owner requests NEW5-minute monitor, updating that SAME timer to5hours only
after actual IPO/DPO training AND successful checkpoint reload/continued update.
Every outer boundary saves adapter+s0/st/st-1+RNG+counts+metrics with atomic
commit/hashes. At most10 updates/segment,18h soft stop,24h allocation.
First0/3 tasks run one update, then another allocation resumes them for one
update, then all six. No blind retries after FAILED/TIMEOUT/ambiguous intent.
Generation/scoring/annotation reuse atomic per-response journals. Human review
of the200-response annotation pilot gates full bulk labels, never fabricated.

## Current execution

[Protocol/code](../experiments/oracle1_semantic/README.md),
[campaign](../experiments/oracle1_semantic/campaigns/oracle1_qwen3_20261015/README.md).
SSH blocked2026-10-19T18:00:06Z: two TCP timeouts, no remote command ran.
Current queue/GPU/progress/errors UNKNOWN; last confirmed snapshot follows.
Launch `/work/users/y/u/yuhe32/ipo/diagnostics/oracle1_qwen3_20261015_v1`;
output `/work/users/y/u/yuhe32/ipo_runs/oracle1_qwen3_20261015_v1`.
Pinned model locks and exact Nemotron score reuse audited; all800 panels pass
Qwen3 full chat/EOS length1537 without dropping/truncation. See campaign details.
Frozen source0e7793f5fdd6afdf2fcf80c4ff40649e8926f5a2 pushed and verified.
Final23 server CPU tests pass without skips, plus shell/contract/config checks.
October15 startup4811149_0/3 and resume4811197_0/3 allCOMPLETED0:0: actual4B
reload/continuedstep2 gate passed, finite metrics, hashes/shared support valid.
At2026-10-19T02:02:34Z sequence0009 array4849728_0..5 allCOMPLETED0:0,
as are all46 prior receipt tasks. Complete steps72/70/70/72/70/70;430 saved states
finite/contiguous/hash-valid, shared initial scores/support, contract unchanged,
latest loss/gradients finite, no errors. DPOreference/feedback verified reload60
then ten more updates. Fresh empty-account preflight/dry-run passed. Guarded
submit created **4852338_0..5%2**, sequence0010, at02:06:07Z, up to10 updates
toward82/80/80/82/80/80. OneH100/task,two concurrent,24h/18h soft stop.
At07:22:24Z0/1 COMPLETED0:0 at82/80;2/3 RUNNING at75/77;4/5 pending70/70.
All460 states/contract/shared support and latest loss/gradients valid, no errors.
IPOfeedback70/DPOordinary72 reload plus five updates each checked; full account
2H100, no unrelated/releasing work. Same0010; no new submission this check.
Never repeat0010 or earlier jobs. No source/scientific change or oldOracle2 action.
Intent/receipt/deployment are in campaign records and local private launch dir.
Baseline/generation/WR/annotation remain unsubmitted;100 rounds not yet complete.
SAME heartbeat `oracle1-qwen3-training-and-resume-checks` now ACTIVE every5HOURS,
tool and stored schedule verified. Do not create another timer or repeat probes.
Next: check UNC VPN/SSH, retry read-only; only fresh successful receipt tasks and
empty full account permit frozen queue advance/dry-run/submit. No blind retry.

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
