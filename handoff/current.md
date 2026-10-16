# Current handoff

Updated2026-10-16, America/Chicago. Read [context index](../doc/INDEX.md).

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
Launch `/work/users/y/u/yuhe32/ipo/diagnostics/oracle1_qwen3_20261015_v1`;
output `/work/users/y/u/yuhe32/ipo_runs/oracle1_qwen3_20261015_v1`.
Model revisions pinned/downloaded: Qwen3-4B1cfa9a720891 and Qwen3-32B9216db5781bf.
Exact existing Nemotron scores reused, pure BT matrices rebuilt; all800 panels
pass Qwen3 full chat/EOS length audit1537 without dropping/truncation.
Frozen source0e7793f5fdd6afdf2fcf80c4ff40649e8926f5a2 pushed and verified.
Final23 server CPU tests pass without skips, plus shell/contract/config checks.
At10:25:58Z first4811149_0/3 and resume4811197_0/3 allCOMPLETED0:0.
Both actual4B objectives passed RESUME_VERIFIED and continued update tostep2,
with finite loss/gradients, valid contiguous hashes/states and matching initial
scores/token support. Resume peak allocated11.89GiB/reserved21.35GiB perGPU.
Full-model startup AND cross-process resume/update gate has passed.
At2026-10-16T09:35:06Z sequence0004 array4814410_0..5 allCOMPLETED0:0,
as are all prior receipts. Complete steps22/20/20/22/20/20;130 saved states
finite/contiguous/hash-valid, shared initial scores/support, contract unchanged,
no errors. DPOreference/feedback also verified reload10 then ten more updates.
Full account empty. Frozen dry-run/guarded submit created **4819533_0..5%2**,
sequence0005, at09:37:17Z, up to10 more updates toward32/30/30/32/30/30.
At14:30:00Z0/1 COMPLETED0:0 at32/30;2/3 RUNNING at24/26;4/5 pending, still20/20.
Account2H100, no unrelated work/errors;158 finite contiguous hash-valid states,
shared initial scores/support and contract pass. IPOfeedback20/DPOordinary22
reloads verified, four further updates each. No new submission; never repeat0005.
Intent/receipt/deployment are in campaign records and local private launch dir.
SAME heartbeat `oracle1-qwen3-training-and-resume-checks` now ACTIVE every5HOURS,
tool and stored schedule verified. Do not create another timer or repeat probes.
Next: inspect live progress; only all prior receipt tasksCOMPLETED0:0 and empty
full account permit frozen queue advance/dry-run/submit for the next segment.

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
