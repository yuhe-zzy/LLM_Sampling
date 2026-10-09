# Experiment collaboration

## Synchronize changes to GitHub

The owner requested on 2026-09-29 that every experiment modification be
synchronized to GitHub. Fetch and integrate collaborator updates first, keep
edits scoped, run relevant checks, commit and push, then verify the remote
commit. Never force-push or discard collaborators' changes. The current cyclic
campaign is shared on `hodge-diagnostics`; do not silently rewrite `main`.
If remote access fails, retain the local work and report that it is not synced.

Include code, configs, tests, reproducibility instructions, dated execution
status and shareable numeric summaries/figures. Exclude credentials, private
raw prompt/response text, model weights, adapters, raw dumps, caches and logs.
Do not use `git add .` without reviewing the complete file list. Do not replace
an immutable campaign source with current HEAD or claim that a documentation
update was used in an earlier run. Publishing code does not authorize GPU jobs.

## GPU and campaign safety

Oracle2 authorization, 2026-10-09: the owner confirmed fixed real-candidate
training plus open-generation WR, .6 Nemotron/.4 Skywork, six IPO/DPO
ordinary/reference/feedback arms, seed0, 100 outer updates, and disjoint
100 calibration/500 train/200 evaluation prompts with four candidates each.
WR must use outer states 0,10,...,100. Code and status live in
`experiments/oracle2/campaigns/oracle2_real_20261009`. Do not duplicate phases;
read deployment/intent/receipt and live owner/UID queue first. Audit the true
mixed matrices and judge interfaces before training, report too few cycles
or saturated scales rather than silently changing temperatures or mixture.
Oracle2 phases are sequential: three-GPU scoring, one-GPU baseline, six
one-GPU policy tasks, six one-GPU generation tasks, three-GPU WR scoring.
This newly approved cached-judge design does not authorize six simultaneous
three-GPU oracle jobs. No extra seed/arm and no cancellation of older jobs.

Oracle2 candidate audit has been submitted as **4773811**, three H100s,
frozen source `975726c30675e9a3df11da1582482f6bd9c07b46`; 96 server CPU tests
passed without skips. At 2026-10-09T16:56:02Z it was RUNNING and was the only
owned job (three allocated GPUs). Judge scores were still initializing, so
this is not inference/audit completion. Six formal training arms have NOT
been submitted; inspect the actual audit before creating its review record.
Never duplicate the audit or silently retune temperatures. No automatic
phase chaining/follow-up was created. See the campaign's receipts/status.

Later 2026-10-09 update supersedes the running snapshot: 4773811 FAILED at
16:57:22Z before scoring any candidate. Transformers 5 chat templates returned
BatchEncoding instead of token IDs; the original judge-length check also
counted dict keys and is invalid. The repair explicitly requests/validates flat
IDs for both preprocessing/scoring. Attempt2 uses separate `_v2` roots and
unchanged experiment parameters. Retry approval was requested; do not submit
it without that approval or duplicate the original. No training is submitted.

At most six allocated GPUs total across all owner programs, not six per array.
Before resource actions inspect both running and pending work using the full
`squeue -a -r` owner/UID listing and `scontrol -a show job`. On Sycamore the owner
is yuhe32, UID448057; `squeue -u` has returned false empty results. Count generic
AllocTRES `gres/gpu` once. There is no installed account-wide hard cap.

The dated status in `experiments/cyclic_history/campaigns/README.md` is a
historical snapshot, not a current allocation guarantee. A/B, fixed-target
probes and Stage C have already been submitted; never duplicate their IDs or
restore old dependencies. Keep oracle tasks serial. Preserve results and
checkpoints. New submissions, reruns or cancellation need user authorization;
the two-week roadmap is a proposal, not such authorization.
