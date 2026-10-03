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
