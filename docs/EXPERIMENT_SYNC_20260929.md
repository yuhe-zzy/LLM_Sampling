# September 29 experiment synchronization

Integrated collaborator commit **dc9f090** by fast-forward before adding this
release. The two-week roadmap, its HTML version and corrections to the
IPO-only pilot claim are preserved unchanged. This publication does not run
the proposed larger roadmap or turn theoretical predictions into observations.

## Added to GitHub

- The previously deployed fixed-target runner and its five CPU tests.
- Stage A, fixed-target, B100 and C100 dated launch scripts, configs,
  deployment hashes and submission receipts (without credentials/logs).
- Portable B100/C100 plans and a non-executing plan exporter, with tests
  matching their contracts to the executed cluster configurations.
- Full local-data audits and plots for A/B/probes, plus public numeric CSVs,
  summary JSON, all-prompt PNG/PDF figures and six pi-versus-iteration panels.
- A CSV-only pi plotting path that needs neither raw private text nor models,
  with tests for complete trajectories, duplicates, finite values and softmax.
- Updated current status, six-GPU budgeting, reproducibility instructions and
  the standing requirement to commit/push future experiment changes.

No production IPO/DPO loss, sampler, reference, model or active training
source was changed. The core of the executed campaigns remains **f0fe034**.
The repository HEAD after publication is not retroactively their training
revision. Stage C submission alone is not a completion or result claim.

## Verification

- Python compilation passed for scripts/experiments/tests.
- Cyclic suite: all **59 tests passed**, without skips, on a separate Linux
  CPU validation copy with Torch. No GPU job was submitted for validation.
- Main experiment suite: all **30 tests passed**, without skips, on that
  validation copy. Earlier local-only checks skipped unavailable ML packages;
  the full Linux checks supersede those partial results.
- Merged Hodge diagnostics: all **23 tests passed** locally with isolated
  pytest dependencies. The server environment lacked pytest and was not
  modified; these tests were therefore checked locally instead.
- All archived and current shell scripts passed Linux `bash -n` checks.
- Cross-platform validation caught a provenance-only line-ending mismatch.
  Portable plans now hash canonical JSON, with a regression test for CRLF/LF
  independence. Original deployment byte hashes and training sources remain
  unchanged.
- Private raw-data audits reran successfully for A (124 snapshots), B (606
  snapshots with 648 raw/log file hashes) and four six-block fixed-target probes.
  Recomputed metrics matched saved values without modifying original downloads.
- CSV-only rendering reproduced all six original pi PNGs **byte for byte**.
  Probability completeness (14,544 points), finite values, unit sums, initial
  equality and sequence-sum softmax reconstruction passed.

The checked-in CI also runs full CPU Torch tests, Hodge tests and shell syntax.
Do not describe an unobserved CI run as passing. Raw inputs and generated QA
files remain outside Git; only sanitized numeric reports and figures are public.

## Publication boundary and future workflow

No passwords, API keys, private keys, raw prompt/response text, adapters,
weights, raw NPZs, queue dumps or Slurm logs are included. Dated job IDs and
cluster paths are non-secret provenance and explicitly labeled as historical.
SSH passwords are supplied only through the caller's environment.

For later experiment changes: fetch, reconcile collaborator edits, validate,
update dated configs/status/provenance, review the staged file list, commit,
push to the shared branch and verify its remote hash. Never force-push.
Report network/authentication blockers rather than claiming synchronization.
Keep already-running deployments immutable. Git synchronization does not
authorize new jobs, restarts, cancellation or extra GPU allocation.
