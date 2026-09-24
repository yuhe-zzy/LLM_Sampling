# Sequence-sum-only release, September 2026

## Scope and provenance

The first server-source release was commit `ee21b60`, based on `83b3157`.
The follow-up removes all token-average training/scoring paths from the current
checkout at the user's request. Git history and existing experiment results
remain intact. It does not rewrite past results or change running jobs.

`SOURCE_MANIFEST.json` retains original imported-source hashes and records
current published hashes. The shared helper module derives from the imported
IPO source, with the changes below; it is not an untouched server snapshot.

## Current entry points

| Entry | Role |
|---|---|
| scripts/run_ipo.py, scripts/run_dpo.py | Non-oracle transitive or cyclic |
| scripts/run_ipo_oracle.py, scripts/run_dpo_oracle.py | Scalar-oracle transitive |
| scripts/run_preference_oracle_core.py | Shared frozen-reference sequence-sum core |
| scripts/sequence_utils.py | Sequence-only scores, static sampler and panel construction |
| experiments/cyclic_history/ | Sequence-sum ordinary/reference/sampling history controls |

The old `scripts/legacy/` trainers and launcher protocol are removed.
`configs/cyclic_legacy.json` is replaced by `configs/cyclic_sampling_sweep.json`.
It keeps the ten parameter combinations but now runs the corrected core,
with 150 updates plus a final evaluation. It is a new experiment, not a
completed-result claim or an exact reproduction of historical runs.

## Intentional changes from ee21b60

1. All likelihood scoring returns response-token sums and token counts only.
   Counts are diagnostics, never divisors of training or sampling scores.
2. Static pair targets use chosen/rejected **sequence-sum** margins, replacing
   the average-margin sampler. The proposal/weighting and clipping law is unchanged.
3. Generated initial candidates are ranked by sequence sums. This can change
   diagnostic panels; archived panels and curves must retain their provenance.
4. Raw panel probabilities use sequence sums. Their entropy/TV/probability
   columns are explicitly named `sequence_*` / `prompt_sequence_*` instead of
   reusing historical average-score column names.
5. The primary relative-sequence entropy is unchanged in definition:
   `softmax(tau*(log_pi_t-log_pi_0))`. Old average entropy is never substituted.
6. New core metrics label the protocol `sequence_sum_only_v3`; config output
   names end in `_v3`, so old and new runs cannot silently overwrite each other.
7. Offline CSV reconstruction prefers sequence scores and can import archived
   average-only CSVs by multiplying by exact response counts. That compatibility
   adapter never feeds training and produces only sequence-based metrics.

Reference caching, IPO/DPO sequence-sum loss formulas, optimizer precision,
learning rate, accumulation normalization and oracle labeling have not been
changed in this follow-up. The history pilot was already sequence-sum throughout
and remains a separate matched-control protocol.

## Existing safeguards

Non-oracle wrappers reject oracle flags, and scalar-oracle training rejects
cyclic labels. Nonfinite scores/loss/gradients fail loudly. The recipe launcher
is dry-run by default, requires an approved allocation to execute, and refuses
existing output directories. Per-run baseline caches avoid concurrent writers.

Only the GitHub source checkout is updated. No production server source,
checkpoint, running allocation, queue dependency or concurrency limit is changed.
CPU validation is not evidence of long-run convergence or absence of collapse.

## Reproducibility limits

Do not combine old average-derived sampling/panel curves with the new protocol
as if they came from matched runs. Retain original code/config and support hashes
with archived results. Adapter-only loading does not restore RNG/support/optimizer
history and is not a lossless continuation. The package snapshot describes the
validation environment, not necessarily every historical training job.
