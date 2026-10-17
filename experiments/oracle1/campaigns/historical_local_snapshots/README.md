# Historical local Oracle1 snapshots

Publication2026-10-13. These are preserved local intermediate exports, NOT a
new server retrieval or a final experiment grid. Directory date identifies the
original export family; later local overwrites are possible, so file timestamps
and names such as `current` are NOT evidence of freshness. Exact published bytes
are recorded in the [import manifest](../../../../docs/result_import_manifest_20261013.json).

For the preferred consolidated result use the
[September11 snapshot](../sequencesum_snapshot_20260911/README.md).

| Archive | Scope / caution |
| --- | --- |
| [DPO August17](oracle_dpo_sequencesum_results_20260817/) | Two partial parameter metrics and entropy/WR previews; not the complete grid |
| [IPO frozen-reference August17](oracle_ipo_frozenref_results_20260817/) | Five local parameter CSVs and previews; older protocol/provenance, not automatically corrected v2 |
| [Sequence-sum August25](oracle_sequencesum_results_20260825/) | Intermediate v2 metric CSVs and entropy/WR previews; entropy column must be checked, not inferred from directory name |
| [Relative-sequence August26](oracle_relative_sequence_results_20260826/) | Reconstructed relative entropy CSVs and previews; may include partial/retried runs absent in the September11 consolidated snapshot |

`prompt_relative_sequence_entropy_mean` is the relative-sequence measure.
`prompt_entropy_mean` is length-normalized legacy entropy in these historical
exports and must not be relabeled or merged with it. Preserve missing WR values.
No inference of final convergence, all16 completion, retry continuity, or current
training-code provenance is made here. This archive is for traceability and
comparison of old exports, not a source of additional independent replicates.

Only existing numeric/enum CSVs and PNG/PDF figures are published. No original
response banks, weights, adapters, credentials or raw scheduler logs are included.
