# Results publication reconciliation: 2026-10-13

The owner requested agent-native project context and publication of shareable
results. The repository is the canonical context root; the private outer
workspace is not uploaded wholesale. No GPU job was changed or started.

## Coverage

Existing published PNG/PDF figures and numeric summaries were matched against
local results. Existing batch paths are retained rather than duplicated.

| Batch | Published coverage |
| --- | --- |
| Original cyclic parameter runs | Final nine-run legacy analysis already present; superseded draft layouts not re-published |
| Historical sampling sweep | Newly published10 runs,12 PNG/PDF files, all500-prompt diagnostics and text-free curve data; DPO invalid tail explicitly marked |
| Calibrated Stage A / probes / B / C | Existing figures/reports retained; original compact numeric per-step metrics added |
| Reference90 and trend sweep | Existing figures/reports/all-prompt summaries retained; numeric metrics added; reused Stage B is not a new replicate |
| Oracle1 | Existing September11 consolidated snapshot retained; older local numeric/figure exports separately archived with protocol/freshness warnings |
| Oracle2 | Existing six-arm policy PNG/PDF and all500-prompt numeric evidence retained and linked prominently; generated WR still absent |
| Anchor90 | Launch/configs and historical completion report available; no dedicated validated trajectory export in this local publication |
| Hodge collaborator results | Existing collaborator files remain in place; proposals stay distinct from executed experiments |

New `metrics_original/<run>.csv` files are small numeric intermediates copied
from local exports with values unchanged, not raw text dumps. Git normalizes
CSV line endings to LF. The manifest distinguishes source-byte SHA256 from
published LF-byte SHA256; figure bytes are unchanged. `step` and `block`
coordinates are checked for contiguous ordering. Fixed-target probe block is
not a new outer update. The [import manifest](result_import_manifest_20261013.json)
records source-relative paths, row coverage and SHA256, and the
[public artifact inventory](result_artifacts_20261013.json) lists figure/table
paths and hashes across experiment packages. Neither manifest contains private
text or guarantees the completeness of server files not retrieved here.
The inventory respects the earlier Oracle1 package's explicit `-text`
attributes, preserving its original CRLF byte hashes instead of rewriting it.

## Reproduction and limits

Run the existing campaign analysis scripts for their documented private inputs.
Public numeric tables and figure pairs are retained for collaborators without
those inputs. The publication importer is an explicit allowlist, not a recursive
workspace uploader; it validates historical sampling statistics against the
original hash-bound local snapshot. New CSVs reject free-text/private columns.
Do not upload `selected_prompts.json`, raw JSONL/NPZ/ZIP archives, models,
adapters, logs, authentication material or caches.

Outstanding data are named, not filled in: Oracle2 WR awaits generation and
recovery followed by scoring; Oracle1's final missing/partial grid needs a
separate provenance reconciliation; Anchor90's results still need dedicated
retrieval/validation; no new non-oracle transitive result package was identified.
This publication is not a claim that every experiment ever attempted is valid,
that every server artifact is public, or that old token-average results satisfy
the current sequence-sum protocol.
