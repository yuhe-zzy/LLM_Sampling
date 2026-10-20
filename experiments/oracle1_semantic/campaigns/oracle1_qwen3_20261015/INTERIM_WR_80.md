# Interim Oracle1 win-rate curves, states0..80

Owner authorization October19, 2026: evaluate saved checkpoints at0,10,...80
for all six arms. TWO additional H100s may operate alongside the two training
GPUs; the entire account must never exceed FOUR. This supersedes the earlier
one-extra-card limit and the blanket nonoverlap rule for this specific sidecar.
It does not authorize a different judge, precision, scientific setting, extra
seed, oldOracle2 work or cancellation of other jobs.

## Immutable extension

Frozen extension root:
`/work/users/y/u/yuhe32/ipo/diagnostics/oracle1_qwen3_interim80_20261020_v2`.
Execution source `caf5ac2b68d0e4d3778260428eedbeb3441a6658`, pushed and verified.
The existing base root remains `oracle1_qwen3_20261015_v1`, frozen computational
source0e7793f5fdd6afdf2fcf80c4ff40649e8926f5a2. `interim.py --freeze`
binds the base deployment, identity, original submission ledger, all54 required
checkpoint commits and the new source files. `verify` checks the actual adapter
files too. No old source/plan/receipt/checkpoint is overwritten or reidentified.
Subsequent training workers exec the ORIGINAL `train.py`, with its original
resume/identity/RNG checks and at most10 updates, not a copied modified trainer.

Two reserved lanes, each at most2 GPUs: training has at most two1GPU workers;
evaluation has a1GPU baseline, up to two1GPU generators, then one2GPU BF16
Nemotron job. Evaluation phases are sequential and require successful preceding
receipts. Both lanes share an exclusive submission lock. Pending/releasing jobs,
array throttles, requested GPUs, full owner/UID queue, generic AllocTRES and sacct
are checked. Unregistered jobs, unknown or failed outcomes, unpaired intents,
changed artifacts or stale locks block action. This is not an administrator cap
against independent outside submissions.

The already submitted legacy array4859197 has a startup guard rejecting all
other jobs. Therefore NO sidecar can even be queued until all its pending tasks
have started, passed reload and written this allocation's manifest, or finished
successfully. This is a one-time compatibility wait, not a need for step100.
Do not cancel/requeue/hold tasks, spoof a receipt or patch frozen source to avoid
the wait. Later training segments use the extension's cooperative launcher and
can coexist with the evaluator under the4GPU ceiling. Do not use original
`queue_campaign.py --advance` while extension lanes remain active.

## Measurement and reuse

Generate800 new initial-Qwen3 baseline responses and54x800 checkpoint responses:
44,000 total. Each bank retains the original full-plan input signature, response
identity, per-response seed, original checkpoint hash and deterministic journal.
Canonical generation directories are reused; partial results are never replaced.
Subset generation writes a `SUBSET_<digest>.json`, NOT full-arm `COMPLETE.json`.
Future90/100 generation can consume these same verified banks without rerunning
them. The initial baseline is not oldOracle2's1.5B baseline.

Score only after ALL requested banks validate. Same Nemotron BF16, temperature1
BT probabilities, all4x4 response comparisons then equal prompt averaging.
Canonical `wr/scores` journals are reused under the same exclusive WR lock.
Subset tables/completion are isolated in `wr/subset_<steps-digest>/`; they never
claim full0..100 WR completion. The later full scorer reuses these exact scores.
No annotation is needed for the WR plot. Generate/score retain18h soft boundaries,
24h allocations and missing-response-only continuation after successful segments.

Plot actual validated54-point summary in IPO/DPO panels, three arms each.
Bands are1.96 prompt SE, not uncertainty across training seeds. No fixed-panel
probability proxy, invented zero, cyclic inference or annotation result. Inspect
rendered PNG/PDF before publishing actual results; test fixtures are not results.

## Operator commands

Use the frozen server Python and extension source. For a known deployed `EXT`:

```text
python interim.py --root EXT --verify
python interim.py --root EXT --inspect
python interim.py --root EXT --advance --lane evaluation
python interim.py --root EXT --advance --lane evaluation --submit
python interim.py --root EXT --advance --lane training
python interim.py --root EXT --advance --lane training --submit
python interim.py --root EXT --plot --output NEW_RESULTS_DIRECTORY
```

Inspect both ledgers first, dry-run then submit exactly once. Immediately check
scheduler, receipt and real startup/progress. Initial new-stage errors need prompt
attention; use the SAME monitor, not a second timer. When legacy guards block,
prepare and wait; do not report submission or inference. Once interim results are
delivered, training reaches100 and BOTH extension lanes have successful terminal
receipts with an empty full account, record handback to the original full pipeline.
Only then may original `queue_campaign.py --advance` generate remaining90/100,
finish full WR and proceed to the human-gated five-trait pilot. Preserve extension
receipts; future original-ledger changes intentionally invalidate extension
`verify`, so publish the interim analysis before handback.

## Current status

AtOctober20T06:44:36Z baseline **4861201** is COMPLETED0:0 with800/800 responses.
Canonical complete-bank hash, full-plan identity, per-response seed/input/row
hashes and bank/journal equality pass. Zero empty responses,398 cap hits
(49.75%), mean196.7925tokens, mean unique ratio0.5932692, max dominant fraction1/3.
Allocated peak8,427,731,968bytes (7.85GiB). These are diagnostics, not quality/WR.
Dry-run passed06:41:03Z; guarded submission created **4861879_0..5%2** at06:41:42Z,
EXTsequence0002, oneH100/task,24h/18h soft stop. Tasks0/1 started06:41:43Z.
Each has17 valid step0 answers, zero empty, four cap hits, no logged errors.
Memory progress is not emitted until50 rows, so no generator peak is claimed.
Tasks2..5 normalPENDING(JobArrayTaskLimit). Full account4H100 and potential4,
two trainers plus two generators, no unrelated/releasing work; full/native-parent
accounting checks pass. Training06:43:56Z:92/90/90/92/84/83,537 valid states.
SAME timer restored5hours after baseline AND first generator actual-response
validation. No WR submitted or complete. Both lanes active; do not duplicate or
call BASE advance. [Audit](records/interim80_generation_startup_20261020T0644.json),
[intent](records/interim_v2_0002_intent.json), [receipt](records/interim_v2_0002_receipt.json).

The following dated baseline startup and preparation snapshots are historical.

Baseline **4861201** submittedOctober20T05:28:21Z, EXTsequence0001, after dry-run
and fresh guarded submit. OneH100,24h allocation/18h soft stop, no duplicate.
At05:33:52Z RUNNING:62/800 actual responses have valid full-plan identity, input
digests, seeds and row hashes; finite diagnostics, zero empty responses,20 cap
hits. Mean length180.21 tokens; mean unique-token ratio0.6226, maximum dominant
token fraction0.2857. These are partial-prefix diagnostics, not a quality verdict.
Progress at50 rows reports8,320,785,408 bytes allocated peak (7.75GiB), not device-
wide reserved/lifetime telemetry. No logged OOM/Traceback/nonfinite error.
Full-account05:33:51Z:3H100, two trainers plus baseline, reserved potential4;
no unrelated/pending/releasing work. [Startup audit](records/interim80_baseline_startup_20261020T0533.json),
exact [intent](records/interim_v2_0001_intent.json) and [receipt](records/interim_v2_0001_receipt.json).
No checkpoint generator or WR submitted. Wait for successful baseline receipt
and complete bank validation before evaluation advance. Keep the SAME5min timer
until baseline AND first checkpoint generator write healthy real responses.

The05:17 [accounting overlap](records/accounting_overlap_20261020T0517.json) cleared after
the final legacy task started05:23:18Z. Fresh raw parent accounting and unchanged
v2 inspector passed05:26:56Z. No row filtering, code/guard or scheduler changes.
Both active legacy jobs verified reload80 and matching RUNNING manifests before
the dry-run passed05:27:41Z. Training05:30:41Z:530 valid states,92/90/90/92/80/80,
0..3 segment-complete and4/5 running, no errors. All54 checkpoint bindings valid.
Historical preparation follows; it does not describe current submission state.

Prepared/deployed at2026-10-20T00:15:51Z, not yet submitted AT THAT TIME. Binding hash
`e56f796f20a6e648a298947ba97bf00a6bdb0f36a7982105bbaf35c0962e0b74`.
All43 server CPU tests passed without skips, shell syntax passed, including real
tiny Qwen3/LoRA resume, cached generation/scoring reuse and synthetic plot render/
pixel checks. This is not full-model sidecar GPU inference or an actual WR plot.
Dry-run verified hashes,54 checkpoints, full-account budget and expanded sacct;
then correctly blocked on the legacy pending-worker startup restriction, before
any intent/sbatch/receipt. Same monitor temporarily5minutes until actual baseline
and first checkpoint generation write valid responses without errors, then5hours.

Preserved v1 (source4cc1506) was never submitted. Real dry-run exposed Slurm's
lazy pending-array accounting: explicit child filters omitted tasks2..5. Adding
`--array` alone still omits lazy tasks with child filters. The tested repair
queries parent array IDs with `sacct -X --array`, validates each expanded child
against receipts, and rejects unregistered children. All58 original receipt
tasks are now visible:52 successful, twoRUNNING and fourPENDING at validation.
No missing status is imputed as success; no GPU retry or scientific change.
Both validation versions and the unused v1 launch remain intact.

At2026-10-20T00:18:18Z the original six steps were86/84/80/82/80/80,
0/1 RUNNING,2..5 PENDING(JobArrayTaskLimit), full account2H100, no logged errors.
All498 committed states valid. No intent/receipt or stale lock in either extension;
generation journals empty, WR absent. See [prepared status](records/interim80_prepared_20261020.json)
and [exact binding](records/interim80_v2_extension.json). This is a dated snapshot,
not permission to submit while legacy startup remains.
