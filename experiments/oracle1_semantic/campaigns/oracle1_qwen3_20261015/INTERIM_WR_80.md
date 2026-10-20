# Interim Oracle1 win-rate curves, states0..80

Owner authorization October19, 2026: evaluate saved checkpoints at0,10,...80
for all six arms. TWO additional H100s may operate alongside the two training
GPUs; the entire account must never exceed FOUR. This supersedes the earlier
one-extra-card limit and the blanket nonoverlap rule for this specific sidecar.
It does not authorize a different judge, precision, scientific setting, extra
seed, oldOracle2 work or cancellation of other jobs.

## Immutable extension

Planned extension root:
`/work/users/y/u/yuhe32/ipo/diagnostics/oracle1_qwen3_interim80_20261020_v1`.
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

Implementation prepared locally; server tests/deployment/submission pending.
At2026-10-20T00:02:38Z the original six steps were86/84/80/82/80/80,
0/1 RUNNING,2..5 PENDING(JobArrayTaskLimit), full account2H100, no logged errors.
This is a dated snapshot, not permission to submit while legacy startup remains.
