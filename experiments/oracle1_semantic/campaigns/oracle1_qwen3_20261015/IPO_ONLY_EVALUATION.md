# Retained IPO ordinary evaluation

Owner authorization October21,2026: finish evaluation and win rate for the
retained first arm, IPO ordinary/index0. Training is already complete100.
Finish original states0,10,...100, including90/100, without retraining or
restoring the other five arms. Shared Qwen3 baseline800 remains valid.
At09:01Z the retained arm had6649 responses:0..70 complete,80 at249/800;
2151 responses remain for its full8800-response checkpoint evaluation.
Final scoring covers9600 answers including the800 shared baseline.

## Independent immutable route

New root: `/work/users/y/u/yuhe32/ipo/diagnostics/oracle1_qwen3_ipo_only_20261021_v1`.
`retained.py` binds unchanged BASE deployment/identity, prior EXT binding,
both historical submission ledgers, reviewed exact71-job terminal disposition,
11 retained checkpoints and its own source inventory. The eight failures in
withdrawn arms remain FAILED1:0. Retained training/generation and shared baseline
prerequisites must be successful. Unknown/duplicate/changed accounting blocks
new work. This is a new owner-authorized scope, not a repair of old receipts.
No old source, checkpoint, plan, scientific setting or other-arm output is edited.

One non-array generation allocation uses1H100; afterward one sequential
Nemotron BF16 scoring allocation uses2H100. The whole owner/UID account must
be empty before submission. Startup permits only the exact registered current
allocation, with verified GPU request and allocation; no unrelated pending,
running or releasing work. Global ceiling remains FOUR, not an administrator cap.
New phases require successful prior receipts; TIMEOUT0:0 is not success.
Submission intent is written before sbatch; ambiguous outcome must not be retried.
24h allocation,18h soft boundary, signal USR1 at1h remaining, no requeue.
Stale writer locks require diagnosis, never automatic removal.

Generation uses the original full-plan signature, seeds, checkpoints and
canonical response journals, preserving every completed row. WR uses the same
canonical score journal and exclusive WR lock, but collects only baseline and
IPOordinary. Tables live in `wr/selection_<arms-and-steps-digest>/` with explicit
`selected_arms` scope, never full-six completion. Scoring formula is unchanged:
Nemotron T1 sigmoid score differences,4x4 comparisons, equal prompt averaging.
Uncertainty bands are prompt SE, not training-seed uncertainty. Cap256 remains.

## Operator sequence

Use frozen source and `/work/users/y/u/yuhe32/h100env312/bin/python`:

```text
python retained.py --root NEW --verify
python retained.py --root NEW --advance
python retained.py --root NEW --advance --submit
python retained.py --root NEW --plot --output NEW_RESULTS_DIRECTORY
```

Inspect all three ledgers plus full owner/UID queue and native-parent sacct
before resource operations. Dry-run first, submit once, then save receipt and
check actual response/score progress, errors and GPU allocation. Successful
soft-stop receipts allow another missing-only segment. Failed new receipts
require diagnosis and an immutable tested repair, not repeated submissions.
Plot only after complete generation, successful scoring receipt and valid
selected-arm tables. Download safe CSV/PNG/PDF and visually inspect before delivery.
Do not invoke BASE/old EXT all-six advancement. Same monitor, no new timer.
Five-trait annotation is separate;200-response pilot still needs human review.

## Verified submission and startup

Execution source `621b7d706c5c0f43f01250c0c9ca366157a43cd2` pushed and verified.
Local54 tests pass with7 GPU-stack-dependent skips; server54 tests pass without
skips (30.493s), including partial generation preserving existing rows, selected
cached WR, isolated completion and plot pixel checks. Shell syntax passes.
These are CPU tests, not a new full-model WR result.

Freeze completed09:21:06Z, binding SHA
`0154a2449ed39ace600f798542fae7cd888b5b52fe0593b96e6277de6212e4d6`.
Fresh dry-run passes09:21:18Z. Full account empty09:22:50Z, then sequence0001
**4878947** submitted, non-array1H100 retained generation,24h/18h boundary.
At09:24:12Z it is RUNNING, the sole owned job; allocated/potential1 GPU.
Worker scope/account/checkpoint/journal guard passed. Step80 advances249 to252
valid responses, zero empty, cap107; progress at250 reports8,365,950,976bytes
allocated peak. No OOM/Traceback/nonfinite. Canonical baseline800 and0..70 each800
remain valid.90/100 not generated yet. No score rows or WR completion yet.

Exact [binding](records/ipo_only_binding.json), [intent](records/ipo_only_0001_intent.json),
[receipt](records/ipo_only_0001_receipt.json), [startup audit](records/ipo_only_startup_20261021.json).
Same5h heartbeat now advances ONLY this frozen retained route after successful
receipts and empty account, through true WR and rendered figure delivery.
Private read-only `oracle1_qwen3_launch_20261015/inspect_ipo_only_readonly.py`
can inspect active work without removing writer locks; full --verify/--advance
bank checks intentionally refuse an active generation writer. Never duplicate4878947.
