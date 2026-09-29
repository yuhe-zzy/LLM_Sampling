# Stage C: remaining mixed-orientation controls, states 0..100

Completion verified September 29 at 13:44 UTC: both tasks COMPLETED, exit 0:0,
all states 0..100. The full account queue was empty, zero allocated GPUs.
See [results and plots](../cyclic_history_stage_c100_results_20260929/REPORT.md).
All startup/launch snapshots below are historical. Do not resubmit these runs.

Submitted as **4606367**, tasks **0-1%2**, at
2026-09-29T05:56:05.170713+00:00. The preflight full account queue was empty.
The immediate scheduler snapshot had both tasks pending, zero allocated GPUs;
this is not a continuing state guarantee. Consult `submission_receipt.json`
and the latest startup inspection. Do not resubmit these two runs.

Startup check, 2026-09-29T05:57:10Z: IPO task 0 is RUNNING on one H100,
passed fresh initial-score calibration and wrote a finite step-0 snapshot.
DPO task 1 is PENDING(Resources), a normal queue state. The complete owned
queue contains only these two tasks; total allocated GPUs = 1. No execution
errors were logged. See `startup_progress.txt`; training completion is not
claimed by this startup check.

The user authorized the remaining planned experiments on September 29, 2026,
after the six Stage B runs finished. The original ten-arm calibrated plan has
only two unrun arms: IPO/DPO mixed orientation. Stage A and B are not repeated.
These two arms use 100 outer updates to match the completed Stage B ordinary
controls, rather than the original plan's 30-round pilot horizon.

| Task | Run | Alpha | Lambda current | Beta train | Seed | Matched completed control |
|---|---|---:|---:|---:|---:|---|
| 0 | ipo_mixed_c100_s0 | .9 | .8 | .2 | 0 | 4605560_0, IPO ordinary |
| 1 | dpo_mixed_c100_s0 | .9 | .8 | .8 | 0 | 4605560_3, DPO ordinary |

Both use ordinary updates (nu=0, kappa=0). Prompt IDs in order are
54, 251, 612, 737, 867, 945. The fixed orientation vector is
[-1, +1, -1, +1, -1, +1], exactly the September 28 prepared Stage C assignment:
reverse the cycle on prompts 54/612/867; keep 251/737/945 unchanged.
No selection based on Stage B outcomes was performed.

The same four responses, prompt texts, initial sequence scores, response roles,
soft cyclic strength (.8/.2 adjacent, .5 opposite), seed and inner training
budget are retained. Only the preference orientation differs from the matched
ordinary control. The transformed support hash necessarily differs because it
includes P. The role-aligned and mixed arms are an orientation-sensitivity test,
not a direct measurement of LoRA feature or gradient coherence.

## Training and interpretation

Initial Qwen2.5-1.5B, not a trained Stage B adapter. States 0 through 100
inclusive; 101 expected metric rows and snapshots. Response sequence sums
including EOS, no token averages. Raw sampler is
mu=.2*uniform+.8*softmax(sequence_sum). All six unordered pairs per prompt,
10 inner epochs and 90 optimizer updates per outer round (9,000 total/run).
AdamW resets each round, LR1e-5, warmup .03, LoRA r16/alpha32/dropout0, BF16
with cuDNN SDPA excluded; fail-fast finite checks remain enabled. Adapter
checkpoints are saved every five rounds and at initialization.

Compare all six prompts' pi-versus-iteration curves, late-round variability,
relative-sequence entropy and operator error with existing ordinary results.
Do not assume mixed orientations must suppress cycling, identify these selected
panels with a representative population, or require exact neural fixed-point
fitting. Each orientation has its own population fixed point/mode basis; do
not silently project mixed results using an incompatible aligned basis.
No winning-rate, open-generation, extra seed, new parameter sweep, or further
fixed-target fit probe is included.

## Source and validation

Frozen training code is the same archive as Stage B at Git commit
f0fe034cc0d8e3bef35b0f5e02806ee81340da76, SHA256
4d09a8748c3b4524fe5e14a1f4786474073573d485d49bfc56af18f1b0c757fb.
No training source was edited. `build_plan.py` asserts that only IDs, paths,
iteration horizon and corresponding contract hashes change from the original
Stage C plan. It also asserts the precise single-variable contrast against the
completed B100 ordinary configurations. Remote preparation verifies raw support,
transformed support, calibration predictions, contract hashes and shell syntax.
Fresh GPU initial-score and EOS checks must pass before training starts.
Both configurations passed remote preparation, and all 46 frozen-source CPU
tests passed (19.700 seconds) before submission.

Remote launch root:
`/work/users/y/u/yuhe32/ipo/diagnostics/history_stage_c100_20260929`

Remote result root:
`/work/users/y/u/yuhe32/ipo_runs/cyclic_history_stage_c100_20260929`

Slurm logs: `ipo_runs/slurm_logs/hist_C100_0929-<array>_<task>.out/.err`.

## Resource guard

One H100 per task, array 0-1%2, two GPUs maximum for this array, six-hour time
limit, no automatic requeue. The account-wide budget remains six allocated
GPUs across all programs. No extra jobs are added to fill unused capacity.
The submission wrapper refuses to submit if the full owner/UID queue has any
other running, pending or releasing job. It checks `squeue -a -r`, ownership
UID448057 and `scontrol -a show job` generic AllocTRES gres/gpu. There is no
installed account-wide hard cap; independent later submissions need accounting.

Receipts: deployment.json, cpu_predictions.json, submission_intent.json,
submission_receipt.json and queue_verified.json. `inspect_stage_c100.py` is
read-only. Previous sources, results and checkpoint directories are untouched.
Estimated runtime after GPU allocation is about 2.3-2.5 hours per run based on
the same-size completed Stage B arms; queue wait is additional.

This launch does not install a timer or create recurring monitoring.
