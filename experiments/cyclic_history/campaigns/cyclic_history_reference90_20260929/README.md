# Empirical 90%-previous reference, September 29, 2026

User requested `r_t = 0*s_0 + 0.1*s_t + 0.9*s_{t-1}`. In the implemented
score convention this is **alpha=1, nu=.9, kappa=0**. The scores are cached
response-token sequence-sum log probabilities including EOS, not model
weights or an arithmetic mixture of probabilities. `previous=current=initial`
at t=0. Thus the first reference is the initial score vector. Training starts
from the same initial model, not a completed Stage B adapter.

## Exactly two arms

| Task | Method | beta_train | alpha | lambda_current | nu | seed | Outer states |
|---|---|---|---|---|---|---|---|
| 0 | IPO | .2 | 1 | .8 | .9 | 0 | 0..100 |
| 1 | DPO | .8 | 1 | .8 | .9 | 0 | 0..100 |

Same six aligned calibrated panels (54,251,612,737,867,945), K=4,
Qwen2.5-1.5B, full unordered pairs, 10 inner epochs/outer update, LR=1e-5,
LoRA and tokenizer settings as Stage B. One H100/task, at most two new GPUs.
Full account running/pending work must be checked against the six-GPU ceiling.
No extra baseline, seed, sweep, Stage B/C rerun or oracle job is authorized.

## Interpretation and validation

Stage B reference used `.1*s_0 + .45*s_t + .45*s_{t-1}`. Both alpha and nu
change here: this is not a nu-only causal contrast. No claim of increased
stability is built into the protocol. Full-refresh can behave differently,
including more oscillation. Report all prompts and actual observed outcomes.

The separate `cyclic_history_empirical_full_refresh_v1` protocol retains
the support/parameter hash, tokenization/EOS and fresh initial-score checks,
sequence-sum losses, cuDNN SDPA exclusion and nonfinite fail-fast behavior.
It intentionally does not invoke the alpha<1 population fixed-point solver
or demand a predicted stable radius. Old calibrated-v2 runs still enforce
their original alpha<1 and population calibration gates. Do not reuse their
fixed points or cyclic eigenvectors for this experiment.

Record raw panel pi versus outer step for every prompt, relative-sequence
entropy separately (`softmax(s_t-s_0)`), adjacent TV, losses and target-fit
residuals. No winning rate is measured. Existing B ordinary/reference/feedback
provide contextual comparisons with the alpha-change caveat. An alpha=1
ordinary baseline would be a separate experiment requiring authorization.

## Provenance and launch

`build_plan.py` derives the two configs from the immutable Stage B plan and
removes obsolete predictions. `queue_reference90.py --prepare` validates the
exact differences, server data, CPU regression tests and frozen source hashes.
Source commit/archive hash are recorded in deployment and submission receipts.
The submit path requires an empty full owner/UID queue and new output directory,
writes an exclusive intent before sbatch and refuses duplicate submission.

Remote launch: `/work/users/y/u/yuhe32/ipo/diagnostics/history_reference90_20260929`.
Remote results: `/work/users/y/u/yuhe32/ipo_runs/cyclic_history_reference90_20260929`.
No model weights, adapters, private prompt/response text or credentials belong
in the public campaign record. Older deployments remain untouched.
