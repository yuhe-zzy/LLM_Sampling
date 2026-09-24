# LLM Sampling: IPO and DPO Experiments

Neural experiments for sampling/reference feedback in iterative preference
optimization. This release separates **scalar-oracle transitive training**,
**fixed-data non-oracle training**, **historical cyclic reproductions**, and
the **new cyclic history pilot**.

The code was synchronized with the experiment server on 24 September 2026.
See [migration and provenance](docs/MIGRATION.md) before comparing with the
older version of this repository. Model weights, data, adapters, result dumps,
credentials, and machine-specific job IDs are deliberately not included.

## 1. Which experiment should I run?

| Setting | Entry points / recipe | Important distinction |
|---|---|---|
| Transitive, online scalar oracle | `scripts/run_ipo_oracle.py`, `scripts/run_dpo_oracle.py`; `configs/oracle.json` | Generate new comparisons; one frozen Nemotron model labels them and evaluates WR |
| Transitive, non-oracle | `scripts/run_ipo.py`, `scripts/run_dpo.py`; `configs/nonoracle_transitive.json` | Fixed HelpSteer scalar-score labels, no reward-model loading |
| Cyclic, standard sequence-sum | Same non-oracle entry points with `--preference_case cyclic`; `configs/cyclic_sequence_sum.json` | Fixed tournament labels; ordinary cached-reference examples |
| Cyclic, historical sampling ablation | `scripts/legacy/run_ipo.py`, `scripts/legacy/run_dpo.py`; `configs/cyclic_legacy.json` | Exact historical runner sources, including legacy IPO averaging and known invalid DPO tail |
| Cyclic, new history experiments | `experiments/cyclic_history/` | Matched ordinary / lagged-reference / feedback-extrapolation arms; prepared, not run |

**Standard entry points use sequence-sum training for both IPO and DPO and
cache the reference for the whole outer round.** The non-oracle standard
entry points newly expose the fixed-data branch of that corrected core.
They are not a claim that historical non-oracle results were produced with
this standardized protocol. The original non-oracle scripts remain under
`scripts/legacy/` for that purpose.

The additional ordinary cyclic example is not a matched control for the
history pilot: the history pilot has its own shared ordinary controls and
pair law. Do not substitute legacy results for either new control.

### Terminology that must not be mixed

| Quantity | Meaning |
|---|---|
| `alpha` | Current-policy weight in the refreshed geometric log-score reference |
| `beta` / `beta_train` | Training-loss coefficient, not automatically the paper's payoff gain |
| Oracle `lambda_on` | Probability of choosing the **initial** response generator |
| Static/legacy `lambda_on` | Weight on the model-induced **pair-margin distribution**, versus uniform pairs |
| History `lambda_current` | Weight on the current **response-panel distribution**, versus uniform responses |
| `prompt_relative_sequence_entropy_mean` | Entropy of normalized sequence likelihood ratios to the initial model |
| `prompt_entropy_mean` | Legacy length-normalized panel entropy; not the primary oracle metric |

Full formulas and interpretation limits are in [PROTOCOLS.md](docs/PROTOCOLS.md).

## 2. Installation

Use Linux, Python 3.12, and a GPU-capable PyTorch installation appropriate for
the cluster's CUDA driver. The experiment pipeline uses Transformers, PEFT,
Accelerate, Datasets, NumPy, Pandas, and Matplotlib.

```bash
git clone https://github.com/yuhe-zzy/LLM_Sampling.git
cd LLM_Sampling
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install the cluster-approved CUDA PyTorch build first.
python -m pip install -r requirements.txt
```

Use [requirements-server.txt](requirements-server.txt) for the recorded
server package versions. It is a provenance snapshot of relevant packages,
not a portable CUDA driver/container lock. Full GPU training has not been
rerun merely to publish this release. CPU tests use tiny randomly initialized
models, not pretrained weights.

Commands below are run from the **repository root**. Avoid concurrent package
upgrades in an environment used by running experiments.

## 3. Data and model preparation

### Download the original HelpSteer release

This is [NVIDIA HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer),
not HelpSteer2. Respect upstream data/model terms and access requirements.

```bash
python scripts/download.py --dataset nvidia/HelpSteer \
  --cache_dir data/hf_cache --save_dir data/raw/HelpSteer
python scripts/export_helpsteer_jsonl.py \
  --input_dir data/raw/HelpSteer --split train --output data/raw/helpsteer.jsonl
```

### Build transitive fixed pairs

Use the scalar average of helpfulness, correctness, coherence, complexity,
and verbosity. Each pair is oriented from larger scalar score to smaller;
exact ties are omitted. `--score_weights` can define a different scalar,
but that is a different protocol and must be recorded.

```bash
python scripts/build_pairs.py \
  --input data/raw/helpsteer.jsonl \
  --out_pairs data/processed/pairs_train.jsonl \
  --out_eval_prompts data/processed/eval_prompt_responses_1000.jsonl \
  --input_format response --prompt_key prompt --response_key response \
  --score_fields helpfulness,correctness,coherence,complexity,verbosity \
  --dedup_responses --keep_exact_k 4 --pair_mode all --eval_prompts 1000 --seed 0
```

### Build cyclic fixed pairs

The same text pool is relabeled with a complete four-response tournament:
`0 > 1 > 2 > 3 > 0`, plus `0 > 2` and `1 > 3`. This is a prescribed synthetic
cycle, not evidence that the original human ratings are cyclic.

```bash
python scripts/build_cyclic_pairs.py \
  --input data/raw/helpsteer.jsonl \
  --out_pairs data/processed/pairs_train_cyclic.jsonl \
  --out_eval_prompts data/processed/eval_prompt_responses_cyclic_1000.jsonl \
  --input_format response --prompt_key prompt --response_key response \
  --score_fields helpfulness,correctness,coherence,complexity,verbosity \
  --dedup_responses --keep_exact_k 4 --comparison_mode full_tournament \
  --order_policy score_desc --eval_prompts 1000 --seed 0
```

Pair rows contain `prompt`, `chosen`, `rejected` and metadata. Panel rows
contain `prompt_id`, `prompt`, and candidate responses (with the cyclic
preference matrix where applicable). Construction does **not** guarantee
that diagnostic prompts are disjoint from training prompts; these panels
are not advertised as independent held-out generalization sets.

For exact historical reproduction, retain the original processed files,
ordering, tokenizer, and support hashes. Rebuilding from a subsequently
changed upstream dataset is not guaranteed bitwise identical.

### Download models

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-1.5B', local_dir='model/Qwen2.5-1.5B')"
# Required only for scalar-oracle runs:
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/Llama-3.1-Nemotron-70B-Reward-HF', local_dir='model/Llama-3.1-Nemotron-70B-Reward-HF')"
```

Authenticate through the usual Hugging Face credential mechanism if required.
Never put tokens or passwords in scripts, configs, commits, or launch commands
saved to version control. Model revisions should be recorded in the final
experiment manifest; model identifiers alone are not immutable revisions.

Optional token-length audit:

```bash
python scripts/analyze_lengths.py --model_path model/Qwen2.5-1.5B \
  --input data/processed/pairs_train_cyclic.jsonl --mode pair --sample 5000
```

## 4. Preview configurations without starting training

`scripts/experiment.py` is **preview-only by default**. Listing or previewing
does not import Torch, load a model, write results, or submit a scheduler job.

```bash
python scripts/experiment.py --config configs/oracle.json --list
python scripts/experiment.py --config configs/oracle.json --index 0
python scripts/experiment.py --config configs/nonoracle_transitive.json --list
python scripts/experiment.py --config configs/cyclic_legacy.json --list
```

Portable overrides: `--model-path`, `--data-root`, `--output-root`, and
`--oracle-model-path`, or environment variables `MODEL_PATH`, `DATA_ROOT`,
`OUTPUT_ROOT`, and `ORACLE_MODEL_PATH`. All paths refer to this checkout or
explicit user-provided locations, not an author's home directory.

Actual execution additionally requires `--execute` inside an approved Slurm
allocation, the expected visible GPU count, and a **new** output directory.
This launcher does not overwrite completed runs or silently resume them.
Low-level Python trainers start training directly; prefer the preview launcher.

## 5. Transitive scalar-oracle IPO / DPO

`configs/oracle.json` defines the intended grid:

| Parameter | Values |
|---|---|
| Objective | IPO, DPO |
| `alpha` | 0.8, 0.9, 0.95, 0.99 |
| `lambda_on` (initial-generator probability) | 0.5, 0.9 |
| `beta`, seed | 1, 0 |
| Outer updates | 80 |
| Saved diagnostic states | 0 through 80 |
| Oracle WR checkpoints | 0, 20, 40, 60, 80 |

There are 16 configurations, eight per objective. `--iters 81` in the current
core means **81 evaluated states and 80 optimizer rounds**: its last iteration
is evaluation only. The legacy runners have different indexing (Section 7).

Each training round selects 500 prompts and generates four responses per
prompt, arranged into two pairs. Each response independently chooses the
initial generator with probability lambda, otherwise the current generator.
Temperature is 0.8, top-p is 0.95, and max new tokens is 256. The single frozen
Nemotron reward model supplies hard chosen/rejected labels; empty responses
and exact reward ties are skipped. The realized pair count can be below 1,000.
This is neither two mixed reward models nor BT-probabilistic oracle labeling.

The 70B oracle is sharded across the visible GPUs. The trainable 1.5B model is
on GPU 0; this is not three-way DDP training. The supplied Slurm template asks
for 3 GPUs, 16 CPU cores, and 320 GB host RAM per task. H100-class hardware was
used for the recorded runs; three arbitrary low-memory GPUs are not equivalent.

```bash
# Only after accounting for existing jobs and dependencies; see Section 10.
sbatch --array=0-15%1 slurm/oracle.sh configs/oracle.json
```

This is a **recipe**, not a command to re-run already finished configurations.
Use `--list` to select only approved missing indices and retain `%1`.

## 6. Transitive non-oracle IPO / DPO

Use `configs/nonoracle_transitive.json` (two illustrative configurations).
No Nemotron model is loaded and no oracle WR is computed. Labels come from the
fixed scalar-score pairs built in Section 3. Sampling uses within-prompt
average-log-probability **pair margins** to mix a model-induced target with
uniform pairs, with the inherited self-normalized/clipped weighting.
Training likelihoods themselves use **sequence sums**, with a cached
outer-round reference.

```bash
python scripts/experiment.py --config configs/nonoracle_transitive.json --index 0
python scripts/experiment.py --config configs/nonoracle_transitive.json --index 1
# After queue preflight and approval:
sbatch --array=0-1%2 slurm/nonoracle.sh configs/nonoracle_transitive.json
```

The low-level `scripts/run_ipo.py` and `scripts/run_dpo.py` force
`--enable_oracle 0 --oracle_train_pairs 0`. Other arguments are shared with the
oracle core; `--preference_case transitive` selects the generated fixed
entropy support. The standardized IPO recipe is **not** the old token-average
non-oracle IPO experiment, and its output must be labeled accordingly.

## 7. Cyclic ordinary IPO / DPO

### Standard sequence-sum examples

```bash
python scripts/experiment.py --config configs/cyclic_sequence_sum.json --list
python scripts/experiment.py --config configs/cyclic_sequence_sum.json --index 0
```

These use fixed cyclic labels, sequence-sum training and cached references,
without a scalar training oracle. The two recipes fix alpha=0.99,
lambda_pair=0.5, beta_train=1 and expose 150 updates (`iters=151`).
They are runnable ordinary examples, not completed experimental claims.

### Reproduce the historical cyclic sampling ablation

```bash
python scripts/experiment.py --config configs/cyclic_legacy.json --list
python scripts/experiment.py --config configs/cyclic_legacy.json --index 0
# Historical reproduction only, after approval and queue preflight:
sbatch --array=0-9%2 slurm/nonoracle.sh configs/cyclic_legacy.json
```

Historical settings: alpha=0.99, tau=1, lambda_pair in
`[1, 0.75, 0.5, 0.25, 0]`, seed=0; IPO beta=10 / 1,000 pairs per round,
DPO beta=1 / 500 pairs per round. There are 150 inner-training rounds and
pre-update snapshots 0 through 149. The original IPO loss uses token averages;
DPO training uses sequence sums, but both displayed legacy panel curves use
token-average probabilities. References are recomputed within minibatches.
The full legacy runners can also consume transitive fixed pairs directly.

**Known invalid data:** the archived DPO lambda_pair=1 run first has nonfinite
loss at round 81; all 500 prompts have invalid saved scores from snapshot 82
onward. The old uniform-softmax fallback must not be plotted as convergence.
Mask those invalid snapshots and inspect raw scores/losses for every other
run. `COMPLETED` is a scheduler status, not a numerical-validity guarantee.
See [historical reproduction notes](docs/LEGACY.md).

## 8. New cyclic history experiments

**Prepared only; no new GPU experiment is started by this release.** One seed
is planned. Six configurations share data, pair proposals, budget and base
parameters: alpha=0.9, lambda_current=0.8, beta_train=1, seed=0.

| Index | Run ID | Objective | Intervention | nu | kappa |
|---:|---|---|---|---:|---:|
| 0 | ipo_baseline_s0 | IPO | ordinary | 0 | 0 |
| 1 | ipo_reference_s0 | IPO | lagged reference | 0.45 | 0 |
| 2 | ipo_sampling_s0 | IPO | feedback extrapolation | 0 | 0.25 |
| 3 | dpo_baseline_s0 | DPO | ordinary | 0 | 0 |
| 4 | dpo_reference_s0 | DPO | lagged reference | 0.45 | 0 |
| 5 | dpo_sampling_s0 | DPO | BT-feedback extrapolation | 0 | 0.25 |

Each method shares its ordinary control across the two comparisons: six runs,
not eight. The plan is in `experiments/cyclic_history/experiment_plan.json`.
The baseline is **not assumed to oscillate**, and no positive stabilization
outcome is assumed.

```bash
python experiments/cyclic_history/run_cyclic_history.py --run-id ipo_baseline_s0
python experiments/cyclic_history/run_cyclic_lagged_reference.py --run-id ipo_reference_s0 --check-data
python experiments/cyclic_history/run_cyclic_lagged_sampling.py --run-id dpo_sampling_s0 --check-data
# Only after explicit pilot approval, with no conflicting allocations:
sbatch --array=0-5%2 slurm/cyclic_history.sh
```

This pilot uses 500 fixed four-response panels, 100 outer updates, 1,000 pairs
per round, sequence-sum scores throughout, BF16, zero dropout, and averaged
gradient accumulation. It samples uniform pair proposals and applies the
positive importance ratio for the specified response-product pair law once.
History is stored as score arrays, not additional resident LLM copies.

The IPO feedback arm implements identity-payoff extrapolation. The DPO arm
extrapolates the **actual DPO BT optimizer**, not the entrywise-logit PsiPO map.
There are no negative loss weights or logits of hard 0/1 labels.
[The detailed history README](experiments/cyclic_history/README.md) provides
the equations, data checks, optimizer conventions, diagnostics, and limitations.

## 9. Metrics, checkpoints, and plots

The standard launcher writes:

```text
outputs/<config-name>/<run-id>/
  run_manifest.json
  oracle_baseline.jsonl              # oracle runs only
  checkpoints/adapters_<tag>/...
  logs/metrics_<tag>.csv
  logs/eval_support_<tag>.jsonl
  logs/iter_dumps_<tag>/iter_XXXX_prompt_metrics.csv
  logs/oracle_response_scores_<tag>.csv   # evaluated oracle checkpoints only
  logs/summary_<tag>.json
```

The current core directly records sequence scores, initial-model sequence
scores, response token counts, relative logits, and relative probabilities.
Primary oracle entropy is `prompt_relative_sequence_entropy_mean`:
`q_rel(i) = softmax_i(log pi_t(y_i|x) - log pi_0(y_i|x))`.
It is finite-panel relative-likelihood concentration in nats, **not full-model
generation entropy**. Keep legacy `prompt_entropy_mean` separate.

Oracle WR compares four current responses against four cached initial-model
responses per prompt, averaging all 16 strict reward comparisons over 500
prompts. Ties count as zero, not half. `oracle_soft_win_rate` is a different
sigmoid-based statistic. Training and evaluation share the same oracle, so WR
is not independent human validation; a WR plateau alone does not prove convergence.

```bash
python scripts/plot_results.py \
  --logs-root outputs/oracle_sequencesum --output-dir results/oracle
```

This creates separate IPO/DPO relative-entropy and WR PNG/PDF plots and a
summary CSV. Nonfinite/missing values remain gaps; the plotter does not fill
unevaluated checkpoints or mix legacy and relative entropy.

For **older compatible oracle dumps** lacking the direct metric:

```bash
python scripts/reconstruct_oracle_relative_entropy.py \
  --logs_root /path/to/historical/logs_oracle \
  --model_path model/Qwen2.5-1.5B --run_glob '*sequencesum_v2' --max_length 1537
```

This older reconstruction tool expects one metrics file and an iteration-dump
directory immediately within each matched run directory. It is not necessary
for fresh standard runs. Match the exact tokenizer, EOS, response-token count,
causal shift and truncation convention; never multiply average scores by a
character count or an arbitrary common length.

History output is under `outputs/cyclic_history/<run-id>/`, with `support.json`,
`manifest.json`, `metrics.csv`, per-round `snapshots/step_XXXX.npz`, and adapters.
Its `panel_entropy_mean` and `relative_sequence_entropy_mean` are distinct.
TV, temporal variance, panel log mass, and inner-update residual qualify the
trajectory interpretation. See the history README for the full artifact schema.

To inspect open-generation degeneration separately from entropy:

```bash
python scripts/generate_adapter_samples.py \
  --model_path model/Qwen2.5-1.5B --adapter_path /path/to/adapter \
  --prompts_path data/processed/eval_prompt_responses_1000.jsonl \
  --output_dir results/generation_check --stage final \
  --num_prompts 50 --num_responses 2
```

Run this only in an approved GPU allocation. Review actual generated CSV rows,
dominant-token share, unique-output ratio, unique-token ratio, and length-cap
hits; checking only for `*` is not a sufficient collapse test.

## 10. Scheduler safety and hardware

Read [SCHEDULING.md](docs/SCHEDULING.md) before submission. The templates do
not inspect or cancel existing work, choose dependencies, or install an
account-wide GPU cap.

- Count **all** allocations owned by the account, including unrelated jobs
  and resources still being released. Inspect pending jobs that can start.
- Sampling/non-oracle phase: at most two concurrent 1-GPU tasks and at most
  two allocated GPUs across all user programs.
- Remaining oracle phase: one 3-GPU task at a time, after the entire sampling
  phase ends. The two-GPU cap does not apply to this oracle phase.
- Separate arrays have independent throttles; `%2` on two arrays can allocate
  four GPUs. Queue subsequent arrays behind existing work.
- Preserve running jobs and checkpoints. No cancellation/requeue is implied.
- The partition/account/GPU type are site-specific. Add local Slurm options;
  the included resource sizes reflect recorded H100 experiments.

Recorded completed-job subsets from the 11 September accounting snapshot:

| Protocol | Jobs | GPUs/job | Wall hours/job, range | Allocated GPU-hours, subset |
|---|---:|---:|---:|---:|
| Legacy cyclic IPO | 5 | 1 | 24.72-25.51 | 124.77 |
| Legacy cyclic DPO | 5 | 1 | 13.28-13.43 | 66.91 |
| Corrected oracle IPO | 6 | 3 | 85.53-111.80 | 1700.54 |
| Corrected oracle DPO | 6 | 3 | 85.88-113.58 | 1862.79 |
| New history pilot | 0 | 1 planned | Not measured | Not measured |

These are not full-campaign totals or predictions for another machine. They
exclude queue waiting; oracle rows exclude later completions and interrupted
attempts. The legacy DPO row includes resources consumed by its numerically
invalid baseline.

## 11. Tests and reproducibility

```bash
python -m compileall -q scripts experiments tests
CUDA_VISIBLE_DEVICES=`` OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=`` OMP_NUM_THREADS=1 python -m unittest discover \
  -s experiments/cyclic_history -p `test_*.py` -v
```

The tests cover recipe expansion/CLI compatibility, synthetic data builders,
sequence-sum masking, frozen references, positive pair losses, history/BT
math, matched controls, and CPU LoRA microsteps. No test downloads real model
weights, submits jobs, or allocates a GPU. Lightweight-only environments may
skip Torch-dependent tests; the full CI environment installs those dependencies.

For a final figure, archive the exact code commit, config, software versions,
data/support hashes, valid iteration mask, and relevant scheduler accounting.
One training seed is not multiple independent runs just because there are many
prompts. Partial runs must retain their actual endpoint.

Adapter-only restart (`--resume_adapter_path` / `--start_iter` in the low-level
core) does not restore RNG, optimizer, diagnostic-support and metric history
as an exact checkpoint continuation. Do not concatenate restarted curves or
call them lossless resumes. The public recipe launcher intentionally refuses
existing run directories.

No experiment result or convergence claim is implied by providing a runnable
configuration. Keep negative and inconclusive outcomes, distinguish measured
results from prepared pilots, and do not force curves into an expected trend.
