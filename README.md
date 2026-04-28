# LLM Sampling with IPO

This repository implements **Identity Preference Optimization (IPO)** for fine-tuning large language models using pairwise preference data from the HelpSteer dataset.

The current codebase supports:

- prompt-aware pair sampling
- sampling mixture controlled by `lambda_on` and `mix_eps`
- evaluation on fixed candidate responses
- optional **augmented evaluation**, where extra model-generated responses are added to selected prompts
- full **token-level probability diagnostics**
- saving **pre-update / post-update adapters** for each iteration
- **training-loss logging**
- optional **fixed validation-pair loss logging** via `--val_pairs_path`

The main training script is:

```bash
run_iterative_ipo_fast.py
```

This is the current recommended version when you want both:
- the usual prompt-level dynamics outputs
- a fixed validation loss curve for convergence diagnostics

---

## Requirements

- Python 3.8+
- PyTorch
- transformers
- peft
- tqdm
- numpy
- pandas

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Step 1: Download a Base Model

Example:

```bash
huggingface-cli download Qwen/Qwen2.5-1.5B --local-dir model/Qwen2.5-1.5B
```

You may also use another compatible causal language model, as long as its architecture matches the LoRA target modules used in the training script.

---

## Step 2: Prepare the Data

Download and export HelpSteer:

```bash
python download.py
python export_helpsteer_jsonl.py
```

Build pairwise training data and a base evaluation set:

```bash
python build_pairs.py \
  --input data/raw/helpsteer.jsonl \
  --out_pairs data/processed/pairs_train.jsonl \
  --out_eval_prompts data/processed/eval_prompt_responses_1000.jsonl \
  --input_format response \
  --dedup_responses \
  --keep_exact_k 4 \
  --pair_mode all \
  --eval_prompts 1000 \
  --seed 0
```

This produces:

- `pairs_train.jsonl`: pairwise training data `(prompt, chosen, rejected, ...)`
- `eval_prompt_responses_1000.jsonl`: evaluation set with 1000 prompts, each with 4 candidate responses from the dataset

---

## Step 3: Prepare a Small Eval Set and a Fixed Validation Set

For quick debugging or small-scale runs, it is useful to:
- keep the full training pair file
- use a **smaller eval prompt file**
- use a **fixed validation pair set** for loss tracking

### Small eval set

Example: sample 100 eval prompts from an existing eval file

```bash
python -c "import json,random; random.seed(0); p='data/processed/eval_prompt_responses_1000.jsonl'; rows=[json.loads(x) for x in open(p,'r',encoding='utf-8') if x.strip()]; rows=random.sample(rows,100); out='data/processed/eval_prompt_responses_100.jsonl'; open(out,'w',encoding='utf-8').write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)); print(f'wrote {len(rows)} rows to {out}')"
```

### Fixed validation pair set

Example: sample 200 validation pairs from the training-pair file

```bash
python -c "import json,random; random.seed(0); p='data/processed/pairs_train.jsonl'; rows=[json.loads(x) for x in open(p,'r',encoding='utf-8') if x.strip()]; rows=random.sample(rows,200); out='data/processed/pairs_val_200.jsonl'; open(out,'w',encoding='utf-8').write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)); print(f'wrote {len(rows)} rows to {out}')"
```

### Optional helper script

If you prefer one script that creates both files:

```python
import os
import json
import random
import argparse

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def sample_rows(rows, k, seed):
    rng = random.Random(seed)
    return rng.sample(rows, k)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_in", type=str, required=True)
    ap.add_argument("--eval_out", type=str, required=True)
    ap.add_argument("--eval_k", type=int, default=100)
    ap.add_argument("--pairs_in", type=str, required=True)
    ap.add_argument("--val_out", type=str, required=True)
    ap.add_argument("--val_k", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    eval_rows = read_jsonl(args.eval_in)
    write_jsonl(args.eval_out, sample_rows(eval_rows, args.eval_k, args.seed))

    pair_rows = read_jsonl(args.pairs_in)
    write_jsonl(args.val_out, sample_rows(pair_rows, args.val_k, args.seed))

if __name__ == "__main__":
    main()
```

---

## Step 4: Run Training

## Recommended script

Use:

```bash
python run_iterative_ipo_fast.py ...
```

This version logs:

- prompt-level dynamics
- training loss
- fixed validation loss, if `--val_pairs_path` is provided

---

## Example: full run

```bash
python run_iterative_ipo_fast.py \
  --model_path /path/to/model \
  --pairs_path /path/to/pairs_train.jsonl \
  --val_pairs_path /path/to/pairs_val_200.jsonl \
  --eval_prompts_path /path/to/eval_prompt_responses_1000.jsonl \
  --out_dir /path/to/checkpoints \
  --log_dir /path/to/logs \
  --seed 0 \
  --auto_stop 1 \
  --max_iters 200 \
  --stop_min_iters 15 \
  --stop_patience 5 \
  --stop_tv_abs 0.005 \
  --exposure_window 10 \
  --min_total_exposure 8 \
  --min_recent_exposure 4 \
  --osc_detect 1 \
  --osc_window 8 \
  --osc_min_switches 4 \
  --osc_tv_floor 0.01 \
  --alpha 0.3 \
  --lambda_on 0.3 \
  --tau 1.0 \
  --beta 10 \
  --mix_eps 0.05 \
  --w_clip_min 0.1 \
  --w_clip_max 10.0 \
  --train_sample_size 500 \
  --pairs_per_prompt 2 \
  --batch_size 2 \
  --grad_accum 8 \
  --lr 5e-5 \
  --warmup_ratio 0.03 \
  --score_batch_size 2 \
  --epochs_per_iter 1 \
  --max_length 1537 \
  --dump_each_iter 1 \
  --save_iter_adapters 1 \
  --save_initial_adapter 1 \
  --save_final_adapter 1 \
  --dump_token_diagnostics 1 \
  --token_diag_max_length 2048 \
  --augment_eval_num_prompts 100 \
  --augment_eval_extra_responses 2 \
  --augment_eval_num_generate_candidates 20 \
  --augment_eval_generate_max_new_tokens 256 \
  --augment_eval_do_sample 1 \
  --augment_eval_temperature 0.8 \
  --augment_eval_top_p 0.95 \
  --augment_eval_select_by avg \
  --augment_eval_seed 123
```

---

## Example: small validation-focused run

This is a good smoke test when you want to inspect training loss and validation loss quickly:

```bash
python run_iterative_ipo_fast.py \
  --model_path /path/to/model \
  --pairs_path /path/to/pairs_train.jsonl \
  --val_pairs_path /path/to/pairs_val_200.jsonl \
  --eval_prompts_path /path/to/eval_prompt_responses_100.jsonl \
  --out_dir /path/to/checkpoints_test_small_val \
  --log_dir /path/to/logs_test_small_val \
  --seed 0 \
  --auto_stop 0 \
  --iters 5 \
  --stop_min_iters 15 \
  --stop_patience 5 \
  --stop_tv_abs 0.005 \
  --exposure_window 10 \
  --min_total_exposure 8 \
  --min_recent_exposure 4 \
  --osc_detect 1 \
  --osc_window 8 \
  --osc_min_switches 4 \
  --osc_tv_floor 0.01 \
  --alpha 0.3 \
  --lambda_on 0.3 \
  --tau 1 \
  --beta 10 \
  --mix_eps 0.05 \
  --w_clip_min 0.1 \
  --w_clip_max 10.0 \
  --train_sample_size 100 \
  --pairs_per_prompt 2 \
  --batch_size 1 \
  --grad_accum 4 \
  --lr 5e-5 \
  --warmup_ratio 0.03 \
  --score_batch_size 1 \
  --epochs_per_iter 1 \
  --max_length 1537 \
  --dump_each_iter 1 \
  --save_iter_adapters 0 \
  --save_initial_adapter 0 \
  --save_final_adapter 0 \
  --augment_eval_num_prompts 0 \
  --augment_eval_extra_responses 0 \
  --dump_token_diagnostics 0
```

---

## What the Current Script Does

At a high level, each outer iteration does the following:

1. Evaluate the current model on the eval candidate responses
2. Compute prompt-level response distributions
3. Compute entropy / TV / KL / top-1 response statistics
4. Optionally evaluate a **fixed validation loss** on `--val_pairs_path`
5. Build a prompt-aware training subset using a sampling distribution controlled by:
   - `tau`
   - `lambda_on`
   - `mix_eps`
6. Train for one or more epochs on that subset
7. Save metrics, dumps, and adapters

If augmented evaluation is enabled, selected prompts first receive additional model-generated responses before the iterative loop begins.

---

# Parameter Reference

## 1. Data and Paths

### `--model_path`
Path to the local base model directory.

### `--pairs_path`
Path to the pairwise training data JSONL file.

### `--val_pairs_path`
Optional path to a **fixed validation pair set**.

If provided, the script evaluates a fixed IPO loss on this set at every outer iteration and stores the resulting validation-loss statistics in the main metrics CSV.

### `--eval_prompts_path`
Path to the evaluation prompts file. This file contains the base evaluation candidate responses.

### `--out_dir`
Directory where model adapters/checkpoints are saved.

### `--log_dir`
Directory where logs, metrics CSVs, summaries, and per-iteration dump files are saved.

---

## 2. Randomness and Run Control

### `--seed`
Global random seed for Python, NumPy, and PyTorch.

### `--auto_stop`
- `1`: use the automatic stopping rule
- `0`: ignore the stopping rule and run a fixed number of outer iterations

### `--max_iters`
Maximum number of outer iterations when `auto_stop=1`.

### `--iters`
Number of outer iterations when `auto_stop=0`.

---

## 3. Convergence and Exposure

### `--stop_min_iters`
Minimum number of outer iterations before convergence checking starts.

### `--stop_patience`
A prompt must satisfy the TV threshold for this many eligible rounds in a row before being marked converged.

### `--stop_tv_abs`
Per-prompt TV threshold for convergence.

### `--exposure_window`
Window size for recent exposure counting.

### `--min_total_exposure`
Minimum cumulative exposure required before a prompt is eligible for convergence checking.

### `--min_recent_exposure`
Minimum recent exposure required before a prompt is eligible for convergence checking.

---

## 4. Oscillation Detection

### `--osc_detect`
- `1`: enable oscillation detection
- `0`: disable oscillation detection

### `--osc_window`
How many recent iterations are used to check oscillation behavior.

### `--osc_min_switches`
Minimum number of top-1 response switches in the window before a prompt can be flagged as oscillatory.

### `--osc_tv_floor`
Minimum mean TV over the oscillation window required to classify switching as real oscillation instead of noise.

---

## 5. Core Dynamics Parameters

### `--alpha`
Reference-policy mixing coefficient used in the training objective.

### `--beta`
IPO loss scale parameter. The target margin is:

```text
1 / (2 * beta)
```

### `--tau`
Softmax sharpness parameter used in:
1. training subset construction
2. evaluation response probabilities

### `--lambda_on`
Strength of preference-based sampling during training subset construction:

```text
base_mix = (1 - lambda_on) * uniform + lambda_on * induced
```

### `--mix_eps`
Additional smoothing parameter:

```text
mixed = (1 - mix_eps) * base_mix + mix_eps * uniform
```

Interpretation:
- `lambda_on` controls how strongly the induced distribution matters
- `mix_eps` controls extra anti-collapse smoothing

---

## 6. Training Subset Construction

### `--train_sample_size`
Approximate total number of training pairs used in each outer iteration.

### `--pairs_per_prompt`
How many pairs are sampled per selected prompt in each outer iteration.

### `--train_prompt_size`
If positive, directly specifies how many training prompts are sampled.
If `0`, the script infers it from:

```text
ceil(train_sample_size / pairs_per_prompt)
```

### `--w_clip_min`
Lower clipping bound for normalized pair weights.

### `--w_clip_max`
Upper clipping bound for normalized pair weights.

---

## 7. Optimization Hyperparameters

### `--batch_size`
Mini-batch size per forward/backward pass.

### `--grad_accum`
Gradient accumulation steps.

Effective batch size is roughly:

```text
batch_size × grad_accum
```

### `--lr`
Learning rate.

### `--warmup_ratio`
Warmup ratio for the linear learning-rate schedule.

### `--score_batch_size`
Batch size used for scoring pairs/responses and for fixed validation evaluation.

### `--epochs_per_iter`
Number of training epochs over the sampled subset in each outer iteration.

### `--max_length`
Maximum sequence length used for training/eval scoring.

---

## 8. LoRA Parameters

### `--lora_r`
LoRA rank.

### `--lora_alpha`
LoRA scaling factor.

### `--lora_dropout`
LoRA dropout.

---

## 9. Iteration Dumps and Checkpoint Saving

### `--dump_each_iter`
- `1`: save per-iteration dumps
- `0`: do not save them

### `--save_iter_adapters`
- `1`: save adapters at each iteration
- `0`: do not save them

### `--save_initial_adapter`
- `1`: save the initial adapter state before training
- `0`: do not save it

### `--save_final_adapter`
- `1`: save the final adapter after training
- `0`: do not save it

---

## 10. Token-Level Diagnostics

### `--dump_token_diagnostics`
- `1`: save token-level diagnostics
- `0`: disable them

### `--token_diag_max_length`
Maximum length used specifically for token-level diagnostics.

---

## 11. Augmented Evaluation Parameters

### `--augment_eval_num_prompts`
How many eval prompts are augmented with generated responses.

### `--augment_eval_extra_responses`
How many generated responses to add per selected prompt.

### `--augment_eval_num_generate_candidates`
How many generation candidates are sampled before selecting the final added responses.

### `--augment_eval_generate_max_new_tokens`
Maximum number of new tokens for each generated candidate response.

### `--augment_eval_do_sample`
- `1`: stochastic generation
- `0`: deterministic generation

### `--augment_eval_temperature`
Sampling temperature.

### `--augment_eval_top_p`
Top-p used in generation.

### `--augment_eval_select_by`
Selection rule for generated candidates:
- `avg`
- `sum`

### `--augment_eval_seed`
Random seed used for augmentation selection/generation randomness.

---

# What Gets Saved

## 1. Main Metrics CSV

A file like:

```text
metrics_alpha{alpha}_lambda{lambda}_tau{tau}_seed{seed}_FAST.csv
```

is written to `--log_dir`.

It stores one row per outer iteration, including:

- prompt entropy / TV / KL summaries
- convergence and oscillation counts
- exposure statistics
- **training loss summaries**
  - `train_loss_mean`
  - `train_loss_std`
  - `train_loss_min`
  - `train_loss_max`
  - `train_loss_last`
  - `train_num_batches`
- **fixed validation loss summaries**, if `--val_pairs_path` is provided
  - `val_loss_mean`
  - `val_loss_std`
  - `val_loss_min`
  - `val_loss_max`
  - `val_num_pairs`

---

## 2. Batch-Level Training Loss CSV

A file like:

```text
train_loss_steps_alpha{alpha}_lambda{lambda}_tau{tau}_seed{seed}.csv
```

is written to `--log_dir`.

It stores step-level training-loss records, including:

- `iter`
- `epoch`
- `batch_in_epoch`
- `global_train_batch_step`
- `loss`

This file is useful for plotting raw and smoothed training-loss curves.

---

## 3. Run Summary JSON

A file like:

```text
convergence_summary_alpha{alpha}_lambda{lambda}_tau{tau}_seed{seed}.json
```

is written to `--log_dir`.

It stores:
- run configuration
- stopping-rule settings
- sampling-rule description
- artifact locations
- final resolved / converged / oscillatory counts

---

## 4. Per-Iteration Prompt Metrics

If `--dump_each_iter 1`, the script writes:

```text
iter_XXXX_prompt_metrics.csv
```

Each row is one prompt and includes:

- `entropy_avg`, `entropy_sum`
- `tv_delta_avg`, `tv_delta_sum`
- `kl_delta_avg`, `kl_delta_sum`
- `top1_idx_avg`, `top1_idx_sum`
- `prob_avg_j`, `prob_sum_j`
- `avg_logprob_j`, `sum_logprob_j`
- `response_j`
- `response_source_j`
- exposure / convergence / oscillation fields

---

## 5. Per-Iteration Token Diagnostics

If `--dump_token_diagnostics 1`, the script writes:

```text
iter_XXXX_token_diagnostics.csv
```

Each row is one `(prompt, response)` pair and includes:

- `token_logprobs_json`
- `token_probs_json`
- `prefix_sum_logprobs_json`
- `prefix_avg_logprobs_json`
- `sum_logprob`
- `avg_logprob`
- `eos_logprob`
- `eos_prob`
- `truncated_by_max_length`

---

## 6. Per-Iteration NPZ Dumps

If `--dump_each_iter 1`, the script also writes:

```text
iter_XXXX.npz
```

This stores array versions of the main prompt-level quantities, including:

- `q_prompt_matrix_avg`
- `q_prompt_matrix_sum`
- `prompt_entropies_avg`
- `prompt_entropies_sum`
- `prompt_tvs_avg`
- `prompt_tvs_sum`
- `prompt_kls_avg`
- `prompt_kls_sum`
- `prompt_top1_avg`
- `prompt_top1_sum`

---

## 7. Adapter Checkpoints

If saving is enabled, adapters are written under a directory like:

```text
adapters_alpha{alpha}_lambda{lambda}_tau{tau}_seed{seed}/
```

including:

- `iter_init`
- `iter_XXXX_preupdate`
- `iter_XXXX_postupdate`
- `final`

### Important note

The per-iteration dumps correspond to:

```text
iter_XXXX_preupdate
```

That is the matching adapter for the saved metrics at iteration `t`.

---

## 8. Augmented Eval Metadata

When augmented evaluation is enabled, the script also saves metadata describing the final evaluation candidate set, including which responses are original vs generated.

---

# Plotting and Analysis

## A. Prompt-level and token-level trajectory plots

Use `plot_dynamics_diagnostics.py` to generate:
- per-prompt conditional probability trajectories
- entropy / TV / KL / top-1 plots
- token-level probability panels
- prompt subsets with or without augmented responses

Typical usage:

```bash
python plot_dynamics_diagnostics.py \
  --dump_dir /path/to/iter_dumps_alpha0.3_lambda0.3_tau1.0_seed0 \
  --out_dir /path/to/diagnostic_plots \
  --n_regular 4 \
  --n_augmented 4 \
  --seed 0 \
  --max_token_iters 3
```

## B. Training loss plots from metrics

If your metrics CSV already contains:
- `train_loss_mean`
- `train_loss_std`
- `train_loss_min`
- `train_loss_max`
- `train_loss_last`

you can use a simple plotting script to visualize:
- mean training loss by outer iteration
- mean ± std
- mean with min-max envelope

## C. Validation loss plots

If you run with `--val_pairs_path`, the metrics CSV also contains:
- `val_loss_mean`
- `val_loss_std`
- `val_loss_min`
- `val_loss_max`

These are the recommended quantities to inspect if you want a fixed-support loss curve that is easier to interpret than dynamic training loss.

---

# Practical Notes

- Token diagnostics can be large, because they are saved for all eval prompts × all responses
- Augmented eval changes the candidate set for selected prompts, so some prompts may have more than 4 responses
- The augmented candidate set is built once at initialization and remains fixed during training
- `train_loss_mean` is measured on the dynamically sampled and dynamically weighted training support for that iteration
- `val_loss_mean` is usually easier to interpret as a convergence diagnostic, because it is computed on a fixed validation pair set

---

# Project Structure

```text
scripts/
  run_iterative_ipo_fast.py

data/
  processed/
    pairs_train.jsonl
    pairs_val_200.jsonl
    eval_prompt_responses_1000.jsonl
    eval_prompt_responses_100.jsonl
```

---

# Summary

This implementation supports:

- IPO training with prompt-aware pair sampling
- meaningful control via `alpha`, `beta`, `lambda_on`, `tau`, and `mix_eps`
- augmented evaluation with additional generated responses
- full token-level probability diagnostics
- comparison of length-normalized and unnormalized response probabilities
- batch-level training-loss logging
- fixed validation-loss logging on a held-out pair set
