

# LLM Sampling with IPO

This repository implements **Identity Preference Optimization (IPO)** for fine-tuning large language models using pairwise preference data from the HelpSteer dataset.

The current training script supports:

- prompt-aware pair sampling
- sampling mixture controlled by `lambda_on` and `mix_eps`
- evaluation on fixed candidate responses
- optional **augmented evaluation**, where additional model-generated responses are added to selected prompts
- full **token-level probability diagnostics**
- saving **pre-update / post-update adapters** for each iteration

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

------

## Step 1: Download Model

Example:

```bash
huggingface-cli download Qwen/Qwen2.5-1.5B --local-dir model/Qwen2.5-1.5B
```

You may also use a different compatible causal language model if the architecture matches the LoRA target modules in the training script.

------

## Step 2: Prepare Dataset

Download and export HelpSteer:

```bash
python download.py
python export_helpsteer_jsonl.py
```

Build pairwise training data and the evaluation set:

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

------

## Step 3: Run Training

Example command:

```bash
python scripts/run_iterative_ipo_fast.py \
  --model_path /path/to/model \
  --pairs_path /path/to/pairs_train.jsonl \
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

------

# What the Current Script Does

At a high level, each outer iteration does the following:

1. Evaluate the current model on the eval candidate responses
2. Compute prompt-level response distributions
3. Compute entropy / TV / KL / top-1 response statistics
4. Build a prompt-aware training subset using a sampling distribution controlled by:
   - `tau`
   - `lambda_on`
   - `mix_eps`
5. Train for one or more epochs on that subset
6. Save metrics, dumps, and adapters

If augmented evaluation is enabled, some prompts first receive extra model-generated responses before the iterative training loop begins.

------

# Parameter Reference

## 1. Data and Paths

### `--model_path`

Path to the local base model directory.

### `--pairs_path`

Path to the pairwise training data JSONL file.

### `--eval_prompts_path`

Path to the evaluation prompts file. This file contains the base evaluation candidate responses (typically 1000 prompts × 4 responses each).

### `--out_dir`

Directory where model adapters/checkpoints are saved.

### `--log_dir`

Directory where logs, metrics CSVs, summaries, and per-iteration dump files are saved.

------

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

------

## 3. Convergence and Exposure

These parameters control when a prompt is considered “resolved” or converged.

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

------

## 4. Oscillation Detection

These parameters control whether a prompt is flagged as oscillatory instead of converged.

### `--osc_detect`

- `1`: enable oscillation detection
- `0`: disable oscillation detection

### `--osc_window`

How many recent iterations are used to check oscillation behavior.

### `--osc_min_switches`

Minimum number of top-1 response switches in the window before a prompt can be flagged as oscillatory.

### `--osc_tv_floor`

Minimum mean TV over the oscillation window required to classify switching as real oscillation instead of noise.

------

## 5. Core Dynamics Parameters

These are the most important theoretical / experimental parameters.

### `--alpha`

Reference-policy mixing coefficient used in the training objective.

The script computes a mixed reference score:

- part from the initial reference model
- part from the current model

Larger `alpha` means the current model contributes more to the reference term.

### `--beta`

IPO loss scale parameter.

The loss uses a target margin of:

```text
1 / (2 * beta)
```

So `beta` changes the loss landscape and the desired training margin.

### `--tau`

Softmax sharpness parameter.

It is used in two places:

1. **Training subset construction**
   - sharper `tau` makes the induced pair distribution more concentrated
2. **Evaluation response probabilities**
   - larger `tau` makes prompt-level conditional distributions sharper

### `--lambda_on`

Controls the strength of preference-based sampling during training subset construction.

The script first builds an induced distribution from pair margins, then mixes it with uniform:

```text
base_mix = (1 - lambda_on) * uniform + lambda_on * induced
```

Interpretation:

- `lambda_on = 0`: pure uniform
- `lambda_on = 1`: pure induced distribution
- intermediate values interpolate between them

### `--mix_eps`

Anti-extreme smoothing parameter.

After `base_mix` is built, the script further smooths it:

```text
mixed = (1 - mix_eps) * base_mix + mix_eps * uniform
```

Interpretation:

- prevents the distribution from becoming too concentrated
- helps avoid near-degenerate sampling such as almost one-hot behavior

### Important distinction: `lambda_on` vs `mix_eps`

- `lambda_on` controls **how strongly the induced distribution matters**
- `mix_eps` controls **how much extra uniform smoothing is injected**

These two parameters are not redundant.

------

## 6. Training Subset Construction

### `--train_sample_size`

Approximate total number of training pairs used in each outer iteration.

### `--pairs_per_prompt`

How many pairs are sampled per selected training prompt in each outer iteration.

### `--train_prompt_size`

If set to a positive value, this directly specifies how many training prompts are sampled each iteration.

If `0`, the script infers it from:

```text
ceil(train_sample_size / pairs_per_prompt)
```

### `--w_clip_min`

Lower clipping bound for pair weights after normalization.

### `--w_clip_max`

Upper clipping bound for pair weights after normalization.

------

## 7. Optimization Hyperparameters

### `--batch_size`

Mini-batch size per forward/backward pass.

### `--grad_accum`

Gradient accumulation steps.

Effective batch size is approximately:

```text
batch_size × grad_accum
```

### `--lr`

Learning rate.

### `--warmup_ratio`

Warmup ratio for the linear learning-rate schedule.

### `--score_batch_size`

Batch size used for scoring responses / pairs during evaluation and subset construction.

If GPU memory is tight, reduce this.

### `--epochs_per_iter`

Number of training epochs over the sampled subset in each outer iteration.

### `--max_length`

Maximum sequence length used for training/eval scoring.

If prompt + response exceed this length, the sequence is truncated from the left.

------

## 8. LoRA Parameters

### `--lora_r`

LoRA rank.

### `--lora_alpha`

LoRA scaling factor.

### `--lora_dropout`

LoRA dropout.

------

## 9. Iteration Dumps and Checkpoint Saving

### `--dump_each_iter`

- `1`: save per-iteration dumps
- `0`: do not save them

Per-iteration dumps include prompt-level metrics and optional token-level diagnostics.

### `--save_iter_adapters`

- `1`: save adapters at each iteration
- `0`: do not save them

### `--save_initial_adapter`

- `1`: save the initial adapter state before training
- `0`: do not save it

### `--save_final_adapter`

- `1`: save the final adapter after training
- `0`: do not save it

------

## 10. Token-Level Diagnostics

These parameters control token-probability saving.

### `--dump_token_diagnostics`

- `1`: save token-level diagnostics
- `0`: disable them

When enabled, the script saves token-level conditional probabilities for **all eval prompts and all responses**.

### `--token_diag_max_length`

Maximum length used specifically for token-level diagnostics.

This can be larger than `--max_length` to reduce truncation in token-level analysis.

------

## 11. Augmented Evaluation Parameters

These parameters control the “4 original responses + generated responses” evaluation augmentation.

### `--augment_eval_num_prompts`

How many eval prompts should be augmented with extra generated responses.

If `0`, no augmentation is performed.

Example:

- `100` means 100 prompts are expanded beyond the original 4 responses.

### `--augment_eval_extra_responses`

How many generated responses to add per selected prompt.

Example:

- `2` means 4 original + 2 generated = 6 responses for augmented prompts

### `--augment_eval_num_generate_candidates`

How many candidate generations to sample initially for each selected prompt before filtering and selecting the final added responses.

Larger values improve coverage but cost more time.

### `--augment_eval_generate_max_new_tokens`

Maximum number of new tokens for each generated candidate response.

### `--augment_eval_do_sample`

- `1`: use stochastic sampling for generation
- `0`: use deterministic generation

### `--augment_eval_temperature`

Temperature used in generation when sampling is enabled.

### `--augment_eval_top_p`

Top-p parameter used in generation when sampling is enabled.

### `--augment_eval_select_by`

Selection rule used to choose the final generated responses from the candidate pool.

Choices:

- `avg`: select by average logprob
- `sum`: select by sum logprob

### `--augment_eval_seed`

Random seed used specifically for selecting augmented prompts and generation-related randomness.

------

# What Gets Saved

## 1. Main Metrics CSV

A file like:

```text
metrics_alpha{alpha}_lambda{lambda}_tau{tau}_seed{seed}_FAST.csv
```

is written to `--log_dir`.

It stores one row per outer iteration, including:

- mean entropy
- mean/max TV
- mean KL
- convergence counts
- oscillation counts
- exposure statistics
- avg-based and sum-based aggregate quantities

------

## 2. Run Summary JSON

A file like:

```text
convergence_summary_alpha{alpha}_lambda{lambda}_tau{tau}_seed{seed}.json
```

is written to `--log_dir`.

It stores:

- run configuration
- stopping rule settings
- sampling rule description
- artifact locations
- final resolved / converged / oscillatory counts

------

## 3. Per-Iteration Prompt Metrics

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

This file supports plotting:

- per-prompt entropy over steps
- per-prompt TV over steps
- per-response conditional probability over steps
- comparison between length-normalized and unnormalized response probabilities

------

## 4. Per-Iteration Token Diagnostics

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

This file supports:

- token-level probability plots
- EOS analysis
- response normalization analysis (`sum` vs `avg`)

------

## 5. Per-Iteration NPZ Dumps

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

------

## 6. Adapter Checkpoints

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

------

## 7. Augmented Eval Metadata

When augmented evaluation is enabled, the script also saves metadata describing the final evaluation candidate set.

This allows you to identify:

- which responses are original
- which responses are generated
- which prompts were augmented

------

# What You Can Analyze From Saved Outputs

Using the saved prompt-level and token-level files, you can recover:

1. **Per-prompt entropy over steps**
2. **Per-prompt TV over steps**
3. **Per-response conditional probability over steps**
   - both `prob_avg`
   - and `prob_sum`
4. **Token-level probability trajectories**
5. **Prefix-sum vs prefix-average behavior**
6. **EOS probability behavior**
7. **Whether generated responses steal probability mass from original responses**

So the current code supports analyzing both:

- prompt-level response distribution dynamics
- token-level response construction dynamics

------

# Practical Notes

- Token diagnostics can be large, because they are saved for **all eval prompts × all responses**
- Augmented eval changes the candidate set for selected prompts, so some prompts may have more than 4 responses
- The augmented candidate set is built **once at initialization** and remains fixed during training

------

# Project Structure

```text
scripts/
  run_iterative_ipo_fast.py

data/
  processed/
    pairs_train.jsonl
    eval_prompt_responses_1000.jsonl
```

------

# Summary

This implementation supports:

- IPO training with prompt-aware pair sampling
- meaningful control via `alpha`, `beta`, `lambda_on`, `tau`, and `mix_eps`
- augmented evaluation with additional generated responses
- full token-level probability diagnostics
- comparison of length-normalized and unnormalized response probabilities
