# LLM Sampling with IPO

This repository implements **Identity Preference Optimization (IPO)** for fine-tuning large language models using pairwise preference data from the HelpSteer dataset.

---

## Requirements

- Python 3.8+
- PyTorch (install separately below)

Install PyTorch first (visit [pytorch.org](https://pytorch.org) to get the right command for your CUDA version):

```bash
# Example for CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then install all other dependencies:

```bash
pip install -r requirements.txt
```

---

## Step 1: Download the Model

Download the Qwen2.5 model from HuggingFace. Choose one of the following:

```bash
# 1.5B version
huggingface-cli download Qwen/Qwen2.5-1.5B --local-dir model/Qwen2.5-1.5B

# or 3B version
huggingface-cli download Qwen/Qwen2.5-3B --local-dir model/Qwen2.5-3B
```

> If `huggingface-cli` is not available, install it with `pip install huggingface_hub`.

Place the downloaded model under the `model/` directory:

```
model/
└── Qwen2.5-1.5B/   (or Qwen2.5-3B/)
    ├── config.json
    ├── tokenizer.json
    └── model.safetensors
    ...
```

---

## Step 2: Download & Prepare the Dataset

### 2a. Download HelpSteer from HuggingFace

```bash
python download.py
```

This will download the [nvidia/HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) dataset and save it to `data/raw/HelpSteer/`.

### 2b. Export to JSONL format

```bash
python export_helpsteer_jsonl.py
```

This converts the dataset to `data/raw/helpsteer.jsonl`.

### 2c. Build training pairs

```bash
python build_pairs.py \
  --input data/raw/helpsteer.jsonl \
  --out_pairs data/processed/pairs_train.jsonl \
  --out_eval_prompts data/processed/eval_prompt_responses.jsonl \
  --input_format response \
  --dedup_responses \
  --keep_exact_k 4 \
  --pair_mode all \
  --eval_prompts 1000 \
  --seed 0
```

This builds pairwise preference data (chosen/rejected) and saves it to `data/processed/`:

- `pairs_train.jsonl` — training pairs `(prompt, chosen, rejected, delta)`
- `eval_prompt_responses.jsonl` — evaluation set of 1000 randomly sampled prompts, each with their 4 candidate responses sorted by score

> **Note:** The size of the evaluation set is fixed at this step by `--eval_prompts 1000`. The training script (`run_iterative_ipo_fast.py`) simply reads this file as-is — there is no way to change the eval set size at training time. If you want a different eval set size, re-run `build_pairs.py` with a different `--eval_prompts` value before training.

---

## Step 3: Run IPO Training

Run the following command, replacing the paths with your own:

```bash
python scripts/run_iterative_ipo_fast.py \
  --model_path /your/path/to/model/model_type \
  --pairs_path /your/path/to/data/processed/pairs_train.jsonl \
  --eval_prompts_path /your/path/to/data/processed/eval_prompt_responses.jsonl \
  --out_dir /your/path/to/checkpoints_model_type \
  --log_dir /your/path/to/logs_model_type \
  --seed 0 \
  --auto_stop 1 \
  --max_iters 150 \
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
  --alpha 0.7 \
  --lambda_on 0.8 \
  --tau 8 \
  --beta 10 \
  --mix_eps 0.05 \
  --w_clip_min 0.1 \
  --w_clip_max 10.0 \
  --train_sample_size 1000 \
  --pairs_per_prompt 2 \
  --batch_size 1 \
  --grad_accum 8 \
  --lr 5e-5 \
  --warmup_ratio 0.03 \
  --score_batch_size 1 \
  --epochs_per_iter 1 \
  --max_length 1537 \
  --dump_each_iter 1
```

> **The 4 paths you must change:** `--model_path`, `--pairs_path`, `--eval_prompts_path`, `--out_dir`, `--log_dir`

---

### Key Parameters to Tune

If you want to reproduce or extend experiments, these are the most important parameters to adjust:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | — | Path to your local Qwen2.5 model directory |
| `--seed` | `0` | Random seed. Change to run different seeds for variance/error bars |
| `--alpha` | `0.7` | Reference policy mixing coefficient |
| `--tau` | `8` | Softmax temperature scale — higher = sharper induced distribution |
| `--beta` | `10` | IPO loss parameter controlling target margin scale |
| `--lambda_on` | `0.8` | Experimental config parameter for theoretical correspondence |
| `--train_sample_size` | `1000` | Number of pairs used per training iteration |

---

### Full Parameter Reference

**Data & Paths**

| Parameter | Description |
|-----------|-------------|
| `--model_path` | Local HuggingFace model directory |
| `--pairs_path` | Pairwise training data `(prompt, chosen, rejected)` |
| `--eval_prompts_path` | Eval set with all responses per prompt (4 responses each) |
| `--out_dir` | Output directory for model checkpoints |
| `--log_dir` | Output directory for logs, metrics, and per-iteration dumps |

**Training Loop**

| Parameter | Description |
|-----------|-------------|
| `--seed` | Random seed |
| `--auto_stop` | `1` = auto stop when converged, `0` = fixed number of iterations |
| `--max_iters` | Max outer iterations (used when `auto_stop=1`) |
| `--iters` | Fixed iterations to run (used when `auto_stop=0`) |

**Convergence**

| Parameter | Description |
|-----------|-------------|
| `--stop_min_iters` | Minimum iterations before convergence check begins |
| `--stop_patience` | Consecutive rounds a prompt must satisfy TV threshold to be marked converged |
| `--stop_tv_abs` | TV convergence threshold per prompt |

**Exposure-Aware Stop Rule**

| Parameter | Description |
|-----------|-------------|
| `--exposure_window` | Sliding window length for recent exposure stats |
| `--min_total_exposure` | Min cumulative exposure for a prompt to be eligible for stop rule |
| `--min_recent_exposure` | Min exposure within the window for a prompt to be eligible |

**Oscillation Detection**

| Parameter | Description |
|-----------|-------------|
| `--osc_detect` | `1` = enable oscillation detection, `0` = disable |
| `--osc_window` | Number of recent rounds used to detect oscillation |
| `--osc_min_switches` | Min top-1 response switches in the window to flag oscillation |
| `--osc_tv_floor` | Min average TV required to flag as true oscillation (not just noise) |

**Theory / Dynamics**

| Parameter | Description |
|-----------|-------------|
| `--alpha` | Reference policy mixing coefficient |
| `--lambda_on` | Experimental config for theoretical parameter correspondence |
| `--tau` | Softmax temperature scale for induced distribution sharpness |
| `--beta` | IPO loss margin scale |

**Resampling**

| Parameter | Description |
|-----------|-------------|
| `--train_sample_size` | Total pairs per iteration |
| `--pairs_per_prompt` | Pairs sampled per selected prompt per iteration |
| `--train_prompt_size` | Explicit prompt count per iteration (inferred if not set) |
| `--mix_eps` | Mix induced distribution with uniform to avoid degenerate sampling (e.g. `0.05` = 95% induced + 5% uniform) |
| `--w_clip_min` | Lower clip bound for pair weights |
| `--w_clip_max` | Upper clip bound for pair weights |

**Training Hyperparameters**

| Parameter | Description |
|-----------|-------------|
| `--batch_size` | Per-GPU batch size |
| `--grad_accum` | Gradient accumulation steps (effective batch = `batch_size × grad_accum`) |
| `--lr` | Learning rate |
| `--warmup_ratio` | Fraction of optimizer steps used for warmup |
| `--epochs_per_iter` | Passes over the training subset per outer iteration |
| `--max_length` | Max tokenized sequence length |
| `--score_batch_size` | Batch size for scoring/eval (set small if GPU memory is tight) |

**Output**

| Parameter | Description |
|-----------|-------------|
| `--dump_each_iter` | `1` = save per-iteration prompt metrics, distributions, and pair support files (needed for trajectory plots) |

---

## Project Structure

```
.
├── model/                        # Downloaded model weights (not tracked by git)
├── data/                         # Datasets (not tracked by git)
│   ├── raw/
│   │   ├── HelpSteer/            # Raw HuggingFace dataset
│   │   └── helpsteer.jsonl       # Exported JSONL
│   └── processed/
│       ├── pairs_train.jsonl     # Training pairs
│       └── eval_prompt_responses.jsonl
├── download.py                   # Step 2a: Download dataset
├── export_helpsteer_jsonl.py     # Step 2b: Export to JSONL
├── build_pairs.py                # Step 2c: Build preference pairs
└── run_iterative_ipo_fast.py               # Step 3: IPO training
```

---

## Notes

- The `model/` and `data/` directories are excluded from git due to file size. You must download them manually following the steps above.
- Windows users: the default paths in `download.py` and `export_helpsteer_jsonl.py` use Windows-style paths. Update them to match your system if you are on Linux/Mac, for example:
  ```python
  # Change this:
  cache_dir = r"C:\yuhe32\dpo\ipo\data\hf_cache"
  # To this:
  cache_dir = "data/hf_cache"
  ```

## Step 4: What gets saved after training

After running `run_iterative_ipo_fast.py`, the script writes several outputs to `--log_dir` (and optionally per-iteration dumps if `--dump_each_iter 1`).

### 1. Main metrics CSV

A file of the form

```text
metrics_alpha{alpha}_lambda{lambda}_tau{tau}_seed{seed}_FAST.csv
```

is written to `--log_dir`.

This is the main per-iteration summary table. Each row corresponds to one outer iteration and records quantities such as:

- `prompt_entropy_mean` — mean entropy over eval prompts
- `prompt_tv_mean`, `prompt_tv_max` — mean / max TV distance between consecutive iterations at the prompt level
- `prompt_kl_mean` — mean prompt-level KL change
- `num_prompts_converged` — number of prompts marked as converged
- `num_prompts_oscillatory` — number of prompts diagnosed as oscillatory
- `num_prompts_resolved` — number of prompts either converged or oscillatory
- `cum_exposure_mean`, `recent_exposure_mean` — exposure statistics used in the stop rule

This CSV is the first file to inspect if you want to track global training dynamics.

------

### 2. Convergence summary JSON

A file of the form

```text
convergence_summary_alpha{alpha}_lambda{lambda}_tau{tau}_seed{seed}.json
```

is also written to `--log_dir`.

This contains a compact summary of the run, including:

- the experiment configuration (`alpha`, `lambda_on`, `tau`, `beta`, `seed`, etc.)
- the total number of eval prompts
- how many prompts were ultimately classified as converged / oscillatory / resolved
- the stopping rule configuration
- the training rule configuration

This file is useful for quickly checking the final status of a run without opening the full metrics CSV.

------

### 3. Per-iteration dump directory

If `--dump_each_iter 1`, the script creates a directory of the form

```text
iter_dumps_alpha{alpha}_lambda{lambda}_tau{tau}_seed{seed}/
```

inside `--log_dir`.

This directory contains detailed files for every iteration.

#### (a) Prompt distribution dump: `iter_XXXX.npz`

For each iteration `t`, the file

```text
iter_XXXX.npz
```

stores numpy arrays such as:

- `q_prompt_matrix` — the prompt-level induced distribution over the full response set for each eval prompt
- `prompt_entropies` — prompt-level entropy values
- `prompt_tvs` — prompt-level TV change relative to the previous iteration
- `prompt_kls` — prompt-level KL change
- `prompt_top1` — index of the highest-probability response
- `prompt_converged_mask` — whether each prompt is currently marked converged
- `prompt_oscillatory_mask` — whether each prompt is flagged as oscillatory
- `prompt_cum_exposure`, `prompt_recent_exposure` — exposure counts used by the stop rule

This `.npz` file is the most convenient source if you want to load arrays directly in Python and make custom plots.

#### (b) Prompt metrics CSV: `iter_XXXX_prompt_metrics.csv`

For each iteration `t`, the file

```text
iter_XXXX_prompt_metrics.csv
```

contains one row per eval prompt.

It includes:

- prompt text / prompt id
- number of candidate responses for that prompt
- entropy
- TV / KL change from the previous iteration
- top-1 response index
- convergence / oscillation / resolved flags
- exposure statistics
- the full per-response probabilities: `prob_0`, `prob_1`, ...

This is the easiest file to use if you want to inspect or plot trajectories for individual prompts across iterations.

#### (c) Training support CSV: `iter_XXXX_train_pair_support.csv`

For each iteration `t`, the file

```text
iter_XXXX_train_pair_support.csv
```

records the prompt-aware pair sampling distribution used to build that iteration’s training subset.

It includes columns such as:

- `prompt`
- `pair_global_idx`
- `margin_avglogprob`
- `induced_pair_prob`
- `mixed_pair_prob`
- `num_pairs_for_prompt`
- `pairs_sampled_for_prompt`

This file is useful for diagnosing how the reweighting / resampling dynamics are behaving during training.

------

## Step 5: Typical post-run analyses

With the saved outputs above, you can do several downstream analyses:

1. **Global convergence curves**
   - Plot `prompt_entropy_mean`, `prompt_tv_mean`, `num_prompts_converged`, etc. from the main metrics CSV.
2. **Single-prompt trajectory plots**
   - Use `iter_XXXX_prompt_metrics.csv` or `iter_XXXX.npz` to visualize, for one prompt, how the response probabilities
     $\pi_t(y \mid x)$
     evolve across iterations.
3. **Oscillation diagnosis**
   - Inspect prompts whose `oscillatory` flag turns on, and plot their top-1 response switches and TV changes across time.
4. **Exposure diagnostics**
   - Use `cum_exposure` and `recent_exposure` to verify that prompts declared converged were sufficiently updated.
5. **Sampling / support diagnostics**
   - Use `iter_XXXX_train_pair_support.csv` to understand which pairs were emphasized at each iteration and how the induced pair distribution changes over time.

------

## Practical note

If you plan to make prompt-level trajectory plots later, make sure to set

```bash
--dump_each_iter 1
```



## Step 6: How to visualize the saved outputs

The most useful files for plotting are the per-iteration `.npz` dumps inside

```text id="um9y7a"
iter_dumps_alpha{alpha}_lambda{lambda}_tau{tau}_seed{seed}/
```

Each file

```text
iter_XXXX.npz
```

contains the prompt-level arrays for iteration `t`, including:

- `q_prompt_matrix`: normalized prompt-level response probabilities
- `prompt_entropies`: entropy for each eval prompt
- `prompt_tvs`: total variation (TV) distance from the previous iteration
- `prompt_kls`: KL divergence from the previous iteration
- `prompt_top1`: index of the highest-probability response
- convergence / oscillation / exposure-related arrays

In particular, `q_prompt_matrix` is already the **normalized conditional distribution**
$\pi_t(y \mid x)$
over the candidate responses for each prompt, so this is the object you should use for plotting response-probability trajectories.

------

### What to use for different types of plots

#### 1. Plotting the conditional probability of each response across iterations

Use `q_prompt_matrix` from each `iter_XXXX.npz`.

For a fixed prompt `x` with 4 candidate responses, read its row from `q_prompt_matrix` at every iteration, and plot

$\pi_t(y_1 \mid x),\ \pi_t(y_2 \mid x),\ \pi_t(y_3 \mid x),\ \pi_t(y_4 \mid x)$

against iteration `t`.

This is the main plot for visualizing whether the prompt-level distribution collapses, remains diffuse, or oscillates.

------

#### 2. Plotting entropy over iterations for a single prompt

Use `prompt_entropies` from each `iter_XXXX.npz`.

For a fixed prompt index `p`, plot

$H_t(x_p)$

against iteration `t`.

This shows whether the prompt-level distribution is becoming sharper or flatter over time.

------

#### 3. Plotting TV change over iterations for a single prompt

Use `prompt_tvs` from each `iter_XXXX.npz`.

For a fixed prompt index `p`, plot the TV distance between consecutive iterations:

$\mathrm{TV}!\left(\pi_t(\cdot \mid x_p), \pi_{t-1}(\cdot \mid x_p)\right)$

against iteration `t`.

This is the main quantity used in the prompt-level convergence rule.

------

#### 4. Plotting global averages across prompts

Use the main run-level CSV

```text
metrics_alpha{alpha}_lambda{lambda}_tau{tau}_seed{seed}_FAST.csv
```

to plot aggregate quantities such as:

- `prompt_entropy_mean`
- `prompt_tv_mean`
- `prompt_tv_max`
- `num_prompts_converged`
- `num_prompts_oscillatory`
- `num_prompts_resolved`

These figures summarize the global dynamics of the run.

------

#### 5. Diagnosing oscillation

To inspect potentially oscillatory prompts, use:

- `q_prompt_matrix` for the response-probability trajectories
- `prompt_top1` for top-response switching
- `prompt_tvs` for whether the changes are substantial
- `prompt_oscillatory_mask` for the final diagnostic flag

A typical oscillation diagnosis plot combines:

- response probabilities across iterations
- top-1 response index across iterations
- prompt-level TV across iterations

------

### Which file should I use: `.npz` or `.csv`?

Use `.npz` if you want to:

- make plots
- load arrays efficiently in Python
- inspect full prompt-level probability trajectories
- compute new statistics after training

Use `.csv` if you want to:

- manually inspect prompts in spreadsheet form
- search for particular prompt ids or prompt texts
- quickly filter prompts by entropy / TV / exposure / convergence flags

In practice, the recommended workflow is:

1. Use the `.csv` files to find prompts of interest.
2. Use the `.npz` files to make the actual plots.

------

### Important note on probabilities

When plotting response trajectories, always use the normalized probabilities from `q_prompt_matrix`.

Do **not** plot raw log-likelihoods directly if your goal is to visualize the prompt-level conditional distribution. The normalization has already been performed when constructing `q_prompt_matrix`, so those values already represent the response probabilities within each prompt.

------

### Example plotting tasks

Typical useful figures include:

1. **Single-prompt probability trajectory**
   - x-axis: iteration
   - y-axis: probability
   - one line per response
2. **Single-prompt entropy trajectory**
   - x-axis: iteration
   - y-axis: entropy
3. **Single-prompt TV trajectory**
   - x-axis: iteration
   - y-axis: TV from previous iteration
4. **Global mean entropy / TV curves**
   - x-axis: iteration
   - y-axis: aggregate metric from the main CSV
5. **Selected oscillatory prompt diagnostics**
   - response-probability lines + TV curve + top-1 switching pattern

------

### Existing plotting scripts

If you use the provided plotting utilities, they should read from the per-iteration dump directory under `--log_dir`.

For example, a prompt-trajectory plotting script typically takes:

- the dump directory containing `iter_XXXX.npz`
- an output directory for figures
- optionally a prompt id / prompt index / prompt text filter

and then plots the response-probability trajectories from `q_prompt_matrix`.

### Plotting a single prompt trace

To visualize the dynamics of a single eval prompt across iterations, use:

```bash id="j8xgch"
python plot_single_prompt_trace.py \
  --dump_dir /path/to/iter_dumps_alpha0.7_lambda0.8_tau8_seed0 \
  --out_dir /path/to/single_prompt_plots \
  --prompt_index 15
```

You may also select a prompt by `--prompt_id` or by substring matching with `--prompt_contains`.

This script reads the per-iteration files

- `iter_XXXX_prompt_metrics.csv`

and produces the following outputs in `--out_dir`:

- `conditional_prob.png`
   normalized prompt-level conditional probability of each response across iterations
- `entropy.png`
   prompt entropy across iterations
- `tv_from_prev.png`
   prompt-level TV distance from the previous iteration
- `kl_from_prev.png`
   prompt-level KL divergence from the previous iteration
- `top1_idx.png`
   index of the highest-probability response over time
- `exposure.png`
   prompt exposure statistics over time
- `flags.png`
   convergence / oscillation / exposure-eligibility flags
- `overview_panel.png`
   a compact multi-panel summary figure for the selected prompt

It also saves:

- `response_mapping.csv` — mapping from response index to response text
- `response_prob_trace.csv` — long-format probability trajectory table
- `prompt_metrics_trace.csv` — per-iteration metrics for the selected prompt
- `meta.json` — metadata of the selected prompt

The main object plotted in `conditional_prob.png` is the normalized prompt-level distribution
$$
\pi_t(y \mid x)
$$
so this figure directly shows whether a prompt collapses, stays diffuse, or oscillates over time.



### Plotting trajectories for all prompts

To generate prompt-level trajectory plots for all eval prompts, use:

```bash id="j8b8fjlwm"
python plot_all_prompt_trajectories.py \
  --dump_dir /path/to/iter_dumps_alpha0.7_lambda0.8_tau8_seed0 \
  --out_dir /path/to/all_prompt_plots
```

Optional filters:

- `--max_prompts N` limits plotting to the first `N` prompts
- `--prompt_contains "keyword"` only plots prompts whose text contains the keyword
- `--only_oscillatory 1` only plots prompts whose final oscillatory flag is on

This script reads the per-iteration files

- `iter_XXXX_prompt_metrics.csv`

and creates one subdirectory per prompt inside `--out_dir`.

Each prompt subdirectory contains:

- `conditional_prob.png`
   normalized prompt-level conditional probability of each response across iterations
- `entropy.png`
   prompt entropy across iterations
- `tv_from_prev.png`
   prompt-level TV distance from the previous iteration
- `kl_from_prev.png`
   prompt-level KL divergence from the previous iteration
- `top1_idx.png`
   top-1 response index across iterations
- `overview_panel.png`
   a compact multi-panel summary figure

and also:

- `response_mapping.csv`
   mapping from response index to response text
- `response_prob_trace.csv`
   long-format table of response probabilities across iterations
- `prompt_metrics_trace.csv`
   full per-iteration metrics for the selected prompt
- `meta.json`
   metadata for the prompt

In addition, the script writes

- `selected_prompts_summary.csv`

to the top-level `--out_dir`, summarizing the final state of every plotted prompt.

This script is useful for:

- visualizing collapse vs. non-collapse at the prompt level
- inspecting whether individual prompts remain diffuse
- identifying top-response switching patterns
- diagnosing oscillatory prompts
- checking how entropy and TV evolve over time for each prompt
