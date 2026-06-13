# Reproducing the IPO/DPO Sampling Experiments

This README explains how to reproduce the real-LLM experiments for the
sampling/reference-feedback project. It covers data download, model download,
preference-data construction, and running IPO/DPO experiments with the external
helpfulness oracle.

The current experiment has two preference settings:

- `transitive`: pairwise labels are induced by a scalar HelpSteer score, so the
  hidden preference matrix is strongly transitive.
- `cyclic`: each prompt keeps four dataset responses, but pairwise labels are
  replaced by a fixed cyclic tournament, so the hidden preference matrix is
  non-transitive.

The current experiment has two training losses:

- IPO: `scripts/run_ipo_oracle.py`
- DPO: `scripts/run_dpo_oracle.py`

Both use the shared implementation in:

```bash
scripts/run_preference_oracle_core.py
```

The older scripts `scripts/run_ipo.py` and `scripts/run_dpo.py` are kept for
backward compatibility. The new oracle experiments should use the `*_oracle.py`
entry points.

## 1. Expected Repository Layout

From the project root:

```bash
ipo/
  data/
    raw/
      helpsteer.jsonl
      HelpSteer/
    processed/
      pairs_train.jsonl
      eval_prompt_responses_1000.jsonl
      pairs_train_cyclic.jsonl
      eval_prompt_responses_cyclic_1000.jsonl
  model/
    Qwen2.5-3B/
    Qwen2.5-1.5B/
  scripts/
    download.py
    export_helpsteer_jsonl.py
    analyze_lengths.py
    build_pairs.py
    build_cyclic_pairs.py
    run_preference_oracle_core.py
    run_ipo_oracle.py
    run_dpo_oracle.py
  run_oracle_experiment.sh
```

On the UNC server, the project root has usually been:

```bash
/nas/longleaf/home/fanyao/ipo
```

The scratch/work output root used by the Slurm script is:

```bash
/work/users/f/a/fanyao/ipo
```

If your paths differ, edit `run_oracle_experiment.sh` or override
`MODEL_PATH` when submitting the job.

## 2. Environment Setup

Use a Python environment with PyTorch, Transformers, PEFT, Accelerate, Datasets,
Pandas, NumPy, and tqdm.

Example:

```bash
python -m venv ~/h100env
source ~/h100env/bin/activate
pip install --upgrade pip
pip install torch transformers accelerate peft datasets pandas numpy tqdm sentencepiece protobuf
```

If your cluster provides CUDA-specific PyTorch wheels or modules, use the
cluster-recommended installation command instead of the generic `pip install
torch`.

Log in to Hugging Face if the environment needs authentication for model
downloads:

```bash
huggingface-cli login
```

## 3. Download the HelpSteer Dataset

The training/evaluation preference pools are built from the original NVIDIA
HelpSteer dataset, not from HelpSteer2. HelpSteer contains prompts, responses,
and five human-scored attributes:

```text
helpfulness, correctness, coherence, complexity, verbosity
```

Download and export the train split to JSONL with the included helper scripts:

```bash
cd /nas/longleaf/home/fanyao/ipo
mkdir -p data/raw data/hf_cache

python scripts/download.py \
  --dataset nvidia/HelpSteer \
  --cache_dir data/hf_cache \
  --save_dir data/raw/HelpSteer

python scripts/export_helpsteer_jsonl.py \
  --input_dir data/raw/HelpSteer \
  --split train \
  --output data/raw/helpsteer.jsonl
```

The scripts are parameterized, so coauthors can use different cache/output
directories without editing Python source files.

Reference: [nvidia/HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer).

## 4. Download the Base Model

The current Slurm script defaults to Qwen2.5-3B:

```bash
mkdir -p model
huggingface-cli download Qwen/Qwen2.5-3B \
  --local-dir model/Qwen2.5-3B \
  --local-dir-use-symlinks False
```

To reproduce smaller 1.5B runs:

```bash
huggingface-cli download Qwen/Qwen2.5-1.5B \
  --local-dir model/Qwen2.5-1.5B \
  --local-dir-use-symlinks False
```

Reference: [Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B).

## 5. Download or Cache the Helpfulness Oracle

The external oracle is:

```text
nvidia/Llama-3.1-Nemotron-70B-Reward-HF
```

This model is a frozen evaluator. We do not train it. It is used only to score
responses and compute within-prompt win rates between checkpoint responses and
initial-model responses.

You can either let Transformers download it automatically on the first run, or
pre-download it:

```bash
huggingface-cli download nvidia/Llama-3.1-Nemotron-70B-Reward-HF \
  --local-dir model/Llama-3.1-Nemotron-70B-Reward-HF \
  --local-dir-use-symlinks False
```

If you pre-download it, pass:

```bash
--oracle_model_path /nas/longleaf/home/fanyao/ipo/model/Llama-3.1-Nemotron-70B-Reward-HF
```

The default Slurm script uses the Hugging Face repo id directly.

Reference:
[nvidia/Llama-3.1-Nemotron-70B-Reward-HF](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Reward-HF).

## 6. Build Transitive Preference Data

The transitive setting uses the scalar average of the five HelpSteer attributes.
For each prompt, the response with the larger scalar score is labeled
`chosen`, and the lower-scored response is labeled `rejected`. Since all pair
labels come from one scalar score, the hidden preference matrix is strongly
transitive.

Build the transitive pair pool and evaluation prompt file:

```bash
mkdir -p data/processed

python scripts/build_pairs.py \
  --input data/raw/helpsteer.jsonl \
  --out_pairs data/processed/pairs_train.jsonl \
  --out_eval_prompts data/processed/eval_prompt_responses_1000.jsonl \
  --input_format response \
  --prompt_key prompt \
  --response_key response \
  --score_fields helpfulness,correctness,coherence,complexity,verbosity \
  --dedup_responses \
  --keep_exact_k 4 \
  --pair_mode all \
  --eval_prompts 1000 \
  --seed 0
```

Expected structure:

- `pairs_train.jsonl`: rows with `prompt`, `chosen`, `rejected`, `delta`, and
  metadata.
- `eval_prompt_responses_1000.jsonl`: prompt-response rows. In the transitive
  real-LLM experiment, only the held-out prompts are used for support
  construction; the fixed support itself is generated by `pi0`.

## 7. Build Cyclic Preference Data

The cyclic setting uses the same HelpSteer prompts and dataset responses, but
the pairwise labels are replaced by a fixed cyclic tournament.

For each prompt, keep four dataset responses:

```text
y0, y1, y2, y3
```

The core cycle is:

```text
y0 > y1 > y2 > y3 > y0
```

We use `full_tournament`, which emits all six pairwise comparisons for four
responses. This keeps the number of pairs per prompt the same as the transitive
case.

Build cyclic data:

```bash
python scripts/build_cyclic_pairs.py \
  --input data/raw/helpsteer.jsonl \
  --out_pairs data/processed/pairs_train_cyclic.jsonl \
  --out_eval_prompts data/processed/eval_prompt_responses_cyclic_1000.jsonl \
  --input_format response \
  --prompt_key prompt \
  --response_key response \
  --score_fields helpfulness,correctness,coherence,complexity,verbosity \
  --dedup_responses \
  --keep_exact_k 4 \
  --comparison_mode full_tournament \
  --order_policy score_desc \
  --eval_prompts 1000 \
  --seed 0
```

Important: cyclic dynamics evaluation uses the four dataset responses stored in
`eval_prompt_responses_cyclic_1000.jsonl`. It does not use `pi0`-generated
fixed supports for the cyclic dynamics metric.

## 8. Quick Data Sanity Check

Run this after building both datasets:

```bash
python - <<'PY'
import json
from collections import Counter, defaultdict

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

for path in [
    "data/processed/pairs_train.jsonl",
    "data/processed/pairs_train_cyclic.jsonl",
]:
    rows = read_jsonl(path)
    by_prompt = defaultdict(int)
    for r in rows:
        by_prompt[r["prompt"]] += 1
    print(path, "rows=", len(rows), "prompts=", len(by_prompt),
          "pairs_per_prompt=", Counter(by_prompt.values()))

for path in [
    "data/processed/eval_prompt_responses_1000.jsonl",
    "data/processed/eval_prompt_responses_cyclic_1000.jsonl",
]:
    rows = read_jsonl(path)
    ks = []
    for r in rows:
        rs = r.get("responses", [])
        ks.append(len(rs))
    print(path, "rows=", len(rows), "K distribution=", Counter(ks))
PY
```

For the main K=4/full-tournament construction, pair counts should usually be
six pairs per prompt.

Optional token-length audit:

```bash
python scripts/analyze_lengths.py \
  --model_path model/Qwen2.5-3B \
  --input data/processed/pairs_train.jsonl \
  --mode pair \
  --sample 5000

python scripts/analyze_lengths.py \
  --model_path model/Qwen2.5-3B \
  --input data/processed/pairs_train_cyclic.jsonl \
  --mode pair \
  --sample 5000
```

Use this to confirm that `--max_length 1537` covers the intended fraction of
prompt-response examples.

## 9. Main Experiment Script

Use:

```bash
run_oracle_experiment.sh
```

Usage:

```bash
sbatch run_oracle_experiment.sh <ipo|dpo> <transitive|cyclic> <alpha> <lambda> <beta>
```

Examples:

```bash
sbatch run_oracle_experiment.sh ipo transitive 0.5 0.5 10
sbatch run_oracle_experiment.sh ipo transitive 0.8 0.8 10
sbatch run_oracle_experiment.sh ipo transitive 1.0 0.8 10

sbatch run_oracle_experiment.sh dpo transitive 0.5 0.5 1
sbatch run_oracle_experiment.sh dpo transitive 0.8 0.5 1
sbatch run_oracle_experiment.sh dpo transitive 1.0 0.8 1

sbatch run_oracle_experiment.sh ipo cyclic 0.8 0.8 10
sbatch run_oracle_experiment.sh dpo cyclic 0.8 0.8 1
```

The Slurm script requests four H100 GPUs:

```bash
#SBATCH --gres=gpu:4
```

The Qwen LoRA training model is loaded on `cuda:0`. The 70B reward oracle uses:

```bash
--oracle_device_map auto
```

so the reward model can be sharded across the visible H100 GPUs.

To use a different base model:

```bash
MODEL_PATH=/nas/longleaf/home/fanyao/ipo/model/Qwen2.5-1.5B \
  sbatch run_oracle_experiment.sh ipo transitive 0.8 0.8 10
```

## 10. Running a Small Smoke Test

Before launching the full 4-H100 oracle run, test the training/evaluation
pipeline without the 70B oracle:

```bash
python scripts/run_ipo_oracle.py \
  --model_path /nas/longleaf/home/fanyao/ipo/model/Qwen2.5-3B \
  --pairs_path data/processed/pairs_train.jsonl \
  --eval_prompts_path data/processed/eval_prompt_responses_1000.jsonl \
  --preference_case transitive \
  --out_dir /work/users/f/a/fanyao/ipo/debug_checkpoints \
  --log_dir /work/users/f/a/fanyao/ipo/debug_logs \
  --iters 2 \
  --alpha 0.5 \
  --lambda_on 0.5 \
  --beta 10 \
  --train_sample_size 20 \
  --pairs_per_prompt 2 \
  --batch_size 1 \
  --grad_accum 1 \
  --score_batch_size 1 \
  --generated_eval_num_prompts 5 \
  --generated_eval_num_candidates 2 \
  --generated_eval_keep_top_k 2 \
  --enable_oracle 0
```

For cyclic smoke test, change:

```bash
--pairs_path data/processed/pairs_train_cyclic.jsonl
--eval_prompts_path data/processed/eval_prompt_responses_cyclic_1000.jsonl
--preference_case cyclic
```

## 11. What the Code Evaluates

### Transitive case

The transitive case is designed to produce an entropy-collapse boundary over
`alpha`, `beta`, and `lambda`.

Evaluation support:

1. Select held-out eval prompts.
2. Before fine-tuning, use `pi0` to generate candidate responses.
3. Deduplicate and keep the top candidates by average log probability.
4. Keep this support fixed for the whole run.
5. At every outer iteration, compute the current model distribution over this
   fixed support.

Main metrics:

- `prompt_entropy_mean`
- `prompt_tv_mean`
- `top1_flip_rate_vs_initial`
- `oracle_win_rate`
- `oracle_soft_win_rate`

The main plot for the transitive setting should use final entropy or entropy
drop over the parameter grid.

### Cyclic case

The cyclic case explores whether real LLM training shows non-convergence or
oscillation under cyclic preference feedback.

Evaluation support:

1. Load the four dataset responses for each cyclic eval prompt.
2. Keep these four responses fixed.
3. At every outer iteration, compute the current model distribution over the
   four responses.

Main metrics:

- `cycle_strength_mean`
- `prompt_tv_mean`
- `prompt_entropy_mean` as auxiliary information
- `oracle_win_rate`
- `oracle_soft_win_rate`

`cycle_strength_mean` is the post-burn-in time variance of the model-induced
probability distribution on the four cyclic responses.

### Oracle trajectory

The oracle trajectory is shared across transitive and cyclic runs.

For each eval prompt:

1. Generate `M=4` responses from the initial model `pi0`.
2. Score them with the frozen Nemotron reward oracle and cache the scores.
3. At every outer iteration, generate `M=4` responses from `pi_t`.
4. Score the checkpoint responses with the same oracle.
5. Compare only responses from the same prompt.
6. Average the `4 x 4 = 16` comparisons per prompt.

The main oracle metrics are:

- `oracle_win_rate`: hard within-prompt win rate of `pi_t` versus `pi0`.
- `oracle_soft_win_rate`: mean sigmoid reward-difference win rate.
- `oracle_reward_delta_mean`: mean reward difference within the same prompt set.

The oracle is not trained. It is only an external trajectory used to show that
the model is changing under a HelpSteer2-derived helpfulness criterion and to
provide an additional empirical convergence signal when the win-rate trajectory
stabilizes.

## 12. Output Files

For a run tag like:

```text
ipo_transitive_alpha0.8_lambda0.8_tau1.0_beta10.0_seed0
```

the output directory contains:

```bash
/work/users/f/a/fanyao/ipo/logs_oracle/<OUT_TAG>/
  metrics_<RUN_TAG>.csv
  oracle_response_scores_<RUN_TAG>.csv
  eval_support_<RUN_TAG>.csv
  eval_support_<RUN_TAG>.jsonl
  generated_eval_generation_scores_<RUN_TAG>.csv
  summary_<RUN_TAG>.json
  iter_dumps_<RUN_TAG>/
    iter_0000_prompt_metrics.csv
    iter_0000_train_pair_support.csv
    ...
```

The most important file is:

```bash
metrics_<RUN_TAG>.csv
```

Key columns:

```text
iter
loss_type
preference_case
alpha
lambda
beta
prompt_entropy_mean
prompt_tv_mean
top1_flip_rate_vs_initial
cycle_strength_mean
oracle_win_rate
oracle_soft_win_rate
oracle_reward_delta_mean
train_loss_mean
```

Oracle baseline caches are stored under:

```bash
/work/users/f/a/fanyao/ipo/oracle_baselines/
```

The baseline cache avoids regenerating and rescoring the initial-model responses
for every grid job.

## 13. Example Grid Submission

IPO transitive grid:

```bash
for a in 0.5 0.8 1.0; do
  for l in 0.5 0.8; do
    for b in 5 10 20; do
      sbatch run_oracle_experiment.sh ipo transitive "$a" "$l" "$b"
    done
  done
done
```

DPO transitive grid:

```bash
for a in 0.5 0.8 1.0; do
  for l in 0.5 0.8; do
    for b in 0.5 1 2; do
      sbatch run_oracle_experiment.sh dpo transitive "$a" "$l" "$b"
    done
  done
done
```

Cyclic exploratory runs:

```bash
for loss in ipo dpo; do
  for a in 0.5 0.8 1.0; do
    for l in 0.5 0.8; do
      if [ "$loss" = "ipo" ]; then
        b=10
      else
        b=1
      fi
      sbatch run_oracle_experiment.sh "$loss" cyclic "$a" "$l" "$b"
    done
  done
done
```

## 14. Collecting Final Rows

After jobs finish, collect the final row of every metrics file:

```bash
python - <<'PY'
import glob
import os
import pandas as pd

root = "/work/users/f/a/fanyao/ipo/logs_oracle"
rows = []
for path in glob.glob(os.path.join(root, "*", "metrics_*.csv")):
    df = pd.read_csv(path)
    if len(df) == 0:
        continue
    row = df.sort_values("iter").iloc[-1].to_dict()
    row["metrics_path"] = path
    rows.append(row)

out = os.path.join(root, "final_metrics_summary.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print("wrote", out, "rows=", len(rows))
PY
```

For the transitive entropy-boundary figure, use columns:

```text
alpha, lambda, beta, prompt_entropy_mean
```

For oracle trajectories, use:

```text
iter, oracle_win_rate, oracle_soft_win_rate
```

For cyclic dynamics, use:

```text
iter, cycle_strength_mean, prompt_tv_mean
```

## 15. Common Issues

### Oracle model OOM

The oracle is a 70B reward model. Use four H100s when possible. If memory is
tight, lower:

```bash
--oracle_batch_size 1
--oracle_generation_batch_size 1
```

### Runs are slow

The most expensive part is per-iteration oracle scoring/generation. For full
paper runs we currently use:

```bash
--oracle_eval_every 1
--oracle_num_prompts 500
--oracle_num_responses 4
```

For debugging, use:

```bash
--enable_oracle 0
```

or reduce:

```bash
--oracle_num_prompts 50
--oracle_num_responses 2
--oracle_eval_every 5
```

### Cyclic eval says no rows with four responses

Make sure the cyclic eval file was built by:

```bash
scripts/build_cyclic_pairs.py
```

and not by the transitive `build_pairs.py` command.

### Transitive eval support looks wrong

In the transitive setting, the runtime support is generated from `pi0`. The
dataset responses in `eval_prompt_responses_1000.jsonl` are not used as the
fixed entropy support.

### Different base model

The default Slurm script uses:

```bash
/nas/longleaf/home/fanyao/ipo/model/Qwen2.5-3B
```

Override it with:

```bash
MODEL_PATH=/path/to/model sbatch run_oracle_experiment.sh ipo transitive 0.8 0.8 10
```

## 16. Reproducibility Notes

- The training pair pool is fixed before training.
- Preference labels are not regenerated during training.
- The current model only changes pair sampling/weighting through the
  mixed-reference/sampling feedback rule.
- The oracle is frozen and is never trained.
- Oracle comparisons are within-prompt only.
- Transitive dynamics and cyclic dynamics use different fixed supports by
  design.
- Inner validation loss is not part of the new main experiment.
