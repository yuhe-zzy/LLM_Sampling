"""List or preview an explicit experiment grid; training requires --execute."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PROTOCOLS = {"oracle", "static", "legacy_cyclic"}
COMMON = {
    "seed": 0, "auto_stop": 0, "epochs_per_iter": 1,
    "iters": 81, "alpha": 0.8, "lambda_on": 0.5, "beta": 1,
    "tau": 1, "mix_eps": 0.05, "w_clip_min": 0.1, "w_clip_max": 10,
    "train_sample_size": 1000, "pairs_per_prompt": 2,
    "batch_size": 1, "grad_accum": 4, "lr": 1e-5, "warmup_ratio": 0.03,
    "score_batch_size": 1, "max_length": 1537,
    "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
    "dump_each_iter": 1, "save_iter_adapters": 0,
    "save_initial_adapter": 0, "save_final_adapter": 1,
    "generated_eval_num_prompts": 500, "generated_eval_num_candidates": 10,
    "generated_eval_keep_top_k": 5, "generated_eval_max_new_tokens": 256,
    "generated_eval_do_sample": 1, "generated_eval_temperature": 0.8,
    "generated_eval_top_p": 0.95, "generated_eval_seed": 123,
}


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("protocol") not in PROTOCOLS:
        raise ValueError("Unknown protocol")
    if config.get("preference_case") not in {"transitive", "cyclic"}:
        raise ValueError("preference_case must be transitive or cyclic")
    if config["protocol"] == "oracle" and config["preference_case"] != "transitive":
        raise ValueError("A scalar training oracle does not preserve cyclic labels")
    if config["protocol"] == "legacy_cyclic" and config["preference_case"] != "cyclic":
        raise ValueError("The legacy_cyclic recipe requires cyclic data")
    grid = config["grid"]
    if not grid or any(not isinstance(v, list) or not v for v in grid.values()):
        raise ValueError("Every grid axis must be a nonempty list")
    allowed_axes = {"method", "alpha", "lambda_on", "beta", "seed"}
    if set(grid) - allowed_axes or "method" not in grid:
        raise ValueError("Unsupported grid axes")
    return config


def enumerate_runs(config):
    axes = list(config["grid"])
    runs = []
    for combination in itertools.product(*(config["grid"][k] for k in axes)):
        axis_values = dict(zip(axes, combination))
        method = axis_values.pop("method")
        if method not in {"ipo", "dpo"}:
            raise ValueError("method must be ipo or dpo")
        params = dict(COMMON)
        params.update(config.get("common", {}))
        params.update(config.get("method_overrides", {}).get(method, {}))
        params.update(axis_values)
        if not 0 <= float(params["alpha"]) <= 1 or not 0 <= float(params["lambda_on"]) <= 1:
            raise ValueError("alpha and lambda_on must lie in [0,1]")
        if not math.isfinite(float(params["beta"])) or float(params["beta"]) <= 0:
            raise ValueError("beta must be positive and finite")
        if int(params["iters"]) < 1:
            raise ValueError("iters must be positive")
        tag = (f"{method}_a{float(params['alpha']):g}_l{float(params['lambda_on']):g}"
               f"_b{float(params['beta']):g}_seed{int(params['seed'])}")
        runs.append({"run_id": tag, "method": method, "parameters": params})
    if len({r["run_id"] for r in runs}) != len(runs):
        raise ValueError("Grid produces duplicate run IDs")
    return runs


def build_command(config, run, model_path, data_root, output_root, oracle_model):
    params = dict(run["parameters"])
    case, protocol, method = config["preference_case"], config["protocol"], run["method"]
    suffix = "_cyclic" if case == "cyclic" else ""
    run_root = Path(output_root).resolve() / config["name"] / run["run_id"]
    params.update(
        model_path=str(model_path),
        pairs_path=str(Path(data_root) / f"pairs_train{suffix}.jsonl"),
        eval_prompts_path=str(Path(data_root) / f"eval_prompt_responses{suffix}_1000.jsonl"),
        log_dir=str(run_root / "logs"), out_dir=str(run_root / "checkpoints"),
    )
    if protocol == "legacy_cyclic":
        entry = ROOT / "scripts" / "legacy" / f"run_{method}.py"
        params.update(eval_support_source="data", eval_response_keep_k=4,
                      compute_oscillation=1, compute_loss_diagnostics=0, track_inner_val_loss=0)
    else:
        entry = ROOT / "scripts" / f"run_{method}{'_oracle' if protocol == 'oracle' else ''}.py"
        params.update(preference_case=case, enable_oracle=int(protocol == "oracle"),
                      oracle_train_pairs=int(protocol == "oracle"), model_torch_dtype="float16")
        if protocol == "oracle":
            params.update(
                oracle_model_path=str(oracle_model), oracle_torch_dtype="bfloat16",
                oracle_device_map="auto", oracle_max_length=4096, oracle_batch_size=1,
                oracle_eval_every=20, oracle_num_prompts=500, oracle_num_responses=4,
                oracle_generation_batch_size=4, oracle_max_new_tokens=256,
                oracle_do_sample=1, oracle_temperature=0.8, oracle_top_p=0.95,
                oracle_seed=777, oracle_train_skip_ties=1, oracle_train_max_new_tokens=256,
                oracle_train_do_sample=1, oracle_train_temperature=0.8, oracle_train_top_p=0.95,
                # Per-run caches avoid cross-support and concurrent-writer collisions.
                oracle_baseline_cache_path=str(run_root / "oracle_baseline.jsonl"),
                oracle_reuse_baseline_cache=1,
            )
    command = [sys.executable, str(entry)]
    for key, value in params.items():
        command.extend([f"--{key}", str(value)])
    return command, run_root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--list", action="store_true", help="Print stable zero-based array indices")
    parser.add_argument("--index", type=int)
    parser.add_argument("--execute", action="store_true", help="Train inside an approved GPU allocation")
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH", str(ROOT / "model/Qwen2.5-1.5B")))
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT", str(ROOT / "data/processed")))
    parser.add_argument("--output-root", default=os.environ.get("OUTPUT_ROOT", str(ROOT / "outputs")))
    parser.add_argument("--oracle-model-path", default=os.environ.get(
        "ORACLE_MODEL_PATH", "nvidia/Llama-3.1-Nemotron-70B-Reward-HF"))
    args = parser.parse_args()
    config = load_config(args.config)
    runs = enumerate_runs(config)
    if args.list:
        if args.execute or args.index is not None:
            parser.error("--list cannot be combined with --execute or --index")
        for i, run in enumerate(runs):
            print(f"{i}\t{run['run_id']}")
        return
    if args.index is None or not 0 <= args.index < len(runs):
        parser.error(f"--index must be in 0..{len(runs)-1}")
    run = runs[args.index]
    command, run_root = build_command(config, run, args.model_path, args.data_root,
                                     args.output_root, args.oracle_model_path)
    print(shlex.join(command), flush=True)
    if not args.execute:
        print("PREVIEW ONLY: no model loaded, no files created, no job submitted.")
        return
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("--execute requires an approved Slurm GPU allocation; never train on a login node")
    import torch
    required_gpus = 3 if config["protocol"] == "oracle" else 1
    if not torch.cuda.is_available() or torch.cuda.device_count() != required_gpus:
        parser.error(f"This recipe requires exactly {required_gpus} visible allocated GPUs")
    if not Path(args.model_path).is_dir():
        parser.error("--model-path must point to a downloaded local model")
    for flag in ("--pairs_path", "--eval_prompts_path"):
        if not Path(command[command.index(flag) + 1]).is_file():
            parser.error(f"Missing input for {flag}")
    run_root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "config": config, "run": run, "command": command,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "state": "RUNNING",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (ROOT / "scripts").rglob("*.py")},
    }
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        manifest["git_commit"] = commit.stdout.strip() if commit.returncode == 0 else None
    except FileNotFoundError:
        manifest["git_commit"] = None
    path = run_root / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except BaseException:
        manifest["state"] = "FAILED_OR_INTERRUPTED"
        raise
    else:
        manifest["state"] = "FINISHED_CHECK_METRICS"
    finally:
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
