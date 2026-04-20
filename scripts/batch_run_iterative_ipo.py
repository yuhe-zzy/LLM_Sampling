#!/usr/bin/env python3
# -*- coding: utf-8 -*-



from __future__ import annotations
import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


# -------------------------------------------------
# Option A: define your parameter groups here
# -------------------------------------------------
EXPERIMENTS: List[Dict[str, Any]] = [
    # Example group 1
    {"name": "a03_l08_t8_b10", "alpha": 0.5, "lambda_on": 0.8, "tau": 10.0, "beta": 5},
    
]


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def fmt_value(x: Any) -> str:
    if isinstance(x, float):
        # compact but stable string
        s = f"{x:.10g}"
        return s
    return str(x)


def load_config_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "experiments" in obj:
        obj = obj["experiments"]
    if not isinstance(obj, list):
        raise ValueError("config_json must contain a list, or a dict with key 'experiments'.")
    for i, item in enumerate(obj):
        if not isinstance(item, dict):
            raise ValueError(f"Experiment #{i} is not a dict.")
        if "name" not in item:
            item["name"] = f"exp_{i}"
    return obj


def build_run_cmd(args, exp: Dict[str, Any], seed: int) -> List[str]:
    cmd = [
        sys.executable,
        args.script,
        "--model_path", args.model_path,
        "--pairs_path", args.pairs_path,
        "--out_dir", args.out_dir,
        "--log_dir", args.log_dir,
        "--seed", str(seed),
    ]

    # shared fixed knobs from CLI
    shared_keys = [
        "iters", "auto_stop", "max_iters", "stop_min_iters", "stop_patience",
        "stop_entropy_abs", "stop_ess_rel", "stop_flip_abs",
        "epochs_per_iter",
        "w_clip_min", "w_clip_max",
        "max_length",
        "train_sample_size", "eval_support_size",
        "batch_size", "grad_accum", "lr", "warmup_ratio",
        "score_batch_size",
        "lora_r", "lora_alpha", "lora_dropout",
        "print_last_k_bench",
        "dump_each_iter",
    ]
    for k in shared_keys:
        v = getattr(args, k)
        cmd.extend([f"--{k}", str(v)])

    # experiment-specific knobs
    exp_keys = ["alpha", "lambda_on", "tau", "beta"]
    for k in exp_keys:
        if k not in exp:
            raise ValueError(f"Experiment '{exp.get('name', '?')}' missing required key '{k}'.")
        cmd.extend([f"--{k}", str(exp[k])])

    return cmd


def metrics_csv_path(log_dir: str, exp: Dict[str, Any], seed: int) -> str:
    return os.path.join(
        log_dir,
        f"metrics_alpha{exp['alpha']}_lambda{exp['lambda_on']}_tau{exp['tau']}_seed{seed}_FAST.csv"
    )


def run_one(args, exp: Dict[str, Any], seed: int) -> str:
    out_csv = metrics_csv_path(args.log_dir, exp, seed)

    if args.skip_existing and os.path.exists(out_csv):
        print(f"[SKIP] existing metrics found: {out_csv}")
        return out_csv

    cmd = build_run_cmd(args, exp, seed)
    print("\n" + "=" * 100)
    print(f"[RUN] exp={exp['name']} seed={seed}")
    print(" ".join(cmd))
    print("=" * 100)

    subprocess.run(cmd, check=True)
    if not os.path.exists(out_csv):
        raise FileNotFoundError(f"Run finished but metrics file not found: {out_csv}")
    return out_csv


def collect_final_rows(log_dir: str, experiments: List[Dict[str, Any]], seeds: List[int]) -> pd.DataFrame:
    rows = []
    for exp in experiments:
        for seed in seeds:
            csv_path = metrics_csv_path(log_dir, exp, seed)
            if not os.path.exists(csv_path):
                print(f"[WARN] missing metrics for exp={exp['name']} seed={seed}: {csv_path}")
                continue
            df = pd.read_csv(csv_path)
            if len(df) == 0:
                print(f"[WARN] empty metrics csv: {csv_path}")
                continue
            last = df.sort_values("iter").iloc[-1].to_dict()
            row = {
                "exp_name": exp["name"],
                "seed": seed,
                "alpha": exp["alpha"],
                "lambda_on": exp["lambda_on"],
                "tau": exp["tau"],
                "beta": exp["beta"],
                "last_iter": int(last["iter"]),
            }
            for k, v in last.items():
                if k != "iter":
                    row[k] = v
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_final_rows(df_final: pd.DataFrame) -> pd.DataFrame:
    if len(df_final) == 0:
        return pd.DataFrame()

    id_cols = ["exp_name", "alpha", "lambda_on", "tau", "beta"]
    numeric_cols = [
        c for c in df_final.columns
        if c not in id_cols + ["seed"]
        and pd.api.types.is_numeric_dtype(df_final[c])
    ]

    agg = df_final.groupby(id_cols, as_index=False)[numeric_cols].agg(["mean", "std", "count"])
    agg.columns = [
        "_".join([x for x in col if x]).rstrip("_")
        if isinstance(col, tuple) else col
        for col in agg.columns
    ]
    return agg.reset_index(drop=True)


def aggregate_trajectories(log_dir: str, experiments: List[Dict[str, Any]], seeds: List[int], out_dir: str) -> None:
    """
    For each experiment:
      read all seed csvs
      outer-join on iter
      output mean/std trajectory over seeds
    """
    ensure_dir(out_dir)

    for exp in experiments:
        seed_dfs = []
        for seed in seeds:
            csv_path = metrics_csv_path(log_dir, exp, seed)
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path).sort_values("iter").reset_index(drop=True)
            df["seed"] = seed
            seed_dfs.append(df)

        if not seed_dfs:
            print(f"[WARN] no trajectories found for exp={exp['name']}")
            continue

        big = pd.concat(seed_dfs, axis=0, ignore_index=True)

        id_cols = ["iter"]
        skip_cols = ["seed"]
        numeric_cols = [
            c for c in big.columns
            if c not in id_cols + skip_cols
            and pd.api.types.is_numeric_dtype(big[c])
        ]

        grp = big.groupby("iter", as_index=False)[numeric_cols].agg(["mean", "std", "count"])
        grp.columns = [
            "_".join([x for x in col if x]).rstrip("_")
            if isinstance(col, tuple) else col
            for col in grp.columns
        ]
        grp = grp.reset_index(drop=True)

        out_csv = os.path.join(out_dir, f"{exp['name']}_trajectory_seed_agg.csv")
        grp.to_csv(out_csv, index=False)
        print(f"[AGG] wrote trajectory agg: {out_csv}")


def main():
    ap = argparse.ArgumentParser()

    # path to base experiment
    ap.add_argument("--script", type=str, required=True, help="Path to run_iterative_ipo_fast.py")
    ap.add_argument("--model_path", type=str, required=True)
    ap.add_argument("--pairs_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="checkpoints_fast")
    ap.add_argument("--log_dir", type=str, default="logs")

    # seeds
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Seeds to run for each parameter group.")
    ap.add_argument("--skip_existing", type=int, default=1, help="Skip a run if final metrics csv already exists.")

    # optional external config
    ap.add_argument("--config_json", type=str, default=None, help="Optional JSON file describing experiment groups.")

    # pass-through knobs to the base script
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--auto_stop", type=int, default=1)
    ap.add_argument("--max_iters", type=int, default=50)
    ap.add_argument("--stop_min_iters", type=int, default=15)
    ap.add_argument("--stop_patience", type=int, default=5)
    ap.add_argument("--stop_entropy_abs", type=float, default=0.005)
    ap.add_argument("--stop_ess_rel", type=float, default=0.01)
    ap.add_argument("--stop_flip_abs", type=float, default=0.005)
    ap.add_argument("--epochs_per_iter", type=int, default=1)

    ap.add_argument("--w_clip_min", type=float, default=0.1)
    ap.add_argument("--w_clip_max", type=float, default=10.0)
    ap.add_argument("--max_length", type=int, default=256)

    ap.add_argument("--train_sample_size", type=int, default=50)
    ap.add_argument("--eval_support_size", type=int, default=4000)

    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)

    ap.add_argument("--score_batch_size", type=int, default=8)

    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    ap.add_argument("--print_last_k_bench", type=int, default=10)
    ap.add_argument("--dump_each_iter", type=int, default=1)

    args = ap.parse_args()

    experiments = load_config_json(args.config_json) if args.config_json else EXPERIMENTS
    if len(experiments) == 0:
        raise ValueError("No experiments to run.")

    ensure_dir(args.out_dir)
    ensure_dir(args.log_dir)

    # run all jobs sequentially
    produced_csvs = []
    for exp in experiments:
        if "name" not in exp:
            raise ValueError(f"Experiment missing 'name': {exp}")
        for seed in args.seeds:
            csv_path = run_one(args, exp, seed)
            produced_csvs.append(csv_path)

    # aggregate final rows
    df_final = collect_final_rows(args.log_dir, experiments, args.seeds)
    final_csv = os.path.join(args.log_dir, "batch_final_metrics_per_seed.csv")
    df_final.to_csv(final_csv, index=False)
    print(f"[BATCH] wrote per-seed final metrics: {final_csv}")

    df_final_agg = aggregate_final_rows(df_final)
    final_agg_csv = os.path.join(args.log_dir, "batch_final_metrics_seed_agg.csv")
    df_final_agg.to_csv(final_agg_csv, index=False)
    print(f"[BATCH] wrote final mean/std metrics: {final_agg_csv}")

    # aggregate trajectories
    traj_dir = os.path.join(args.log_dir, "batch_trajectory_seed_agg")
    aggregate_trajectories(args.log_dir, experiments, args.seeds, traj_dir)

    print("\n[DONE] Batch run complete.")


if __name__ == "__main__":
    main()
