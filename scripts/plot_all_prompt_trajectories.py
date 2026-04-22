import os
import re
import glob
import json
import argparse
import textwrap
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def short_text(s: str, width: int = 120) -> str:
    s = str(s).replace("\n", " ").replace("\r", " ").strip()
    if len(s) <= width:
        return s
    return s[: width - 3] + "..."


def wrapped_title(s: str, width: int = 90) -> str:
    s = str(s).replace("\n", " ").replace("\r", " ").strip()
    return "\n".join(textwrap.wrap(s, width=width))


def safe_filename(s: str, max_len: int = 80) -> str:
    bad = '\\/:*?"<>|'
    out = "".join("_" if c in bad else c for c in str(s))
    out = out.replace("\n", " ").replace("\r", " ").strip()
    out = "_".join(out.split())
    return out[:max_len] if len(out) > max_len else out


def extract_iter(path: str) -> int:
    name = os.path.basename(path)
    m = re.search(r"iter_(\d+)", name)
    if not m:
        raise ValueError(f"Cannot parse iteration from filename: {path}")
    return int(m.group(1))


def load_all_prompt_metric_csvs(dump_dir: str) -> pd.DataFrame:
    csvs = sorted(
        glob.glob(os.path.join(dump_dir, "iter_*_prompt_metrics.csv")),
        key=extract_iter,
    )
    if not csvs:
        raise FileNotFoundError(f"No iter_*_prompt_metrics.csv files found in {dump_dir}")

    dfs = []
    for p in csvs:
        it = extract_iter(p)
        df = pd.read_csv(p)
        df["iter"] = it
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.sort_values(["prompt_index", "iter"]).reset_index(drop=True)
    return all_df


def find_indexed_cols(df: pd.DataFrame, prefix: str) -> List[Tuple[int, str]]:
    cols = []
    pat = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    for c in df.columns:
        m = pat.fullmatch(str(c))
        if m:
            cols.append((int(m.group(1)), c))
    cols.sort(key=lambda x: x[0])
    return cols


def auto_find_cols(df: pd.DataFrame, candidate_prefixes: List[str]) -> Tuple[Optional[str], List[Tuple[int, str]]]:
    for p in candidate_prefixes:
        cols = find_indexed_cols(df, p)
        if cols:
            return p, cols
    return None, []


def normalize_rows(arr: np.ndarray, mode: str = "auto") -> np.ndarray:
    x = np.asarray(arr, dtype=float)

    if mode == "auto":
        # 如果全非负，则按和归一化；否则按 softmax
        if np.nanmin(x) >= 0:
            mode = "sum"
        else:
            mode = "softmax"

    if mode == "sum":
        x = np.clip(x, 0.0, None)
        row_sum = x.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return x / row_sum

    if mode == "softmax":
        x = x - np.nanmax(x, axis=1, keepdims=True)
        ex = np.exp(x)
        row_sum = ex.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return ex / row_sum

    raise ValueError(f"Unknown normalization mode: {mode}")


def entropy_from_probs(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    q = np.clip(p, eps, 1.0)
    return -(q * np.log(q)).sum(axis=1)


def tv_from_probs(p: np.ndarray) -> np.ndarray:
    tv = np.full(p.shape[0], np.nan, dtype=float)
    for t in range(1, p.shape[0]):
        tv[t] = 0.5 * np.abs(p[t] - p[t - 1]).sum()
    return tv


def build_prompt_dirname(one_prompt_df: pd.DataFrame) -> str:
    row0 = one_prompt_df.sort_values("iter").iloc[0]
    prompt_index = int(row0["prompt_index"])
    prompt_id = row0["prompt_id"] if "prompt_id" in one_prompt_df.columns else None

    parts = [f"prompt_{prompt_index:04d}"]
    if prompt_id is not None and pd.notna(prompt_id):
        try:
            parts.append(f"pid_{int(prompt_id):04d}")
        except Exception:
            parts.append(f"pid_{safe_filename(prompt_id, 20)}")
    return "_".join(parts)


def extract_matrix(df: pd.DataFrame, indexed_cols: List[Tuple[int, str]]) -> np.ndarray:
    if not indexed_cols:
        return np.empty((len(df), 0))
    col_names = [c for _, c in indexed_cols]
    return df[col_names].to_numpy(dtype=float)


def build_long_df(df: pd.DataFrame, matrix: np.ndarray, indexed_cols: List[Tuple[int, str]], value_name: str) -> pd.DataFrame:
    rows = []
    resp_indices = [j for j, _ in indexed_cols]
    iters = df["iter"].tolist()

    for r, it in enumerate(iters):
        for c, resp_idx in enumerate(resp_indices):
            rows.append(
                {
                    "iter": int(it),
                    "resp_idx": int(resp_idx),
                    value_name: float(matrix[r, c]) if np.isfinite(matrix[r, c]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_pi_trajectories(long_df: pd.DataFrame, y_col: str, out_path: str, title: str, ylabel: str):
    if long_df.empty:
        return

    plt.figure(figsize=(11, 6))
    for resp_idx in sorted(long_df["resp_idx"].unique()):
        sub = long_df[long_df["resp_idx"] == resp_idx].sort_values("iter")
        if len(sub) == 0:
            continue
        plt.plot(
            sub["iter"],
            sub[y_col],
            marker="o",
            linewidth=1.7,
            markersize=3,
            label=f"resp {resp_idx}",
        )

    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(wrapped_title(title))
    if plt.gca().lines:
        plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_tv_entropy(iters: np.ndarray, tv: np.ndarray, ent: np.ndarray, out_path: str, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(iters, tv, marker="o", linewidth=1.8, markersize=3)
    axes[0].set_title("TV from previous iteration")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("TV distance")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iters, ent, marker="o", linewidth=1.8, markersize=3)
    axes[1].set_title("Entropy")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Entropy")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(wrapped_title(title), fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def choose_raw_and_norm(
    one_prompt_df: pd.DataFrame,
    raw_prefix: Optional[str],
    norm_prefix: Optional[str],
) -> Tuple[str, List[Tuple[int, str]], str, List[Tuple[int, str]]]:
    raw_candidates = [raw_prefix] if raw_prefix else [
        "raw_prob_",
        "unnorm_prob_",
        "pi_raw_",
        "raw_pi_",
        "score_",
        "logit_",
    ]
    norm_candidates = [norm_prefix] if norm_prefix else [
        "prob_",
        "pi_",
        "norm_prob_",
        "normalized_prob_",
    ]

    raw_found_prefix, raw_cols = auto_find_cols(one_prompt_df, raw_candidates)
    norm_found_prefix, norm_cols = auto_find_cols(one_prompt_df, norm_candidates)

    if not raw_cols and not norm_cols:
        raise ValueError(
            "Cannot find any indexed probability-like columns. "
            "Please pass --raw_prefix / --norm_prefix explicitly."
        )

    # 只有 normalized，没有 raw：那 raw 图就退化成直接用 normalized
    if not raw_cols and norm_cols:
        raw_found_prefix = f"{norm_found_prefix}(fallback)"
        raw_cols = norm_cols

    # 只有 raw，没有 normalized：后面自动从 raw 算 normalized
    if raw_cols and not norm_cols:
        norm_found_prefix = "computed_from_raw"
        norm_cols = []

    return raw_found_prefix, raw_cols, norm_found_prefix, norm_cols


def save_one_prompt_outputs(
    one_prompt_df: pd.DataFrame,
    base_out_dir: str,
    raw_prefix: Optional[str],
    norm_prefix: Optional[str],
    norm_mode: str,
):
    one_prompt_df = one_prompt_df.sort_values("iter").reset_index(drop=True)

    raw_found_prefix, raw_cols, norm_found_prefix, norm_cols = choose_raw_and_norm(
        one_prompt_df, raw_prefix, norm_prefix
    )

    raw_mat = extract_matrix(one_prompt_df, raw_cols)

    if norm_cols:
        norm_mat = extract_matrix(one_prompt_df, norm_cols)
    else:
        norm_mat = normalize_rows(raw_mat, mode=norm_mode)

    # 再保险归一化一次，避免原始 prob_ 因数值误差不严格和为1
    norm_mat = normalize_rows(norm_mat, mode="sum")

    # TV / entropy
    if "tv_delta" in one_prompt_df.columns:
        tv = one_prompt_df["tv_delta"].to_numpy(dtype=float)
    else:
        tv = tv_from_probs(norm_mat)

    if "entropy" in one_prompt_df.columns:
        ent = one_prompt_df["entropy"].to_numpy(dtype=float)
    else:
        ent = entropy_from_probs(norm_mat)

    row0 = one_prompt_df.iloc[0]
    prompt_index = int(row0["prompt_index"])
    prompt_id = row0["prompt_id"] if "prompt_id" in one_prompt_df.columns else None
    prompt_text = str(row0["prompt"]) if "prompt" in one_prompt_df.columns else ""
    prompt_short = short_text(prompt_text, 150)

    prompt_dir = os.path.join(base_out_dir, build_prompt_dirname(one_prompt_df))
    ensure_dir(prompt_dir)

    meta = {
        "prompt_index": prompt_index,
        "prompt_id": int(prompt_id) if (prompt_id is not None and pd.notna(prompt_id)) else None,
        "prompt": prompt_text,
        "num_iterations": int(one_prompt_df["iter"].nunique()),
        "num_responses": int(norm_mat.shape[1]),
        "raw_prefix_used": raw_found_prefix,
        "norm_prefix_used": norm_found_prefix,
        "normalization_mode_for_computed_norm": norm_mode,
    }
    with open(os.path.join(prompt_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    out_trace = one_prompt_df[["iter"]].copy()
    for j, (_, col) in enumerate(raw_cols):
        out_trace[f"raw_resp_{j}"] = raw_mat[:, j]
    for j in range(norm_mat.shape[1]):
        out_trace[f"norm_resp_{j}"] = norm_mat[:, j]
    out_trace["tv"] = tv
    out_trace["entropy"] = ent
    out_trace.to_csv(os.path.join(prompt_dir, "prompt_dynamics.csv"), index=False)

    raw_long = build_long_df(one_prompt_df, raw_mat, raw_cols, "value")
    norm_long = build_long_df(
        one_prompt_df,
        norm_mat,
        [(j, f"norm_{j}") for j in range(norm_mat.shape[1])],
        "value",
    )

    title_prefix = f"Prompt index = {prompt_index}"
    if prompt_id is not None and pd.notna(prompt_id):
        try:
            title_prefix += f", prompt_id = {int(prompt_id)}"
        except Exception:
            title_prefix += f", prompt_id = {prompt_id}"
    title_prefix += f"\nPrompt: {prompt_short}"

    plot_pi_trajectories(
        raw_long,
        "value",
        os.path.join(prompt_dir, "pi_raw_over_time.png"),
        f"{title_prefix}\nUnnormalized π over iterations",
        "Unnormalized π",
    )

    plot_pi_trajectories(
        norm_long,
        "value",
        os.path.join(prompt_dir, "pi_normalized_over_time.png"),
        f"{title_prefix}\nNormalized π over iterations",
        "Normalized π",
    )

    plot_tv_entropy(
        one_prompt_df["iter"].to_numpy(dtype=int),
        tv,
        ent,
        os.path.join(prompt_dir, "tv_entropy_over_time.png"),
        f"{title_prefix}\nTV and entropy over iterations",
    )

    return {
        "prompt_index": prompt_index,
        "prompt_id": int(prompt_id) if (prompt_id is not None and pd.notna(prompt_id)) else np.nan,
        "prompt_short": prompt_short,
        "num_iterations": int(one_prompt_df["iter"].nunique()),
        "num_responses": int(norm_mat.shape[1]),
        "plot_dir": prompt_dir,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, required=True, help="Directory containing iter_*_prompt_metrics.csv")
    parser.add_argument("--out_dir", type=str, default=None, help="Default: <dump_dir>/prompt_pi_plots")
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--prompt_contains", type=str, default=None)
    parser.add_argument("--raw_prefix", type=str, default=None, help="Example: raw_prob_")
    parser.add_argument("--norm_prefix", type=str, default=None, help="Example: prob_")
    parser.add_argument(
        "--norm_mode",
        type=str,
        default="auto",
        choices=["auto", "sum", "softmax"],
        help="How to normalize raw π if normalized columns are absent",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.dump_dir, "prompt_pi_plots")
    ensure_dir(out_dir)

    all_df = load_all_prompt_metric_csvs(args.dump_dir)

    prompt_groups = []
    for _, g in all_df.groupby("prompt_index", sort=True):
        g = g.sort_values("iter").reset_index(drop=True)

        if args.prompt_contains is not None:
            ptxt = str(g["prompt"].iloc[0]) if "prompt" in g.columns else ""
            if args.prompt_contains.lower() not in ptxt.lower():
                continue

        prompt_groups.append(g)

    if args.max_prompts is not None:
        prompt_groups = prompt_groups[: args.max_prompts]

    print(f"[INFO] total prompts to plot: {len(prompt_groups)}")
    print(f"[INFO] outputs will be written to: {out_dir}")

    summaries = []
    for i, g in enumerate(prompt_groups, start=1):
        row0 = g.iloc[0]
        prompt_index = int(row0["prompt_index"])
        try:
            summary = save_one_prompt_outputs(
                g,
                out_dir,
                raw_prefix=args.raw_prefix,
                norm_prefix=args.norm_prefix,
                norm_mode=args.norm_mode,
            )
            summaries.append(summary)
        except Exception as e:
            print(f"[WARN] failed for prompt_index={prompt_index}: {e}")

        if i % 50 == 0 or i == len(prompt_groups):
            print(f"[INFO] processed {i}/{len(prompt_groups)} prompts")

    if summaries:
        pd.DataFrame(summaries).to_csv(
            os.path.join(out_dir, "selected_prompts_summary.csv"),
            index=False,
        )

    print(f"[DONE] all plots written to: {out_dir}")


if __name__ == "__main__":
    main()