import os
import re
import glob
import json
import math
import argparse
import textwrap

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


def safe_filename(s: str, max_len: int = 100) -> str:
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


def load_all_prompt_metric_csvs(dump_dir: str):
    csvs = sorted(
        glob.glob(os.path.join(dump_dir, "iter_*_prompt_metrics.csv")),
        key=extract_iter,
    )
    if not csvs:
        raise FileNotFoundError(
            f"No iter_*_prompt_metrics.csv files found in {dump_dir}"
        )

    dfs = []
    for p in csvs:
        it = extract_iter(p)
        df = pd.read_csv(p)
        df["iter"] = it
        dfs.append(df)

    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.sort_values(["iter", "prompt_index"]).reset_index(drop=True)
    return all_df


def choose_prompt(df: pd.DataFrame, prompt_index=None, prompt_id=None, prompt_contains=None):
    sub = df.copy()

    if prompt_index is not None:
        sub = sub[sub["prompt_index"] == prompt_index]
    if prompt_id is not None and "prompt_id" in sub.columns:
        sub = sub[sub["prompt_id"] == prompt_id]
    if prompt_contains is not None:
        sub = sub[sub["prompt"].astype(str).str.contains(prompt_contains, case=False, na=False)]

    if len(sub) == 0:
        raise ValueError("No prompt matched the given selection criteria.")

    prompt_indices = sorted(sub["prompt_index"].unique())
    if len(prompt_indices) > 1:
        preview = (
            sub[["prompt_index", "prompt"]]
            .drop_duplicates("prompt_index")
            .head(10)
            .copy()
        )
        preview["prompt"] = preview["prompt"].map(lambda x: short_text(x, 100))
        raise ValueError(
            "Selection matched multiple prompts. Please make it more specific.\n"
            + preview.to_string(index=False)
        )

    pid = prompt_indices[0]
    return df[df["prompt_index"] == pid].sort_values("iter").reset_index(drop=True)


def infer_prob_cols(df: pd.DataFrame):
    prob_cols = []
    for c in df.columns:
        m = re.fullmatch(r"prob_(\d+)", str(c))
        if m:
            prob_cols.append((int(m.group(1)), c))
    prob_cols = sorted(prob_cols, key=lambda x: x[0])
    return prob_cols


def save_response_mapping(one_prompt_df: pd.DataFrame, out_dir: str):
    row0 = one_prompt_df.sort_values("iter").iloc[0]
    prob_cols = infer_prob_cols(one_prompt_df)

    rows = []
    for j, prob_col in prob_cols:
        resp_col = f"response_{j}"
        resp_text = row0[resp_col] if resp_col in one_prompt_df.columns else ""
        rows.append({
            "resp_idx": j,
            "response": resp_text,
            "response_short": short_text(resp_text, 200),
        })

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "response_mapping.csv"), index=False)


def build_long_prob_df(one_prompt_df: pd.DataFrame):
    prob_cols = infer_prob_cols(one_prompt_df)
    rows = []
    for _, row in one_prompt_df.iterrows():
        it = int(row["iter"])
        for j, prob_col in prob_cols:
            resp_col = f"response_{j}"
            resp_text = row[resp_col] if resp_col in one_prompt_df.columns else ""
            prob = row[prob_col]
            rows.append({
                "iter": it,
                "resp_idx": j,
                "response": resp_text,
                "response_short": short_text(resp_text, 120),
                "prob": float(prob) if pd.notna(prob) else np.nan,
            })
    long_df = pd.DataFrame(rows)
    return long_df


def plot_prob_trajectories(long_df: pd.DataFrame, out_path: str, title: str):
    plt.figure(figsize=(11, 6))
    for resp_idx in sorted(long_df["resp_idx"].unique()):
        sub = long_df[long_df["resp_idx"] == resp_idx].sort_values("iter")
        label = f"resp {resp_idx}"
        plt.plot(sub["iter"], sub["prob"], marker="o", linewidth=1.8, markersize=3, label=label)

    plt.xlabel("Iteration")
    plt.ylabel("Conditional probability")
    plt.title(wrapped_title(title))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_metric_line(df: pd.DataFrame, y_col: str, out_path: str, title: str, ylabel: str):
    if y_col not in df.columns:
        return
    tmp = df[["iter", y_col]].dropna()
    if len(tmp) == 0:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(tmp["iter"], tmp[y_col], marker="o", linewidth=1.8, markersize=3)
    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.title(wrapped_title(title))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_top1(df: pd.DataFrame, out_path: str, title: str):
    if "top1_idx" not in df.columns:
        return
    tmp = df[["iter", "top1_idx"]].dropna()
    if len(tmp) == 0:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(tmp["iter"], tmp["top1_idx"], marker="o", linewidth=1.8, markersize=3)
    plt.xlabel("Iteration")
    plt.ylabel("Top-1 response index")
    plt.title(wrapped_title(title))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_exposure(df: pd.DataFrame, out_path: str, title: str):
    cols = [c for c in ["exposure_iter", "cum_exposure", "recent_exposure"] if c in df.columns]
    if not cols:
        return

    plt.figure(figsize=(11, 6))
    for c in cols:
        tmp = df[["iter", c]].dropna()
        if len(tmp) > 0:
            plt.plot(tmp["iter"], tmp[c], marker="o", linewidth=1.8, markersize=3, label=c)

    plt.xlabel("Iteration")
    plt.ylabel("Exposure")
    plt.title(wrapped_title(title))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_flags(df: pd.DataFrame, out_path: str, title: str):
    cols = [c for c in ["converged", "oscillatory", "resolved", "exposure_eligible"] if c in df.columns]
    if not cols:
        return

    plt.figure(figsize=(11, 6))
    for c in cols:
        tmp = df[["iter", c]].dropna()
        if len(tmp) > 0:
            plt.step(tmp["iter"], tmp[c], where="post", linewidth=1.8, label=c)

    plt.xlabel("Iteration")
    plt.ylabel("Flag value")
    plt.ylim(-0.05, 1.05)
    plt.title(wrapped_title(title))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_overview_panel(one_prompt_df: pd.DataFrame, long_df: pd.DataFrame, out_path: str, title: str):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. probability trajectories
    ax = axes[0, 0]
    for resp_idx in sorted(long_df["resp_idx"].unique()):
        sub = long_df[long_df["resp_idx"] == resp_idx].sort_values("iter")
        ax.plot(sub["iter"], sub["prob"], marker="o", linewidth=1.5, markersize=2.5, label=f"resp {resp_idx}")
    ax.set_title("Conditional prob")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Prob")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 2. entropy
    ax = axes[0, 1]
    if "entropy" in one_prompt_df.columns:
        tmp = one_prompt_df[["iter", "entropy"]].dropna()
        if len(tmp) > 0:
            ax.plot(tmp["iter"], tmp["entropy"], marker="o", linewidth=1.5, markersize=2.5)
    ax.set_title("Entropy")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Entropy")
    ax.grid(True, alpha=0.3)

    # 3. TV
    ax = axes[0, 2]
    if "tv_delta" in one_prompt_df.columns:
        tmp = one_prompt_df[["iter", "tv_delta"]].dropna()
        if len(tmp) > 0:
            ax.plot(tmp["iter"], tmp["tv_delta"], marker="o", linewidth=1.5, markersize=2.5)
    ax.set_title("TV from prev")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("TV")
    ax.grid(True, alpha=0.3)

    # 4. KL
    ax = axes[1, 0]
    if "kl_delta" in one_prompt_df.columns:
        tmp = one_prompt_df[["iter", "kl_delta"]].dropna()
        if len(tmp) > 0:
            ax.plot(tmp["iter"], tmp["kl_delta"], marker="o", linewidth=1.5, markersize=2.5)
    ax.set_title("KL from prev")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("KL")
    ax.grid(True, alpha=0.3)

    # 5. top1
    ax = axes[1, 1]
    if "top1_idx" in one_prompt_df.columns:
        tmp = one_prompt_df[["iter", "top1_idx"]].dropna()
        if len(tmp) > 0:
            ax.plot(tmp["iter"], tmp["top1_idx"], marker="o", linewidth=1.5, markersize=2.5)
    ax.set_title("Top-1 index")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Index")
    ax.grid(True, alpha=0.3)

    # 6. exposure
    ax = axes[1, 2]
    for c in ["exposure_iter", "cum_exposure", "recent_exposure"]:
        if c in one_prompt_df.columns:
            tmp = one_prompt_df[["iter", c]].dropna()
            if len(tmp) > 0:
                ax.plot(tmp["iter"], tmp[c], marker="o", linewidth=1.5, markersize=2.5, label=c)
    ax.set_title("Exposure")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle(wrapped_title(title), fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, required=True, help="Directory containing iter_*_prompt_metrics.csv")
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--prompt_index", type=int, default=None)
    parser.add_argument("--prompt_id", type=int, default=None)
    parser.add_argument("--prompt_contains", type=str, default=None)

    parser.add_argument("--save_trace_csv", type=int, default=1)
    args = parser.parse_args()

    ensure_dir(args.out_dir)

    all_df = load_all_prompt_metric_csvs(args.dump_dir)
    one_prompt_df = choose_prompt(
        all_df,
        prompt_index=args.prompt_index,
        prompt_id=args.prompt_id,
        prompt_contains=args.prompt_contains,
    )

    one_prompt_df = one_prompt_df.sort_values("iter").reset_index(drop=True)
    long_df = build_long_prob_df(one_prompt_df)

    prompt_index = int(one_prompt_df["prompt_index"].iloc[0])
    prompt_id = one_prompt_df["prompt_id"].iloc[0] if "prompt_id" in one_prompt_df.columns else None
    prompt_text = str(one_prompt_df["prompt"].iloc[0])
    prompt_short = short_text(prompt_text, 150)

    meta = {
        "prompt_index": prompt_index,
        "prompt_id": int(prompt_id) if pd.notna(prompt_id) else None,
        "prompt": prompt_text,
        "num_iterations": int(one_prompt_df["iter"].nunique()),
        "num_responses": int(long_df["resp_idx"].nunique()),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    save_response_mapping(one_prompt_df, args.out_dir)

    if args.save_trace_csv == 1:
        long_df.to_csv(os.path.join(args.out_dir, "response_prob_trace.csv"), index=False)
        one_prompt_df.to_csv(os.path.join(args.out_dir, "prompt_metrics_trace.csv"), index=False)

    title_prefix = f"Prompt index = {prompt_index}"
    if prompt_id is not None and not pd.isna(prompt_id):
        title_prefix += f", prompt_id = {int(prompt_id)}"
    title_prefix += f"\nPrompt: {prompt_short}"

    plot_prob_trajectories(
        long_df,
        os.path.join(args.out_dir, "conditional_prob.png"),
        f"{title_prefix}\nConditional probability of each response across iterations",
    )

    plot_metric_line(
        one_prompt_df,
        "entropy",
        os.path.join(args.out_dir, "entropy.png"),
        f"{title_prefix}\nPrompt entropy across iterations",
        "Entropy",
    )

    plot_metric_line(
        one_prompt_df,
        "tv_delta",
        os.path.join(args.out_dir, "tv_from_prev.png"),
        f"{title_prefix}\nTV distance from previous iteration",
        "TV distance",
    )

    plot_metric_line(
        one_prompt_df,
        "kl_delta",
        os.path.join(args.out_dir, "kl_from_prev.png"),
        f"{title_prefix}\nKL divergence from previous iteration",
        "KL divergence",
    )

    plot_top1(
        one_prompt_df,
        os.path.join(args.out_dir, "top1_idx.png"),
        f"{title_prefix}\nTop-1 response index across iterations",
    )

    plot_exposure(
        one_prompt_df,
        os.path.join(args.out_dir, "exposure.png"),
        f"{title_prefix}\nExposure statistics across iterations",
    )

    plot_flags(
        one_prompt_df,
        os.path.join(args.out_dir, "flags.png"),
        f"{title_prefix}\nConvergence / oscillation / exposure flags",
    )

    plot_overview_panel(
        one_prompt_df,
        long_df,
        os.path.join(args.out_dir, "overview_panel.png"),
        f"{title_prefix}\nSingle-prompt dynamics overview",
    )

    print(f"[DONE] plots written to: {args.out_dir}")
    print(f"[DONE] selected prompt_index = {prompt_index}")
    if prompt_id is not None and not pd.isna(prompt_id):
        print(f"[DONE] selected prompt_id = {int(prompt_id)}")
    print(f"[DONE] prompt: {prompt_short}")


if __name__ == "__main__":
    main()