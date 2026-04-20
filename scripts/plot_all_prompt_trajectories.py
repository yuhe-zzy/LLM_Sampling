import os
import re
import glob
import json
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


def safe_filename(s: str, max_len: int = 120) -> str:
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


def infer_prob_cols(df: pd.DataFrame):
    prob_cols = []
    for c in df.columns:
        m = re.fullmatch(r"prob_(\d+)", str(c))
        if m:
            prob_cols.append((int(m.group(1)), c))
    prob_cols = sorted(prob_cols, key=lambda x: x[0])
    return prob_cols


def build_long_prob_df(one_prompt_df: pd.DataFrame) -> pd.DataFrame:
    prob_cols = infer_prob_cols(one_prompt_df)
    rows = []
    for _, row in one_prompt_df.iterrows():
        it = int(row["iter"])
        for j, prob_col in prob_cols:
            resp_col = f"response_{j}"
            resp_text = row[resp_col] if resp_col in one_prompt_df.columns else ""
            rows.append(
                {
                    "iter": it,
                    "resp_idx": j,
                    "response": resp_text,
                    "response_short": short_text(resp_text, 120),
                    "prob": float(row[prob_col]) if pd.notna(row[prob_col]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def save_response_mapping(one_prompt_df: pd.DataFrame, out_dir: str):
    row0 = one_prompt_df.sort_values("iter").iloc[0]
    prob_cols = infer_prob_cols(one_prompt_df)

    rows = []
    for j, _ in prob_cols:
        resp_col = f"response_{j}"
        resp_text = row0[resp_col] if resp_col in one_prompt_df.columns else ""
        rows.append(
            {
                "resp_idx": j,
                "response": resp_text,
                "response_short": short_text(resp_text, 200),
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "response_mapping.csv"), index=False)


def plot_prob_trajectories(long_df: pd.DataFrame, out_path: str, title: str):
    plt.figure(figsize=(11, 6))
    for resp_idx in sorted(long_df["resp_idx"].unique()):
        sub = long_df[long_df["resp_idx"] == resp_idx].sort_values("iter")
        plt.plot(
            sub["iter"],
            sub["prob"],
            marker="o",
            linewidth=1.8,
            markersize=3,
            label=f"resp {resp_idx}",
        )
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


def plot_overview_panel(one_prompt_df: pd.DataFrame, long_df: pd.DataFrame, out_path: str, title: str):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    ax = axes[0, 0]
    for resp_idx in sorted(long_df["resp_idx"].unique()):
        sub = long_df[long_df["resp_idx"] == resp_idx].sort_values("iter")
        ax.plot(sub["iter"], sub["prob"], marker="o", linewidth=1.5, markersize=2.5, label=f"resp {resp_idx}")
    ax.set_title("Conditional prob")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Prob")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    if "entropy" in one_prompt_df.columns:
        tmp = one_prompt_df[["iter", "entropy"]].dropna()
        if len(tmp) > 0:
            ax.plot(tmp["iter"], tmp["entropy"], marker="o", linewidth=1.5, markersize=2.5)
    ax.set_title("Entropy")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Entropy")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    if "tv_delta" in one_prompt_df.columns:
        tmp = one_prompt_df[["iter", "tv_delta"]].dropna()
        if len(tmp) > 0:
            ax.plot(tmp["iter"], tmp["tv_delta"], marker="o", linewidth=1.5, markersize=2.5)
    ax.set_title("TV from prev")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("TV")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if "kl_delta" in one_prompt_df.columns:
        tmp = one_prompt_df[["iter", "kl_delta"]].dropna()
        if len(tmp) > 0:
            ax.plot(tmp["iter"], tmp["kl_delta"], marker="o", linewidth=1.5, markersize=2.5)
    ax.set_title("KL from prev")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("KL")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    if "top1_idx" in one_prompt_df.columns:
        tmp = one_prompt_df[["iter", "top1_idx"]].dropna()
        if len(tmp) > 0:
            ax.plot(tmp["iter"], tmp["top1_idx"], marker="o", linewidth=1.5, markersize=2.5)
    ax.set_title("Top-1 index")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Index")
    ax.grid(True, alpha=0.3)

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


def build_prompt_dirname(one_prompt_df: pd.DataFrame) -> str:
    row0 = one_prompt_df.sort_values("iter").iloc[0]
    prompt_index = int(row0["prompt_index"])
    prompt_id = row0["prompt_id"] if "prompt_id" in one_prompt_df.columns else None
    prompt = row0["prompt"] if "prompt" in one_prompt_df.columns else ""

    parts = [f"prompt_{prompt_index:04d}"]
    if prompt_id is not None and pd.notna(prompt_id):
        try:
            parts.append(f"pid_{int(prompt_id):04d}")
        except Exception:
            parts.append(f"pid_{safe_filename(prompt_id, 20)}")
    if isinstance(prompt, str) and len(prompt.strip()) > 0:
        parts.append(safe_filename(short_text(prompt, 60), 70))
    return "_".join(parts)


def save_one_prompt_outputs(one_prompt_df: pd.DataFrame, base_out_dir: str):
    one_prompt_df = one_prompt_df.sort_values("iter").reset_index(drop=True)
    long_df = build_long_prob_df(one_prompt_df)

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
        "num_responses": int(long_df["resp_idx"].nunique()),
    }
    with open(os.path.join(prompt_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    save_response_mapping(one_prompt_df, prompt_dir)
    long_df.to_csv(os.path.join(prompt_dir, "response_prob_trace.csv"), index=False)
    one_prompt_df.to_csv(os.path.join(prompt_dir, "prompt_metrics_trace.csv"), index=False)

    title_prefix = f"Prompt index = {prompt_index}"
    if prompt_id is not None and pd.notna(prompt_id):
        try:
            title_prefix += f", prompt_id = {int(prompt_id)}"
        except Exception:
            title_prefix += f", prompt_id = {prompt_id}"
    title_prefix += f"\nPrompt: {prompt_short}"

    plot_prob_trajectories(
        long_df,
        os.path.join(prompt_dir, "conditional_prob.png"),
        f"{title_prefix}\nConditional probability of each response across iterations",
    )
    plot_metric_line(
        one_prompt_df,
        "entropy",
        os.path.join(prompt_dir, "entropy.png"),
        f"{title_prefix}\nPrompt entropy across iterations",
        "Entropy",
    )
    plot_metric_line(
        one_prompt_df,
        "tv_delta",
        os.path.join(prompt_dir, "tv_from_prev.png"),
        f"{title_prefix}\nTV distance from previous iteration",
        "TV distance",
    )
    plot_metric_line(
        one_prompt_df,
        "kl_delta",
        os.path.join(prompt_dir, "kl_from_prev.png"),
        f"{title_prefix}\nKL divergence from previous iteration",
        "KL divergence",
    )
    plot_top1(
        one_prompt_df,
        os.path.join(prompt_dir, "top1_idx.png"),
        f"{title_prefix}\nTop-1 response index across iterations",
    )
    plot_overview_panel(
        one_prompt_df,
        long_df,
        os.path.join(prompt_dir, "overview_panel.png"),
        f"{title_prefix}\nSingle-prompt dynamics overview",
    )

    final_row = one_prompt_df.sort_values("iter").iloc[-1]
    summary = {
        "prompt_index": prompt_index,
        "prompt_id": int(prompt_id) if (prompt_id is not None and pd.notna(prompt_id)) else None,
        "prompt_short": prompt_short,
        "num_iterations": int(one_prompt_df["iter"].nunique()),
        "num_responses": int(long_df["resp_idx"].nunique()),
        "final_entropy": float(final_row["entropy"]) if "entropy" in final_row and pd.notna(final_row["entropy"]) else np.nan,
        "final_tv_delta": float(final_row["tv_delta"]) if "tv_delta" in final_row and pd.notna(final_row["tv_delta"]) else np.nan,
        "final_kl_delta": float(final_row["kl_delta"]) if "kl_delta" in final_row and pd.notna(final_row["kl_delta"]) else np.nan,
        "final_top1_idx": int(final_row["top1_idx"]) if "top1_idx" in final_row and pd.notna(final_row["top1_idx"]) else np.nan,
        "final_converged": int(final_row["converged"]) if "converged" in final_row and pd.notna(final_row["converged"]) else np.nan,
        "final_oscillatory": int(final_row["oscillatory"]) if "oscillatory" in final_row and pd.notna(final_row["oscillatory"]) else np.nan,
        "plot_dir": prompt_dir,
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, required=True, help="Directory containing iter_*_prompt_metrics.csv")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save all prompt plots")
    parser.add_argument("--max_prompts", type=int, default=None, help="Optional cap on number of prompts to plot")
    parser.add_argument("--prompt_contains", type=str, default=None, help="Only plot prompts whose text contains this substring")
    parser.add_argument("--only_oscillatory", type=int, default=0, help="If 1, only plot prompts whose final oscillatory flag is 1")
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    all_df = load_all_prompt_metric_csvs(args.dump_dir)

    prompt_groups = []
    for pid, g in all_df.groupby("prompt_index", sort=True):
        g = g.sort_values("iter").reset_index(drop=True)

        if args.prompt_contains is not None:
            ptxt = str(g["prompt"].iloc[0]) if "prompt" in g.columns else ""
            if args.prompt_contains.lower() not in ptxt.lower():
                continue

        if args.only_oscillatory == 1 and "oscillatory" in g.columns:
            final_osc = g.sort_values("iter").iloc[-1]["oscillatory"]
            if pd.isna(final_osc) or int(final_osc) != 1:
                continue

        prompt_groups.append(g)

    if args.max_prompts is not None:
        prompt_groups = prompt_groups[: args.max_prompts]

    print(f"[INFO] total prompts to plot: {len(prompt_groups)}")

    summaries = []
    for i, g in enumerate(prompt_groups, start=1):
        try:
            summary = save_one_prompt_outputs(g, args.out_dir)
            summaries.append(summary)
        except Exception as e:
            row0 = g.iloc[0]
            print(f"[WARN] failed for prompt_index={row0['prompt_index']}: {e}")

        if i % 50 == 0 or i == len(prompt_groups):
            print(f"[INFO] plotted {i}/{len(prompt_groups)} prompts")

    if summaries:
        pd.DataFrame(summaries).to_csv(
            os.path.join(args.out_dir, "selected_prompts_summary.csv"),
            index=False,
        )

    print(f"[DONE] all plots written to: {args.out_dir}")


if __name__ == "__main__":
    main()