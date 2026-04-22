import os
import re
import glob
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
    prob_cols.sort(key=lambda x: x[0])
    return prob_cols


def build_long_prob_df(one_prompt_df: pd.DataFrame) -> pd.DataFrame:
    prob_cols = infer_prob_cols(one_prompt_df)
    rows = []

    for _, row in one_prompt_df.iterrows():
        it = int(row["iter"])
        for j, prob_col in prob_cols:
            rows.append(
                {
                    "iter": it,
                    "resp_idx": j,
                    "prob": float(row[prob_col]) if pd.notna(row[prob_col]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_output_filename(one_prompt_df: pd.DataFrame) -> str:
    row0 = one_prompt_df.sort_values("iter").iloc[0]
    prompt_index = int(row0["prompt_index"])
    prompt_id = row0["prompt_id"] if "prompt_id" in one_prompt_df.columns else None

    if prompt_id is not None and pd.notna(prompt_id):
        try:
            return f"prompt_{prompt_index:04d}_pid_{int(prompt_id):04d}.png"
        except Exception:
            return f"prompt_{prompt_index:04d}_pid_{safe_filename(prompt_id, 20)}.png"
    return f"prompt_{prompt_index:04d}.png"


def plot_single_prompt_figure(one_prompt_df: pd.DataFrame, out_path: str):
    one_prompt_df = one_prompt_df.sort_values("iter").reset_index(drop=True)
    long_df = build_long_prob_df(one_prompt_df)

    row0 = one_prompt_df.iloc[0]
    prompt_index = int(row0["prompt_index"])
    prompt_id = row0["prompt_id"] if "prompt_id" in one_prompt_df.columns else None
    prompt_text = str(row0["prompt"]) if "prompt" in one_prompt_df.columns else ""
    prompt_short = short_text(prompt_text, 140)

    fig, axes = plt.subplots(3, 1, figsize=(13, 14), sharex=True)

    # 1) pi trajectories
    ax = axes[0]
    if not long_df.empty:
        for resp_idx in sorted(long_df["resp_idx"].unique()):
            sub = long_df[long_df["resp_idx"] == resp_idx].sort_values("iter")
            if len(sub) > 0:
                ax.plot(
                    sub["iter"],
                    sub["prob"],
                    marker="o",
                    linewidth=1.5,
                    markersize=2.8,
                    label=f"resp {resp_idx}",
                )
    ax.set_ylabel("pi")
    ax.set_title("pi over iterations")
    ax.grid(True, alpha=0.3)
    if ax.lines:
        ax.legend(fontsize=8, ncol=2)

    # 2) entropy
    ax = axes[1]
    if "entropy" in one_prompt_df.columns:
        tmp = one_prompt_df[["iter", "entropy"]].dropna()
        if len(tmp) > 0:
            ax.plot(
                tmp["iter"],
                tmp["entropy"],
                marker="o",
                linewidth=1.7,
                markersize=3,
            )
    ax.set_ylabel("entropy")
    ax.set_title("entropy over iterations")
    ax.grid(True, alpha=0.3)

    # 3) tv
    ax = axes[2]
    if "tv_delta" in one_prompt_df.columns:
        tmp = one_prompt_df[["iter", "tv_delta"]].dropna()
        if len(tmp) > 0:
            ax.plot(
                tmp["iter"],
                tmp["tv_delta"],
                marker="o",
                linewidth=1.7,
                markersize=3,
            )
    ax.set_xlabel("iteration")
    ax.set_ylabel("tv")
    ax.set_title("TV from previous iteration")
    ax.grid(True, alpha=0.3)

    title_prefix = f"Prompt index = {prompt_index}"
    if prompt_id is not None and pd.notna(prompt_id):
        try:
            title_prefix += f", prompt_id = {int(prompt_id)}"
        except Exception:
            title_prefix += f", prompt_id = {prompt_id}"
    title_prefix += f"\nPrompt: {prompt_short}"

    fig.suptitle(wrapped_title(title_prefix), fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump_dir", type=str, required=True, help="Directory containing iter_*_prompt_metrics.csv")
    parser.add_argument("--out_dir", type=str, default=None, help="Default: <dump_dir_basename>_prompt_figures")
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument("--prompt_contains", type=str, default=None)
    args = parser.parse_args()

    dump_dir_norm = os.path.normpath(args.dump_dir)
    dump_base = os.path.basename(dump_dir_norm)

    out_dir = args.out_dir or os.path.join(os.path.dirname(dump_dir_norm), f"{dump_base}_prompt_figures")
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

    summary_rows = []
    for i, g in enumerate(prompt_groups, start=1):
        row0 = g.iloc[0]
        prompt_index = int(row0["prompt_index"])

        try:
            filename = build_output_filename(g)
            out_path = os.path.join(out_dir, filename)
            plot_single_prompt_figure(g, out_path)

            summary_rows.append(
                {
                    "prompt_index": prompt_index,
                    "prompt_id": row0["prompt_id"] if "prompt_id" in g.columns else np.nan,
                    "file": filename,
                }
            )
        except Exception as e:
            print(f"[WARN] failed for prompt_index={prompt_index}: {e}")

        if i % 50 == 0 or i == len(prompt_groups):
            print(f"[INFO] processed {i}/{len(prompt_groups)} prompts")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(out_dir, "figure_index.csv"),
            index=False,
        )

    print(f"[DONE] all figures written to: {out_dir}")


if __name__ == "__main__":
    main()