"""Plot corrected sequence-sum metrics without substituting legacy entropy."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ENTROPY = "prompt_relative_sequence_entropy_mean"


def collect(root):
    runs, keys = [], set()
    for path in sorted(Path(root).rglob("metrics_*.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        required = {"iter", "loss_type", "alpha", "lambda", "beta", "seed", ENTROPY}
        if required - set(frame):
            raise ValueError(f"{path}: not a corrected metric table; missing {sorted(required-set(frame))}")
        if "outer_reference_frozen" in frame:
            if not frame.outer_reference_frozen.astype(str).str.lower().eq("true").all():
                raise ValueError(f"{path}: non-frozen references must not be pooled with this protocol")
        identity = ["loss_type", "alpha", "lambda", "beta", "seed"]
        if len(frame[identity].drop_duplicates()) != 1 or frame["iter"].duplicated().any():
            raise ValueError(f"{path}: mixed runs or duplicate iteration indices")
        key = tuple(frame.iloc[0][identity].tolist())
        if key in keys:
            raise ValueError(f"Duplicate configuration {key}; select one attempt, do not merge restarts")
        keys.add(key)
        frame = frame.sort_values("iter")
        if frame.loss_type.iloc[0] not in {"ipo", "dpo"}:
            raise ValueError(f"{path}: unknown loss")
        runs.append((path, frame))
    if not runs:
        raise ValueError(f"No corrected metrics found under {root}")
    return runs


def render(runs, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = []
    for path, frame in runs:
        finite_h = frame[np.isfinite(frame[ENTROPY])]
        row = frame.iloc[-1]
        item = {key: row[key] for key in ("loss_type", "alpha", "lambda", "beta", "seed")}
        item.update(metrics_path=str(path), last_recorded_iter=int(row["iter"]),
                    last_valid_entropy_iter=int(finite_h.iloc[-1]["iter"]) if len(finite_h) else None,
                    last_valid_relative_entropy=float(finite_h.iloc[-1][ENTROPY]) if len(finite_h) else None)
        wr = frame[np.isfinite(frame.oracle_win_rate)] if "oracle_win_rate" in frame else frame.iloc[:0]
        item.update(last_wr_iter=int(wr.iloc[-1]["iter"]) if len(wr) else None,
                    last_wr=float(wr.iloc[-1].oracle_win_rate) if len(wr) else None)
        h80 = frame.loc[frame["iter"].eq(80), ENTROPY]
        wr80 = wr.loc[wr["iter"].eq(80), "oracle_win_rate"] if len(wr) else pd.Series(dtype=float)
        item["iter80_and_wr80_complete"] = bool(
            len(h80) == 1 and np.isfinite(h80.iloc[0]) and len(wr80) == 1)
        summary.append(item)
    pd.DataFrame(summary).to_csv(output / "summary.csv", index=False)
    for method in ("ipo", "dpo"):
        selected = [(p, f) for p, f in runs if f.loss_type.iloc[0] == method]
        if not selected:
            continue
        lambdas = sorted({float(f["lambda"].iloc[0]) for _, f in selected})
        for metric, name, ylabel, stride in [
            (ENTROPY, "relative_sequence_entropy", "Relative-sequence entropy (nats)", 1),
            ("oracle_win_rate", "winning_rate", "Oracle winning rate", 20),
        ]:
            if not any(metric in f and np.isfinite(f[metric]).any() for _, f in selected):
                continue
            fig, axes = plt.subplots(1, len(lambdas), figsize=(5.0 * len(lambdas), 3.7), squeeze=False)
            for ax, lam in zip(axes[0], lambdas):
                for _, frame in selected:
                    if float(frame["lambda"].iloc[0]) != lam or metric not in frame:
                        continue
                    row = frame.iloc[0]
                    steps = np.arange(0, int(frame["iter"].max()) + 1, stride)
                    values = frame.set_index("iter")[metric].reindex(steps)
                    values = values.where(np.isfinite(values))
                    ax.plot(steps, values, marker="o" if stride > 1 else None,
                            markersize=4, linewidth=1.6,
                            label=f"alpha={row.alpha:g}, beta={row.beta:g}, seed={int(row.seed)}")
                ax.set(title=f"{method.upper()} | lambda={lam:g}", xlabel="Outer update", ylabel=ylabel)
                if metric == "oracle_win_rate":
                    ax.set_ylim(0, 1)
                ax.grid(alpha=0.2)
                ax.legend(fontsize=8)
            fig.tight_layout()
            for extension in ("png", "pdf"):
                fig.savefig(output / f"{method}_{name}.{extension}", dpi=200, bbox_inches="tight")
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    render(collect(args.logs_root), args.output_dir)


if __name__ == "__main__":
    main()
