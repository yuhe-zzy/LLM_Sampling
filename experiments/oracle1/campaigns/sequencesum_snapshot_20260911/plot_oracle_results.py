from pathlib import Path
import json
import math
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ENTROPY = "prompt_relative_sequence_entropy_mean"
PATTERN = re.compile(r"^(dpo|ipo)_transitive_a([0-9.]+)_l([0-9.]+)_b1_")
ALPHAS = [0.8, 0.9, 0.95, 0.99]
LAMBDAS = [0.5, 0.9]
CHECKPOINTS = [0, 20, 40, 60, 80]
COLORS = {0.8: "#2563eb", 0.9: "#c95037", 0.95: "#07836f", 0.99: "#8136b3"}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def load_runs():
    runs = {}
    for path in sorted(ROOT.glob("*_relative_sequence_metrics.csv")):
        match = PATTERN.match(path.name)
        if not match:
            continue
        method, alpha, lam = match.groups()
        key = (method, float(alpha), float(lam))
        frame = pd.read_csv(path).sort_values("iter")
        frame = frame.loc[frame["iter"].between(0, 80)].copy()
        if frame.empty or frame["iter"].duplicated().any():
            raise ValueError(f"Empty or duplicate iterations: {path}")
        if not (frame["primary_entropy_metric"] == ENTROPY).all():
            raise ValueError(f"Unexpected entropy definition: {path}")
        if not (frame["beta"] == 1).all():
            raise ValueError(f"Unexpected beta: {path}")
        if not np.isfinite(frame[ENTROPY]).all():
            raise ValueError(f"Nonfinite entropy: {path}")
        if not frame[ENTROPY].between(-1e-8, math.log(5) + 1e-8).all():
            raise ValueError(f"Entropy outside support bounds: {path}")
        wr = frame.dropna(subset=["oracle_win_rate"])
        if not wr["iter"].isin(CHECKPOINTS).all() or not wr["oracle_win_rate"].between(0, 1).all():
            raise ValueError(f"Unexpected WR checkpoints or values: {path}")
        raw_path = ROOT / path.name.replace("_relative_sequence_metrics.csv", "_training_metrics.csv")
        raw = pd.read_csv(raw_path)
        if not raw["outer_reference_frozen"].astype(str).str.lower().eq("true").all():
            raise ValueError(f"Outer reference is not frozen: {path}")
        raw_wr = raw.dropna(subset=["oracle_win_rate"]).set_index("iter")["oracle_win_rate"]
        for row in wr.itertuples():
            if not np.isclose(row.oracle_win_rate, raw_wr.loc[row.iter]):
                raise ValueError(f"WR disagrees with original metrics: {path}")
        final = frame.loc[frame["iter"] == 80]
        complete = bool(not final.empty and final["oracle_win_rate"].notna().all())
        if complete and set(frame["iter"]) != set(range(81)):
            raise ValueError(f"Missing entropy iterations: {path}")
        if complete and set(wr["iter"]) != set(CHECKPOINTS):
            raise ValueError(f"Missing WR checkpoints: {path}")
        if key in runs:
            raise ValueError(f"Duplicate run: {key}")
        runs[key] = dict(frame=frame, wr=wr, last=int(frame["iter"].max()), complete=complete, source=path.name)
    return runs


def plot_metric(runs, method, metric):
    is_entropy = metric == "entropy"
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.9), sharey=True, dpi=200)
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.20, top=0.80, wspace=0.15)
    title = "Relative-Sequence Entropy" if is_entropy else "Oracle Winning Rate"
    fig.suptitle(f"Oracle {method.upper()} | {title}", fontsize=18, y=0.97)
    fig.text(0.5, 0.905, "Sequence-sum training | beta = 1 | snapshot: 2026-09-11", ha="center", color="#535b65", fontsize=10)
    for ax, lam in zip(axes, LAMBDAS):
        pending = []
        for alpha in ALPHAS:
            run = runs.get((method, alpha, lam))
            if run is None:
                pending.append(f"{alpha:g}")
                continue
            frame = run["frame"] if is_entropy else run["wr"]
            label = rf"$\alpha={alpha:g}$"
            if not run["complete"]:
                label += f" (running: iter {run['last']})"
            y = frame[ENTROPY] if is_entropy else frame["oracle_win_rate"] * 100
            ax.plot(frame["iter"], y, color=COLORS[alpha], label=label,
                    lw=2.1, ls="-" if run["complete"] else "--",
                    marker=None if is_entropy else "o", ms=5)
            if is_entropy and not frame.empty:
                ax.scatter([frame["iter"].iloc[-1]], [y.iloc[-1]], color=COLORS[alpha], s=22, zorder=3)
        ax.set_title(rf"$\lambda={lam:g}$", pad=12)
        ax.set_xlabel("Outer iteration")
        ax.set_xlim(-1, 81)
        ax.set_xticks(CHECKPOINTS)
        ax.grid(color="#dce1e6", alpha=0.8, linewidth=0.65)
        ax.set_axisbelow(True)
        legend_location = ("upper right" if method == "dpo" else "lower left") if is_entropy else ("lower right" if method == "dpo" else "upper left")
        ax.legend(loc=legend_location, fontsize=9, framealpha=0.92)
        if is_entropy:
            ax.set_ylim(0, 1.66)
            ax.set_yticks(np.arange(0, 1.61, 0.4))
        else:
            ax.set_ylim(45, 100)
            ax.set_yticks([50, 60, 70, 80, 90, 100])
            ax.axhline(50, lw=0.85, color="#7b8490", ls=":")
        if pending:
            ax.text(0.5, -0.18, "Awaiting rerun: alpha = " + ", ".join(pending),
                    transform=ax.transAxes, ha="center", fontsize=9, color="#666f79")
    axes[0].set_ylabel("Relative-sequence entropy (nats)" if is_entropy else "Oracle winning rate (%)")
    footer = (
        r"Fixed-support entropy: $q_t(y|x) \propto \exp(\log\pi_t(y|x)-\log\pi_0(y|x))$."
        if is_entropy else "Markers show measured checkpoints; connecting lines are visual guides."
    )
    fig.text(0.075, 0.045, footer, fontsize=9, color="#535b65")
    fig.text(0.075, 0.012, "Dashed = incomplete run. Previously cancelled runs are excluded from these current-run figures.",
             fontsize=8.5, color="#535b65")
    stem = ROOT / f"oracle_{method}_{'relative_sequence_entropy' if is_entropy else 'winning_rate'}_20260911"
    fig.savefig(stem.with_suffix(".png"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def export_tables(runs):
    rows, curves = [], []
    for method in ("dpo", "ipo"):
        for lam in LAMBDAS:
            for alpha in ALPHAS:
                run = runs.get((method, alpha, lam))
                row = dict(method=method, alpha=alpha, lambda_value=lam, beta=1)
                if run is None:
                    row["status"] = "pending_rerun"
                else:
                    frame, wr = run["frame"], run["wr"]
                    row.update(status="complete" if run["complete"] else "running", latest_iter=run["last"],
                               entropy_initial=frame[ENTROPY].iloc[0], entropy_latest=frame[ENTROPY].iloc[-1],
                               latest_wr_iter=int(wr["iter"].iloc[-1]), latest_wr=wr["oracle_win_rate"].iloc[-1],
                               source_file=run["source"])
                    for checkpoint in CHECKPOINTS:
                        vals=wr.loc[wr["iter"] == checkpoint, "oracle_win_rate"]
                        row[f"wr_{checkpoint}"] = vals.iloc[0] if len(vals) else math.nan
                    curves.append(frame.assign(method=method, source_file=run["source"])[
                        ["method", "alpha", "lambda", "beta", "iter", ENTROPY, "oracle_win_rate", "source_file"]])
                rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT / "oracle_parameter_summary_20260911.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(ROOT / "oracle_curves_20260911.csv", index=False)
    summary[["method", "alpha", "lambda_value", "status"] + [f"wr_{k}" for k in CHECKPOINTS]].to_csv(
        ROOT / "oracle_winning_rate_checkpoints_20260911.csv", index=False)
    print(summary.drop(columns=["source_file", "entropy_initial"]).to_string(index=False))


if __name__ == "__main__":
    current_runs = load_runs()
    for current_method in ("dpo", "ipo"):
        for current_metric in ("entropy", "winning_rate"):
            plot_metric(current_runs, current_method, current_metric)
    export_tables(current_runs)
    print(json.dumps({"runs_plotted": len(current_runs), "complete": sum(r["complete"] for r in current_runs.values()),
                      "primary_entropy_metric": ENTROPY, "png_files": len(list(ROOT.glob("*.png")))}))
