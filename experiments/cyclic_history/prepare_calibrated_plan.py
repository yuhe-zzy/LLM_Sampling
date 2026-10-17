"""Build a CPU-calibrated, unapproved micropilot from existing Qwen score dumps.

Raw text stays in the user's original source dataset. Output plans store panel
IDs, score arrays, roles, hashes, and explicit predictions, not credentials.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from calibrated_protocol import (PROTOCOL, STABLE_MAX, UNSTABLE_MIN, contract_hash,
                                 verify_calibration)
from history_math import softmax, support_hash
from population_calibration import analyze, balanced_cycle, exact_trajectory


def prepare(snapshot_path, support_path, manifest_path, out, min_panels=6):
    out = Path(out)
    if out.exists():
        raise FileExistsError("Use a new output directory; do not overwrite frozen plans")
    source = json.loads(Path(support_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if support_hash(source) != manifest["support_sha256"]:
        raise ValueError("Source manifest/support hash mismatch")
    with np.load(snapshot_path, allow_pickle=False) as data:
        scores = data["sequence_sum_logprob"].copy()
        lengths = data["response_token_count"].copy()
    if scores.shape != (len(source), 4) or lengths.shape != scores.shape:
        raise ValueError("Snapshot/support shape mismatch")
    q = softmax(scores)
    eligible = np.flatnonzero(q.min(1) >= .005)
    candidates, selected = [], []
    # Candidate coefficients are disclosed, fixed before any new neural output.
    beta_by_method = dict(ipo=.2, dpo=.8)
    roles = [np.lexsort((np.arange(4), scores[i], lengths[i])).tolist() for i in range(len(source))]
    for i in eligible:
        row = dict(panel_index=int(i), prompt_id=source[i]["prompt_id"], predictions={})
        passed = True
        for method, beta in beta_by_method.items():
            for sign in (1, -1):
                key = f"{method}_{sign:+d}"
                try:
                    result = analyze(method, balanced_cycle(roles=roles[i], orientation=sign),
                                     scores[i], .9, .8, beta, .45, .5)
                    row["predictions"][key] = result
                    r = result["radii"]
                    passed &= (r["ordinary"] >= UNSTABLE_MIN and r["lagged_reference"] <= STABLE_MAX
                               and r["oracle_feedback_extrapolation"] <= STABLE_MAX)
                except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                    row["predictions"][key] = {"unresolved": str(exc)}
                    passed = False
        row["passed"] = bool(passed)
        candidates.append(row)
        if passed:
            selected.append(int(i))
    report = dict(status="BLOCKED_INSUFFICIENT_SUPPORT", input_panels=len(source),
                  initial_probability_floor=.005, eligible_count=len(eligible),
                  selected_indices=selected, selected_prompt_ids=[source[i]["prompt_id"] for i in selected],
                  minimum_panels=min_panels, beta_by_method=beta_by_method,
                  alpha=.9, lambda_current=.8, nu=.45, kappa=.5,
                  thresholds=dict(ordinary_min=UNSTABLE_MIN, intervention_max=STABLE_MAX),
                  source_snapshot_sha256=hashlib.sha256(Path(snapshot_path).read_bytes()).hexdigest(),
                  source_support_sha256=manifest["support_sha256"], candidates=candidates,
                  interpretation="Selected synthetic mechanism micropilot, not representative LLM evidence",
                  orientation_role="token length, initial sequence score, then source index; fixed once",
                  coherence_caveat="Aligned/mixed feature-role labels do not prove LoRA tangent coherence")
    out.mkdir(parents=True)
    if len(selected) < min_panels or len(selected) < 2:
        (out / "calibration_report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    panels = [copy.deepcopy(source[i]) for i in selected]
    n = len(panels)
    common = dict(manifest["config"])
    for key in ("run_id", "method", "scheme", "nu", "kappa", "protocol", "plan_status"):
        common.pop(key, None)
    common.update(model_path="model/Qwen2.5-1.5B",
                  eval_path="data/processed/eval_prompt_responses_cyclic_1000.jsonl",
                  output_root="outputs/cyclic_history_calibrated_v2",
                  alpha=.9, lambda_current=.8, seed=0, num_prompts=n,
                  panel_ids=[p["prompt_id"] for p in panels], iters=30,
                  pair_mode="all_unordered", pairs_per_prompt=6, epochs_per_iter=10,
                  cycle_probability=.8, checkpoint_every=5)
    common["calibration"] = dict(source_panel_sha256=support_hash(panels),
                                initial_sequence_scores=scores[selected].tolist(),
                                response_token_counts=lengths[selected].tolist(),
                                response_roles=[roles[i] for i in selected],
                                initial_centered_score_atol=.01,
                                source_snapshot_sha256=report["source_snapshot_sha256"])
    mixed = np.ones(n, dtype=int)
    mixed[np.random.default_rng(123).permutation(n)[:n // 2]] = -1
    runs = []
    for method, beta in beta_by_method.items():
        for arm, scheme, nu, kappa, factor, signs, role in (
                ("ordinary", "ordinary", 0., 0., 1., np.ones(n, dtype=int), "ordinary_unstable"),
                ("reference", "lagged_reference", .45, 0., 1., np.ones(n, dtype=int), "ordinary_unstable"),
                ("feedback", "oracle_feedback_extrapolation", 0., .5, 1., np.ones(n, dtype=int), "ordinary_unstable"),
                ("stable", "ordinary", 0., 0., 2., np.ones(n, dtype=int), "ordinary_stable"),
                ("mixed", "ordinary", 0., 0., 1., mixed, "ordinary_unstable")):
            row = dict(run_id=f"{method}_{arm}_calibrated_s0", method=method, scheme=scheme,
                       nu=nu, kappa=kappa, beta_train=beta * factor,
                       orientations=signs.tolist(), prediction_role=role)
            transformed = copy.deepcopy(panels)
            for panel, i, sign in zip(transformed, selected, signs):
                panel["preference_matrix"] = balanced_cycle(roles=roles[i], orientation=int(sign)).tolist()
            row["transformed_support_sha256"] = support_hash(transformed)
            cfg = dict(common, **row)
            row["calibration_contract_sha256"] = contract_hash(cfg)
            cfg.update(row)
            predictions = verify_calibration(transformed, cfg)
            row["predictions"] = predictions
            runs.append(row)
    plan = dict(status="CALIBRATED_MICROPILOT_REQUIRES_EXPLICIT_APPROVAL", protocol=PROTOCOL,
                common=common, runs=runs)
    report.update(status="CALIBRATED_MICROPILOT_NOT_APPROVED", prepared_runs=len(runs),
                  full_scale_status="NOT_READY: only a small selected subset; no representative claim",
                  optimizer_steps_per_round=int(np.ceil(n * 6 * common["epochs_per_iter"] / common["grad_accum"])))
    # Save both actual-start and small-perturbation exact trajectories. Neither
    # changes the LLM initialization; perturbations are CPU-only local checks.
    trajectories = {}
    for run in runs:
        for j, i in enumerate(selected):
            p = balanced_cycle(roles=roles[i], orientation=run["orientations"][j])
            fixed = np.asarray(run["predictions"][j]["fixed_logits"])
            for start_name, initial in (("actual_start", scores[i]),
                                         ("local_probe", fixed + np.array([1., -2., 3., -2.]) * 1e-4)):
                key = f"{run['run_id']}_prompt{source[i]['prompt_id']}_{start_name}"
                trajectories[key] = exact_trajectory(run["method"], p, scores[i], initial,
                    .9, .8, run["beta_train"], steps=200, nu=run["nu"], kappa=run["kappa"])
    np.savez_compressed(out / "exact_population_trajectories.npz", **trajectories)
    (out / "experiment_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (out / "calibration_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path)
    parser.add_argument("--support", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--min-panels", type=int, default=6)
    args = parser.parse_args()
    if args.min_panels < 2:
        parser.error("At least two distinct panels are needed for the orientation control")
    report = prepare(args.snapshot, args.support, args.manifest, args.out, args.min_panels)
    print(json.dumps({k: v for k, v in report.items() if k != "candidates"}, indent=2))


if __name__ == "__main__":
    main()
