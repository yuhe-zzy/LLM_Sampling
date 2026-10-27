"""Recompute exact-population trajectories from a frozen plan, without data/LLM/GPU."""
import argparse
import json
from pathlib import Path

import numpy as np

from calibrated_protocol import contract_hash
from population_calibration import analyze, balanced_cycle, exact_trajectory


def replay(plan_path, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plan = json.loads(Path(plan_path).read_text())
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    summary, trajectories = [], {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for run in plan['runs']:
        cfg = dict(plan['common'], **run)
        if contract_hash(cfg) != cfg['calibration_contract_sha256']:
            raise ValueError('Plan parameters changed after calibration')
        distances, radii = [], []
        for pid, ref, roles, sign, stored in zip(cfg['panel_ids'], cfg['calibration']['initial_sequence_scores'],
                cfg['calibration']['response_roles'], cfg['orientations'], cfg['predictions']):
            p = balanced_cycle(cfg['cycle_probability'], sign, roles)
            result = analyze(cfg['method'], p, ref, cfg['alpha'], cfg['lambda_current'],
                             cfg['beta_train'], cfg['nu'], cfg['kappa'])
            np.testing.assert_allclose(result['fixed_logits'], stored['fixed_logits'], atol=1e-6, rtol=0)
            radius = result['radii'][cfg['scheme']]
            radii.append(radius)
            trajectory = exact_trajectory(cfg['method'], p, ref, ref, cfg['alpha'], cfg['lambda_current'],
                                           cfg['beta_train'], steps=200, nu=cfg['nu'], kappa=cfg['kappa'])
            distance = np.linalg.norm(trajectory - result['fixed_logits'], axis=1)
            distances.append(distance)
            trajectories[f"{run['run_id']}_prompt{pid}"] = trajectory
        distances = np.asarray(distances)
        expected_stable = cfg['scheme'] != 'ordinary' or cfg['prediction_role'] == 'ordinary_stable'
        if expected_stable and np.max(distances[:, -1]) > .002:
            raise ValueError(f"Actual-start recovery check failed for {run['run_id']}")
        if not expected_stable and np.min(distances[:, -1]) < .1:
            raise ValueError(f"Ordinary expansion not observed for {run['run_id']}")
        summary.append(dict(run_id=run['run_id'], radius_min=min(radii), radius_max=max(radii),
                            final_distance_max=float(distances[:, -1].max()),
                            final_distance_min=float(distances[:, -1].min()),
                            actual_start_check='PASS', steps=200))
        ax = axes[0 if cfg['method'] == 'ipo' else 1]
        ax.semilogy(np.maximum(np.median(distances, axis=0), 1e-12), label=run['run_id'].split('_')[1])
    for ax, name in zip(axes, ['IPO', 'Actual DPO']):
        ax.set(title=name, xlabel='Exact population outer step')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel('Median centered-logit distance to own fixed point')
    fig.suptitle('Selected 6-panel synthetic control; not neural training')
    fig.tight_layout()
    fig.savefig(out / 'population_recovery.png', dpi=170)
    fig.savefig(out / 'population_recovery.pdf')
    plt.close(fig)
    np.savez_compressed(out / 'population_replay.npz', **trajectories)
    (out / 'population_replay_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.plan, args.out), indent=2))
