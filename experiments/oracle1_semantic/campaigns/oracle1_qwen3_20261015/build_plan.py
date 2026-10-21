"""Build the explicitly approved Oracle1-only, five-trait, Qwen3 campaign plan."""
import argparse
import json
from pathlib import Path


def build(root):
    old = Path(__file__).resolve().parents[3]/'oracle2/campaigns/oracle2_real_20261009/plan_two_gpu.json'
    original = json.loads(old.read_text())
    training = dict(original['training'])
    # Chat-template overhead is checked without dropping/replacing approved panels.
    # If the inherited1537 limit fails preflight, stop for review, do not truncate.
    return dict(protocol='oracle1_qwen3_generated_dynamics_v1', authorization='2026-10-15',
        output_root='/work/users/y/u/yuhe32/ipo_runs/oracle1_qwen3_20261015_v1',
        launch_root=str(root).replace('\\', '/'), account_gpu_limit=4, phase_concurrency=2,
        dataset=dict(splits=dict(calibration=100, train=500, evaluation=200), keep_k=4,
                     source='reuse_hash_verified_oracle2_candidates_and_nemotron_scores_only'),
        oracle=dict(name='oracle1', component='nemotron', temperature=1., scoring_gpus=2, max_length=4096),
        training=training, runs=original['runs'],
        evaluation=dict(original['evaluation'], primary_metric='oracle1_expected_win_rate',
                        prompt_format='qwen3_chat_non_thinking_same_as_training', top_k=0),
        annotation=dict(model='Qwen/Qwen3-32B', enable_thinking=False, do_sample=False,
            max_new_tokens=256, traits=['elaboration', 'linguistic_complexity', 'formality', 'politeness', 'hedging'],
            input='response_only', scale=[0,10], pilot_panels=50, pilot_seed=0,
            human_review_required_before_full_annotation=True),
        continuation=dict(checkpoint_every_outer=1, max_updates_per_segment=10,
            initial_probe_indices=[0,3], initial_probe_updates=1,
            resume_test_updates=1, soft_wall_seconds=64800, slurm_time='1-00:00:00',
            optimizer='reset_each_outer_iteration', reuse_completed_responses=True),
        interpretation=dict(goal='empirical_generated_output_dynamics',
            dpo_feedback='exact_null_under_single_BT_reward',
            primary_axes='raw_five_traits_then_discovery_frozen_PCA_not_claimed_cyclic',
            seed_count=1, no_cycle_acceptance_gate=True))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    with a.output.open('x', encoding='utf-8') as f:
        json.dump(build(a.root), f, indent=2)
