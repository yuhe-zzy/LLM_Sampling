"""Train one approved real-panel arm using the tested sequence-sum history engine."""
import argparse
from pathlib import Path

from common import jsonl, load_json, require_gpu, sha256, verify_model_lock
from history_math import validate_preferences
from run_cyclic_history import train, validate_config
from oracle_math import wr_steps


def configuration(plan, run_id):
    run = next(r for r in plan['runs'] if r['run_id'] == run_id)
    cfg = dict(plan['training'], **run, protocol=plan['protocol'], output_root=plan['output_root'])
    validate_config(cfg)
    if plan['evaluation']['steps'] != wr_steps(cfg['iters'], plan['evaluation']['every_outer_iterations']):
        raise ValueError('WR checkpoint grid inconsistent')
    if cfg['checkpoint_every'] != plan['evaluation']['every_outer_iterations']:
        raise ValueError('Every WR checkpoint must be saved')
    if cfg['epochs_per_iter'] != 1 or cfg['pair_mode'] != 'all_unordered':
        raise ValueError('This first campaign requires one epoch over all soft pairs')
    return cfg


def validate_review(audit, review, audit_digest):
    if review.get('decision') != 'APPROVE_SIX_ARMS' or review.get('audit_sha256') != audit_digest:
        raise ValueError('A recorded review of this exact audit is required; no automatic temperature tuning')
    scope = review.get('analysis_scope', 'cyclic_vs_transitive')
    if scope not in ('cyclic_vs_transitive', 'empirical_stability_trends'):
        raise ValueError('Unrecognized analysis scope')
    if scope == 'empirical_stability_trends':
        observed = {split: {g: audit['splits'][split]['groups'].get(g, 0)
                           for g in ('cyclic', 'transitive', 'ambiguous')}
                    for split in ('calibration', 'train', 'evaluation')}
        if (review.get('user_authorized_sparse_cycles') is not True or
                review.get('observed_groups') != observed or
                review.get('interpretation_limits') != 'no_cyclic_subgroup_claim_or_convergence_proof'):
            raise ValueError('Empirical scope requires explicit user authorization and exact group-count acknowledgement')
    else:
        for split in ('train', 'evaluation'):
            if any(audit['splits'][split]['groups'].get(g, 0) == 0 for g in ('cyclic', 'transitive')):
                raise ValueError('Missing comparison group; review experimental design before training')
    if audit['splits']['train']['bt_solver_failures']:
        raise ValueError('DPO projection failed preflight; do not clip probabilities silently')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--review', type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    plan = load_json(a.plan)
    cfg = configuration(plan, a.run_id)
    scored = Path(plan['data_root']) / 'scored'
    audit = load_json(scored / 'audit.json')
    review = load_json(a.review)
    validate_review(audit, review, sha256(scored / 'audit.json'))
    if audit['plan_sha256'] != sha256(a.plan) or audit['model_lock_sha256'] != sha256(plan['model_lock']):
        raise ValueError('Plan or judge lock changed since audit')
    if sha256(scored / 'train.jsonl') != audit['files_sha256']['train.jsonl']:
        raise ValueError('Frozen training support changed')
    panels = jsonl(scored / 'train.jsonl')
    if len(panels) != cfg['num_prompts']:
        raise ValueError('Training size mismatch')
    for panel in panels:
        validate_preferences(panel['preference_matrix'])
    lock = load_json(plan['model_lock'])
    verify_model_lock(lock)
    cfg.update(model_path=lock['policy']['path'], source_commit=a.source_commit,
               plan_sha256=sha256(a.plan), audit_sha256=sha256(scored / 'audit.json'),
               model_lock_sha256=sha256(plan['model_lock']),
               initial_score_policy='fresh finite EOS-included sequence sums on verified real candidates; no synthetic target',
               oracle=plan['oracle'], evaluation=plan['evaluation'],
               audit_review=review, audit_review_sha256=sha256(a.review))
    if not a.execute:
        print('CHECKED ONLY: no model loaded, no training started')
        return
    require_gpu(1)
    # The original history engine is unchanged. This protocol intentionally has
    # no synthetic fixed-point calibration gate; the real data audit replaces it.
    train(cfg, panels)


if __name__ == '__main__':
    main()
