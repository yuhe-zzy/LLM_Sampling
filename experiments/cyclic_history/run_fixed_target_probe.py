"""Fixed-target inner-fit probe from completed Stage A adapters; no outer updates."""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import time

import numpy as np

from history_math import build_outer_state, centered, pair_distribution, support_hash
from run_cyclic_history import (encode_panel, score_panel, train_round, validate_config,
                                write_json, write_metrics)

RUNS = ('ipo_ordinary_calibrated_s0', 'ipo_stable_calibrated_s0',
        'dpo_ordinary_calibrated_s0', 'dpo_stable_calibrated_s0')
CORE = ('history_math.py', 'population_calibration.py', 'calibrated_protocol.py',
        'run_cyclic_history.py')
SOURCE_STEP = 20
BLOCKS = 6
SCORE_ATOL = .01


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(path):
    with np.load(path, allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in archive.files}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError('Non-finite source snapshot')
    return result


def frozen_state(cfg, matrices, initial, current, previous, saved_target_state):
    state = build_outer_state(initial, current, previous, matrices, cfg['method'],
                              cfg['alpha'], cfg['lambda_current'], cfg['beta_train'],
                              cfg['nu'], cfg['kappa'])
    for key, value in state.items():
        np.testing.assert_allclose(value, saved_target_state['training_' + key],
                                   rtol=0, atol=1e-7, err_msg='Saved target differs: ' + key)
        value.setflags(write=False)
    return state


def prepare(source):
    source = Path(source).resolve()
    manifest = json.loads((source / 'manifest.json').read_text())
    cfg = manifest['config']
    validate_config(cfg)
    if source.name not in RUNS or cfg['run_id'] != source.name:
        raise ValueError('Only the four approved Stage A source runs are allowed')
    if manifest['state'] != 'COMPLETED' or manifest['last_complete_step'] != 30:
        raise ValueError('Stage A source must be complete; never probe an active run')
    expected = dict(alpha=.9, lambda_current=.8, seed=0, epochs_per_iter=10,
                    pair_mode='all_unordered', num_prompts=6, keep_k=4,
                    scheme='ordinary', nu=0, kappa=0, batch_size=1, grad_accum=4)
    if any(cfg[key] != value for key, value in expected.items()):
        raise ValueError('Source config is outside the approved probe protocol')
    beta = dict(zip(RUNS, (.2, .4, .8, 1.6)))[source.name]
    if cfg['beta_train'] != beta:
        raise ValueError('Unexpected source beta')
    for name in CORE:
        if sha(Path(__file__).parent / name) != manifest['source_sha256'][name]:
            raise ValueError('Source-matched training code required: ' + name)
    panels = json.loads((source / 'support.json').read_text())
    if support_hash(panels) != manifest['support_sha256']:
        raise ValueError('Support hash mismatch')
    states = {step: snapshot(source / 'snapshots' / f'step_{step:04d}.npz')
              for step in (0, SOURCE_STEP - 1, SOURCE_STEP, SOURCE_STEP + 1)}
    matrices = np.array([panel['preference_matrix'] for panel in panels])
    state = frozen_state(cfg, matrices, states[0]['sequence_sum_logprob'],
                         states[20]['sequence_sum_logprob'], states[19]['sequence_sum_logprob'],
                         states[21])
    adapter = source / 'adapters' / f'step_{SOURCE_STEP:04d}'
    for name in ('adapter_config.json', 'adapter_model.safetensors'):
        if not (adapter / name).is_file() or (adapter / name).stat().st_size == 0:
            raise ValueError('Missing/nonempty checkpoint required: ' + name)
    return dict(source=source, manifest=manifest, cfg=cfg, panels=panels, states=states,
                matrices=matrices, state=state, adapter=adapter)


def pair_objectives(scores, matrices, state, cfg):
    result = []
    for x, pref, mu, ref in zip(scores, matrices, state['mu'], state['effective_reference']):
        left, right, weights = pair_distribution(mu)
        delta = (x[left] - x[right]) - (ref[left] - ref[right])
        p = pref[left, right]
        if cfg['method'] == 'ipo':
            a = 1 / (2 * cfg['beta_train'])
            loss = p * (delta - a)**2 + (1-p) * (delta + a)**2
        else:
            logits = cfg['beta_train'] * delta
            loss = np.logaddexp(0, logits) - p * logits
        result.append(np.sum(weights * loss))
    return np.asarray(result)


def measure(scores, start, target):
    current = centered(scores)
    intended = target - centered(start)
    error = current - target
    actual = current - centered(start)
    rms = lambda x: float(np.sqrt(np.mean(x**2)))
    intended_norm = float(np.linalg.norm(intended))
    actual_norm = float(np.linalg.norm(actual))
    scale = intended_norm**2
    fraction = float(np.sum(actual * intended)/scale) if scale else None
    cosine = (float(np.sum(actual * intended)/(actual_norm*intended_norm))
              if actual_norm and intended_norm else None)
    return dict(target_error_rms=rms(error), intended_update_rms=rms(intended),
                error_to_intended_ratio=float(np.linalg.norm(error)/intended_norm) if intended_norm else None,
                actual_update_rms=rms(actual), update_cosine=cosine,
                effective_update_fraction=fraction,
                off_direction_rms=rms(actual-fraction*intended) if fraction is not None else None), error


def train_blocks(state, train, record, blocks=BLOCKS):
    immutable = {key: value.copy() for key, value in state.items()}
    for block in range(1, blocks + 1):
        diagnostics = train(block)
        for key, value in state.items():
            np.testing.assert_array_equal(value, immutable[key], err_msg='Frozen state mutated')
        record(block, diagnostics)


def execute(context, out):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Exactly one approved Slurm GPU is required')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('Source requires BF16 support')
    source, cfg, state = context['source'], context['cfg'], context['state']
    versions = {p: version(p) for p in ('torch', 'transformers', 'peft', 'numpy')}
    if versions != context['manifest']['versions']:
        raise RuntimeError('Runtime versions differ from Stage A; review before training')
    out = Path(out).resolve()
    if out == source or source in out.parents or out in source.parents:
        raise ValueError('Probe output must be separate from the source run')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'snapshots').mkdir()
    manifest = dict(protocol='cyclic_fixed_target_probe_v1', state='INITIALIZING',
                    source_run=str(source), source_step=SOURCE_STEP, config=cfg,
                    blocks=BLOCKS, epochs_per_block=10, optimizer_steps_per_block=90,
                    outer_updates=0, reset_optimizer_each_block=True,
                    source_step_score_atol=SCORE_ATOL,
                    slurm_job_id=os.environ['SLURM_JOB_ID'], completed_blocks=0,
                    versions=versions,
                    source_sha256={p: sha(Path(__file__).parent/p) for p in (*CORE, Path(__file__).name)},
                    checkpoint_sha256={p.name: sha(p) for p in context['adapter'].glob('adapter*') if p.is_file()})
    write_json(out / 'manifest.json', manifest)
    np.savez_compressed(out / 'frozen_outer_state.npz', **state)
    write_json(out / 'support.json', context['panels'])
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cudnn.benchmark = False
    try:
        tok = AutoTokenizer.from_pretrained(cfg['model_path'], local_files_only=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        encoded = [row for p in context['panels'] for row in encode_panel(tok, p, cfg['max_length'])]
        audit = json.loads((source / 'tokenization_audit.json').read_text())
        if [row['response_tokens'] for row in encoded] != audit['response_token_counts']:
            raise ValueError('Response tokenization differs from source')
        if [row['truncated_prompt_tokens'] for row in encoded] != audit['truncated_prompt_tokens']:
            raise ValueError('Prompt tokenization differs from source')
        base = AutoModelForCausalLM.from_pretrained(cfg['model_path'], torch_dtype=torch.bfloat16,
                    local_files_only=True, attn_implementation='sdpa').to('cuda')
        base.config.use_cache = False
        model = PeftModel.from_pretrained(base, str(context['adapter']), is_trainable=True,
                                         local_files_only=True)
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
        if not any(p.requires_grad for p in model.parameters()):
            raise RuntimeError('Loaded adapter is not trainable')
        start = context['states'][20]['sequence_sum_logprob']
        target = state['target_logits']
        lengths = context['states'][20]['response_token_count']
        score = lambda: score_panel(model, encoded, tok.pad_token_id, 'cuda', cfg['score_batch_size'], start.shape)
        scores, counts = score()
        np.testing.assert_array_equal(counts, lengths)
        np.testing.assert_allclose(scores, start, rtol=0, atol=SCORE_ATOL,
                                   err_msg='Checkpoint replay score mismatch; no training performed')
        manifest['checkpoint_replay_max_abs_error'] = float(abs(scores-start).max())
        manifest['state'] = 'RUNNING'
        write_json(out / 'manifest.json', manifest)
        rows, prompt_rows = [], []
        optimum = pair_objectives(target, context['matrices'], state, cfg)

        def record(block, diagnostics, cached=None):
            current, token_counts = score() if cached is None else cached
            np.testing.assert_array_equal(token_counts, lengths)
            summary, error = measure(current, start, target)
            objectives = pair_objectives(current, context['matrices'], state, cfg)
            row = dict(block=block, cumulative_epochs=block*10, cumulative_optimizer_steps=block*90,
                       **summary, pair_objective_mean=float(objectives.mean()),
                       population_minimum_mean=float(optimum.mean()),
                       excess_pair_objective_mean=float((objectives-optimum).mean()), **diagnostics)
            if block == 1:
                row['original_step21_replay_max_abs_error'] = float(abs(
                    current-context['states'][21]['sequence_sum_logprob']).max())
            rows.append(row)
            for j, panel in enumerate(context['panels']):
                values, _ = measure(current[j], start[j], target[j])
                prompt_rows.append(dict(block=block, cumulative_epochs=block*10,
                    prompt_id=panel['prompt_id'], **values, pair_objective=float(objectives[j]),
                    excess_pair_objective=float(objectives[j]-optimum[j])))
            np.savez_compressed(out/'snapshots'/f'block_{block:02d}.npz',
                                sequence_sum_logprob=current, target_logits=target,
                                target_error=error, response_token_count=token_counts)
            write_metrics(out, rows)
            write_json(out/'per_prompt_metrics.json', prompt_rows)
            if block in (1, 3, 6):
                checkpoint = out/'adapters'/f'epochs_{block*10:04d}'
                model.save_pretrained(checkpoint)
                tok.save_pretrained(checkpoint)
            manifest['completed_blocks'] = block
            write_json(out/'manifest.json', manifest)
            print(json.dumps(row, allow_nan=False), flush=True)

        record(0, {}, (scores, counts))

        def one_block(block):
            started = time.monotonic()
            result = train_round(model, encoded, context['matrices'], state, cfg,
                                 tok.pad_token_id, 'cuda', SOURCE_STEP)
            if result['optimizer_steps'] != 90:
                raise RuntimeError('Unexpected inner block budget')
            return dict(**result, block_train_seconds=time.monotonic()-started)

        train_blocks(state, one_block, record)
        manifest['state'] = 'COMPLETED'
        write_json(out/'manifest.json', manifest)
    except BaseException as exc:
        manifest.update(state='FAILED', error=f'{type(exc).__name__}: {exc}')
        write_json(out/'manifest.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    context = prepare(args.source_run)
    if args.execute:
        if args.output is None:
            parser.error('--execute requires a separate --output')
        execute(context, args.output)
    else:
        print(json.dumps(dict(preflight='PASS', run=context['cfg']['run_id'],
                              step=SOURCE_STEP, blocks=BLOCKS, outer_updates=0)))


if __name__ == '__main__':
    main()
