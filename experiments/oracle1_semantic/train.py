"""Outer-boundary resumable training, preserving s0, st, st-1 and RNG state."""
import argparse
import os
from pathlib import Path
import random
import signal
import time
import numpy as np

from runtime import (checked, checkpoints, contract, digest, encode, exclusive, jsonl,
                     load_json, require_gpu, seal, staged, utc, write_json)
from history_math import build_outer_state, centered, sampler
from run_cyclic_history import score_panel, train_round, save_snapshot, validate_config


def configuration(plan, run_id):
    arm = next(r for r in plan['runs'] if r['run_id'] == run_id)
    cfg = dict(plan['training'], **arm, protocol=plan['protocol'])
    validate_config(cfg)
    return cfg


def outer(initial, current, previous, panels, cfg):
    matrices = np.asarray([p['preference_matrix'] for p in panels])
    if cfg['method'] == 'ipo':
        return build_outer_state(initial, current, previous, matrices, 'ipo', cfg['alpha'],
            cfg['lambda_current'], cfg['beta_train'], cfg['nu'], cfg['kappa'])
    # A single BT reward has an exact optimizer independent of sampling weights.
    # In this setting DPO feedback is a null intervention, not numerical solver noise.
    delta = centered(np.asarray([p['nemotron_scores'] for p in panels]))/cfg['beta_train']
    reference = (1-cfg['alpha'])*initial+(cfg['alpha']-cfg['nu'])*current+cfg['nu']*previous
    return dict(mu=sampler(current, cfg['lambda_current']), reference=reference,
        effective_reference=reference, feedback_delta=delta, previous_feedback_delta=delta.copy(),
        offset=np.zeros_like(delta), target_logits=centered(reference+delta),
        solver_residual=np.zeros(len(panels)))


def rng_state(torch):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all())


def restore_rng(torch, state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def commit(root, step, identity, model, torch, initial, current, previous, lengths, row, outer_state=None):
    temp = staged(root, f'step_{step:04d}')
    model.save_pretrained(temp/'adapter')
    np.savez_compressed(temp/'state.npz', initial=initial, current=current, previous=previous, lengths=lengths)
    torch.save(rng_state(torch), temp/'rng.pt')
    metrics = save_snapshot(temp, step, current, initial, None if step == 0 else previous, lengths, outer_state)
    write_json(temp/'metrics.json', dict(metrics, **row))
    seal(temp, root/f'step_{step:04d}', identity, dict(step=step, created=utc(),
         job=os.environ['SLURM_JOB_ID'], optimizer='reset_each_outer_iteration_no_mid_round_resume'))


def run(root, run_id, max_updates, wall_seconds=64800):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    require_gpu(1)
    plan, lock, identity = contract(root)
    cfg = configuration(plan, run_id)
    target = Path(plan['output_root'])/'training'/run_id
    started = time.monotonic()
    stop = {'requested': False}
    signal.signal(signal.SIGUSR1, lambda *_: stop.update(requested=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(requested=True))
    with exclusive(target):
        committed = checkpoints(target, identity)
        if len(committed) == cfg['iters']+1:
            raise ValueError('Training already complete; no duplicate work')
        panels = jsonl(root/'data/train.jsonl')
        random.seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        torch.manual_seed(cfg['seed'])
        torch.cuda.manual_seed_all(cfg['seed'])
        torch.backends.cudnn.benchmark = False
        tok = AutoTokenizer.from_pretrained(lock['policy']['path'], local_files_only=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        encoded = [row for panel in panels for row in encode(tok, panel, cfg['max_length'])]
        base = AutoModelForCausalLM.from_pretrained(lock['policy']['path'], local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
        base.config.use_cache = False
        if committed:
            model = PeftModel.from_pretrained(base, committed[-1]/'adapter', is_trainable=True)
        else:
            model = get_peft_model(base, LoraConfig(r=cfg['lora_r'], lora_alpha=cfg['lora_alpha'],
                lora_dropout=0., bias='none', task_type='CAUSAL_LM',
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
        shape = (len(panels), 4)
        fresh, lengths = score_panel(model, encoded, tok.pad_token_id, 'cuda', cfg['score_batch_size'], shape)
        if committed:
            step = len(committed)-1
            with np.load(committed[-1]/'state.npz', allow_pickle=False) as state:
                initial, current, previous, prior_lengths = [state[k].copy() for k in ('initial', 'current', 'previous', 'lengths')]
            if not np.array_equal(lengths, prior_lengths) or not np.allclose(fresh, current, atol=.02, rtol=0):
                raise ValueError('Reloaded adapter does not reproduce saved sequence sums/support')
            restore_rng(torch, torch.load(committed[-1]/'rng.pt', map_location='cpu', weights_only=False))
            print(f'RESUME_VERIFIED {run_id} outer={step}', flush=True)
        else:
            step = 0
            initial = fresh.copy()
            current, previous = initial.copy(), initial.copy()
            row = dict(step=0, finite=True)
            commit(target, 0, identity, model, torch, initial, current, previous, lengths, row)
        manifest = dict(identity=identity, run_id=run_id, config=cfg, last_complete_step=step,
            state='RUNNING', job=os.environ['SLURM_JOB_ID'], started=utc(),
            dpo_feedback_exact_null=cfg['method']=='dpo' and cfg['scheme']=='oracle_feedback_extrapolation')
        write_json(target/'manifest.json', manifest)
        stop_at = min(cfg['iters'], step+max_updates)
        matrices = np.asarray([p['preference_matrix'] for p in panels])
        try:
            while step < stop_at and not stop['requested'] and time.monotonic()-started < wall_seconds:
                begin = time.monotonic()
                state = outer(initial, current, previous, panels, cfg)
                diagnostics = train_round(model, encoded, matrices, state, cfg, tok.pad_token_id, 'cuda', step)
                updated, new_lengths = score_panel(model, encoded, tok.pad_token_id, 'cuda', cfg['score_batch_size'], shape)
                if not np.array_equal(lengths, new_lengths):
                    raise ValueError('Token support changed')
                step += 1
                previous, current = current.copy(), updated.copy()
                row = dict(step=step, **diagnostics, wall_seconds=time.monotonic()-begin,
                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
                commit(target, step, identity, model, torch, initial, current, previous, lengths, row, state)
                manifest.update(last_complete_step=step, last_metrics=row)
                write_json(target/'manifest.json', manifest)
                print(__import__('json').dumps(dict(run_id=run_id, **row)), flush=True)
            manifest['state'] = 'COMPLETE' if step == cfg['iters'] else 'SEGMENT_COMPLETE'
            write_json(target/'manifest.json', manifest)
        except BaseException as exc:
            manifest.update(state='FAILED', error=f'{type(exc).__name__}: {exc}')
            write_json(target/'manifest.json', manifest)
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--max-updates', type=int, default=10)
    a = p.parse_args()
    if not 1 <= a.max_updates <= 10:
        p.error('Segment must have1..10 outer updates')
    run(a.root, a.run_id, a.max_updates)
