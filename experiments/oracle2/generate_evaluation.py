"""Generate independent baseline/checkpoint response banks at outer states 0,10,...,100."""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import numpy as np

from common import jsonl, load_json, require_gpu, sha256, verify_model_lock, write_json
from run_cyclic_history import stable_attention


def generation_seed(base, step, prompt_id, draw):
    return int(np.random.SeedSequence([base, step, prompt_id, draw]).generate_state(1)[0])


def generate(plan_path, run_id=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    require_gpu(1)
    plan = load_json(plan_path)
    spec = plan['evaluation']
    lock = load_json(plan['model_lock'])
    verify_model_lock(lock)
    data = Path(plan['data_root']) / 'scored'
    audit = load_json(data / 'audit.json')
    if audit['plan_sha256'] != sha256(plan_path) or sha256(data / 'evaluation.jsonl') != audit['files_sha256']['evaluation.jsonl']:
        raise ValueError('Evaluation support or plan changed')
    panels = jsonl(data / 'evaluation.jsonl')
    if len(panels) != plan['dataset']['evaluation']:
        raise ValueError('Evaluation prompt count mismatch')
    output_root = Path(plan['output_root'])
    if run_id is not None:
        if run_id not in {r['run_id'] for r in plan['runs']}:
            raise ValueError('Unknown arm')
        run_root = output_root / run_id
        manifest = load_json(run_root / 'manifest.json')
        if manifest['state'] != 'COMPLETED' or manifest['last_complete_step'] != plan['training']['iters']:
            raise ValueError('Do not invent missing checkpoints; full training required by this batch evaluator')
        if manifest['config']['audit_sha256'] != sha256(data / 'audit.json'):
            raise ValueError('Checkpoint was trained against another oracle/support')
        steps = spec['steps']
    else:
        run_root, steps = None, [0]
    target = output_root / 'generations' / (run_id or 'baseline')
    target.mkdir(parents=True, exist_ok=False)
    status = dict(state='RUNNING', plan_sha256=sha256(plan_path),
                  audit_sha256=sha256(data / 'audit.json'), run_id=run_id or 'baseline', files={},
                  model_lock_sha256=sha256(plan['model_lock']), settings=spec)
    write_json(target / 'manifest.json', status)
    tok = AutoTokenizer.from_pretrained(lock['policy']['path'], local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    try:
        for step in steps:
            model = AutoModelForCausalLM.from_pretrained(lock['policy']['path'], local_files_only=True,
                torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
            adapter = None
            if run_root is not None:
                adapter = run_root / ('adapter_initial' if step == 0 else f'adapters/step_{step:04d}')
                model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
            model.eval()
            path = target / f'step_{step:04d}.jsonl'
            diagnostic_rows = []
            with path.open('x', encoding='utf-8') as handle, torch.inference_mode(), stable_attention():
                for prompt in panels:
                    ids = tok.encode(prompt['prompt'], add_special_tokens=False)
                    if len(ids)+spec['max_new_tokens'] > plan['dataset']['max_policy_length']:
                        raise ValueError('Generation prompt exceeds preflight length limit')
                    inputs = torch.tensor([ids], device='cuda', dtype=torch.long)
                    for draw in range(spec['responses_per_prompt']):
                        seed = generation_seed(spec['baseline_seed'] if run_id is None else spec['checkpoint_seed'],
                                               step, prompt['prompt_id'], draw)
                        torch.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                        result = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                            do_sample=True, temperature=spec['temperature'], top_p=spec['top_p'],
                            max_new_tokens=spec['max_new_tokens'], pad_token_id=tok.pad_token_id,
                            eos_token_id=tok.eos_token_id, use_cache=True)
                        tokens = result[0, len(ids):].tolist()
                        eos = bool(tokens and tokens[-1] == tok.eos_token_id)
                        content = tokens[:-1] if eos else tokens
                        maximum = (len(tokens) == spec['max_new_tokens'] and not eos)
                        counts = Counter(content)
                        stats = dict(response_tokens=len(content), hit_max_new_tokens=maximum,
                            dominant_token_fraction=max(counts.values(), default=0)/max(1, len(content)),
                            unique_token_ratio=len(counts)/max(1, len(content)), ended_with_eos=eos)
                        row = dict(id=f'{run_id or "baseline"}:{step}:{prompt["prompt_key"]}:{draw}',
                            run_id=run_id or 'baseline', step=step, draw=draw, seed=seed,
                            prompt_key=prompt['prompt_key'], prompt_id=prompt['prompt_id'],
                            group=prompt['oracle2_group'], prompt=prompt['prompt'],
                            response=tok.decode(tokens, skip_special_tokens=True), **stats)
                        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                        handle.flush()
                        diagnostic_rows.append(stats)
                        del result
            status['files'][path.name] = dict(sha256=sha256(path), count=len(diagnostic_rows),
                adapter_sha256=sha256(adapter / 'adapter_model.safetensors') if adapter else None,
                hit_max_new_tokens_fraction=float(np.mean([r['hit_max_new_tokens'] for r in diagnostic_rows])),
                dominant_token_ge_0_95_fraction=float(np.mean([r['dominant_token_fraction'] >= .95 for r in diagnostic_rows])),
                mean_unique_token_ratio=float(np.mean([r['unique_token_ratio'] for r in diagnostic_rows])))
            write_json(target / 'manifest.json', status)
            print(json.dumps(dict(run=run_id or 'baseline', completed_outer_state=step)), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        status['state'] = 'COMPLETE'
        write_json(target / 'manifest.json', status)
    except BaseException as exc:
        status.update(state='FAILED', error=f'{type(exc).__name__}: {exc}')
        write_json(target / 'manifest.json', status)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--run-id')
    a = p.parse_args()
    generate(a.plan, a.run_id)
