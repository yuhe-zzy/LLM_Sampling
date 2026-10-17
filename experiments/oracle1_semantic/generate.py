"""Restartable open-response banks; completed responses never regenerated."""
from collections import Counter
import gc
import json
from pathlib import Path
import time
import numpy as np

from runtime import (Journal, boundary_stop, checked, contract, digest, exclusive, jsonl, load_json,
                     qwen_prefix, require_gpu, sha256, write_json)
from generation_recovery import generation_seed
from run_cyclic_history import stable_attention
from common import save_jsonl


def expected(plan, panels, arm, step):
    spec = plan['evaluation']
    for panel in panels:
        for draw in range(spec['responses_per_prompt']):
            yield dict(id=f'{arm}:{step}:{panel["prompt_key"]}:{draw}', run_id=arm, step=step,
                prompt_key=panel['prompt_key'], prompt_id=panel['prompt_id'], prompt=panel['prompt'],
                draw=draw, seed=generation_seed(spec['baseline_seed'] if arm=='baseline' else spec['checkpoint_seed'],
                                               step, panel['prompt_id'], draw))


def bank(root, identity, requests, input_identity):
    """Validate a bank against exact response identities, not only its row count."""
    root = Path(root)
    path = root/'bank.jsonl'
    manifest = load_json(root/'COMPLETE.json')
    rows = jsonl(path)
    if manifest['identity'] != identity or manifest['input_identity'] != input_identity or manifest['sha256'] != sha256(path):
        raise ValueError('Generation bank provenance mismatch')
    if len(rows) != len(requests) or manifest['count'] != len(rows):
        raise ValueError('Generation bank count mismatch')
    for row, request in zip(rows, requests):
        if any(row[k] != v for k, v in request.items()) or not isinstance(row['response'], str):
            raise ValueError('Generation response identity mismatch')
    return rows


def run(root, arm, wall_seconds=64800):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    require_gpu(1)
    plan, lock, identity = contract(root)
    if arm not in ['baseline']+[r['run_id'] for r in plan['runs']]:
        raise ValueError('Unknown generation arm')
    spec = plan['evaluation']
    panels = jsonl(root/'data/evaluation.jsonl')
    steps = [0] if arm == 'baseline' else spec['steps']
    target = Path(plan['output_root'])/'generations'/arm
    started = time.monotonic()
    stop = boundary_stop()
    with exclusive(target):
        tok = AutoTokenizer.from_pretrained(lock['policy']['path'], local_files_only=True)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        for step in steps:
            adapter, adapter_hash = None, None
            if arm != 'baseline':
                checkpoint = Path(plan['output_root'])/'training'/arm/f'step_{step:04d}'
                checked(checkpoint, identity)
                adapter = checkpoint/'adapter'
                adapter_hash = sha256(checkpoint/'COMMIT.json')
            input_identity = digest(dict(identity=identity, arm=arm, step=step, adapter=adapter_hash, settings=spec))
            directory = target/f'step_{step:04d}'
            requests = list(expected(plan, panels, arm, step))
            if (directory/'COMPLETE.json').exists():
                bank(directory, identity, requests, input_identity)
                continue
            journal = Journal(directory/'rows', input_identity)
            rows = []
            model = None
            for request in requests:
                signature = digest(request)
                row = journal.get(request['id'], signature)
                if row is None:
                    if stop['requested'] or time.monotonic()-started >= wall_seconds:
                        write_json(target/'progress.json', dict(state='SEGMENT_COMPLETE', step=step, rows=len(rows)))
                        return
                    if model is None:
                        model = AutoModelForCausalLM.from_pretrained(lock['policy']['path'], local_files_only=True,
                            torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda')
                        if adapter:
                            model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
                        model.eval()
                    ids = qwen_prefix(tok, request['prompt'])
                    if len(ids)+spec['max_new_tokens'] > plan['training']['max_length']:
                        raise ValueError('Generation would exceed no-truncation limit')
                    torch.manual_seed(request['seed'])
                    torch.cuda.manual_seed_all(request['seed'])
                    inputs = torch.tensor([ids], device='cuda')
                    with torch.inference_mode(), stable_attention():
                        result = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                            do_sample=True, temperature=spec['temperature'], top_p=spec['top_p'], top_k=spec['top_k'],
                            max_new_tokens=spec['max_new_tokens'], pad_token_id=tok.pad_token_id,
                            eos_token_id=tok.eos_token_id, use_cache=True)
                    tokens = result[0, len(ids):].tolist()
                    eos = bool(tokens and tokens[-1] == tok.eos_token_id)
                    content = tokens[:-1] if eos else tokens
                    counts = Counter(content)
                    row = dict(**request, response=tok.decode(tokens, skip_special_tokens=True),
                        response_tokens=len(content), ended_with_eos=eos,
                        hit_max_new_tokens=len(tokens)==spec['max_new_tokens'] and not eos,
                        unique_token_ratio=len(counts)/max(1,len(content)),
                        dominant_token_fraction=max(counts.values(), default=0)/max(1,len(content)))
                    journal.put(request['id'], signature, row)
                    del result, inputs
                rows.append(row)
                if len(rows)%50 == 0:
                    progress = dict(arm=arm, step=step, rows=len(rows), expected=len(requests),
                                    cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated())
                    write_json(target/'progress.json', progress)
                    print(json.dumps(progress), flush=True)
            path = directory/'bank.jsonl'
            if path.exists():
                if jsonl(path) != rows:
                    raise ValueError('Uncommitted bank differs from preserved response journal')
            else:
                save_jsonl(path, rows)
            write_json(directory/'COMPLETE.json', dict(identity=identity, input_identity=input_identity,
                       count=len(rows), sha256=sha256(path), adapter_commit_sha256=adapter_hash))
            if model is not None:
                del model
                gc.collect()
                torch.cuda.empty_cache()
        write_json(target/'COMPLETE.json', dict(identity=identity, steps=steps))


def collect(root):
    plan, _, identity = contract(root)
    panels = jsonl(root/'data/evaluation.jsonl')
    rows = []
    for arm in ['baseline']+[r['run_id'] for r in plan['runs']]:
        for step in ([0] if arm == 'baseline' else plan['evaluation']['steps']):
            directory = Path(plan['output_root'])/'generations'/arm/f'step_{step:04d}'
            adapter_hash = None
            if arm != 'baseline':
                checkpoint = Path(plan['output_root'])/'training'/arm/f'step_{step:04d}'
                checked(checkpoint, identity)
                adapter_hash = sha256(checkpoint/'COMMIT.json')
            input_identity = digest(dict(identity=identity, arm=arm, step=step, adapter=adapter_hash,
                                         settings=plan['evaluation']))
            rows.extend(bank(directory, identity, list(expected(plan, panels, arm, step)), input_identity))
    return rows
