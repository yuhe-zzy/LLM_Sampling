"""Score private prompt/response records sequentially with frozen Nemotron and Skywork."""
import argparse
import gc
import json
from pathlib import Path
import time

from common import judge_token_ids, jsonl, load_json, require_gpu, sha256, verify_model_lock, write_json


def score(plan, records_path, output):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
    require_gpu(plan['oracle']['scoring_gpus'])
    lock = load_json(plan['model_lock'])
    verify_model_lock(lock)
    records = jsonl(records_path)
    if not records or len({r['id'] for r in records}) != len(records):
        raise ValueError('Empty records or duplicate IDs')
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(state='SCORING', records_sha256=sha256(records_path), count=len(records),
                    model_lock_sha256=sha256(plan['model_lock']), components={},
                    templates='own model chat template, tokenize=True, no duplicate BOS',
                    truncation_allowed=False)
    write_json(output / 'manifest.json', manifest)
    try:
        for name in ('nemotron', 'skywork'):
            started = time.monotonic()
            tok = AutoTokenizer.from_pretrained(lock[name]['path'], local_files_only=True,
                                                 trust_remote_code=False)
            kwargs = dict(local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16,
                          attn_implementation='sdpa')
            if name == 'nemotron':
                model = AutoModelForCausalLM.from_pretrained(lock[name]['path'], device_map='auto',
                    max_memory={i: '65GiB' for i in range(torch.cuda.device_count())}, **kwargs)
            else:
                model = AutoModelForSequenceClassification.from_pretrained(lock[name]['path'],
                    num_labels=1, device_map={'': 0}, **kwargs)
            if any(str(v) in ('cpu', 'disk') for v in getattr(model, 'hf_device_map', {}).values()):
                raise RuntimeError('Unexpected CPU/disk model offload; review allocation')
            model.eval()
            model.requires_grad_(False)
            device = model.get_input_embeddings().weight.device
            path = output / f'{name}.jsonl'
            lengths, scores = [], []
            with path.open('x', encoding='utf-8') as handle, torch.inference_mode():
                for index, row in enumerate(records):
                    ids = judge_token_ids(tok, row['prompt'], row['response'])
                    if len(ids) > plan['dataset']['max_judge_length'] or not ids:
                        raise ValueError(f'Judge input length violation: {row["id"]}')
                    inputs = torch.tensor([ids], dtype=torch.long, device=device)
                    if name == 'nemotron':
                        result = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                            do_sample=False, max_new_tokens=1, return_dict_in_generate=True, output_scores=True,
                            pad_token_id=tok.pad_token_id or tok.eos_token_id)
                        value = float(result.scores[0][0, 0].float())
                    else:
                        result = model(input_ids=inputs, attention_mask=torch.ones_like(inputs))
                        if result.logits.shape != (1, 1):
                            raise ValueError('Skywork must return one scalar per response')
                        value = float(result.logits[0, 0].float())
                    if not torch.isfinite(torch.tensor(value)):
                        raise FloatingPointError('Nonfinite judge score')
                    handle.write(json.dumps(dict(id=row['id'], reward=value, input_tokens=len(ids))) + '\n')
                    handle.flush()
                    lengths.append(len(ids))
                    scores.append(value)
                    if (index+1) % 50 == 0 or index == 0:
                        print(json.dumps(dict(component=name, scored=index+1, total=len(records),
                                              elapsed_seconds=time.monotonic()-started)), flush=True)
                    del result, inputs
            manifest['components'][name] = dict(count=len(scores), min_reward=min(scores), max_reward=max(scores),
                max_input_tokens=max(lengths), seconds=time.monotonic()-started, sha256=sha256(path),
                max_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())])
            write_json(output / 'manifest.json', manifest)
            del model, tok
            gc.collect()
            torch.cuda.empty_cache()
        manifest['state'] = 'COMPLETE'
        write_json(output / 'manifest.json', manifest)
    except BaseException as exc:
        manifest.update(state='FAILED', error=f'{type(exc).__name__}: {exc}')
        write_json(output / 'manifest.json', manifest)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--records', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    score(load_json(a.plan), a.records, a.output)


if __name__ == '__main__':
    main()
