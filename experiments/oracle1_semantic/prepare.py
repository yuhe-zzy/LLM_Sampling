"""Pin Qwen3 models and reuse exact Nemotron candidate scores without rejudging."""
import argparse
from pathlib import Path
import numpy as np

from runtime import encode, jsonl, load_json, qwen_prefix, save_new, sha256, utc
from common import save_jsonl, verify_model_lock
from prepare_models import fingerprint
from oracle_math import sigmoid

POLICY = 'Qwen/Qwen3-4B'
RATER = 'Qwen/Qwen3-32B'


def models(root, model_root, old_lock):
    from huggingface_hub import HfApi, snapshot_download
    root.mkdir(parents=True, exist_ok=True)
    pin = root/'model_revisions.json'
    if not pin.exists():
        save_new(pin, {key: dict(repo_id=repo, revision=HfApi().model_info(repo).sha)
                       for key, repo in [('policy', POLICY), ('rater', RATER)]})
    revisions = load_json(pin)
    old = load_json(old_lock)
    verify_model_lock({'nemotron': old['nemotron']})
    lock = {'nemotron': old['nemotron']}
    for key, item in revisions.items():
        target = model_root/(item['repo_id'].split('/')[1]+'-'+item['revision'][:12])
        print(f'DOWNLOAD {key} {item["revision"]}', flush=True)
        snapshot_download(item['repo_id'], revision=item['revision'], local_dir=str(target), max_workers=2,
            allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', '*.model', 'README.md', 'LICENSE*'])
        lock[key] = dict(**item, **fingerprint(target))
    output = root/'model_lock.json'
    if output.exists():
        if load_json(output) != lock:
            raise ValueError('Existing model lock differs')
    else:
        save_new(output, lock)
    print('MODEL_LOCK_COMPLETE', flush=True)


def data(root, old_scored, old_lock):
    from transformers import AutoTokenizer
    plan, lock = load_json(root/'plan.json'), load_json(root/'model_lock.json')
    if lock['nemotron'] != load_json(old_lock)['nemotron']:
        raise ValueError('Cannot reuse scores from a different Oracle1')
    source_audit = load_json(old_scored/'audit.json')
    tok = AutoTokenizer.from_pretrained(lock['policy']['path'], local_files_only=True)
    report = dict(created=utc(), plan_sha256=sha256(root/'plan.json'),
        model_lock_sha256=sha256(root/'model_lock.json'),
        source_audit_sha256=sha256(old_scored/'audit.json'), oracle='nemotron_only_BT_T1',
        truncation=False, files_sha256={}, lengths={})
    destination = root/'data'
    destination.mkdir(exist_ok=False)
    for split, count in plan['dataset']['splits'].items():
        source = old_scored/f'{split}.jsonl'
        if sha256(source) != source_audit['files_sha256'][source.name]:
            raise ValueError('Prior audited support changed')
        panels = jsonl(source)
        if len(panels) != count:
            raise ValueError('Unexpected split size')
        lengths = []
        for panel in panels:
            r = np.asarray(panel['nemotron_scores'], dtype=float)
            if r.shape != (4,) or not np.isfinite(r).all():
                raise ValueError('Invalid cached Oracle1 rewards')
            clean = {k: panel[k] for k in ('prompt_id', 'prompt_key', 'prompt', 'responses')}
            clean.update(nemotron_scores=r.tolist(), preference_matrix=sigmoid(r[:, None]-r[None, :]).tolist())
            panel.clear()
            panel.update(clean)
            rows = encode(tok, panel, plan['training']['max_length'])
            lengths.extend(len(row['input_ids']) for row in rows)
            if split == 'evaluation' and len(qwen_prefix(tok, panel['prompt']))+plan['evaluation']['max_new_tokens'] > plan['training']['max_length']:
                raise ValueError('Evaluation prompt exceeds no-truncation policy limit')
        save_jsonl(destination/source.name, panels)
        report['files_sha256'][source.name] = sha256(destination/source.name)
        report['lengths'][split] = dict(panels=len(panels), max_total_tokens=max(lengths))
    save_new(destination/'audit.json', report)
    print('DATA_AUDIT_COMPLETE', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['models', 'data'])
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--model-root', type=Path)
    p.add_argument('--old-lock', type=Path, required=True)
    p.add_argument('--old-scored', type=Path)
    a = p.parse_args()
    if a.phase == 'models':
        models(a.root, a.model_root, a.old_lock)
    else:
        data(a.root, a.old_scored, a.old_lock)
