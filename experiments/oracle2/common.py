"""Artifact integrity and shared imports for the isolated real-candidate campaign."""
import hashlib
import json
import os
from pathlib import Path
import sys

HISTORY = Path(__file__).resolve().parents[1] / 'cyclic_history'
sys.path.insert(0, str(HISTORY))
from run_cyclic_history import write_json  # noqa: E402


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def jsonl(path):
    with open(path, encoding='utf-8') as handle:
        return [json.loads(line) for line in handle if line.strip()]


def save_jsonl(path, rows):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temp = Path(str(path) + '.tmp')
    with temp.open('x', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(temp, path)


def verify_model_lock(lock):
    from prepare_models import fingerprint
    for item in lock.values():
        now = fingerprint(item['path'])
        for key in ('files_sha256', 'weights'):
            if now[key] != item[key]:
                raise ValueError(f'Model changed since locking: {item["path"]}: {key}')


def require_gpu(count):
    import torch
    if not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count() != count:
        raise RuntimeError(f'Requires an approved Slurm allocation with exactly {count} visible GPUs')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('BF16 support required')
