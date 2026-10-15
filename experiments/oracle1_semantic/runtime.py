"""Integrity primitives for the independently frozen Oracle1/Qwen3 campaign."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
import signal
from pathlib import Path
import sys
import uuid

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS / 'oracle2'))
from common import jsonl, load_json, sha256, verify_model_lock, require_gpu
from run_cyclic_history import write_json


def utc():
    return datetime.now(timezone.utc).isoformat()


def boundary_stop():
    state = {'requested': False}
    for sig in (signal.SIGTERM, getattr(signal, 'SIGUSR1', signal.SIGTERM)):
        signal.signal(sig, lambda *_: state.update(requested=True))
    return state


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def scheduler():
    path = EXPERIMENTS / 'oracle2/campaigns/oracle2_real_20261009/queue.py'
    spec = importlib.util.spec_from_file_location('oracle1_budget_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def contract(root):
    root = Path(root)
    deployment = load_json(root / 'deployment.json')
    for name, expected in deployment['files_sha256'].items():
        if sha256(root / name) != expected:
            raise ValueError(f'Frozen deployment changed: {name}')
    plan = load_json(root / 'plan.json')
    audit = load_json(root / 'data/audit.json')
    if audit['plan_sha256'] != sha256(root / 'plan.json'):
        raise ValueError('Data audit belongs to another plan')
    if audit['model_lock_sha256'] != sha256(root / 'model_lock.json'):
        raise ValueError('Data audit belongs to another model lock')
    for split, count in plan['dataset']['splits'].items():
        path = root / 'data' / f'{split}.jsonl'
        if sha256(path) != audit['files_sha256'][path.name] or len(jsonl(path)) != count:
            raise ValueError('Frozen split hash/count mismatch')
    lock = load_json(root / 'model_lock.json')
    verify_model_lock(lock)
    identity = digest(dict(plan=sha256(root/'plan.json'), lock=sha256(root/'model_lock.json'),
                           data=sha256(root/'data/audit.json'), source=deployment['source_commit']))
    return plan, lock, identity


@contextmanager
def exclusive(root):
    """Never steal a stale lock automatically; accounting must first prove its owner exited."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'writer.lock'
    save_new(path, dict(job=os.environ.get('SLURM_JOB_ID'), pid=os.getpid(), created=utc()))
    try:
        yield
    finally:
        path.unlink()


def staged(root, name):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'.pending-{name}-{uuid.uuid4().hex}'
    path.mkdir()
    return path


def seal(temp, destination, identity, metadata):
    temp, destination = Path(temp), Path(destination)
    hashes = {str(p.relative_to(temp)).replace('\\', '/'): sha256(p)
              for p in sorted(temp.rglob('*')) if p.is_file()}
    save_new(temp/'COMMIT.json', dict(identity=identity, files=hashes, **metadata))
    if destination.exists():
        raise FileExistsError('Committed artifact already exists')
    os.rename(temp, destination)


def checked(directory, identity):
    directory = Path(directory)
    info = load_json(directory/'COMMIT.json')
    if info['identity'] != identity:
        raise ValueError('Artifact experiment identity mismatch')
    actual = {str(p.relative_to(directory)).replace('\\', '/')
              for p in directory.rglob('*') if p.is_file() and p.name != 'COMMIT.json'}
    if actual != set(info['files']):
        raise ValueError('Missing/extra artifact files')
    for name, expected in info['files'].items():
        if sha256(directory/name) != expected:
            raise ValueError(f'Artifact hash mismatch: {name}')
    return info


def checkpoints(root, identity):
    paths = sorted(Path(root).glob('step_[0-9][0-9][0-9][0-9]'))
    for expected, path in enumerate(paths):
        info = checked(path, identity)
        if info['step'] != expected or path.name != f'step_{expected:04d}':
            raise ValueError('Non-contiguous checkpoint chain')
    return paths


def qwen_prefix(tok, prompt):
    ids = tok.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=True,
        add_generation_prompt=True, enable_thinking=False, return_dict=False)
    if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids):
        raise TypeError('Expected flat chat-template IDs')
    return ids


def encode(tok, panel, maximum):
    prefix = qwen_prefix(tok, panel['prompt'])
    rows = []
    for response in panel['responses']:
        answer = tok.encode(response, add_special_tokens=False) + [tok.eos_token_id]
        if len(prefix)+len(answer) > maximum:
            raise ValueError('Full chat prompt/response exceeds length limit; no truncation')
        rows.append(dict(input_ids=prefix+answer, labels=[-100]*len(prefix)+answer,
                         response_tokens=len(answer), truncated_prompt_tokens=0))
    return rows


class Journal:
    """Per-response atomic records. A kill cannot lose or overwrite a completed response."""
    def __init__(self, root, identity):
        self.root, self.identity = Path(root), identity
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root/'IDENTITY.json'
        if path.exists():
            if load_json(path) != {'identity': identity}:
                raise ValueError('Journal identity changed')
        else:
            save_new(path, {'identity': identity})

    def path(self, key):
        return self.root/(hashlib.sha256(key.encode()).hexdigest()+'.json')

    def get(self, key, input_digest):
        path = self.path(key)
        if not path.exists():
            return None
        envelope = load_json(path)
        row = envelope['row']
        if (envelope['identity'] != self.identity or envelope['input_digest'] != input_digest
                or row['id'] != key or envelope['row_sha256'] != digest(row)):
            raise ValueError('Journal row identity/content mismatch')
        return row

    def put(self, key, input_digest, row):
        if row['id'] != key:
            raise ValueError('Response key mismatch')
        if self.path(key).exists():
            raise FileExistsError('Never replace an existing response')
        write_json(self.path(key), dict(identity=self.identity, input_digest=input_digest,
                                       row=row, row_sha256=digest(row)))
