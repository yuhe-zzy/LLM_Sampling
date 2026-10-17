"""Pin/download Skywork; inventory existing policy/Nemotron without replacing them."""
import argparse
import hashlib
import json
from pathlib import Path

SKYWORK = 'Skywork/Skywork-Reward-Llama-3.1-8B-v0.2'


def fingerprint(root):
    root = Path(root)
    small = {}
    weights = {}
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.name.startswith('.'):
            continue
        if path.suffix == '.safetensors':
            meta = root / '.cache/huggingface/download' / (path.name + '.metadata')
            if not meta.is_file():
                raise ValueError(f'Missing download provenance: {meta}')
            lines = meta.read_text().splitlines()
            weights[path.name] = dict(bytes=path.stat().st_size, revision=lines[0], etag=lines[1])
        elif path.suffix in ('.json', '.py', '.jinja', '.txt', '.model'):
            small[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not weights or 'config.json' not in small:
        raise ValueError(f'Incomplete model directory: {root}')
    return dict(path=str(root.resolve()), files_sha256=small, weights=weights,
                weight_verification='download_metadata_etag_and_size; not a fresh weight rehash')


def main():
    from huggingface_hub import HfApi, snapshot_download
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve the existing model lock; do not repin silently')
    revision = HfApi().model_info(SKYWORK).sha
    print(json.dumps(dict(skywork_revision=revision)), flush=True)
    target = a.model_root / ('Skywork-Reward-Llama-3.1-8B-v0.2-' + revision[:12])
    snapshot_download(SKYWORK, revision=revision, local_dir=str(target), max_workers=2,
                      allow_patterns=['*.json', '*.safetensors', '*.jinja', '*.txt', 'README.md', 'LICENSE*'])
    result = {
        'skywork': dict(repo_id=SKYWORK, revision=revision, **fingerprint(target)),
        'nemotron': dict(repo_id='nvidia/Llama-3.1-Nemotron-70B-Reward-HF',
                         **fingerprint(a.model_root / 'Llama-3.1-Nemotron-70B-Reward-HF')),
        'policy': fingerprint(a.model_root / 'Qwen2.5-1.5B'),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(dict(model_lock=str(a.output), complete=True)), flush=True)


if __name__ == '__main__':
    main()
