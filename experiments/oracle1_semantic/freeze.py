"""Seal one newly deployed source tree. Never edit an existing deployment."""
import argparse
from pathlib import Path
from runtime import contract, save_new, sha256, utc


def freeze(root, commit):
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('Full source commit required')
    if (root/'deployment.json').exists():
        raise FileExistsError('Existing immutable deployment')
    paths = [p for p in (root/'source').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths += [root/'plan.json', root/'model_lock.json', root/'data/audit.json']
    save_new(root/'deployment.json', dict(source_commit=commit, created=utc(),
        files_sha256={str(p.relative_to(root)):sha256(p) for p in sorted(paths)}))
    plan, _, identity = contract(root)
    from train import configuration
    for arm in plan['runs']:
        configuration(plan, arm['run_id'])
    print('FROZEN_AND_VERIFIED', identity, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--commit', required=True)
    a = p.parse_args()
    freeze(a.root, a.commit)
