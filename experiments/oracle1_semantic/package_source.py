"""Build a reviewed small source archive, excluding text data, weights and results."""
import argparse
from pathlib import Path
import subprocess


def pack(repo, output):
    repo, output = repo.resolve(), output.resolve()
    tracked = subprocess.check_output(['git','ls-files','experiments/oracle1_semantic',
        'experiments/oracle2','experiments/cyclic_history'],cwd=repo,text=True).splitlines()
    paths = {p for p in tracked if Path(p).suffix in ('.py','.sh')}
    paths.add('experiments/oracle2/campaigns/oracle2_real_20261009/plan_two_gpu.json')
    changed = subprocess.check_output(['git','diff','HEAD','--name-only','--',*sorted(paths)],cwd=repo,text=True)
    if changed.strip():
        raise ValueError('Commit reviewed source before packaging')
    if output.exists():
        raise FileExistsError('Never replace a source archive')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Archive Git blob bytes, not platform-converted working-tree line endings.
    subprocess.run(['git','archive','--format=tar.gz','--output',str(output),'HEAD',*sorted(paths)],
                   cwd=repo,check=True)
    print(f'PACKAGED {len(paths)} files {output.stat().st_size} bytes')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    pack(a.repo, a.output)
