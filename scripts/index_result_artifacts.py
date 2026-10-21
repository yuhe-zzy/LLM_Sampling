"""Generate a dated path/size/hash inventory, without copying result contents."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess


def inventory(repo):
    files = []
    paths = [p for p in sorted((repo/'experiments').rglob('*'))
             if p.is_file() and p.suffix.lower() in ('.png', '.pdf', '.csv')]
    names = [p.relative_to(repo).as_posix() for p in paths]
    result = subprocess.run(['git', 'check-attr', '-z', '--stdin', 'text'], cwd=repo,
                            input=('\0'.join(names)+'\0').encode(), capture_output=True, check=True)
    fields = result.stdout.decode().strip('\0').split('\0')
    attrs = {fields[i]: fields[i+2] for i in range(0, len(fields), 3)}
    for path in paths:
        data = path.read_bytes()
        name = path.relative_to(repo).as_posix()
        mode = 'lf_normalized' if path.suffix.lower() == '.csv' and attrs[name] != 'unset' else 'source_bytes'
        if mode == 'lf_normalized':
            data = data.replace(b'\r\n', b'\n')
        files.append(dict(path=name, bytes=len(data), hash_encoding=mode,
                          sha256=hashlib.sha256(data).hexdigest()))
    return dict(scope='Public experiment PNG/PDF/CSV artifacts; not raw server coverage',
                hash_encoding='Per-file Git text attributes respected; binary and explicitly -text files unchanged',
                file_count=len(files), total_bytes=sum(f['bytes'] for f in files),
                by_extension=dict(sorted(Counter(Path(f['path']).suffix for f in files).items())),
                files=files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--date', required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    result = dict(publication_date=args.date, **inventory(repo))
    target = repo / 'docs' / f'result_artifacts_{args.date.replace("-", "")}.json'
    text = json.dumps(result, indent=2) + '\n'
    if target.exists():
        assert target.read_text(encoding='utf-8') == text, 'Use a new dated inventory for changed artifacts'
    else:
        target.write_text(text, encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'files'}))
