"""Export text-free, lossless numeric trajectories without changing experiment files."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np


FIELDS = ('sequence_sum_logprob', 'panel_probability', 'relative_sequence_entropy',
          'panel_entropy', 'tv', 'response_token_count')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def export(root, destination):
    plan_raw = (root / 'plan.json').read_bytes()
    plan = json.loads(plan_raw)
    deployment = json.loads((root / 'deployment.json').read_bytes())
    output = Path(plan['output_root'])
    report = dict(exported_at_utc=datetime.now(timezone.utc).isoformat(),
                  execution_source_commit=deployment['source_commit'],
                  plan_sha256=digest(plan_raw), runs={}, files_sha256={},
                  raw_text_included=False, weights_included=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    initial = None
    canonical_support = None
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        def put(name, raw):
            archive.writestr(name, raw)
            report['files_sha256'][name] = digest(raw)

        for arm in plan['runs']:
            run_id = arm['run_id']
            directory = output / run_id
            manifest_raw = (directory / 'manifest.json').read_bytes()
            manifest = json.loads(manifest_raw)
            cfg = manifest['config']
            assert manifest['state'] == 'COMPLETED'
            assert manifest['last_complete_step'] == cfg['iters'] == 100
            assert cfg['source_commit'] == deployment['source_commit']
            assert cfg['plan_sha256'] == digest(plan_raw)
            support_raw = (directory / 'support.json').read_bytes()
            support = json.loads(support_raw)
            assert len(support) == 500
            if canonical_support is None:
                canonical_support = digest(support_raw)
                metadata = [dict(panel_index=i, prompt_id=p['prompt_id'],
                                 prompt_key=p['prompt_key'], group=p['oracle2_group'],
                                 preference_matrix=p['preference_matrix'])
                            for i, p in enumerate(support)]
                put('prompt_metadata.json', json.dumps(metadata, indent=2).encode())
            else:
                assert digest(support_raw) == canonical_support
            arrays = {k: [] for k in FIELDS}
            hashes = {}
            for step in range(101):
                path = directory / 'snapshots' / f'step_{step:04d}.npz'
                raw = path.read_bytes()
                hashes[path.name] = digest(raw)
                with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                    for key in FIELDS:
                        assert np.isfinite(z[key]).all()
                        arrays[key].append(z[key].copy())
            arrays = {k: np.stack(v) for k, v in arrays.items()}
            assert arrays['sequence_sum_logprob'].shape == (101, 500, 4)
            if initial is None:
                initial = arrays['sequence_sum_logprob'][0]
            else:
                assert np.array_equal(initial, arrays['sequence_sum_logprob'][0])
            buffer = io.BytesIO()
            np.savez_compressed(buffer, steps=np.arange(101), **arrays)
            put(f'{run_id}/policy_numeric.npz', buffer.getvalue())
            metrics_raw = (directory / 'metrics.csv').read_bytes()
            rows = list(csv.DictReader(io.StringIO(metrics_raw.decode())))
            assert [int(r['step']) for r in rows] == list(range(101))
            put(f'{run_id}/metrics.csv', metrics_raw)
            report['runs'][run_id] = dict(state=manifest['state'], steps=101,
                prompts=500, candidates=4, config={k: cfg[k] for k in
                ('method', 'alpha', 'lambda_current', 'beta_train', 'nu', 'kappa', 'seed')},
                manifest_sha256=digest(manifest_raw), support_sha256=digest(support_raw),
                snapshot_sha256=hashes, audit_sha256=cfg['audit_sha256'])
            print(f'Exported {run_id}: 101 x 500 x 4', flush=True)
        wr = output / 'wr_results' / 'wr_summary.csv'
        report['wr_summary_exists'] = wr.exists()
        if wr.exists():
            put('wr_summary.csv', wr.read_bytes())
            put('wr_manifest.json', (wr.parent / 'manifest.json').read_bytes())
        archive.writestr('export_manifest.json', json.dumps(report, indent=2))
    print(json.dumps(dict(archive=str(destination), sha256=digest(destination.read_bytes()),
                          bytes=destination.stat().st_size)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    export(args.root, args.destination)
