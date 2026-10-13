"""Explicit, text-free publication allowlist for the 2026-10-13 reconciliation."""
import argparse
import csv
import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CAMPAIGNS = REPO / 'experiments/cyclic_history/campaigns'
LEGACY = REPO / 'experiments/legacy_sampling_analysis/snapshot_20260911'
ARCHIVE = REPO / 'experiments/oracle1/campaigns/historical_local_snapshots'
METRICS = {
    'cyclic_fixed_target_results_20260928': 'cyclic_fixed_target_results_20260928',
    'cyclic_history_stage_a_analysis_20260928': 'cyclic_history_stage_a_analysis_20260928',
    'cyclic_history_stage_b100_results_20260929': 'cyclic_history_stage_b100_results_20260929',
    'cyclic_history_stage_c100_results_20260929': 'cyclic_history_stage_c100_results_20260929',
    'cyclic_reference90_results_20261003': 'cyclic_history_reference90_20260929',
    'cyclic_history_trend_sweep_results_20261004': 'cyclic_history_trend_sweep_20261003',
}
ARCHIVES = ('oracle_dpo_sequencesum_results_20260817',
            'oracle_ipo_frozenref_results_20260817',
            'oracle_sequencesum_results_20260825',
            'oracle_relative_sequence_results_20260826')
FIGURES = ('dpo_policy_comparison', 'dpo_policy_trajectories',
           'ipo_policy_comparison', 'ipo_policy_trajectories',
           'population_diagnostics', 'population_time_series')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def published_sha(path):
    data = path.read_bytes()
    if path.suffix == '.csv':
        data = data.replace(b'\r\n', b'\n')
    return hashlib.sha256(data).hexdigest()


def check_csv(path, numeric_only=False):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)), path
        forbidden = {'prompt', 'response', 'text', 'chosen', 'rejected', 'password', 'token'}
        assert not forbidden.intersection(reader.fieldnames), path
        rows = list(reader)
    assert rows, path
    for row in rows:
        assert None not in row and all(v is not None for v in row.values()), path
        for value in row.values():
            if value in ('', 'True', 'False'):
                continue
            try:
                float(value)
            except ValueError:
                identifier = re.fullmatch(r'[A-Za-z0-9_.:/=+\-]+', value)
                checkpoints = re.fullmatch(r'\d+:[0-9.]+(?:; \d+:[0-9.]+)*', value)
                assert not numeric_only and (identifier or checkpoints), (path, value)
    return rows


def copy_checked(source, target):
    if source.suffix == '.csv':
        check_csv(source)
    assert source.suffix in ('.png', '.pdf', '.csv'), source
    assert source.stat().st_size < 20_000_000, source
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert sha(source) == sha(target), f'Refusing changed destination {target}'
    else:
        shutil.copy2(source, target)


def write_json(path, value):
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_text(encoding='utf-8') == text, f'Refusing changed destination {path}'
    else:
        path.write_text(text, encoding='utf-8')


def export_legacy(workspace):
    root = workspace / 'analysis_sampling_20260911'
    script = root / 'analyze_sampling.py'
    spec = importlib.util.spec_from_file_location('legacy_analysis', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    snapshot = root / 'snapshot_20260911.zip'
    assert sha(snapshot) == 'c8e694d30a78fc328cdb8a0f08102098495ef53e6491448029b9201331dfc81c'
    runs, manifest = module.load_runs(snapshot)
    table = pd.read_csv(root / 'figures/run_diagnostics.csv')
    selections = json.loads((root / 'figures/selected_prompts.json').read_text(encoding='utf-8'))
    selected_rows, population = [], []
    for run in runs:
        stop = 82 if run['method'] == 'dpo' else 150
        diag = module.diagnostics(run['data']['prob'][:stop], 30)
        expected = table[(table.method == run['method']) & (table.rho == run['rho'])].iloc[0]
        for field, values in diag.items():
            assert np.isclose(values.mean(), expected['compare_' + field]), (run['run'], field)
        assert run['invalid_rows'] == int(expected.invalid_rows)
        prob = run['data']['prob']
        for item in selections[run['method']]:
            pid = item['prompt_id']
            j = list(run['data']['prompt_id']).index(pid)
            for step, p in enumerate(prob[:, j]):
                selected_rows.append(dict(method=run['method'], rho=run['rho'], prompt_id=pid,
                    percentile=item['percentile'], step=step, valid=bool(np.isfinite(p).all()),
                    **{f'p{k}': p[k] for k in range(4)}))
        for step, p in enumerate(prob):
            valid = bool(np.isfinite(p).all())
            entropy = float(-(p * np.log(np.clip(p, 1e-300, 1))).sum(-1).mean()) if valid else np.nan
            tv = float(np.abs(p - prob[step-1]).sum(-1).mean()/2) if step and valid else np.nan
            population.append(dict(method=run['method'], rho=run['rho'], step=step,
                                   valid=valid, mean_panel_entropy=entropy, mean_adjacent_tv=tv))
    LEGACY.mkdir(parents=True, exist_ok=True)
    for name, rows in (('selected_policy.csv', selected_rows), ('population_time_series.csv', population)):
        path = LEGACY / 'metrics' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        text = pd.DataFrame(rows).to_csv(index=False, lineterminator='\n')
        if path.exists():
            assert path.read_text(encoding='utf-8') == text, path
        else:
            with path.open('w', encoding='utf-8', newline='') as handle:
                handle.write(text)
        check_csv(path)
    for name in ('run_diagnostics.csv', 'prompt_diagnostics.csv', 'paired_comparisons.csv'):
        copy_checked(root / 'figures' / name, LEGACY / 'metrics' / name)
    for stem in FIGURES:
        for ext in ('.png', '.pdf'):
            copy_checked(root / 'figures' / (stem+ext), LEGACY / 'figures' / (stem+ext))
    write_json(LEGACY / 'provenance.json', {
        'captured_at_utc': manifest['captured_at_utc'], 'private_snapshot_sha256': sha(snapshot),
        'analysis_script_sha256': sha(script), 'training_commit': None,
        'training_commit_note': 'Unknown; never substitute the current repository commit.',
        'metric': 'legacy softmax of average response log-probability; NOT sequence-sum',
        'runs': [{k: r[k] for k in ('run', 'method', 'lambda', 'rho', 'seed', 'iterations', 'prompts', 'support_sha256')}
                 for r in manifest['runs']],
        'validation': 'Recomputed all common-window metrics from private numeric snapshot; tables match.',
        'private_text_exported': False, 'dpo_baseline_invalid_from': 82,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True, type=Path)
    args = parser.parse_args()
    root = args.workspace.resolve()
    copies = []
    export_legacy(root)
    for local, public in METRICS.items():
        for source in sorted((root / local / 'raw').glob('*/metrics.csv')):
            rows = check_csv(source, numeric_only=True)
            key = 'step' if 'step' in rows[0] else 'block'
            steps = [int(r[key]) for r in rows]
            assert steps == list(range(len(steps))), source
            target = CAMPAIGNS / public / 'metrics_original' / (source.parent.name + '.csv')
            copy_checked(source, target)
            copies.append({'path': target.relative_to(REPO).as_posix(), 'local_source': source.relative_to(root).as_posix(),
                           'sha256': published_sha(target), 'source_sha256': sha(source),
                           'rows': len(rows), 'coordinate': key, 'last': steps[-1]})
    for directory in ARCHIVES:
        for source in sorted((root / directory).iterdir()):
            if source.suffix in ('.csv', '.png', '.pdf'):
                target = ARCHIVE / directory / source.name
                copy_checked(source, target)
                copies.append({'path': target.relative_to(REPO).as_posix(), 'local_source': source.relative_to(root).as_posix(),
                               'sha256': published_sha(target), 'source_sha256': sha(source),
                               'status': 'historical_local_snapshot_not_final'})
    write_json(REPO / 'docs/result_import_manifest_20261013.json', {
        'publication_date': '2026-10-13', 'scope': 'Reviewed local result reconciliation, no GPU action',
        'hash_encoding': 'sha256 uses Git LF-normalized CSV bytes; source_sha256 uses original local bytes',
        'files': copies, 'legacy_sampling': LEGACY.relative_to(REPO).as_posix(),
        'excluded': ['raw prompt/response text', 'weights/adapters', 'credentials', 'raw archives/logs',
                     'superseded draft legacy figures (final version already published)'],
    })
    print(json.dumps({'copied_metrics_and_historical_files': len(copies), 'legacy_runs_validated': 10}))


if __name__ == '__main__':
    main()
