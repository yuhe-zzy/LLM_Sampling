"""Read-only provenance checks and copy-on-recovery for interrupted response banks."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from common import jsonl, load_json, sha256


def generation_seed(base, step, prompt_id, draw):
    return int(np.random.SeedSequence([base, step, prompt_id, draw]).generate_state(1)[0])


def bank_info(path, rows, adapter_digest):
    return dict(sha256=sha256(path), count=len(rows), adapter_sha256=adapter_digest,
        hit_max_new_tokens_fraction=float(np.mean([r['hit_max_new_tokens'] for r in rows])),
        dominant_token_ge_0_95_fraction=float(np.mean([r['dominant_token_fraction'] >= .95 for r in rows])),
        mean_unique_token_ratio=float(np.mean([r['unique_token_ratio'] for r in rows])))


def read_prefix(path, panels, run_id, step, spec):
    """Only newline-terminated, ordered, validated rows can be reused."""
    raw = path.read_bytes()
    end = raw.rfind(b'\n') + 1
    prefix, tail = raw[:end], raw[end:]
    lines = prefix.splitlines()
    draws = spec['responses_per_prompt']
    if len(lines) > len(panels) * draws:
        raise ValueError('Too many responses in recovery bank')
    rows = []
    for index, line in enumerate(lines):
        row = json.loads(line)
        prompt, draw = panels[index // draws], index % draws
        expected = dict(id=f'{run_id}:{step}:{prompt["prompt_key"]}:{draw}',
            run_id=run_id, step=step, draw=draw, prompt_id=prompt['prompt_id'],
            prompt_key=prompt['prompt_key'], prompt=prompt['prompt'], group=prompt['oracle2_group'],
            seed=generation_seed(spec['checkpoint_seed'], step, prompt['prompt_id'], draw))
        if any(row.get(k) != v for k, v in expected.items()):
            raise ValueError(f'Recovery response identity/order/seed mismatch at row {index}')
        if not isinstance(row.get('response'), str):
            raise ValueError('Missing generated response text')
        length = row.get('response_tokens')
        if type(length) is not int or not 0 <= length <= spec['max_new_tokens']:
            raise ValueError('Invalid response token count')
        for key in ('ended_with_eos', 'hit_max_new_tokens'):
            if type(row.get(key)) is not bool:
                raise ValueError('Invalid response termination diagnostic')
        for key in ('dominant_token_fraction', 'unique_token_ratio'):
            value = row.get(key)
            if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Nonfinite or invalid response diagnostic')
        if row['hit_max_new_tokens'] != (length == spec['max_new_tokens'] and not row['ended_with_eos']):
            raise ValueError('Inconsistent response termination diagnostic')
        rows.append(row)
    if tail and len(rows) == len(panels) * draws:
        raise ValueError('Unexpected trailing bytes after a full bank')
    return rows, prefix, dict(original_bytes=len(raw), reusable_bytes=len(prefix),
        preserved_trailing_bytes=len(tail), trailing_sha256=hashlib.sha256(tail).hexdigest())


def checked_contract(plan_path, recovery_path):
    plan, contract = load_json(plan_path), load_json(recovery_path)
    data = Path(plan['data_root']) / 'scored'
    bindings = dict(plan_sha256=sha256(plan_path), audit_sha256=sha256(data/'audit.json'),
        model_lock_sha256=sha256(plan['model_lock']), evaluation_support_sha256=sha256(data/'evaluation.jsonl'))
    if contract.get('version') != 1 or any(contract.get(k) != v for k, v in bindings.items()):
        raise ValueError('Recovery plan/model/audit/support changed')
    audit = load_json(data/'audit.json')
    if audit['plan_sha256'] != bindings['plan_sha256'] or audit['files_sha256']['evaluation.jsonl'] != bindings['evaluation_support_sha256']:
        raise ValueError('Evaluation support is not the audited support')
    runs = [arm['run_id'] for arm in contract['arms']]
    if not runs or len(runs) > 2 or len(set(runs)) != len(runs) or not set(runs) <= {r['run_id'] for r in plan['runs']}:
        raise ValueError('Recovery is restricted to one or two existing training arms')
    original = Path(contract['original_launch_root'])
    if sha256(original/'deployment.json') != contract['original_deployment_sha256']:
        raise ValueError('Original deployment changed')
    deployment = load_json(original/'deployment.json')
    if deployment['source_commit'] != contract['original_source_commit']:
        raise ValueError('Original execution source changed')
    for name, digest in deployment['files_sha256'].items():
        if sha256(original/name) != digest:
            raise ValueError(f'Original frozen file changed: {name}')
    if sha256(original/'generate_submission_receipt.json') != contract['original_receipt_sha256']:
        raise ValueError('Original generation receipt changed')
    if sha256(original/'plan.json') != bindings['plan_sha256']:
        raise ValueError('Recovery is not using the byte-identical original plan')
    receipt = load_json(original/'generate_submission_receipt.json')
    if receipt['source_commit'] != contract['original_source_commit'] or receipt['phase'] != 'generate':
        raise ValueError('Original receipt/source mismatch')
    destinations = set()
    output = Path(plan['output_root']).resolve()
    for arm in contract['arms']:
        expected_job = receipt['job_id'].split(';')[0] + '_' + str(next(
            i for i, r in enumerate(plan['runs']) if r['run_id'] == arm['run_id']))
        if arm['original_job_id'] != expected_job:
            raise ValueError('Recovery job does not match original array mapping')
        source, target = Path(arm['source_dir']).resolve(), Path(arm['target_dir']).resolve()
        if source != output/'generations'/arm['run_id']:
            raise ValueError('Recovery source is not the original generation bank')
        if target == output or output in target.parents or target == output.parent or target in output.parents:
            raise ValueError('Recovery must use a separate output tree')
        if any(target == old or old in target.parents or target in old.parents for old in destinations):
            raise ValueError('Recovery destinations overlap')
        destinations.add(target)
    return plan, contract


def check_failed_jobs(contract):
    for arm in contract['arms']:
        job = arm['original_job_id']
        raw = subprocess.check_output(['sacct', '-X', '-n', '-P', '-j', job,
            '-o', 'JobID,State,ExitCode'], text=True).strip()
        rows = [line.split('|') for line in raw.splitlines() if line.strip()]
        parent = [r for r in rows if r[0] == job]
        if len(parent) != 1 or len(parent[0]) != 3 or parent[0][1] != 'TIMEOUT':
            raise ValueError('Recovery requires a verified terminal TIMEOUT, not a stale manifest')


def inspect_arm(plan, contract, arm):
    run_id, spec = arm['run_id'], plan['evaluation']
    output = Path(plan['output_root'])
    panels = jsonl(Path(plan['data_root'])/'scored/evaluation.jsonl')
    if len(panels) != plan['dataset']['evaluation'] or len({p['prompt_key'] for p in panels}) != len(panels):
        raise ValueError('Bad evaluation panel count or identities')
    source = Path(arm['source_dir'])
    if sha256(source/'manifest.json') != arm['source_manifest_sha256']:
        raise ValueError('Original generation manifest changed')
    manifest = load_json(source/'manifest.json')
    for key in ('plan_sha256', 'audit_sha256', 'model_lock_sha256'):
        if manifest[key] != contract[key]:
            raise ValueError('Original generation provenance mismatch')
    if manifest['run_id'] != run_id or manifest['settings'] != spec or manifest['state'] == 'COMPLETE':
        raise ValueError('Wrong settings/arm or already-complete generation')
    train = output/run_id/'manifest.json'
    if sha256(train) != arm['training_manifest_sha256']:
        raise ValueError('Training manifest changed')
    training = load_json(train)
    if training['state'] != 'COMPLETED' or training['last_complete_step'] != plan['training']['iters'] or training['config']['audit_sha256'] != contract['audit_sha256']:
        raise ValueError('Training incomplete or wrong audit')
    for key, digest in [('source_commit', contract['original_source_commit']),
                        ('plan_sha256', contract['plan_sha256']), ('model_lock_sha256', contract['model_lock_sha256'])]:
        if training['config'].get(key) != digest:
            raise ValueError('Training source/plan/model provenance mismatch')
    expected = {f'step_{s:04d}.jsonl' for s in spec['steps']}
    found = {p.name for p in source.iterdir() if p.name != 'manifest.json'}
    if found != set(arm['source_files_sha256']) or not found <= expected or not set(manifest['files']) <= found:
        raise ValueError('Missing/extra/changed original banks')
    adapter_files = set()
    for step in spec['steps']:
        directory = output/run_id/('adapter_initial' if step == 0 else f'adapters/step_{step:04d}')
        adapter_files.update(str(p) for p in directory.iterdir() if p.is_file())
        for name in ('adapter_model.safetensors', 'adapter_config.json'):
            if not (directory/name).is_file():
                raise ValueError('Required adapter file missing')
    if adapter_files != set(arm['adapter_files_sha256']):
        raise ValueError('Adapter file set changed')
    for path, digest in arm['adapter_files_sha256'].items():
        if sha256(path) != digest:
            raise ValueError('Adapter hash changed')
    banks, stopped = {}, False
    for step in spec['steps']:
        name = f'step_{step:04d}.jsonl'
        path = source/name
        if not path.exists():
            stopped = True
            continue
        if stopped or sha256(path) != arm['source_files_sha256'][name]:
            raise ValueError('Non-prefix or changed response banks')
        rows, prefix, info = read_prefix(path, panels, run_id, step, spec)
        adapter = output/run_id/('adapter_initial' if step == 0 else f'adapters/step_{step:04d}')/'adapter_model.safetensors'
        complete = len(rows) == len(panels)*spec['responses_per_prompt']
        if name in manifest['files']:
            if not complete or info['preserved_trailing_bytes'] or manifest['files'][name] != bank_info(path, rows, sha256(adapter)):
                raise ValueError('Completed bank count/hash/adapter/diagnostics mismatch')
        elif not complete:
            stopped = True
        banks[name] = dict(rows=rows, prefix=prefix, info=info, complete=complete,
                           adapter_sha256=sha256(adapter))
    if len(banks) == len(expected) and all(b['complete'] for b in banks.values()):
        raise ValueError('No missing responses to recover')
    return banks


def prepare_recovery(plan_path, recovery_path, run_id):
    plan, contract = checked_contract(plan_path, recovery_path)
    check_failed_jobs(contract)
    arm = next(a for a in contract['arms'] if a['run_id'] == run_id)
    banks = inspect_arm(plan, contract, arm)
    target = Path(arm['target_dir'])
    target.mkdir(parents=True, exist_ok=False)
    files, reused = {}, {}
    for name, bank in banks.items():
        with (target/name).open('xb') as handle:
            handle.write(bank['prefix'])
        reused[name] = dict(count=len(bank['rows']), **bank['info'])
        if bank['complete']:
            files[name] = bank_info(target/name, bank['rows'], bank['adapter_sha256'])
    status = dict(state='RUNNING', run_id=run_id, files=files, settings=plan['evaluation'],
        plan_sha256=contract['plan_sha256'], audit_sha256=contract['audit_sha256'],
        model_lock_sha256=contract['model_lock_sha256'], recovery_sha256=sha256(recovery_path),
        recovered_from=arm['source_dir'], reused=reused,
        original_source_commit=contract['original_source_commit'])
    return target, status, {name: b['rows'] for name, b in banks.items()}


def generation_directories(plan_path, recovery_path=None):
    plan = load_json(plan_path)
    roots = {name: Path(plan['output_root'])/'generations'/name
             for name in ['baseline']+[r['run_id'] for r in plan['runs']]}
    if recovery_path is not None:
        plan, contract = checked_contract(plan_path, recovery_path)
        for arm in contract['arms']:
            banks = inspect_arm(plan, contract, arm)
            target = Path(arm['target_dir'])
            manifest = load_json(target/'manifest.json')
            if manifest.get('recovery_sha256') != sha256(recovery_path) or manifest.get('recovered_from') != arm['source_dir']:
                raise ValueError('Recovery destination provenance mismatch')
            if manifest['state'] != 'COMPLETE':
                raise ValueError('Recovered generation is incomplete')
            panels = jsonl(Path(plan['data_root'])/'scored/evaluation.jsonl')
            expected = {f'step_{s:04d}.jsonl' for s in plan['evaluation']['steps']}
            if set(manifest['files']) != expected:
                raise ValueError('Recovered checkpoint grid incomplete')
            for step in plan['evaluation']['steps']:
                name = f'step_{step:04d}.jsonl'
                rows, _, info = read_prefix(target/name, panels, arm['run_id'], step, plan['evaluation'])
                adapter = Path(plan['output_root'])/arm['run_id']/('adapter_initial' if step == 0 else f'adapters/step_{step:04d}')/'adapter_model.safetensors'
                if len(rows) != len(panels)*plan['evaluation']['responses_per_prompt'] or info['preserved_trailing_bytes']:
                    raise ValueError('Recovered response count incomplete')
                if manifest['files'][name] != bank_info(target/name, rows, sha256(adapter)):
                    raise ValueError('Recovered bank hash/count/adapter/diagnostics mismatch')
            for name, bank in banks.items():
                with (target/name).open('rb') as handle:
                    if handle.read(len(bank['prefix'])) != bank['prefix']:
                        raise ValueError('Recovered bank did not preserve its original response prefix')
            roots[arm['run_id']] = target
    return roots


def capture(original, target_root, run_ids, destination):
    plan_path = original/'plan.json'
    plan = load_json(plan_path)
    data, output = Path(plan['data_root'])/'scored', Path(plan['output_root'])
    receipt = load_json(original/'generate_submission_receipt.json')
    contract = dict(version=1, original_launch_root=str(original),
        original_deployment_sha256=sha256(original/'deployment.json'),
        original_source_commit=load_json(original/'deployment.json')['source_commit'],
        original_receipt_sha256=sha256(original/'generate_submission_receipt.json'),
        plan_sha256=sha256(plan_path), audit_sha256=sha256(data/'audit.json'),
        model_lock_sha256=sha256(plan['model_lock']),
        evaluation_support_sha256=sha256(data/'evaluation.jsonl'), arms=[])
    for run_id in run_ids:
        index = next(i for i, r in enumerate(plan['runs']) if r['run_id'] == run_id)
        source = output/'generations'/run_id
        adapters = [output/run_id/('adapter_initial' if step == 0 else f'adapters/step_{step:04d}')
                    for step in plan['evaluation']['steps']]
        contract['arms'].append(dict(run_id=run_id, original_job_id=f'{receipt["job_id"].split(";")[0]}_{index}',
            source_dir=str(source), target_dir=str(target_root/run_id),
            source_manifest_sha256=sha256(source/'manifest.json'),
            training_manifest_sha256=sha256(output/run_id/'manifest.json'),
            source_files_sha256={p.name: sha256(p) for p in source.iterdir() if p.name != 'manifest.json'},
            adapter_files_sha256={str(p): sha256(p) for d in adapters for p in d.iterdir() if p.is_file()}))
    check_failed_jobs(contract)
    summary = {}
    for arm in contract['arms']:
        banks = inspect_arm(plan, contract, arm)
        summary[arm['run_id']] = {n: dict(rows=len(b['rows']), complete=b['complete'], **b['info']) for n, b in banks.items()}
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(contract, handle, indent=2)
    checked_contract(plan_path, destination)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-root', type=Path, required=True)
    p.add_argument('--target-root', type=Path, required=True)
    p.add_argument('--run-id', action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(capture(a.original_root, a.target_root, a.run_id, a.output)), flush=True)
