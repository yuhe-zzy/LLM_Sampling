import contextlib
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from common import load_json, save_jsonl, sha256, write_json
from evaluate_wr import collect
from generate_evaluation import generate
from generation_recovery import (bank_info, capture, checked_contract, check_failed_jobs,
    generation_directories, generation_seed, inspect_arm, prepare_recovery, read_prefix)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = self.root/'old'
        self.old.mkdir()
        self.out = self.root/'out'
        self.data = self.root/'data/scored'
        self.data.mkdir(parents=True)
        self.run = 'ipo_ordinary'
        self.source = self.out/'generations'/self.run
        self.source.mkdir(parents=True)
        self.spec = dict(steps=[0, 10, 20], responses_per_prompt=4, max_new_tokens=4,
                         checkpoint_seed=100780, baseline_seed=777, temperature=.8, top_p=.95)
        self.panels = [dict(prompt_id=i, prompt_key=f'key{i}', prompt=f'private{i}', oracle2_group='transitive') for i in range(2)]
        save_jsonl(self.data/'evaluation.jsonl', self.panels)
        self.lock = self.root/'lock.json'
        write_json(self.lock, dict(policy=dict(path='policy')))
        self.plan = dict(output_root=str(self.out), data_root=str(self.root/'data'), model_lock=str(self.lock),
            evaluation=self.spec, dataset=dict(evaluation=2, max_policy_length=20),
            training=dict(iters=100), runs=[dict(run_id=self.run)])
        self.plan_path = self.old/'plan.json'
        write_json(self.plan_path, self.plan)
        write_json(self.data/'audit.json', dict(plan_sha256=sha256(self.plan_path),
            files_sha256={'evaluation.jsonl': sha256(self.data/'evaluation.jsonl')}))
        self.runroot = self.out/self.run
        self.runroot.mkdir()
        write_json(self.runroot/'manifest.json', dict(state='COMPLETED', last_complete_step=100,
            config=dict(audit_sha256=sha256(self.data/'audit.json'), source_commit='original',
                plan_sha256=sha256(self.plan_path), model_lock_sha256=sha256(self.lock))))
        for step in self.spec['steps']:
            directory = self.adapter(step)
            directory.mkdir(parents=True)
            (directory/'adapter_model.safetensors').write_bytes(f'weights{step}'.encode())
            write_json(directory/'adapter_config.json', dict(step=step))
        save_jsonl(self.source/'step_0000.jsonl', self.rows(0))
        save_jsonl(self.source/'step_0010.jsonl', self.rows(10)[:3])
        with (self.source/'step_0010.jsonl').open('ab') as handle:
            handle.write(b'{"unfinished')
        write_json(self.source/'manifest.json', dict(state='RUNNING', run_id=self.run,
            plan_sha256=sha256(self.plan_path), audit_sha256=sha256(self.data/'audit.json'),
            model_lock_sha256=sha256(self.lock), settings=self.spec,
            files={'step_0000.jsonl': bank_info(self.source/'step_0000.jsonl', self.rows(0),
                sha256(self.adapter(0)/'adapter_model.safetensors'))}))
        write_json(self.old/'deployment.json', dict(source_commit='original',
            files_sha256={'plan.json': sha256(self.plan_path)}))
        write_json(self.old/'generate_submission_receipt.json', dict(job_id='123', phase='generate', source_commit='original'))
        self.recovery = self.root/'generation_recovery.json'
        with patch('generation_recovery.subprocess.check_output', return_value='123_0|TIMEOUT|0:0'):
            capture(self.old, self.root/'repaired', [self.run], self.recovery)
        self.contract = load_json(self.recovery)

    def adapter(self, step):
        return self.runroot/('adapter_initial' if step == 0 else f'adapters/step_{step:04d}')

    def rows(self, step):
        return [dict(id=f'{self.run}:{step}:{p["prompt_key"]}:{draw}', run_id=self.run,
            step=step, draw=draw, seed=generation_seed(self.spec['checkpoint_seed'], step, p['prompt_id'], draw),
            prompt_id=p['prompt_id'], prompt_key=p['prompt_key'], prompt=p['prompt'], group=p['oracle2_group'],
            response='answer', response_tokens=1, ended_with_eos=True, hit_max_new_tokens=False,
            dominant_token_fraction=1., unique_token_ratio=1.) for p in self.panels for draw in range(4)]

    def inspect(self):
        return inspect_arm(self.plan, self.contract, self.contract['arms'][0])

    def test_reuses_full_bank_and_valid_prefix_without_touching_original(self):
        hashes = {p: sha256(p) for p in self.source.iterdir()}
        with patch('generation_recovery.check_failed_jobs'):
            target, status, rows = prepare_recovery(self.plan_path, self.recovery, self.run)
        self.assertEqual(status['files']['step_0000.jsonl']['count'], 8)
        self.assertEqual(len(rows['step_0010.jsonl']), 3)
        self.assertGreater(status['reused']['step_0010.jsonl']['preserved_trailing_bytes'], 0)
        self.assertEqual(sha256(target/'step_0000.jsonl'), sha256(self.source/'step_0000.jsonl'))
        self.assertEqual({p: sha256(p) for p in hashes}, hashes)
        self.assertTrue((target/'step_0010.jsonl').read_bytes().endswith(b'\n'))
        with patch('generation_recovery.check_failed_jobs'), self.assertRaises(FileExistsError):
            prepare_recovery(self.plan_path, self.recovery, self.run)

    def test_complete_rows_without_manifest_entry_are_reused(self):
        path = self.source/'step_0010.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in self.rows(10)), encoding='utf-8')
        self.contract['arms'][0]['source_files_sha256'][path.name] = sha256(path)
        bank = self.inspect()[path.name]
        self.assertTrue(bank['complete'])
        self.assertEqual(len(bank['rows']), 8)

    def test_changed_source_manifest_bank_and_adapter_fail_closed(self):
        paths = [self.source/'manifest.json', self.source/'step_0010.jsonl',
                 self.adapter(10)/'adapter_model.safetensors', self.adapter(10)/'adapter_config.json']
        for path in paths:
            with self.subTest(path=path.name):
                old = path.read_bytes()
                path.write_bytes(old+b' ')
                with self.assertRaises(ValueError):
                    self.inspect()
                path.write_bytes(old)

    def test_unknown_extra_bank_fails(self):
        (self.source/'step_0020.jsonl').write_bytes(b'')
        with self.assertRaises(ValueError):
            self.inspect()

    def test_wrong_identity_seed_order_and_nonfinite_rejected(self):
        path = self.root/'bad.jsonl'
        for field, value in [('seed', 0), ('draw', 2), ('step', 20), ('prompt', 'wrong'),
                             ('run_id', 'wrong'), ('unique_token_ratio', float('nan'))]:
            rows = self.rows(10)[:1]
            rows[0][field] = value
            path.write_text(json.dumps(rows[0])+'\n', encoding='utf-8')
            with self.subTest(field=field), self.assertRaises(ValueError):
                read_prefix(path, self.panels, self.run, 10, self.spec)
        rows = self.rows(10)[:2][::-1]
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
        with self.assertRaises(ValueError):
            read_prefix(path, self.panels, self.run, 10, self.spec)

    def test_malformed_complete_line_is_not_silently_discarded(self):
        path = self.root/'bad.jsonl'
        path.write_bytes(b'{bad}\n')
        with self.assertRaises(ValueError):
            read_prefix(path, self.panels, self.run, 0, self.spec)

    def test_only_terminal_timeout_may_be_recovered(self):
        for text in ('', '123_0|RUNNING|0:0', '123_0|COMPLETED|0:0', '123_0|FAILED|1:0'):
            with patch('generation_recovery.subprocess.check_output', return_value=text), self.assertRaises(ValueError):
                check_failed_jobs(self.contract)
        with patch('generation_recovery.subprocess.check_output', return_value='123_0|TIMEOUT|0:0'):
            check_failed_jobs(self.contract)

    def test_contract_binds_settings_and_prevents_original_output_overwrite(self):
        for mutate in (lambda c: c.update(plan_sha256='wrong'),
                       lambda c: c['arms'][0].update(target_dir=str(self.source)),
                       lambda c: c['arms'][0].update(original_job_id='123_1')):
            bad = copy.deepcopy(self.contract)
            mutate(bad)
            write_json(self.recovery, bad)
            with self.assertRaises(ValueError):
                checked_contract(self.plan_path, self.recovery)
        write_json(self.recovery, self.contract)

    def test_end_to_end_mock_gpu_generates_only_missing_responses(self):
        calls, seeds, adapters = [], [], []

        class Tensor:
            def __init__(self, value): self.value = value
            def __getitem__(self, index):
                row, sl = index
                return Tensor(self.value[row][sl])
            def tolist(self): return self.value

        class Model:
            def to(self, device): return self
            def eval(self): return self
            def generate(self, **kwargs):
                calls.append(kwargs)
                return Tensor([kwargs['input_ids'].value[0]+[4, 9]])

        def load_adapter(model, path, **kwargs):
            adapters.append(str(path))
            return model

        tok = SimpleNamespace(pad_token_id=9, eos_token_id=9, encode=lambda *a, **k: [1, 2], decode=lambda *a, **k: 'answer')
        torch = SimpleNamespace(bfloat16='bf16', long='long', tensor=lambda value, **kw: Tensor(value),
            ones_like=lambda value: value, inference_mode=contextlib.nullcontext, manual_seed=seeds.append,
            cuda=SimpleNamespace(manual_seed_all=lambda s: None, empty_cache=lambda: None))
        modules = dict(torch=torch,
            transformers=SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tok),
                AutoModelForCausalLM=SimpleNamespace(from_pretrained=lambda *a, **k: Model())),
            peft=SimpleNamespace(PeftModel=SimpleNamespace(from_pretrained=load_adapter)))
        original_hashes = {p: sha256(p) for p in self.source.iterdir()}
        with patch.dict(sys.modules, modules), patch('generate_evaluation.require_gpu'), \
             patch('generate_evaluation.verify_model_lock'), patch('generate_evaluation.stable_attention', contextlib.nullcontext), \
             patch('generation_recovery.check_failed_jobs'):
            generate(self.plan_path, self.run, self.recovery)
        self.assertEqual(len(calls), 13)
        self.assertEqual(seeds, [r['seed'] for r in self.rows(10)[3:]+self.rows(20)])
        self.assertEqual(adapters, [str(self.adapter(10)), str(self.adapter(20))])
        for kwargs in calls:
            self.assertEqual((kwargs['temperature'], kwargs['top_p'], kwargs['max_new_tokens']), (.8, .95, 4))
        self.assertEqual({p: sha256(p) for p in original_hashes}, original_hashes)
        target = generation_directories(self.plan_path, self.recovery)[self.run]
        manifest = load_json(target/'manifest.json')
        self.assertEqual(manifest['state'], 'COMPLETE')
        self.assertEqual([v['count'] for v in manifest['files'].values()], [8, 8, 8])
        baseline = self.out/'generations/baseline'
        baseline.mkdir()
        base_rows = self.rows(0)
        for r in base_rows:
            r['id'] = 'baseline:'+r['id']
            r['run_id'] = 'baseline'
        save_jsonl(baseline/'step_0000.jsonl', base_rows)
        write_json(baseline/'manifest.json', dict(state='COMPLETE', plan_sha256=sha256(self.plan_path),
            files={'step_0000.jsonl': dict(count=8, sha256=sha256(baseline/'step_0000.jsonl'))}))
        collect(self.plan_path, self.root/'records.jsonl', self.recovery)
        self.assertEqual(load_json(self.root/'records.jsonl.manifest.json')['count'], 32)
        bad = load_json(target/'manifest.json')
        bad['files'].pop('step_0020.jsonl')
        write_json(target/'manifest.json', bad)
        with self.assertRaises(ValueError):
            collect(self.plan_path, self.root/'incomplete.jsonl', self.recovery)
        self.assertFalse((self.root/'incomplete.jsonl').exists())

    def test_queue_blocks_recovery_until_other_generation_complete(self):
        spec = importlib.util.spec_from_file_location('recovery_queue', Path(__file__).parent/'campaigns/oracle2_real_20261009/queue.py')
        queue = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(queue)
        launch = self.root/'new_launch'
        launch.mkdir()
        (launch/'plan.json').write_bytes(self.plan_path.read_bytes())
        (launch/'generation_recovery.json').write_bytes(self.recovery.read_bytes())
        for phase in ('audit', 'baseline', 'train', 'generate'):
            with self.subTest(phase=phase), self.assertRaises(ValueError):
                queue.phase_prerequisites(launch, phase)
        with patch('generation_recovery.subprocess.check_output', return_value='123_0|TIMEOUT|0:0'), self.assertRaises(FileNotFoundError):
            queue.phase_prerequisites(launch, 'recover')
        self.assertEqual(queue.RESOURCES['recover']['gpus'], 1)
        self.assertEqual(queue.RESOURCES['recover']['concurrency'], 2)

    def test_queue_requires_successful_scheduler_state_not_exit_code_alone(self):
        spec = importlib.util.spec_from_file_location('recovery_queue', Path(__file__).parent/'campaigns/oracle2_real_20261009/queue.py')
        queue = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(queue)
        for result in ('123_2|RUNNING|0:0', '123_2|TIMEOUT|0:0', '123_2|COMPLETED|1:0', ''):
            with patch.object(queue, 'run', return_value=result), self.assertRaises(ValueError):
                queue.require_successful_jobs(['123_2'])
        with patch.object(queue, 'run', return_value='123_2|COMPLETED|0:0'):
            queue.require_successful_jobs(['123_2'])

    def test_busy_account_prevents_submission_even_after_preflight(self):
        spec = importlib.util.spec_from_file_location('recovery_queue', Path(__file__).parent/'campaigns/oracle2_real_20261009/queue.py')
        queue = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(queue)
        snapshot = dict(allocated_gpus=2, owned_queue=['456|yuhe32|448057|PENDING|gpu:1|Resources'], owned_scontrol=[])
        with patch.object(queue, 'verify', return_value=dict(source_commit='new')), \
             patch.object(queue, 'checked_contract', return_value=(self.plan, self.contract)), \
             patch.object(queue, 'phase_prerequisites'), patch.object(queue, 'owned_snapshot', return_value=snapshot), \
             patch.object(queue, 'run', return_value='CHECKED ONLY') as commands, \
             patch.object(sys, 'argv', ['queue.py', '--root', str(self.old), '--phase', 'recover', '--submit']), \
             self.assertRaises(RuntimeError):
            queue.main()
        self.assertFalse(any(call.args[0][0] == 'sbatch' for call in commands.call_args_list))
        self.assertFalse((self.old/'recover_submission_intent.json').exists())


if __name__ == '__main__':
    unittest.main()
