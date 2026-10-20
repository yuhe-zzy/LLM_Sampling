import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from campaigns.oracle1_qwen3_20261015.build_plan import build
from runtime import Journal, digest, save_new, sha256
from generate import collect, completion_name, expected, selected_steps
from interim import (STEPS, accounting, check_account, choice, control_ids,
                     legacy_ready, receipts)
from measure import score_wr, wr_directory


def receipt(job, phase='train', indices=None):
    return dict(job_id=job,phase=phase,indices=[0,1] if indices is None else indices)


def snapshot(rows, gpus):
    return dict(allocated_gpus=gpus,owned_scontrol=rows,
        owned_queue=[f'{job}|yuhe32|448057|RUNNING|gres/gpu:1|node'
                     for row in rows for job in control_ids(row)])


def control(job, tasks=None, gpu=1, throttle=2):
    return (f'JobId={job} JobState=RUNNING ReqTRES=cpu=8,gres/gpu={gpu} '
            f'AllocTRES=gres/gpu={gpu},gres/gpu:h100={gpu} '+
            (f'ArrayJobId={job} ArrayTaskId={tasks} ArrayTaskThrottle={throttle}' if tasks else ''))


def budget(snapshot):
    if snapshot['allocated_gpus'] is None or snapshot['allocated_gpus']>4:
        raise RuntimeError('Unknown/excess allocation')


HELPERS=SimpleNamespace(check_live_budget=budget)


class InterimTests(unittest.TestCase):
    def test_exact_subset_and_full_completion_separate(self):
        plan=build(Path('/base'))
        self.assertEqual(selected_steps(plan,STEPS),STEPS)
        self.assertNotEqual(completion_name(plan,STEPS),'COMPLETE.json')
        self.assertEqual(completion_name(plan,plan['evaluation']['steps']),'COMPLETE.json')
        for bad in ([],[0,0],[10,0],[5],[True]):
            with self.assertRaises(ValueError):
                selected_steps(plan,bad)

    def test_six_arm_generation_seed_and_identity_unchanged(self):
        plan=build(Path('/base'))
        before=copy.deepcopy(plan)
        panels=[dict(prompt_id=3,prompt_key='key',prompt='synthetic unit test')]
        for arm in ['baseline']+[r['run_id'] for r in plan['runs']]:
            for step in STEPS:
                a=list(expected(plan,panels,arm,step))
                selected_steps(plan,STEPS)
                b=list(expected(plan,panels,arm,step))
                self.assertEqual(a,b)
                self.assertEqual(len({r['seed'] for r in a}),4)
        self.assertEqual(plan,before)

    def test_collection_reuses_full_spec_signature(self):
        plan=build(Path('/base'))
        panels=[dict(prompt_id=3,prompt_key='key',prompt='synthetic unit test')]
        calls=[]
        def bank(directory,identity,requests,signature):
            calls.append((directory,signature))
            return requests
        with patch('generate.contract',return_value=(plan,{},'identity')),patch('generate.jsonl',return_value=panels), \
             patch('generate.checked'),patch('generate.sha256',return_value='adapter'),patch('generate.bank',side_effect=bank):
            rows=collect(Path('/base'),steps=STEPS,arms=['ipo_ordinary'])
        self.assertEqual(len(rows),36)
        for (path,signature),step in zip(calls,STEPS):
            self.assertEqual(signature,digest(dict(identity='identity',arm='ipo_ordinary',step=step,
                adapter='adapter',settings=plan['evaluation'])))

    def test_collapsed_array_range_is_fully_expanded(self):
        self.assertEqual(control_ids(control('12','2-5%2')),{'12_2','12_3','12_4','12_5'})
        for value in ('5-2','1-99999','2:4'):
            with self.assertRaises(ValueError):
                control_ids(control('12',value))

    def test_two_lanes_allow_four_including_pending(self):
        train,wr=receipt('10'),receipt('20','wrscore',[0])
        states={'10_0':['RUNNING','0:0'],'10_1':['PENDING','0:0'],'20':['PENDING','0:0']}
        snap=snapshot([control('10','0-1'),control('20',gpu=2)],2)
        self.assertEqual(check_account(snap,[train,wr],states,HELPERS),4)

    def test_one_lane_allows_second_two_gpu_lane(self):
        train=receipt('10')
        states={f'10_{i}':['RUNNING','0:0'] for i in range(2)}
        self.assertEqual(check_account(snapshot([control('10','0-1')],2),[train],states,HELPERS,'evaluation'),4)
        with self.assertRaisesRegex(RuntimeError,'still active'):
            check_account(snapshot([control('10','0-1')],2),[train],states,HELPERS,'training')

    def test_third_allocation_same_lane_blocked(self):
        rs=[receipt('10'),receipt('20','generate'),receipt('30','baseline',[0])]
        states={j:['PENDING','0:0'] for j in ['10_0','10_1','20_0','20_1','30']}
        with self.assertRaisesRegex(RuntimeError,'Multiple allocations'):
            check_account(snapshot([],0),rs,states,HELPERS)

    def test_unknown_pending_and_releasing_block(self):
        r=receipt('10')
        states={'10_0':['RUNNING','0:0'],'10_1':['RUNNING','0:0']}
        for extra in (control('77',gpu=1),control('77',gpu=1).replace('RUNNING','COMPLETING')):
            with self.assertRaisesRegex(RuntimeError,'Unregistered'):
                check_account(snapshot([control('10','0-1'),extra],3),[r],states,HELPERS)

    def test_throttle_gpu_count_extra_index_unknown_memory_fail_closed(self):
        r=receipt('10')
        states={'10_0':['RUNNING','0:0'],'10_1':['RUNNING','0:0']}
        for row in (control('10','0-1',throttle=3),control('10','0-1',gpu=2),control('10','0-2')):
            with self.assertRaises((RuntimeError,ValueError)):
                check_account(snapshot([row],2),[r],states,HELPERS)
        for gpus in (None,5):
            with self.assertRaises(RuntimeError):
                check_account(snapshot([control('10','0-1')],gpus),[r],states,HELPERS)

    def test_legacy_pending_blocks_sidecar_even_with_free_cards(self):
        plan=build(Path('/base'))
        with self.assertRaisesRegex(RuntimeError,'Legacy pending'):
            legacy_ready(Path('/base'),plan,[receipt('10')],
                         {'10_0':['COMPLETED','0:0'],'10_1':['PENDING','0:0']},snapshot([],0))

    def test_legacy_loaded_running_is_compatible_but_stale_manifest_is_not(self):
        plan=build(Path('/base'))
        r=receipt('10',indices=[0])
        states={'10_0':['RUNNING','0:0']}
        snap=snapshot([control('10','0')],1)
        with patch('interim.load_json',return_value=dict(job='10',state='RUNNING')):
            legacy_ready(Path('/base'),plan,[r],states,snap)
        with patch('interim.load_json',return_value=dict(job='9',state='SEGMENT_COMPLETE')):
            with self.assertRaisesRegex(RuntimeError,'not verified'):
                legacy_ready(Path('/base'),plan,[r],states,snap)

    def test_timeout_zero_exit_missing_accounting_block(self):
        for raw in ('10_0|TIMEOUT|0:0','10_0|FAILED|1:0','', '10_0|COMPLETED|1:0'):
            with self.assertRaises(RuntimeError):
                accounting(SimpleNamespace(run=lambda _,value=raw:value),[receipt('10',indices=[0])])

    def test_unknown_intent_or_changed_receipt_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            save_new(path/'0001_intent.json',dict(phase='baseline'))
            with self.assertRaises(RuntimeError):
                receipts(path)
            save_new(path/'0001_receipt.json',dict(phase='baseline',job_id='1'))
            self.assertEqual(len(receipts(path)),1)
            with patch('interim.load_json',side_effect=[{},dict(phase='train',job_id='1')]):
                with self.assertRaises(ValueError):
                    receipts(path)

    def test_training_selection_only_missing_states(self):
        plan=build(Path('/base'))
        with patch('interim.training_steps',return_value=[100,90,100,100,90,90]):
            result=choice(Path('/base'),plan,'identity','training')
            self.assertEqual(result,dict(phase='train',indices=[1,4,5],updates=10))
        with patch('interim.training_steps',return_value=[100]*6):
            self.assertIsNone(choice(Path('/base'),plan,'identity','training'))

    def test_subset_result_cannot_be_full_measurement(self):
        plan=build(Path('/base'))
        self.assertNotEqual(wr_directory(plan,STEPS),Path(plan['output_root'])/'wr')
        self.assertEqual(wr_directory(plan,plan['evaluation']['steps']),Path(plan['output_root'])/'wr')

    def test_completed_generation_subset_skips_inference_and_full_marker(self):
        try:
            import transformers
            import peft
        except ImportError:
            self.skipTest('Torch/Transformers/PEFT mandatory on server')
        from generate import run
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            plan=build(root)
            plan['output_root']=str(root/'outputs')
            target=Path(plan['output_root'])/'generations/ipo_ordinary'
            for step in STEPS:
                save_new(target/f'step_{step:04d}/COMPLETE.json',{'preserved':step})
            hashes={p:sha256(p) for p in target.glob('step_*/COMPLETE.json')}
            with patch('generate.contract',return_value=(plan,dict(policy=dict(path='unused')),'test')), \
                 patch('generate.require_gpu'),patch('generate.jsonl',return_value=[]),patch('generate.checked'), \
                 patch('generate.sha256',return_value='adapter'),patch('generate.bank') as bank, \
                 patch.object(transformers.AutoTokenizer,'from_pretrained'), \
                 patch.object(transformers.AutoModelForCausalLM,'from_pretrained',side_effect=AssertionError('No repeated generation')):
                run(root,'ipo_ordinary',steps=STEPS)
            self.assertEqual(bank.call_count,9)
            self.assertFalse((target/'COMPLETE.json').exists())
            self.assertTrue((target/completion_name(plan,STEPS)).exists())
            self.assertEqual(hashes,{p:sha256(p) for p in target.glob('step_*/COMPLETE.json')})

    def test_cached_cpu_scoring_subset_then_full_reuses_exact_journal_and_plot(self):
        try:
            import torch
            import transformers
        except ImportError:
            self.skipTest('Torch/Transformers mandatory on server')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            plan=build(root)
            plan['output_root']=str(root/'outputs')
            plan['dataset']['splits']['evaluation']=2
            panels=[dict(prompt_key=f'key{i}',prompt_id=i,prompt='synthetic test') for i in range(2)]
            records=[]
            for arm in ['baseline']+[r['run_id'] for r in plan['runs']]:
                for step in ([0] if arm=='baseline' else plan['evaluation']['steps']):
                    records += [dict(r,response='synthetic response') for r in expected(plan,panels,arm,step)]
            target=Path(plan['output_root'])/'wr'
            journal=Journal(target/'scores',digest(dict(identity='test',oracle=plan['oracle'])))
            for row in records:
                journal.put(row['id'],digest(dict(prompt=row['prompt'],response=row['response'])),
                            dict(id=row['id'],reward=float(row['draw'])+row['step']/100))
            hashes={p.name:sha256(p) for p in (target/'scores').glob('*.json')}
            def collect_scope(*args,steps=None):
                return [r for r in records if r['run_id']=='baseline' or r['step'] in steps]
            with patch('measure.contract',return_value=(plan,dict(nemotron=dict(path='unused')),'test')), \
                 patch('measure.require_gpu'),patch('measure.collect',side_effect=collect_scope), \
                 patch('measure.memory_snapshot',return_value=[]), \
                 patch.object(transformers.AutoTokenizer,'from_pretrained'), \
                 patch.object(transformers.AutoModelForCausalLM,'from_pretrained',side_effect=AssertionError('No inference for cached scores')):
                score_wr(root,steps=STEPS)
                self.assertFalse((target/'COMPLETE.json').exists())
                subset=wr_directory(plan,STEPS)
                self.assertEqual(json.loads((subset/'COMPLETE.json').read_text())['count'],440)
                from analyze import plot
                with patch('analyze.contract',return_value=(plan,{},'test')):
                    plot(root,root/'plot',steps=STEPS)
                from PIL import Image
                pixels=np.asarray(Image.open(root/'plot/oracle1_wr.png').convert('RGB'))
                self.assertGreater(pixels.shape[0],500)
                self.assertGreater(pixels.std(),10)
                score_wr(root)
                self.assertEqual(json.loads((target/'COMPLETE.json').read_text())['count'],536)
            self.assertEqual(hashes,{p.name:sha256(p) for p in (target/'scores').glob('*.json')})


if __name__=='__main__':
    unittest.main()
