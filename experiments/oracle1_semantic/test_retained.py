from contextlib import nullcontext
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from campaigns.oracle1_qwen3_20261015.build_plan import build
from runtime import Journal, digest, load_json, save_new, sha256
from generate import expected
from measure import score_wr, selected_arms, wr_directory
from retained import ARM, STEPS, AUDIT, account, advance, completion, freeze, history, new_receipts, query, validate_banks, verify, worker
from test_interim import HELPERS, control, snapshot


class RetainedTests(unittest.TestCase):
    def test_freeze_and_verify_bind_only_selected_checkpoints_and_both_old_ledgers(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'new';base=Path(folder)/'base';prior=Path(folder)/'prior'
            plan=build(base);plan['output_root']=str(Path(folder)/'output')
            save_new(base/'deployment.json',dict(source_commit='a'*40))
            save_new(prior/'extension.json',dict(identity='test',base_root=str(base)))
            save_new(root/'source'/AUDIT,dict(source=dict(binding_sha256=sha256(prior/'extension.json'))))
            for step in STEPS:
                save_new(Path(plan['output_root'])/'training'/ARM/f'step_{step:04d}/COMMIT.json',dict(step=step))
            save_new(Path(plan['output_root'])/'training'/ARM/'manifest.json',dict(state='COMPLETE',last_complete_step=100))
            helpers=SimpleNamespace(owned_snapshot=lambda:snapshot([],0),check_live_budget=HELPERS.check_live_budget)
            with patch('retained.contract',return_value=(plan,{},'test')),patch('retained.checked'), \
                 patch('retained.scheduler',return_value=helpers),patch('retained.history',return_value={}), \
                 patch('retained.validate_banks'),patch('builtins.print'):
                freeze(root,base,prior,'b'*40)
                self.assertEqual(len(verify(root)[2]['checkpoint_commits']),11)
                with self.assertRaises(FileExistsError):
                    freeze(root,base,prior,'b'*40)
                original=sha256
                for name in ('deployment.json','extension.json','COMMIT.json',Path(AUDIT).name):
                    with patch('retained.sha256',side_effect=lambda p,n=name:'changed' if Path(p).name==n else original(p)):
                        with self.assertRaises(ValueError):
                            verify(root)
                save_new(prior/'submissions/0001_intent.json',{})
                with self.assertRaisesRegex(ValueError,'ledger changed'):
                    verify(root)

    def test_scope_validation_and_isolated_completion(self):
        plan=build(Path('/base'))
        for bad in ([],[ARM,ARM],['baseline'],['missing'],['ipo_reference',ARM]):
            with self.assertRaises(ValueError):
                selected_arms(plan,bad)
        self.assertNotEqual(wr_directory(plan,STEPS,[ARM]),wr_directory(plan,STEPS))
        self.assertNotEqual(wr_directory(plan,STEPS,[ARM]),wr_directory(plan,STEPS,['ipo_reference']))

    def test_account_empty_and_single_worker_only(self):
        self.assertEqual(account(HELPERS,snapshot([],0)),0)
        self.assertEqual(account(HELPERS,snapshot([control('10',gpu=2)],2),'10',2),2)
        for snap in (snapshot([control('11')],1),snapshot([control('11')],0),snapshot([],1),snapshot([],None)):
            with self.assertRaises(RuntimeError):
                account(HELPERS,snap)
        for snap in (snapshot([control('10',gpu=1)],1),snapshot([control('10','0')],1),
                     snapshot([control('10',gpu=2),control('11')],3),snapshot([],0)):
            with self.assertRaises((RuntimeError,ValueError)):
                account(HELPERS,snap,'10',2)

    def test_accounting_duplicates_extra_missing_are_never_imputed(self):
        for raw in ('','10|COMPLETED|0:0\n10|COMPLETED|0:0','11|COMPLETED|0:0','bad'):
            with self.assertRaises(ValueError):
                query(SimpleNamespace(run=lambda _,v=raw:v),['10'],['10'])
        self.assertEqual(query(SimpleNamespace(run=lambda _:'10|FAILED|1:0'),['10'],['10']),{'10':['FAILED','1:0']})

    def test_reviewed_withdrawn_failure_is_preserved_not_success(self):
        original=[dict(job_id='10',phase='train',indices=[0])]
        extra=[dict(job_id='20',phase='generate',indices=[0,2],extension_sha256='hash')]
        approved=dict(accounting=dict(original_successful_tasks=1,extension={
            '20_0':['COMPLETED','0:0'],'20_2':['FAILED','1:0']}))
        expected_states={'10_0':['COMPLETED','0:0'],**approved['accounting']['extension']}
        with patch('retained.receipts',side_effect=[original,extra]),patch('retained.sha256',return_value='hash'), \
             patch('retained.query',return_value=expected_states):
            actual=history(Path('/base'),Path('/prior'),HELPERS,approved)
            self.assertEqual(actual['20_2'],['FAILED','1:0'])
        approved['accounting']['extension']['20_0']=['FAILED','1:0']
        expected_states['20_0']=['FAILED','1:0']
        with patch('retained.receipts',side_effect=[original,extra]),patch('retained.sha256',return_value='hash'), \
             patch('retained.query',return_value=expected_states):
            with self.assertRaisesRegex(RuntimeError,'Retained prerequisite'):
                history(Path('/base'),Path('/prior'),HELPERS,approved)

    def test_new_failed_timeout_pending_missing_success_blocks_advance(self):
        r=dict(job_id='30',phase='retained_generate',arms=[ARM],steps=STEPS,
               binding_sha256='hash',identity='test',source_commit='a'*40)
        b=dict(identity='test',source_commit='a'*40)
        for outcome in (['FAILED','1:0'],['TIMEOUT','0:0'],['PENDING','0:0'],['COMPLETED','1:0']):
            with patch('retained.receipts',return_value=[r]),patch('retained.sha256',return_value='hash'), \
                 patch('retained.query',return_value={'30':outcome}):
                with self.assertRaises(RuntimeError):
                    new_receipts(Path('/new'),b,HELPERS)
        with patch('retained.receipts',return_value=[r]),patch('retained.sha256',return_value='hash'), \
             patch('retained.query',return_value={'30':['RUNNING','0:0']}):
            self.assertEqual(new_receipts(Path('/new'),b,HELPERS,'30')[0],[r])

    def test_advance_is_nonarray_single_arm_and_unknown_submission_not_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            plan=build(root)
            b=dict(identity='test',source_commit='a'*40,prior_root='/prior',historical_disposition={})
            calls=[]
            helpers=SimpleNamespace(owned_snapshot=lambda:snapshot([],0),check_live_budget=HELPERS.check_live_budget,
                                    run=lambda c:calls.append(c) or 'ambiguous submission')
            with patch('retained.verify',return_value=(Path('/base'),plan,b)),patch('retained.scheduler',return_value=helpers), \
                 patch('retained.history',return_value={}),patch('retained.load_json',return_value={}), \
                 patch('retained.new_receipts',return_value=([],{})),patch('retained.sha256',return_value='hash'), \
                 patch('retained.validate_banks',return_value={ARM:dict(complete=False)}),patch('builtins.print'):
                advance(root)
                self.assertEqual(calls,[])
                self.assertFalse((root/'submissions').exists())
                with self.assertRaisesRegex(RuntimeError,'Unknown submission'):
                    advance(root,True)
            self.assertEqual(len(calls),1)
            self.assertTrue((root/'submissions/0001_intent.json').exists())
            self.assertFalse((root/'submissions/0001_receipt.json').exists())
            self.assertIn('--gres=gpu:1',calls[0])
            self.assertFalse(any(s.startswith('--array') for s in calls[0]))

    def test_worker_dispatch_cannot_train_or_generate_other_arms(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            save_new(root/'submissions/0001_receipt.json',dict(job_id='30'))
            b=dict(identity='test',prior_root='/prior',historical_disposition={})
            with patch('retained.verify',return_value=(Path('/base'),{},b)),patch('retained.scheduler',return_value=SimpleNamespace(owned_snapshot=lambda:{})), \
                 patch('retained.new_receipts',return_value=([dict(phase='retained_generate')],{})), \
                 patch('retained.history',return_value={}),patch('retained.account'),patch('retained.validate_banks',return_value={}), \
                 patch('retained.load_json',return_value=dict(job_id='30')),patch.dict(os.environ,SLURM_JOB_ID='30'), \
                 patch('generate.run') as run,patch('builtins.print'):
                worker(root)
            run.assert_called_once_with(Path('/base'),ARM,steps=STEPS)

    def test_prefix_validation_rejects_foreign_row_and_never_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            plan=build(root)
            plan['output_root']=str(root/'output')
            panels=[dict(prompt_id=1,prompt_key='test',prompt='synthetic')]
            for arm in ['baseline',ARM]:
                p=root/'output/generations'/arm/'step_0000'
                signature=digest(dict(identity='test',arm=arm,step=0,adapter=None if arm=='baseline' else 'hash',settings=plan['evaluation']))
                journal=Journal(p/'rows',signature)
                for request in list(expected(plan,panels,arm,0))[:4 if arm=='baseline' else 1]:
                    row=dict(**request,response='synthetic',response_tokens=2,unique_token_ratio=1.,dominant_token_fraction=.5,hit_max_new_tokens=False)
                    journal.put(row['id'],digest(request),row)
                if arm=='baseline':
                    save_new(p/'COMPLETE.json',{})
            before={str(p):sha256(p) for p in root.rglob('*') if p.is_file()}
            # Real bank order is request order; use the journal for that order here.
            def ordered(p,identity,requests,signature):
                j=Journal.__new__(Journal);j.root=p/'rows';j.identity=signature
                return [j.get(r['id'],digest(r)) for r in requests]
            with patch('retained.jsonl',return_value=panels),patch('retained.sha256',return_value='hash'),patch('retained.bank',side_effect=ordered):
                result=validate_banks(root,plan,'test')
                self.assertEqual(result[ARM+':0']['rows'],1)
                self.assertEqual(len(result),12)
            self.assertEqual(before,{str(p):sha256(p) for p in root.rglob('*') if p.is_file()})
            save_new(root/'output/generations'/ARM/'step_0000/rows'/('a'*64+'.json'),{})
            with patch('retained.jsonl',return_value=panels),patch('retained.sha256',return_value='hash'),patch('retained.bank',side_effect=ordered):
                with self.assertRaisesRegex(ValueError,'Unexpected journal'):
                    validate_banks(root,plan,'test')

    def test_partial_generation_only_missing_rows_keep_full_signature(self):
        try:
            import torch, transformers, peft
        except ImportError:
            self.skipTest('Required server GPU-stack libraries for CPU mock')
        from generate import run
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); plan=build(root); plan['output_root']=str(root/'output')
            before=copy.deepcopy(plan)
            panels=[dict(prompt_id=1,prompt_key='test',prompt='synthetic')]
            requests=list(expected(plan,panels,ARM,80))
            signature=digest(dict(identity='test',arm=ARM,step=80,adapter='hash',settings=plan['evaluation']))
            directory=root/'output/generations'/ARM/'step_0080'
            journal=Journal(directory/'rows',signature)
            first=dict(**requests[0],response='preserved',response_tokens=1,ended_with_eos=True,hit_max_new_tokens=False,unique_token_ratio=1.,dominant_token_fraction=1.)
            journal.put(first['id'],digest(requests[0]),first)
            old_hash=sha256(journal.path(first['id']))
            class Model:
                calls=0
                def to(self,*a): return self
                def eval(self): return self
                def generate(self,**kwargs):
                    self.calls+=1
                    return torch.tensor([[1,3,9]])
            model=Model(); tensor=torch.tensor
            tok=SimpleNamespace(pad_token_id=9,eos_token_id=9,decode=lambda *a,**k:'synthetic')
            with patch('generate.contract',return_value=(plan,dict(policy=dict(path='unused')),'test')),patch('generate.require_gpu'), \
                 patch('generate.checked'),patch('generate.sha256',return_value='hash'),patch('generate.jsonl',return_value=panels), \
                 patch('generate.qwen_prefix',return_value=[1]),patch('generate.stable_attention',return_value=nullcontext()), \
                 patch.object(torch,'tensor',side_effect=lambda *a,**k:tensor(*a,**{x:y for x,y in k.items() if x!='device'})), \
                 patch.object(torch.cuda,'manual_seed_all'),patch.object(torch.cuda,'empty_cache'), \
                 patch.object(transformers.AutoTokenizer,'from_pretrained',return_value=tok), \
                 patch.object(transformers.AutoModelForCausalLM,'from_pretrained',return_value=model), \
                 patch.object(peft.PeftModel,'from_pretrained',return_value=model):
                run(root,ARM,steps=[80])
            self.assertEqual(model.calls,3)
            self.assertEqual(sha256(journal.path(first['id'])),old_hash)
            self.assertEqual(plan,before)
            self.assertFalse((root/'output/generations'/ARM/'COMPLETE.json').exists())

    def test_cached_selected_wr_and_plot_are_complete_only_for_one_arm(self):
        try:
            import transformers
        except ImportError:
            self.skipTest('Required server libraries')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); plan=build(root); plan['output_root']=str(root/'output')
            plan['dataset']['splits']['evaluation']=2
            before=copy.deepcopy(plan)
            panels=[dict(prompt_id=i,prompt_key=str(i),prompt='synthetic') for i in range(2)]
            rows=[]
            journal=Journal(root/'output/wr/scores',digest(dict(identity='test',oracle=plan['oracle'])))
            for arm in ['baseline',ARM]:
                for step in ([0] if arm=='baseline' else STEPS):
                    for r in expected(plan,panels,arm,step):
                        row=dict(**r,response='synthetic');rows.append(row)
                        journal.put(row['id'],digest(dict(prompt=row['prompt'],response=row['response'])),
                                    dict(id=row['id'],reward=row['draw']+.01*step))
            old={p.name:sha256(p) for p in journal.root.glob('*.json')}
            with patch('measure.contract',return_value=(plan,dict(nemotron=dict(path='unused')),'test')),patch('measure.require_gpu'), \
                 patch('measure.collect',return_value=rows) as collect,patch('measure.memory_snapshot',return_value=[]), \
                 patch.object(transformers.AutoTokenizer,'from_pretrained'), \
                 patch.object(transformers.AutoModelForCausalLM,'from_pretrained',side_effect=AssertionError('No repeated scores')):
                score_wr(root,arms=[ARM])
                collect.assert_called_once_with(root,steps=STEPS,arms=['baseline',ARM])
            self.assertTrue(completion(plan,'test'))
            self.assertFalse((root/'output/wr/COMPLETE.json').exists())
            self.assertEqual(old,{p.name:sha256(p) for p in journal.root.glob('*.json')})
            self.assertEqual(plan,before)
            from analyze import plot,read_csv
            with patch('analyze.contract',return_value=(plan,{},'test')):
                plot(root,root/'plot',arms=[ARM])
            self.assertEqual(len(read_csv(root/'plot/wr_summary.csv')),11)
            from PIL import Image
            pixels=np.asarray(Image.open(root/'plot/oracle1_wr.png').convert('RGB'))
            self.assertGreater(pixels.std(),10)
            self.assertEqual(load_json(root/'plot/coverage.json')['arms'],[ARM])


if __name__=='__main__':
    unittest.main()
