import copy
import json
import os
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from runtime import (Journal, checked, checkpoints, digest, encode, exclusive, load_json,
                     qwen_prefix, save_new, seal, staged)
from train import configuration, outer, rng_state, restore_rng
from generate import expected
from measure import TRAITS, parse_labels, wr_metrics
from campaigns.oracle1_qwen3_20261015.build_plan import build
from queue_campaign import receipt_jobs, select_phase, prior_receipts
from history_math import build_outer_state
from oracle_math import sigmoid
from run_cyclic_history import validate_config


class Tokenizer:
    eos_token_id = 9
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        assert kwargs['return_dict'] is False
        return [1,2,3]
    def encode(self, text, **kwargs):
        return [4]*len(text)


class CampaignTests(unittest.TestCase):
    def test_discovery_pca_is_fixed_orthonormal_and_handles_constant_traits(self):
        from analyze import fit_axes
        values=np.random.default_rng(0).normal(size=(100,5))
        values[:,4]=3.
        center,scale,axes,variance=fit_axes(values)
        np.testing.assert_allclose(axes@axes.T,np.eye(2),atol=1e-12)
        np.testing.assert_allclose(((values-center)/scale@axes.T).mean(axis=0),0,atol=1e-12)
        self.assertEqual(scale[4],1.)
        self.assertLessEqual(variance.sum(),1.)

    def test_six_configs_and_oracle1(self):
        plan = build(Path('/launch'))
        self.assertEqual(plan['oracle']['component'],'nemotron')
        self.assertEqual(plan['evaluation']['steps'],list(range(0,101,10)))
        self.assertEqual(plan['annotation']['traits'],list(TRAITS))
        for arm in plan['runs']:
            cfg = configuration(plan, arm['run_id'])
            self.assertEqual(cfg['run_id'], arm['run_id'])
            validate_config(cfg)

    def test_encode_preserves_chat_prefix_and_eos_sum(self):
        rows = encode(Tokenizer(),dict(prompt='x',responses=['a','ab']),7)
        self.assertEqual(rows[0]['labels'],[-100,-100,-100,4,9])
        self.assertEqual(rows[1]['response_tokens'],3)

    def test_no_silent_truncation(self):
        with self.assertRaises(ValueError):
            encode(Tokenizer(),dict(prompt='x',responses=['abcde']),7)

    def test_bad_chat_template(self):
        tok = Tokenizer()
        tok.apply_chat_template = lambda *a,**k: {'input_ids':[1]}
        with self.assertRaises(TypeError):
            qwen_prefix(tok,'x')

    def test_five_simultaneous_labels(self):
        self.assertEqual(len(parse_labels(json.dumps(dict.fromkeys(TRAITS,5)))),5)
        self.assertIsNone(parse_labels(json.dumps(dict.fromkeys(TRAITS,None)))['hedging'])

    def test_bad_labels_fail(self):
        for value in (True,11,-1,3.5,'5'):
            with self.assertRaises(ValueError):
                parse_labels(json.dumps(dict.fromkeys(TRAITS,value)))
        with self.assertRaises(ValueError):
            parse_labels('{}')

    def test_dpo_feedback_exactly_null(self):
        r=np.array([.2,1.,-1.,2.])
        panels=[dict(nemotron_scores=r.tolist(),preference_matrix=sigmoid(r[:,None]-r[None,:]).tolist())]
        initial=np.array([[-1.,-2.,-3.,-4.]])
        current=initial+np.array([[0.,2.,0.,-1.]])
        previous=initial+np.array([[1.,0.,-2.,0.]])
        cfg=dict(method='dpo',alpha=.9,lambda_current=.8,beta_train=.8,nu=0,kappa=.5)
        fb=outer(initial,current,previous,panels,cfg)
        normal=outer(initial,current,previous,panels,dict(cfg,kappa=0))
        for key in normal:
            np.testing.assert_array_equal(fb[key],normal[key])
        numerical=build_outer_state(initial,current,previous,np.array([panels[0]['preference_matrix']]),
                                   'dpo',.9,.8,.8,kappa=.5)
        np.testing.assert_allclose(fb['target_logits'],numerical['target_logits'],atol=1e-8)

    def test_ipo_history_uses_previous(self):
        r=np.array([.2,1.,-1.,2.])
        panels=[dict(nemotron_scores=r.tolist(),preference_matrix=sigmoid(r[:,None]-r[None,:]).tolist())]
        x=np.array([[-1.,-2.,-3.,-4.]])
        cfg=dict(method='ipo',alpha=.9,lambda_current=.8,beta_train=.2,nu=0,kappa=.5)
        result=outer(x,x+np.array([[0.,2.,0.,0.]]),x,panels,cfg)
        self.assertGreater(np.linalg.norm(result['offset']),0)

    def test_oracle1_wr_ties_and_direction(self):
        self.assertEqual(wr_metrics([0]*4,[0]*4)['oracle1_expected_win_rate'],.5)
        self.assertEqual(wr_metrics([0]*4,[0]*4)['oracle1_majority_win_rate'],.5)
        a=wr_metrics([1,2,3,4],[0,1,2,3])['oracle1_expected_win_rate']
        b=wr_metrics([0,1,2,3],[1,2,3,4])['oracle1_expected_win_rate']
        self.assertGreater(a,.5)
        self.assertAlmostEqual(a+b,1)
        with self.assertRaises(ValueError):
            wr_metrics([1],[0])

    def test_generation_seeds_match_arms_not_baseline(self):
        plan=build(Path('/launch'))
        panels=[dict(prompt='private fixture',prompt_id=1,prompt_key='key')]
        a=list(expected(plan,panels,'ipo_ordinary',10))
        b=list(expected(plan,panels,'ipo_reference',10))
        c=list(expected(plan,panels,'baseline',0))
        self.assertEqual([r['seed'] for r in a],[r['seed'] for r in b])
        self.assertNotEqual(a[0]['seed'],c[0]['seed'])
        self.assertEqual(len(a),4)

    def test_journal_never_overwrites_and_checks_input(self):
        with tempfile.TemporaryDirectory() as folder:
            j=Journal(folder,'contract')
            j.put('x','input',dict(id='x',reward=1.))
            self.assertEqual(j.get('x','input')['reward'],1.)
            with self.assertRaises(FileExistsError):
                j.put('x','input',dict(id='x',reward=2.))
            with self.assertRaises(ValueError):
                j.get('x','different input')
            with self.assertRaises(ValueError):
                Journal(folder,'changed contract')

    def test_journal_detects_corruption(self):
        with tempfile.TemporaryDirectory() as folder:
            j=Journal(folder,'contract')
            j.put('x','input',dict(id='x',reward=1.))
            row=load_json(j.path('x'))
            row['row']['reward']=99
            j.path('x').write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                j.get('x','input')

    def test_atomic_commit_ignores_partial_and_checks_chain(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            incomplete=staged(root,'step_0001')
            temp=staged(root,'step_0000')
            save_new(temp/'state.json',{'step':0})
            seal(temp,root/'step_0000','contract',dict(step=0))
            self.assertEqual(len(checkpoints(root,'contract')),1)
            self.assertTrue(incomplete.exists())
            with self.assertRaises(ValueError):
                checked(root/'step_0000','other')
            (root/'step_0000/state.json').write_text('{}')
            with self.assertRaises(ValueError):
                checked(root/'step_0000','contract')

    def test_checkpoint_gap_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            temp=staged(root,'step_0001')
            seal(temp,root/'step_0001','contract',dict(step=1))
            with self.assertRaises(ValueError):
                checkpoints(root,'contract')

    def test_writer_lock_not_stolen(self):
        with tempfile.TemporaryDirectory() as folder:
            with exclusive(folder):
                with self.assertRaises(FileExistsError):
                    with exclusive(folder):
                        pass
            self.assertFalse((Path(folder)/'writer.lock').exists())

    def test_phase_startup_then_resume_then_all_six(self):
        plan=build(Path('/launch'))
        for steps, stage, indices, updates in [([-1]*6,'startup_probe',[0,3],1),
            ([1,-1,-1,1,-1,-1],'resume_probe',[0,3],1),
            ([2,-1,-1,2,-1,-1],'training_segment',list(range(6)),10)]:
            with patch('queue_campaign.training_steps',return_value=steps):
                result=select_phase(Path('/launch'),plan,'identity')
                self.assertEqual((result['stage'],result['indices'],result['updates']),(stage,indices,updates))

    def test_array_receipt_indices(self):
        self.assertEqual(receipt_jobs(dict(job_id='123;cluster',phase='train',indices=[0,3])),['123_0','123_3'])
        self.assertEqual(receipt_jobs(dict(job_id='123',phase='wrscore',indices=[0])),['123'])

    def test_ambiguous_intent_blocks_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            save_new(root/'submissions/0001_intent.json',{})
            with self.assertRaisesRegex(RuntimeError,'Unresolved'):
                prior_receipts(root,None)

    def test_failed_accounting_blocks_continuation(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            save_new(root/'submissions/0001_intent.json',{})
            save_new(root/'submissions/0001_receipt.json',dict(job_id='1',phase='train',indices=[0]))
            with self.assertRaisesRegex(RuntimeError,'not successful'):
                prior_receipts(root,SimpleNamespace(run=lambda _: '1_0|TIMEOUT|0:0'))


class ResumeTensorTests(unittest.TestCase):
    def test_actual_qwen_lora_history_round_resume_on_cpu(self):
        try:
            import torch
            from transformers import Qwen3Config, Qwen3ForCausalLM
            from peft import LoraConfig, PeftModel, get_peft_model
        except ImportError:
            self.skipTest('Torch/Transformers/PEFT mandatory on server')
        from train import commit
        from run_cyclic_history import score_panel, train_round
        torch.manual_seed(12)
        model_config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
            head_dim=8, pad_token_id=0, eos_token_id=2)
        base = Qwen3ForCausalLM(model_config)
        weights = copy.deepcopy(base.state_dict())
        model = get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],
                                             lora_dropout=0.,task_type='CAUSAL_LM'))
        encoded=[dict(input_ids=[1,3,4+i,2],labels=[-100,-100,4+i,2]) for i in range(8)]
        cfg=configuration(build(Path('/launch')),'ipo_reference')
        cfg.update(num_prompts=2,score_batch_size=2)
        r=np.array([[.1,.5,-.3,1.],[.2,-.2,.6,1.2]])
        matrices=sigmoid(r[:,:,None]-r[:,None,:])
        panels=[dict(nemotron_scores=a.tolist(),preference_matrix=p.tolist()) for a,p in zip(r,matrices)]
        initial,lengths=score_panel(model,encoded,0,'cpu',2,(2,4))
        state=outer(initial,initial,initial,panels,cfg)
        train_round(model,encoded,matrices,state,cfg,0,'cpu',0)
        current,_=score_panel(model,encoded,0,'cpu',2,(2,4))
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,SLURM_JOB_ID='cpu-test'):
            root=Path(folder)
            commit(root,1,'test',model,torch,initial,current,initial,lengths,dict(step=1),state)
            checked(root/'step_0001','test')
            base2=Qwen3ForCausalLM(model_config)
            base2.load_state_dict(weights)
            resumed=PeftModel.from_pretrained(base2,root/'step_0001/adapter',is_trainable=True)
            scores,_=score_panel(resumed,encoded,0,'cpu',2,(2,4))
            np.testing.assert_allclose(scores,current,atol=1e-6,rtol=0)
            state2=outer(initial,current,initial,panels,cfg)
            train_round(model,encoded,matrices,state2,cfg,0,'cpu',1)
            restore_rng(torch,torch.load(root/'step_0001/rng.pt',weights_only=False))
            train_round(resumed,encoded,matrices,state2,cfg,0,'cpu',1)
            for (name,p),(other,q) in zip(model.named_parameters(),resumed.named_parameters()):
                self.assertEqual(name,other)
                torch.testing.assert_close(p,q,rtol=1e-5,atol=1e-6)

    def test_cpu_rng_roundtrip(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch unavailable locally; mandatory server CPU test')
        random.seed(3)
        np.random.seed(3)
        torch.manual_seed(3)
        with patch.object(torch.cuda,'get_rng_state_all',return_value=[]),patch.object(torch.cuda,'set_rng_state_all'):
            state=rng_state(torch)
            expected=(random.random(),np.random.rand(),torch.rand(4))
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'rng.pt'
                torch.save(state,path)
                restore_rng(torch,torch.load(path,weights_only=False))
            self.assertEqual(random.random(),expected[0])
            self.assertEqual(np.random.rand(),expected[1])
            torch.testing.assert_close(torch.rand(4),expected[2],rtol=0,atol=0)

    def test_two_round_toy_training_matches_checkpoint_resume(self):
        try:
            import torch
        except ImportError:
            self.skipTest('Torch unavailable locally; mandatory server CPU test')
        # Every outer update deliberately creates a fresh optimizer, as production does.
        def round_(parameter, history, index):
            optimizer=torch.optim.AdamW([parameter],lr=.02)
            target=history[0]+.9*history[1]-.2*history[2]
            loss=((parameter-target)**2).sum()
            loss.backward()
            optimizer.step()
            parameter.grad=None
            return [history[0],parameter.detach().clone(),history[1]]
        initial=torch.tensor([1.,-2.])
        p=torch.nn.Parameter(initial.clone())
        h=[initial.clone() for _ in range(3)]
        h=round_(p,h,0)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'state.pt'
            torch.save(dict(parameter=p.detach().clone(),history=h),path)
            h=round_(p,h,1)
            state=torch.load(path,weights_only=True)
            q=torch.nn.Parameter(state['parameter'])
            recovered=round_(q,state['history'],1)
            torch.testing.assert_close(p,q,rtol=0,atol=0)
            for a,b in zip(h,recovered):
                torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
