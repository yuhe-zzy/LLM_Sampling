"""Empirical trend sweep must change only named factors and keep existing gates."""
import copy
import importlib.util
from pathlib import Path
import unittest
import numpy as np

from calibrated_protocol import EMPIRICAL_PROTOCOL, TREND_PROTOCOL, contract_hash
from history_math import build_outer_state
from run_cyclic_history import validate_config

PATH = Path(__file__).parent/'campaigns/cyclic_history_trend_sweep_20261003/build_plan.py'
spec = importlib.util.spec_from_file_location('trend_plan',PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TrendSweepTest(unittest.TestCase):
    def setUp(self):
        self.plan = module.make_plan()
        self.cfg = dict(self.plan['common'],**self.plan['runs'][0],protocol=TREND_PROTOCOL)

    def test_counts_and_unique_grid(self):
        p = self.plan
        self.assertEqual((len(p['runs']),len(p['reused_runs']),len(p['comparisons'])),(44,6,50))
        self.assertEqual(len({r['run_id'] for r in p['comparisons']}),50)
        for method in ('ipo','dpo'):
            selected=[r for r in p['comparisons'] if r['method']==method]
            self.assertEqual(sum(r['arm'].startswith('reference') for r in selected),10)
            self.assertEqual(sum(r['arm'].startswith('feedback') for r in selected),10)
            self.assertEqual(sum(r['arm']=='ordinary' for r in selected),5)
        self.assertEqual(sorted(r['task_index'] for r in p['comparisons'] if r['status']=='NEW'),list(range(44)))

    def test_every_arm_has_matched_baseline(self):
        lookup={r['run_id']:r for r in self.plan['comparisons']}
        for row in lookup.values():
            base=lookup[row['baseline_run_id']]
            for key in ('method','alpha','lambda_current','beta_train'):
                self.assertEqual(row[key],base[key])
            self.assertGreater(1-row['alpha'],0)

    def test_only_one_base_factor_changes(self):
        center=module.BASES[0]
        for base in module.BASES[1:]:
            self.assertEqual(sum(base[k]!=center[k] for k in ('alpha','lambda_current','beta_scale')),1)

    def test_hash_and_config_validation(self):
        for run in self.plan['runs']:
            cfg=dict(self.plan['common'],**run,protocol=TREND_PROTOCOL)
            validate_config(cfg)
            self.assertEqual(cfg['calibration_contract_sha256'],contract_hash(cfg))
            self.assertEqual((cfg['iters'],cfg['seed'],cfg['num_prompts'],cfg['epochs_per_iter']),(100,0,6,10))

    def test_partial_only_and_full_refresh_gate_preserved(self):
        bad=copy.deepcopy(self.cfg)
        bad['alpha']=1
        with self.assertRaises(ValueError): validate_config(bad)
        bad=dict(self.cfg,protocol=EMPIRICAL_PROTOCOL)
        with self.assertRaises(ValueError): validate_config(bad)
        bad=dict(self.cfg,scheme='lagged_sampling',nu=0,kappa=.5)
        with self.assertRaises(ValueError): validate_config(bad)

    def test_max_lag_keeps_initial_anchor(self):
        cfg=self.cfg
        initial=np.asarray(cfg['calibration']['initial_sequence_scores'])
        current=initial+np.arange(4)
        previous=initial-np.arange(4)
        matrix=np.full((6,4,4),.5)
        state=build_outer_state(initial,current,previous,matrix,'ipo',.9,.8,.2,nu=.9)
        np.testing.assert_allclose(state['reference'],.1*initial+.9*previous)
        np.testing.assert_array_equal(state['offset'],np.zeros_like(initial))


if __name__=='__main__': unittest.main()
