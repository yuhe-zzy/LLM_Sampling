import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_plan import CODE, RUNS, make_plan
from history_math import build_outer_state
import queue_anchor90


class AnchorPlanTests(unittest.TestCase):
    def test_exact_arms_and_matched_baselines(self):
        prior = json.loads((CODE/'campaigns/cyclic_history_stage_b100_launch_20260929/experiment_plan_b100.json').read_text())
        plan = make_plan()
        self.assertEqual([r['run_id'] for r in plan['runs']], RUNS)
        for row in plan['runs']:
            old = next(r for r in prior['runs'] if r['run_id']==f"{row['method']}_ordinary_b100_s0")
            expected = dict(prior['common'], **old)
            actual = dict(plan['common'], **row)
            changed = {k for k in set(expected)|set(actual) if expected.get(k)!=actual.get(k)}
            self.assertEqual(changed, {'alpha', 'run_id', 'output_root', 'predictions',
                                      'prediction_role', 'calibration_contract_sha256'})
            self.assertEqual(actual['alpha'], .1)
            self.assertEqual((actual['nu'], actual['kappa'], actual['lambda_current']), (0., 0., .8))
            self.assertEqual(actual['beta_train'], .2 if row['method']=='ipo' else .8)
            self.assertEqual(actual['panel_ids'], [54, 251, 612, 737, 867, 945])
            self.assertEqual((actual['seed'], actual['iters'], actual['epochs_per_iter']), (0, 100, 10))
            self.assertEqual(actual['support_probability'], 'softmax_sequence_sum')

    def test_serialized_plan_is_exact(self):
        self.assertEqual(json.loads((HERE/'experiment_plan.json').read_text()), make_plan())

    def test_reference_only_intervention(self):
        initial = np.array([[-2., -3., -4., -5.]])
        current = initial + np.array([[.3, -.2, .1, -.4]])
        previous = initial - .2
        p = np.full((1, 4, 4), .5)
        p[0, 0, 1], p[0, 1, 0] = .8, .2
        for method, beta in [('ipo', .2), ('dpo', .8)]:
            old = build_outer_state(initial, current, previous, p, method, .9, .8, beta)
            new = build_outer_state(initial, current, previous, p, method, .1, .8, beta)
            np.testing.assert_allclose(new['reference'], .9*initial+.1*current)
            np.testing.assert_array_equal(new['mu'], old['mu'])
            np.testing.assert_array_equal(new['feedback_delta'], old['feedback_delta'])
            np.testing.assert_array_equal(new['offset'], np.zeros_like(initial))
            alternate = build_outer_state(initial, current, previous+10., p, method, .1, .8, beta)
            np.testing.assert_array_equal(new['reference'], alternate['reference'])
            self.assertFalse(np.allclose(new['reference'], old['reference']))

    def test_submission_refuses_existing_intent(self):
        with patch.object(queue_anchor90, 'verify', return_value={}), \
                patch.object(Path, 'exists', return_value=True), \
                patch.object(queue_anchor90, 'account_snapshot') as account:
            with self.assertRaisesRegex(RuntimeError, 'Already attempted'):
                queue_anchor90.submit()
            account.assert_not_called()

    def test_submission_refuses_pending_owned_work(self):
        with patch.object(queue_anchor90, 'verify', return_value={}), \
                patch.object(Path, 'exists', return_value=False), \
                patch.object(queue_anchor90, 'account_snapshot', return_value={
                    'owned_jobs': [{'state': 'PENDING'}], 'allocated_gpus': 0}), \
                patch.object(queue_anchor90, 'command') as command:
            with self.assertRaisesRegex(RuntimeError, 'Account is not empty'):
                queue_anchor90.submit()
            command.assert_not_called()

    def test_submission_refuses_existing_outputs(self):
        with patch.object(queue_anchor90, 'verify', return_value={}), \
                patch.object(Path, 'exists', side_effect=[False, False, True]), \
                patch.object(queue_anchor90, 'account_snapshot', return_value={
                    'owned_jobs': [], 'allocated_gpus': 0}), \
                patch.object(queue_anchor90, 'command') as command:
            with self.assertRaisesRegex(RuntimeError, 'Output root exists'):
                queue_anchor90.submit()
            command.assert_not_called()


if __name__ == '__main__':
    unittest.main()
