import unittest

import numpy as np

from history_math import build_outer_state
from run_fixed_target_probe import frozen_state, measure, pair_objectives, train_blocks


class FixedTargetTests(unittest.TestCase):
    def setUp(self):
        self.p = np.array([[[.5, .8, .5, .2], [.2, .5, .8, .5],
                            [.5, .2, .5, .8], [.8, .5, .2, .5]]])
        self.start = np.array([[-10., -11., -12., -11.5]])

    def state(self, method):
        cfg = dict(method=method, alpha=.9, lambda_current=.8,
                   beta_train=.2 if method == 'ipo' else .8, nu=0, kappa=0)
        state = build_outer_state(self.start, self.start, self.start, self.p,
                                  method, .9, .8, cfg['beta_train'])
        saved = {'training_'+key: value.copy() for key, value in state.items()}
        return cfg, frozen_state(cfg, self.p, self.start, self.start, self.start, saved)

    def test_target_is_objective_minimum_for_both_losses(self):
        for method in ('ipo', 'dpo'):
            cfg, state = self.state(method)
            target = state['target_logits']
            best = pair_objectives(target, self.p, state, cfg)
            for direction in np.eye(4):
                for scale in (-.3, .3):
                    loss = pair_objectives(target+scale*direction, self.p, state, cfg)
                    self.assertTrue(np.all(loss > best))
            np.testing.assert_allclose(pair_objectives(target+12, self.p, state, cfg), best)

    def test_target_and_metrics(self):
        _, state = self.state('ipo')
        target = state['target_logits']
        before, _ = measure(self.start, self.start, target)
        after, _ = measure(target, self.start, target)
        self.assertAlmostEqual(before['error_to_intended_ratio'], 1.)
        self.assertIsNone(before['update_cosine'])
        self.assertAlmostEqual(after['target_error_rms'], 0.)
        self.assertAlmostEqual(after['update_cosine'], 1.)

    def test_blocks_reuse_identical_frozen_state(self):
        _, state = self.state('dpo')
        seen = []
        train_blocks(state, lambda b: {'block': b}, lambda b, d: seen.append((b, d['block'])))
        self.assertEqual(seen, [(i, i) for i in range(1, 7)])
        with self.assertRaises(ValueError):
            state['mu'][0, 0] += .1

    def test_mutation_is_detected(self):
        _, state = self.state('ipo')
        mutable = {key: value.copy() for key, value in state.items()}
        def corrupt(_):
            mutable['target_logits'][0, 0] += .1
            return {}
        with self.assertRaises(AssertionError):
            train_blocks(mutable, corrupt, lambda *args: None)

    def test_source_target_mismatch_rejected(self):
        cfg, state = self.state('dpo')
        saved = {'training_'+key: value.copy() for key, value in state.items()}
        saved['training_mu'][0, 0] += .1
        with self.assertRaises(AssertionError):
            frozen_state(cfg, self.p, self.start, self.start, self.start, saved)


if __name__ == '__main__':
    unittest.main()
