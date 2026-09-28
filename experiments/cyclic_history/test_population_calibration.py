"""CPU regression tests for actual operators, matched support, and launch gates."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from calibrated_protocol import (contract_hash, mode_coordinates, transform_panels,
                                 verify_calibration)
from history_math import (build_outer_state, centered, load_panels, population_delta,
                          sample_pairs, sampler, support_hash)
from population_calibration import (analyze, balanced_cycle, basis, exact_trajectory,
                                    feedback_jacobian, local_matrices, spectral_radius)
from run_cyclic_history import main, resolve_config, save_snapshot, validate_config

HERE = Path(__file__).parent
PLAN = HERE / 'calibration/2026-09-28/experiment_plan.json'
HARD = np.array([[.5, 1, 1, 0], [0, .5, 1, 1], [0, 0, .5, 1], [1, 0, 0, .5]])


class PopulationTests(unittest.TestCase):
    def test_balanced_oracle_and_reverse_keep_direction_zero(self):
        p = balanced_cycle(roles=[2, 0, 3, 1])
        np.testing.assert_allclose((p - .5).sum(1), 0, atol=1e-15)
        np.testing.assert_allclose(p + p.T, 1)
        np.testing.assert_allclose(balanced_cycle(orientation=-1, roles=[2, 0, 3, 1]), p.T)
        self.assertEqual(np.sum(p == .8), 4)

    def test_balanced_bt_derivative_is_not_entrywise_logit(self):
        p = balanced_cycle()
        q = basis(4)
        actual = feedback_jacobian('dpo', p, np.ones(4) / 4, 1.)
        np.testing.assert_allclose(q.T @ actual @ q, q.T @ (4 * (p - .5)) @ q, atol=1e-10)
        actual_radius = analyze('dpo', p, np.zeros(4), .84, .8, 1.)['radii']['ordinary']
        proxy = q.T @ (.84 * np.eye(4) + .8 / 4 * np.log(p / (1 - p))) @ q
        self.assertAlmostEqual(actual_radius ** 2, .936, places=9)
        self.assertGreater(spectral_radius(proxy), 1)

    def test_hard_pilot_ipo_and_dpo_do_not_share_frontier(self):
        ipo = analyze('ipo', HARD, np.zeros(4), .9, .8, 1., .45, .25)
        dpo = analyze('dpo', HARD, np.zeros(4), .9, .8, 1., .45, .25)
        self.assertAlmostEqual(ipo['radii']['ordinary'] ** 2, .8632870842, places=8)
        self.assertAlmostEqual(dpo['radii']['ordinary'], 1.288720526, places=7)

    def test_general_bt_derivative_matches_finite_difference(self):
        mu = np.array([.12, .18, .27, .43])
        q = basis(4)
        derivative = feedback_jacobian('dpo', HARD, mu, .8)
        for direction in q.T:
            plus = population_delta('dpo', HARD, mu + 1e-4 * direction, .8)[0]
            minus = population_delta('dpo', HARD, mu - 1e-4 * direction, .8)[0]
            np.testing.assert_allclose((plus - minus) / 2e-4, derivative @ direction, atol=2e-5)

    def test_augmented_jacobians_match_actual_target(self):
        p = balanced_cycle()
        reference = np.array([.4, -.7, .2, .1])
        q = basis(4)
        for method, beta in [('ipo', .2), ('dpo', .8)]:
            prediction = analyze(method, p, reference, .9, .8, beta)
            x = np.asarray(prediction['fixed_logits'])
            for name, nu, kappa in [('lagged_reference', .45, 0),
                                     ('oracle_feedback_extrapolation', 0, .5)]:
                expected = local_matrices(method, p, reference, x, .9, .8, beta, nu, kappa)[name]
                def target(y):
                    current, previous = q @ y[:3], q @ y[3:]
                    state = build_outer_state(reference[None], current[None], previous[None],
                                              p[None], method, .9, .8, beta, nu, kappa)
                    return np.r_[q.T @ state['target_logits'][0], y[:3]]
                y = np.r_[q.T @ x, q.T @ x]
                numerical = np.column_stack([(target(y + e * 1e-4) - target(y - e * 1e-4)) / 2e-4
                                             for e in np.eye(6)])
                np.testing.assert_allclose(numerical, expected, atol=2e-6)

    def test_balanced_positive_and_negative_controls(self):
        p = balanced_cycle()
        for method, beta in [('ipo', .2), ('dpo', .8)]:
            positive = analyze(method, p, np.zeros(4), .9, .8, beta)
            self.assertAlmostEqual(positive['radii']['ordinary'] ** 2, 1.17, places=8)
            self.assertLess(positive['radii']['lagged_reference'], .98)
            self.assertLess(positive['radii']['oracle_feedback_extrapolation'], .98)
            self.assertLess(analyze(method, p, np.zeros(4), .9, .8, beta * 2)['radii']['ordinary'], .98)
            initial = np.array([1., -2., 3., -2.]) * 1e-4
            trajectory = exact_trajectory(method, p, np.zeros(4), initial, .9, .8, beta, steps=40)
            self.assertGreater(np.linalg.norm(trajectory[-1]), np.linalg.norm(initial) * 2)

    def test_full_pair_enumeration_uses_weights_once(self):
        pairs = sample_pairs(np.array([[.1, .2, .3, .4]]), 6, 0, 0, mode='all_unordered')
        self.assertEqual(len({x[1:3] for x in pairs}), 6)
        self.assertAlmostEqual(sum(x[3] for x in pairs) / 6, 1)
        with self.assertRaises(ValueError):
            sample_pairs(np.ones((1, 4)) / 4, 2, 0, 0, mode='all_unordered')

    def test_explicit_support_selection_does_not_resample(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.jsonl'
            rows = [dict(prompt_id=i, prompt=str(i), responses=['a', 'b', 'c', 'd'],
                         preference_matrix=HARD.tolist()) for i in range(4)]
            path.write_text('\n'.join(json.dumps(x) for x in rows))
            result = load_panels(path, count=2, seed=999, prompt_ids=[3, 1])
            self.assertEqual([p['prompt_id'] for p in result], [3, 1])
            with self.assertRaises(ValueError):
                load_panels(path, count=2, prompt_ids=[3, 99])

    def test_frozen_plan_all_ten_predictions_pass(self):
        plan = json.loads(PLAN.read_text())
        self.assertEqual(len(plan['runs']), 10)
        for run in plan['runs']:
            cfg = resolve_config(PLAN, run['run_id'])
            validate_config(cfg)
            self.assertEqual(contract_hash(cfg), cfg['calibration_contract_sha256'])
            for p in run['predictions']:
                radius = p['radii'][run['scheme']]
                self.assertLess(radius, .98) if run['prediction_role'] == 'ordinary_stable' or run['scheme'] != 'ordinary' else self.assertGreater(radius, 1.03)

    def synthetic_config(self):
        cfg = resolve_config(PLAN, 'ipo_reference_calibrated_s0')
        rows = [dict(prompt_id=i, prompt=str(i), responses=['a', 'b', 'c', 'd'],
                     preference_matrix=HARD.tolist()) for i in cfg['panel_ids']]
        cfg['calibration']['source_panel_sha256'] = support_hash(rows)
        transformed = copy.deepcopy(rows)
        for p, role in zip(transformed, cfg['calibration']['response_roles']):
            p['preference_matrix'] = balanced_cycle(roles=role).tolist()
        cfg['transformed_support_sha256'] = support_hash(transformed)
        cfg['calibration_contract_sha256'] = contract_hash(cfg)
        return cfg, rows, transformed

    def test_fresh_scores_tokenization_and_parameter_tampering_fail(self):
        cfg, raw, panels = self.synthetic_config()
        self.assertEqual(transform_panels(raw, cfg), panels)
        scores = np.array(cfg['calibration']['initial_sequence_scores'])
        lengths = np.array(cfg['calibration']['response_token_counts'])
        predictions = verify_calibration(panels, cfg, scores, lengths)
        mode = mode_coordinates(panels, cfg, predictions)
        self.assertEqual(mode['left_modes'].shape, (6, 4))
        with tempfile.TemporaryDirectory() as tmp:
            save_snapshot(Path(tmp), 0, scores, scores, None, lengths, calibration=mode)
            with np.load(Path(tmp) / 'step_0000.npz') as data:
                self.assertIn('cyclic_mode_phase', data.files)
                self.assertIn('fixed_point_error', data.files)
        wrong = scores.copy(); wrong[0, 0] += .1
        with self.assertRaises(ValueError):
            verify_calibration(panels, cfg, wrong, lengths)
        with self.assertRaises(ValueError):
            verify_calibration(panels, cfg, scores, lengths + 1)
        cfg['beta_train'] *= 2
        with self.assertRaises(ValueError):
            verify_calibration(panels, cfg)

    def test_execution_requires_explicit_new_protocol_approval(self):
        with patch('sys.argv', ['runner', '--plan', str(PLAN), '--run-id', 'ipo_reference_calibrated_s0', '--execute']):
            with patch('run_cyclic_history.train') as train, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main()
                train.assert_not_called()

    def test_legacy_execution_requires_explicit_override(self):
        with patch('sys.argv', ['runner', '--run-id', 'ipo_baseline_s0', '--execute']):
            with patch('run_cyclic_history.train') as train, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main()
                train.assert_not_called()


if __name__ == '__main__':
    unittest.main()
