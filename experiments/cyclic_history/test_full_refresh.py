"""Full-refresh is an empirical protocol, not the old fixed-point certificate."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from calibrated_protocol import (EMPIRICAL_PROTOCOL, contract_hash,
                                 verify_support_calibration)
from history_math import build_outer_state, support_hash
from population_calibration import balanced_cycle
from run_cyclic_history import resolve_config, save_snapshot, validate_config

HERE = Path(__file__).parent


class FullRefreshTests(unittest.TestCase):
    def config(self):
        c = resolve_config(HERE/'calibration/2026-09-28/experiment_plan.json',
                           'ipo_reference_calibrated_s0')
        c.update(protocol=EMPIRICAL_PROTOCOL, alpha=1., nu=.9,
                 prediction_role='empirical_unclassified')
        c['calibration_contract_sha256'] = contract_hash(c)
        return c

    def test_explicit_protocol_only(self):
        c = self.config()
        validate_config(c)
        c['protocol'] = 'cyclic_history_calibrated_sequence_v2'
        with self.assertRaises(ValueError):
            validate_config(c)

    def test_reject_wrong_arm_role_and_coefficients(self):
        for changes in (dict(alpha=1.01), dict(alpha=.9), dict(nu=1.1),
                        dict(scheme='ordinary'), dict(prediction_role='ordinary_unstable')):
            c = self.config()
            c.update(changes)
            with self.assertRaises(ValueError):
                validate_config(c)

    def test_exact_reference_and_no_initial_anchor_for_both_losses(self):
        initial = np.array([[-9., -12., -4., -6.]])
        current = np.array([[-3., -1., -5., -9.]])
        previous = np.array([[-5., -8., -3., -4.]])
        for method, beta in [('ipo', .2), ('dpo', .8)]:
            args = (current, previous, balanced_cycle()[None], method, 1., .8, beta, .9, 0.)
            result = build_outer_state(initial, *args)
            np.testing.assert_allclose(result['reference'], .1*current+.9*previous)
            changed = build_outer_state(initial*100, *args)
            np.testing.assert_allclose(result['target_logits'], changed['target_logits'])
            np.testing.assert_allclose(result['effective_reference'], result['reference'])

    def test_first_round_uses_initial_as_previous(self):
        initial = np.array([[-9., -12., -4., -6.]])
        state = build_outer_state(initial, initial, initial, balanced_cycle()[None],
                                  'ipo', 1., .8, .2, .9, 0.)
        np.testing.assert_allclose(state['reference'], initial)

    def test_support_gate_checks_initialization_without_population_analysis(self):
        c = self.config()
        panels = [dict(prompt_id=i, prompt=str(i), responses=['a','b','c','d'],
                       preference_matrix=balanced_cycle().tolist()) for i in c['panel_ids']]
        c['transformed_support_sha256'] = support_hash(panels)
        c['calibration_contract_sha256'] = contract_hash(c)
        scores = np.asarray(c['calibration']['initial_sequence_scores'])
        lengths = np.asarray(c['calibration']['response_token_counts'])
        with patch('calibrated_protocol.analyze', side_effect=AssertionError('No theory gate')):
            verify_support_calibration(panels, c, scores, lengths)
        bad_scores = scores.copy()
        bad_scores[0, 0] += 10
        with self.assertRaises(ValueError):
            verify_support_calibration(panels, c, bad_scores, lengths)
        with self.assertRaises(ValueError):
            verify_support_calibration(panels, c, scores, lengths+1)
        c['nu'] = .8
        with self.assertRaises(ValueError):
            verify_support_calibration(panels, c)

    def test_no_borrowed_fixed_point_or_mode_fields(self):
        scores = np.array([[-9., -12., -4., -6.]])
        with tempfile.TemporaryDirectory() as tmp:
            row = save_snapshot(Path(tmp), 0, scores, scores, None, np.ones_like(scores))
            with np.load(Path(tmp)/'step_0000.npz') as snap:
                self.assertNotIn('population_fixed_logits', snap.files)
                self.assertNotIn('cyclic_mode_amplitude', snap.files)
            self.assertAlmostEqual(row['relative_sequence_entropy_mean'], np.log(4))


if __name__ == '__main__':
    unittest.main()
