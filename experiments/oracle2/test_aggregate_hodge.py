"""CPU-only regression tests; run with unittest, no model dependencies."""
import json
from pathlib import Path
import unittest

import numpy as np

from aggregate_hodge import aggregate


class AverageHodgeTests(unittest.TestCase):
    def test_opposite_cycles_cancel_not_norms(self):
        c = np.array([[0, .2, -.2, 0], [-.2, 0, .2, 0],
                      [.2, -.2, 0, 0], [0, 0, 0, 0]])
        result = aggregate([.5+c, .5-c])['links']['identity']
        self.assertAlmostEqual(result['hodge_of_mean_link_flow']['cyclic_spectral_norm'], 0)
        self.assertGreater(result['mean_per_prompt_cyclic_spectral_norm'], .3)

    def test_potential_flow_has_no_residual(self):
        u = np.array([-.1, -.03, .03, .1])
        result = aggregate([.5 + u[:, None] - u[None, :]])['links']['identity']
        self.assertLess(result['hodge_of_mean_link_flow']['cyclic_spectral_norm'], 1e-15)

    def test_logit_average_operations_are_distinct(self):
        p = np.array([[.5, .9, .8], [.1, .5, .6], [.2, .4, .5]])
        entry = aggregate([p, p*.5+.25])['links']['logit']
        self.assertFalse(np.allclose(entry['hodge_of_mean_link_flow']['C'],
                                     entry['hodge_of_link_of_mean_P']['C']))

    def test_no_silent_logit_clipping(self):
        p = np.array([[.5, 1], [0, .5]])
        entry = aggregate([p, 1-p])['links']['logit']
        self.assertEqual(entry['status'], 'UNDEFINED_UNCLIPPED_PER_PROMPT_LOGIT')
        self.assertNotIn('hodge_of_mean_link_flow', entry)
        saturated = aggregate([p])['links']['logit']
        self.assertEqual(saturated['mean_P_status'], 'UNDEFINED_UNCLIPPED_LOGIT')

    def test_common_relabeling_preserves_mean_norm(self):
        p = np.array([[.5, .8, .3], [.2, .5, .7], [.7, .3, .5]])
        permutation = [2, 0, 1]
        r = aggregate([p, p*.5+.25])
        q = aggregate([v[np.ix_(permutation, permutation)] for v in [p, p*.5+.25]])
        for link in ('identity', 'logit'):
            self.assertAlmostEqual(r['links'][link]['hodge_of_mean_link_flow']['cyclic_spectral_norm'],
                                   q['links'][link]['hodge_of_mean_link_flow']['cyclic_spectral_norm'])

    def test_invalid_input_rejected(self):
        for p in (np.zeros((1, 4, 4)), np.full((1, 4, 4), np.nan), np.empty((0, 4, 4))):
            with self.assertRaises(ValueError):
                aggregate(p)

    def test_frozen_500_panel_regression(self):
        path = Path(__file__).parent / 'campaigns/oracle2_real_20261009/policy_results_20261013/prompt_metadata.json'
        rows = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(len(rows), 500)
        result = aggregate([row['preference_matrix'] for row in rows])
        for link, mean_norm, norm_mean in [('identity', .221815660192, .022618963780),
                                           ('logit', 1.145352103763, .052292636504)]:
            entry = result['links'][link]
            self.assertAlmostEqual(entry['mean_per_prompt_cyclic_spectral_norm'], mean_norm, places=10)
            self.assertAlmostEqual(entry['hodge_of_mean_link_flow']['cyclic_spectral_norm'], norm_mean, places=10)
            self.assertLess(entry['linearity_max_error'], 1e-12)


if __name__ == '__main__':
    unittest.main()
