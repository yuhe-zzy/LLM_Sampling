import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from plot_policy_results import choose_prompts, entropy, plot_wr, softmax, validate


class PolicyPlotsTest(unittest.TestCase):
    def test_selection_ignores_input_order(self):
        metadata = [dict(panel_index=i, prompt_id=10-i, prompt_key=str(i), group=g)
                    for i, g in enumerate(['cyclic']+['transitive']*4+['ambiguous']*3)]
        self.assertEqual(choose_prompts(metadata), [0, 4, 3, 2, 7, 6])
        self.assertEqual(choose_prompts(metadata[::-1]), [0, 4, 3, 2, 7, 6])

    def test_missing_wr_does_not_create_figure(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            plot_wr([], output)
            self.assertFalse(list(output.glob('*.png')))
            self.assertEqual(json.loads((output/'winning_rate_status.json').read_text())['state'], 'NOT_AVAILABLE')

    def test_softmax_extreme_sequence_sums(self):
        p = softmax(np.array([[-10000., -10001., -10002., -10003.]]))
        self.assertAlmostEqual(float(p.sum()), 1.)
        self.assertTrue(np.isfinite(entropy(p)).all())

    def test_reconstruct_and_reject_relative_as_raw(self):
        s = np.broadcast_to(np.array([-1., -2., -3., -4.]), (101, 500, 4)).copy()
        p = softmax(s)
        a = dict(steps=np.arange(101), sequence_sum_logprob=s, panel_probability=p,
                 panel_entropy=entropy(p), relative_sequence_entropy=entropy(softmax(s-s[0])),
                 tv=np.zeros((101, 500)))
        validate({'example': a})
        a['panel_probability'] = softmax(s-s[0])
        with self.assertRaises(AssertionError):
            validate({'example': a})


if __name__ == '__main__':
    unittest.main()
