"""Focused numerical tests for the empirical-only plotting definitions."""
import unittest

import numpy as np
from plot_results import entropy, softmax, summaries


class PlotDefinitionsTest(unittest.TestCase):
    def test_softmax_is_shift_invariant_and_normalized(self):
        scores = np.array([[-10000., -10001., -9999., -10002.]])
        np.testing.assert_allclose(softmax(scores), softmax(scores + 10000), atol=1e-14)
        np.testing.assert_allclose(softmax(scores).sum(-1), 1)

    def test_relative_entropy_is_not_raw_entropy(self):
        initial = np.array([[0., -4., -8., -12.]])
        self.assertAlmostEqual(entropy(softmax(initial-initial))[0], np.log(4))
        self.assertLess(entropy(softmax(initial))[0], .2)

    def test_nonfinite_rejected(self):
        for value in (np.nan, np.inf, -np.inf):
            with self.assertRaises(ValueError):
                softmax([[0., value]])

    def test_zeros_in_entropy_are_finite(self):
        self.assertEqual(entropy(np.array([[1., 0., 0., 0.]]))[0], 0.)

    def test_summary_does_not_fill_incomplete_window(self):
        with self.assertRaises(ValueError):
            summaries(dict(last=99), 'ipo', 'reference90')


if __name__ == '__main__':
    unittest.main()
