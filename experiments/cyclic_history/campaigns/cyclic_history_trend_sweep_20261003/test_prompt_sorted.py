import unittest

import matplotlib.pyplot as plt
import numpy as np

from plot_prompt_sorted import draw_prompt, sorted_columns
from test_all_variants import fixture


def configured_fixture():
    data = fixture()
    for run in data.values():
        r = run['record']
        r['nu'] = r['alpha']/2 if r['arm'] == 'reference_half' else (
            r['alpha'] if r['arm'] == 'reference_max' else 0.)
        r['kappa'] = .5 if r['arm'] == 'feedback_half' else (
            1. if r['arm'] == 'feedback_one' else 0.)
    return data


class PromptSortedTest(unittest.TestCase):
    def test_ordinary_first_and_history_sorted(self):
        for family, field in [('reference','nu'), ('feedback','kappa')]:
            columns = sorted_columns(configured_fixture(), 'ipo', family)
            self.assertEqual([b for b,_,_ in columns[:5]], [1,2,3,4,5])
            self.assertEqual([arm for _,_,arm in columns[:5]], ['ordinary']*5)
            keys = [(run['record'][field], b) for b,run,_ in columns[5:]]
            self.assertEqual(keys, sorted(keys))
            self.assertEqual(len(keys), 10)

    def test_all_panels_have_identical_dimensions_and_data(self):
        data = configured_fixture()
        fig, mapping = draw_prompt(data, 'ipo', 'reference', 0)
        try:
            self.assertEqual(len(fig.axes), 15)
            dimensions = [(ax.get_position().width, ax.get_position().height) for ax in fig.axes]
            np.testing.assert_allclose(dimensions, np.tile(dimensions[0], (15,1)), atol=1e-12)
            for ax, row in zip(fig.axes, mapping):
                self.assertEqual(ax.get_xlim(), (0.,100.))
                self.assertEqual(ax.get_ylim(), (0.,1.))
                for k, line in enumerate(ax.lines):
                    np.testing.assert_array_equal(line.get_ydata(), data[row['run_id']]['pi'][:,0,k])
        finally:
            plt.close(fig)


if __name__ == '__main__':
    unittest.main()
