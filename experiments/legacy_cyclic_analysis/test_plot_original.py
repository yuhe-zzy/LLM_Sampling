import copy
import unittest
import numpy as np
import matplotlib.pyplot as plt
from plot_original import select,figure


def examples():
    runs=[]
    for method in ('ipo','dpo'):
        prob=np.zeros((6,5,4))+ .25
        for pid in range(5):
            prob[:,pid,0] += np.arange(6)*pid*.003
            prob[:,pid,1] -= np.arange(6)*pid*.003
        runs.append(dict(method=method,alpha=.99,lambda_on=.8,beta=1.,last_step=5,
            step=np.arange(6),prompt_id=np.arange(5),support_sha256=np.array(['same']*5),
            prob=prob,valid=np.ones((6,5),dtype=bool)))
    return runs


class PlotTests(unittest.TestCase):
    def test_fixed_quantile_selection(self):
        selected,audit=select(examples())
        self.assertEqual([s['prompt_id'] for s in selected],[1,2,4])
        self.assertEqual(audit['selection_window'],[0,5])

    def test_different_response_support_is_rejected(self):
        runs=examples()
        runs[1]['support_sha256'][2]='oops'
        with self.assertRaises(ValueError):
            select(runs)

    def test_nonfinite_baseline_prompt_is_excluded_from_selection(self):
        runs=examples()
        runs[1]['valid'][4,2]=False
        selected,audit=select(runs)
        self.assertEqual(audit['eligible_prompts'],4)
        self.assertNotIn(2,[s['prompt_id'] for s in selected])

    def test_invalid_probability_fallback_is_not_drawn(self):
        runs=examples()
        runs[0]['valid'][2,1]=False
        fig=figure(runs,'ipo',[(.99,.8,1.)],[dict(prompt_id=1)],'Test')
        self.assertTrue(np.isnan(fig.axes[0].lines[0].get_ydata()[2]))
        self.assertEqual(fig.axes[0].get_ylim(),(0.,1.))
        plt.close(fig)


if __name__=='__main__':
    unittest.main()
