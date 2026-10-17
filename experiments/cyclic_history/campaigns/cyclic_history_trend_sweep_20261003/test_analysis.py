import unittest
import numpy as np
from analyze_results import window_metrics, aggregate_and_pair


class DescriptiveMetricsTest(unittest.TestCase):
    def test_constant_policy_has_zero_motion(self):
        q=np.full((101,6,4),.25)
        values=window_metrics(q,np.full((101,6),np.log(4)),51,100)
        for key in ('tv_mean','probability_temporal_sd','leading_response_switches'):
            np.testing.assert_array_equal(values[key],np.zeros(6))
        np.testing.assert_allclose(values['relative_entropy_mean'],np.log(4))

    def test_alternation_counts_boundary_transition(self):
        q=np.zeros((101,6,4))
        for t in range(101): q[t,:,t%2]=1
        values=window_metrics(q,np.zeros((101,6)),51,100)
        np.testing.assert_allclose(values['tv_mean'],1)
        np.testing.assert_allclose(values['probability_temporal_sd'],.25)
        np.testing.assert_allclose(values['leading_response_switches'],50)

    def test_never_extend_partial_window(self):
        with self.assertRaises(ValueError):
            window_metrics(np.full((80,6,4),.25),np.zeros((80,6)),51,100)

    def test_pairing_does_not_cross_base_or_method(self):
        fields=('tv_mean','probability_temporal_sd','mean_max_probability',
                'leading_response_switches','relative_entropy_mean')
        rows=[]
        for method in ('ipo','dpo'):
            for base in ('center','alpha08'):
                for arm,value in [('ordinary',2),('reference_half',1)]:
                    for prompt in range(6):
                        rows.append(dict(method=method,base_id=base,arm=arm,prompt_id=prompt,
                            start_step=51,end_step=100,**{f:value for f in fields}))
        means,pairs=aggregate_and_pair(rows)
        self.assertEqual((len(means),len(pairs)),(8,4))
        self.assertTrue(all(p['tv_mean_difference']==-1 and p['tv_mean_percent_change']==-50 for p in pairs))


if __name__=='__main__': unittest.main()
