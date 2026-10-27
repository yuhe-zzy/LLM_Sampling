import copy
import unittest
import numpy as np

from plot_ordinary_comparison import MATCHED, validate_pair


def fixture():
    cfg = {key:0 for key in MATCHED}
    cfg.update(alpha=.9,nu=0.,kappa=0.)
    ordinary = dict(manifest=dict(config=cfg,state='COMPLETED',
        calibration_gate='PASSED_FRESH_INITIAL_SCORES'),last=100,
        support=[1,2,3],scores=np.zeros((101,6,4)))
    reference = copy.deepcopy(ordinary)
    reference['manifest']['config'].update(alpha=1.,nu=.9)
    return ordinary, reference


class OrdinaryComparisonTest(unittest.TestCase):
    def test_expected_alpha_difference_is_explicitly_allowed(self):
        validate_pair(*fixture())

    def test_extra_parameter_difference_is_rejected(self):
        a,b = fixture()
        b['manifest']['config']['beta_train'] = 2
        with self.assertRaises(ValueError):
            validate_pair(a,b)

    def test_not_the_alpha09_nu09_sweep_arm(self):
        a,b = fixture()
        b['manifest']['config']['alpha'] = .9
        with self.assertRaises(ValueError):
            validate_pair(a,b)

    def test_response_mismatch_is_rejected(self):
        a,b = fixture()
        b['support'] = [3,2,1]
        with self.assertRaises(ValueError):
            validate_pair(a,b)


if __name__=='__main__':
    unittest.main()
