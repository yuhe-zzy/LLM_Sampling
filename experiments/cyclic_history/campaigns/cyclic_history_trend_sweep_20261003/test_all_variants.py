import unittest

import numpy as np

from plot_all_variants import BASES, FAMILIES, select_grid


def fixture():
    data = {}
    for b, base in enumerate(BASES):
        for arm in ('ordinary',) + FAMILIES['reference'] + FAMILIES['feedback']:
            key = f'ipo_{base}_{arm}_b100_s0'
            data[key] = dict(record=dict(run_id=key, arm=arm, method='ipo',
                base_id=base, alpha=.8+b*.01, lambda_current=.8, beta_train=.2),
                pi=np.full((101,6,4),.25))
    return data


class AllVariantsTest(unittest.TestCase):
    def test_all_ten_arms_have_exact_matching_baselines(self):
        for family in FAMILIES:
            ordinary, variants = select_grid(fixture(), 'ipo', family)
            self.assertEqual(len(ordinary), 5)
            self.assertEqual([b for b,_ in variants], [1,1,2,2,3,3,4,4,5,5])
            self.assertEqual([r['record']['arm'] for _,r in variants], list(FAMILIES[family])*5)

    def test_parameter_mismatch_is_rejected(self):
        data = fixture()
        data['ipo_center_reference_half_b100_s0']['record']['alpha'] = .99
        with self.assertRaises(ValueError):
            select_grid(data, 'ipo', 'reference')

    def test_missing_variant_is_not_silently_omitted(self):
        data = fixture()
        del data['ipo_beta15_feedback_one_b100_s0']
        with self.assertRaises(KeyError):
            select_grid(data, 'ipo', 'feedback')

    def test_partial_or_nonfinite_data_is_rejected(self):
        for pi in (np.full((100,6,4),.25), np.full((101,6,4),np.nan)):
            data = fixture()
            data['ipo_center_ordinary_b100_s0']['pi'] = pi
            with self.assertRaises(ValueError):
                select_grid(data, 'ipo', 'reference')


if __name__ == '__main__':
    unittest.main()
