import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from make_100_round_plan import make_plan
from population_calibration import balanced_cycle

PATH = Path(__file__).resolve().parent/'campaigns/cyclic_history_stage_c100_results_20260929/analyze_stage_c.py'
spec = importlib.util.spec_from_file_location('stage_c_audit', PATH)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def example(stage):
    plan = make_plan(stage)
    cfg = dict(plan['common'], **plan['runs'][0])
    panels = [dict(prompt_id=pid, prompt='synthetic', responses=['a','b','c','d'],
                   preference_matrix=balanced_cycle(.8, orientation, role).tolist())
              for pid, orientation, role in zip(C.PIDS, cfg['orientations'], cfg['calibration']['response_roles'])]
    return dict(cfg=cfg, panels=panels, data=[dict(sequence_sum_logprob=np.zeros((6,4)),
                                                response_token_count=np.ones((6,4)))])


class StageCAnalysisTests(unittest.TestCase):
    def test_constant_complex_phase_is_arbitrary(self):
        z = np.array([[1+2j, 2-1j], [3-1j, 1+4j], [-1+2j, 3j]])
        np.testing.assert_allclose(C.canonical_phase(z),
                                   C.canonical_phase(z*np.exp(1j*np.array([.7, np.pi]))), atol=1e-12)
        np.testing.assert_allclose(abs(C.canonical_phase(z)), abs(z))

    def test_zero_initial_mode_fails_closed(self):
        with self.assertRaises(ValueError):
            C.canonical_phase(np.array([[0j], [1j]]))

    def test_declared_orientation_contrast_passes(self):
        C.validate_contrast(example('B'), example('C'))

    def test_wrong_preference_or_response_is_rejected(self):
        for key in ('preference_matrix', 'responses'):
            mixed = copy.deepcopy(example('C'))
            mixed['panels'][1][key] = (balanced_cycle(.8, -1).tolist() if key=='preference_matrix'
                                      else ['changed','b','c','d'])
            with self.assertRaises(AssertionError):
                C.validate_contrast(example('B'), mixed)

    def test_unmatched_initial_score_is_rejected(self):
        mixed = example('C')
        mixed['data'][0]['sequence_sum_logprob'][0,0] = 1
        with self.assertRaises(AssertionError):
            C.validate_contrast(example('B'), mixed)


if __name__ == '__main__':
    unittest.main()
