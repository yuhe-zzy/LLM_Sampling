import json
from pathlib import Path
import unittest

from calibrated_protocol import contract_hash
from make_100_round_plan import make_plan, plan_fingerprint
from run_cyclic_history import validate_config

ROOT = Path(__file__).resolve().parent


class CampaignPlanTests(unittest.TestCase):
    def test_portable_plans_match_executed_contracts(self):
        for stage in ('B', 'C'):
            plan = make_plan(stage)
            saved = json.loads((ROOT / 'plans' / f'stage_{stage.lower()}100.json').read_text())
            self.assertEqual(plan, saved)
            launch = ROOT / 'campaigns' / f'cyclic_history_stage_{stage.lower()}100_launch_20260929'
            actual = json.loads((launch / f'experiment_plan_{stage.lower()}100.json').read_text())
            self.assertEqual(len(plan['runs']), 6 if stage == 'B' else 2)
            for row, recorded in zip(plan['runs'], actual['runs']):
                self.assertEqual(row, recorded)
                cfg = dict(plan['common'], **row, protocol=plan['protocol'])
                validate_config(cfg)
                self.assertEqual(contract_hash(cfg), row['calibration_contract_sha256'])
                self.assertEqual(cfg['iters'], 100)
            for key in ('eval_path', 'model_path', 'output_root'):
                self.assertFalse(Path(plan['common'][key]).is_absolute())

    def test_mixed_contrast_has_only_the_prespecified_orientation_change(self):
        b, c = make_plan('B'), make_plan('C')
        for mixed in c['runs']:
            ordinary = next(r for r in b['runs'] if r['run_id'] == f"{mixed['method']}_ordinary_b100_s0")
            changed = {k for k in mixed if mixed[k] != ordinary[k]}
            self.assertEqual(changed, {'run_id', 'orientations', 'predictions',
                                      'transformed_support_sha256', 'calibration_contract_sha256'})
            self.assertEqual(mixed['orientations'], [-1, 1, -1, 1, -1, 1])
            self.assertEqual(mixed['nu'], 0)
            self.assertEqual(mixed['kappa'], 0)

    def test_relocation_does_not_change_mathematics(self):
        for stage in ('B', 'C'):
            self.assertEqual(make_plan(stage)['runs'],
                             make_plan(stage, 'elsewhere/model', 'elsewhere/panels', 'elsewhere/out')['runs'])

    def test_new_stage_is_not_silently_invented(self):
        with self.assertRaises(ValueError):
            make_plan('D')

    def test_provenance_is_independent_of_json_line_endings(self):
        original = {'alpha': .9, 'values': [0, 1, 2]}
        windows = json.dumps(original, indent=2).replace('\n', '\r\n')
        self.assertEqual(plan_fingerprint(original), plan_fingerprint(json.loads(windows)))
        self.assertNotEqual(plan_fingerprint(original), plan_fingerprint(dict(original, alpha=.8)))


if __name__ == '__main__':
    unittest.main()
