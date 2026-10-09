import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from common import judge_token_ids, load_json, save_jsonl, sha256, write_json
from oracle_math import classify_panel, generated_wr, preference_matrix, probability, wr_steps
from prepare_data import prompt_key, split_panels
from generate_evaluation import generation_seed
from train_oracle2 import configuration
from evaluate_wr import summarize
from history_math import build_outer_state, pair_distribution

HERE = Path(__file__).resolve().parent
PLAN = HERE / 'campaigns/oracle2_real_20261009/plan.json'


class OracleMathTests(unittest.TestCase):
    def test_unequal_mixture_can_be_cyclic(self):
        p = preference_matrix([0, 1, 2], [0, -3, -6])
        self.assertEqual(classify_panel(p)['group'], 'cyclic')
        np.testing.assert_allclose(p+p.T, 1)
        np.testing.assert_equal(np.diag(p), .5)

    def test_half_mixture_has_no_strict_majority_cycle(self):
        rng = np.random.default_rng(0)
        for _ in range(100):
            p = preference_matrix(rng.normal(size=4)*3, rng.normal(size=4)*3, .5)
            self.assertEqual(classify_panel(p, 0)['majority_triangle_count'], 0)

    def test_not_sigmoid_of_mixed_scalar(self):
        from oracle_math import sigmoid
        self.assertGreater(abs(float(probability(1, -3)) - float(sigmoid(np.array(-.6)))), .05)

    def test_groups_are_not_forced(self):
        self.assertEqual(classify_panel(np.full((4, 4), .5))['group'], 'ambiguous')
        self.assertEqual(classify_panel(preference_matrix([0, 1, 2, 3], [0, 1, 2, 3]))['group'], 'transitive')

    def test_strict_cycle_margin_boundary(self):
        p = np.array([[.5, .52, .48], [.48, .5, .52], [.52, .48, .5]])
        self.assertEqual(classify_panel(p)['group'], 'ambiguous')

    def test_reject_invalid_input(self):
        for t in ((0, 1), (1, float('nan')), (-1, 1)):
            with self.assertRaises(ValueError):
                probability(0, 0, temperatures=t)
        with self.assertRaises(ValueError):
            preference_matrix([0, float('inf'), 1], [0, 0, 1])

    def test_ties_and_reciprocal_wr(self):
        result = generated_wr([1]*4, [2]*4, [1]*4, [2]*4)
        self.assertEqual(result['oracle2_expected_win_rate'], .5)
        self.assertEqual(result['oracle2_majority_win_rate'], .5)
        self.assertEqual(result['pair_count'], 16)
        p = generated_wr([0, 1, 2, 3], [1, 0, 4, 2], [0, 1, 2, 3], [1, 0, 4, 2])
        self.assertAlmostEqual(p['oracle2_expected_win_rate'], .5)

    def test_cross_pair_not_paired_draw_wr(self):
        result = generated_wr([2, -1], [1, 2], [0, 3], [0, -2])
        p = probability(np.array([[2, -1], [-1, -4]]), np.array([[1, 3], [2, 4]]))
        self.assertAlmostEqual(result['oracle2_expected_win_rate'], p.mean())
        self.assertEqual(result['pair_count'], 4)

    def test_cadence_is_outer_0_through_100(self):
        self.assertEqual(wr_steps(), list(range(0, 101, 10)))
        self.assertEqual(len(wr_steps()), 11)
        with self.assertRaises(ValueError):
            wr_steps(99, 10)

    def test_full_probability_history_operator(self):
        p = preference_matrix([0, 1, 2, 0], [0, -3, -6, 1])[None]
        s = np.array([[-2., -3., -4., -5.]])
        for method, beta in [('ipo', .2), ('dpo', .8)]:
            ordinary = build_outer_state(s, s, s, p, method, .9, .8, beta)
            feedback = build_outer_state(s, s, s, p, method, .9, .8, beta, kappa=.5)
            np.testing.assert_allclose(feedback['offset'], 0)
            np.testing.assert_allclose(feedback['reference'], ordinary['reference'])
            current = s + [[0, .1, -.2, .3]]
            ref = build_outer_state(s, current, s, p, method, .9, .8, beta, nu=.45)
            np.testing.assert_allclose(ref['reference'], .1*s+.45*current+.45*s)
            self.assertLess(ref['solver_residual'].max(), 1e-10)

    def test_all_pairs_strictly_positive(self):
        _, _, w = pair_distribution(np.array([.85, .05, .05, .05]))
        self.assertEqual(len(w), 6)
        self.assertAlmostEqual(w.sum(), 1)
        self.assertTrue((w > 0).all())


class DataAndPlanTests(unittest.TestCase):
    def test_chat_template_mapping_default_is_explicitly_disabled(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return [1, 2, 3] if kwargs.get('return_dict') is False else {'input_ids': [1, 2, 3]}
        self.assertEqual(judge_token_ids(Tokenizer(), 'Hello', 'Hi'), [1, 2, 3])

    def test_bad_chat_template_output_cannot_pass_length_audit(self):
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return {'input_ids': [1, 2, 3], 'attention_mask': [1, 1, 1]}
        with self.assertRaises(TypeError):
            judge_token_ids(Tokenizer(), 'Hello', 'Hi')

    def test_split_disjoint_and_deterministic(self):
        rows = [dict(prompt_key=str(i)) for i in range(1000)]
        sizes = dict(calibration=100, train=500, evaluation=200)
        splits = split_panels(rows, sizes, 0)
        self.assertEqual(splits, split_panels(rows, sizes, 0))
        keys = [r['prompt_key'] for rs in splits.values() for r in rs]
        self.assertEqual(len(set(keys)), 800)
        self.assertEqual({k: len(v) for k, v in splits.items()}, sizes)

    def test_short_pool_fails_without_reducing_size(self):
        with self.assertRaises(ValueError):
            split_panels([dict(prompt_key='x')], dict(train=500), 0)

    def test_normalized_duplicates_cannot_leak(self):
        self.assertEqual(prompt_key(' A\n B '), prompt_key('A B'))
        with self.assertRaises(ValueError):
            split_panels([dict(prompt_key='x')]*2, dict(train=2), 0)

    def test_six_configurations_only(self):
        plan = load_json(PLAN)
        self.assertEqual(len(plan['runs']), 6)
        for run in plan['runs']:
            cfg = configuration(plan, run['run_id'])
            self.assertEqual(cfg['checkpoint_every'], 10)
            self.assertEqual(cfg['iters'], 100)
            self.assertEqual(cfg['num_prompts'], 500)
            self.assertEqual(cfg['support_probability'], 'softmax_sequence_sum')
            self.assertEqual(cfg['beta_train'], .2 if run['method'] == 'ipo' else .8)

    def test_repair_attempt_changes_paths_not_experimental_parameters(self):
        first = load_json(PLAN)
        second = load_json(PLAN.with_name('plan_attempt2.json'))
        for field in ('data_root', 'output_root', 'model_lock'):
            first.pop(field)
            second.pop(field)
        self.assertEqual(first, second)

    def test_generation_rng_matched_between_arms_independent_from_baseline(self):
        self.assertEqual(generation_seed(100780, 10, 5, 2), generation_seed(100780, 10, 5, 2))
        self.assertNotEqual(generation_seed(777, 0, 5, 2), generation_seed(100780, 0, 5, 2))
        self.assertEqual(len({generation_seed(777, 0, 5, r) for r in range(4)}), 4)

    def test_no_artifact_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'a.jsonl'
            save_jsonl(path, [dict(a=1)])
            with self.assertRaises(FileExistsError):
                save_jsonl(path, [dict(a=2)])

    def test_prompt_level_wr_summary_integration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = dict(model_lock=str(root/'lock.json'), runs=[dict(run_id='ipo_ordinary')],
                        dataset=dict(evaluation=2), evaluation=dict(steps=[0]),
                        oracle=dict(nemotron_weight=.6, temperatures=[1, 1]))
            write_json(root/'lock.json', {})
            write_json(root/'plan.json', plan)
            rows = [dict(id=f'{arm}:{key}:{draw}', run_id=arm, prompt_key=key, draw=draw, step=0,
                         group='cyclic' if key == 'x' else 'transitive')
                    for arm in ('baseline', 'ipo_ordinary') for key in ('x', 'y') for draw in range(4)]
            save_jsonl(root/'records.jsonl', rows)
            manifest = dict(state='COMPLETE', records_sha256=sha256(root/'records.jsonl'),
                            model_lock_sha256=sha256(root/'lock.json'), components={})
            for name in ('nemotron', 'skywork'):
                save_jsonl(root/f'{name}.jsonl', [dict(id=r['id'], reward=0) for r in rows])
                manifest['components'][name] = dict(sha256=sha256(root/f'{name}.jsonl'))
            write_json(root/'manifest.json', manifest)
            summarize(root/'plan.json', root/'records.jsonl', root, root/'results')
            import csv
            with (root/'results/wr_summary.csv').open(encoding='utf-8') as handle:
                output = list(csv.DictReader(handle))
            self.assertEqual(output[0]['prompt_count'], '2')
            self.assertEqual(output[0]['oracle2_expected_win_rate'], '0.5')
            self.assertEqual(output[-1]['prompt_count'], '0')
            self.assertEqual(output[-1]['oracle2_expected_win_rate'], '')


class QueueTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('oracle2_queue', HERE/'campaigns/oracle2_real_20261009/queue.py')
        self.queue = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.queue)

    def test_full_owner_or_uid_detection(self):
        outputs = ['123|alias|448057|RUNNING|gpu:1|n1\n124|yuhe32|1|PENDING|gpu:1|(Resources)',
                   'JobId=123 UserId=alias(448057) AllocTRES=gres/gpu=1,gres/gpu:h100=1']
        with patch.object(self.queue, 'run', side_effect=outputs):
            result = self.queue.owned_snapshot()
        self.assertEqual(len(result['owned_queue']), 2)
        self.assertEqual(len(result['owned_scontrol']), 1)
        self.assertIsNone(result['allocated_gpus'])

    def test_budget_and_readonly_preview_resources(self):
        for resource in self.queue.RESOURCES.values():
            self.assertLessEqual(resource['tasks']*resource['gpus'], 6)

    def test_terminal_scontrol_is_not_live_gpu_allocation(self):
        outputs = ['', 'JobId=123 UserId=yuhe32(448057) JobState=FAILED AllocTRES=gres/gpu=3,gres/gpu:h100=3']
        with patch.object(self.queue, 'run', side_effect=outputs):
            result = self.queue.owned_snapshot()
        self.assertEqual(result['owned_scontrol'], [])
        self.assertEqual(result['allocated_gpus'], 0)
        self.assertEqual(len(result['recent_terminal_scontrol']), 1)

    def test_existing_receipt_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'receipt.json'
            self.queue.save_new(path, dict(job_id='123'))
            with self.assertRaises(FileExistsError):
                self.queue.save_new(path, dict(job_id='456'))


if __name__ == '__main__':
    unittest.main()
