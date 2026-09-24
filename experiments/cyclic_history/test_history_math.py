"""CPU-only algebra/protocol tests. No LLM weights or experimental runs."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from history_math import (bt_objective_gradient_hessian, bt_scores, build_outer_state,
                          centered, describe_distribution, load_panels, pair_distribution,
                          population_delta, sample_pairs, sampler, sigmoid, softmax,
                          validate_preferences)
from run_cyclic_history import encode_panel, encode_response, main, resolve_config, validate_config

P = np.array([[.5, 1, 1, 0], [0, .5, 1, 1], [0, 0, .5, 1], [1, 0, 0, .5]])
PLAN = Path(__file__).with_name("experiment_plan.json")


def expected_loss(delta, p, mu, beta, method):
    i, j, weights = pair_distribution(mu)
    margins = delta[i] - delta[j]
    if method == "dpo":
        return np.sum(weights * (np.logaddexp(0, beta * margins) - p[i, j] * beta * margins))
    return np.sum(weights * (p[i, j] * (margins - 1 / (2 * beta)) ** 2
                             + (1 - p[i, j]) * (margins + 1 / (2 * beta)) ** 2))


class HistoryTests(unittest.TestCase):
    def test_legacy_diagonal(self):
        old = P.copy()
        np.fill_diagonal(old, 0)
        np.testing.assert_equal(validate_preferences(old), P)

    def test_missing_edge_rejected(self):
        incomplete = P.copy()
        incomplete[0, 2] = 0
        with self.assertRaises(ValueError):
            validate_preferences(incomplete)

    def test_nonfinite_rejected(self):
        with self.assertRaises(FloatingPointError):
            softmax([0, np.nan])
        with self.assertRaises(FloatingPointError):
            centered([0, np.inf])

    def test_sampler_is_positive_and_has_defined_lambda(self):
        mu = sampler(np.array([0., -1000, -2000, -3000]), .8)
        np.testing.assert_allclose(mu, [.85, .05, .05, .05])
        with self.assertRaises(ValueError):
            sampler(np.zeros(4), 1)

    def test_pair_weights_match_product_law(self):
        mu = np.array([.1, .2, .3, .4])
        i, j, w = pair_distribution(mu)
        np.testing.assert_allclose(w, mu[i] * mu[j] / np.sum(mu[i] * mu[j]))
        self.assertAlmostEqual(float(w.sum()), 1)
        np.testing.assert_allclose(np.mean(len(w) * w * np.arange(6)), np.sum(w * np.arange(6)))

    def test_bt_recovers_realizable_scores(self):
        truth = np.array([.6, -.8, .1, .4])
        p = sigmoid(truth[:, None] - truth[None, :])
        for mu in (np.ones(4) / 4, np.array([.05, .1, .7, .15])):
            v, info = bt_scores(p, mu)
            np.testing.assert_allclose(v, centered(truth), atol=1e-7)
            self.assertLess(info["residual"], 1e-9)

    def test_bt_handles_hard_cyclic_without_logit_infinities(self):
        result, info = bt_scores(P, np.ones(4) / 4)
        np.testing.assert_allclose(result, np.log(3) / 2 * np.array([1, 1, -1, -1]), atol=1e-8)
        self.assertLess(info["residual"], 1e-9)

    def test_bt_rejects_separable_hard_order(self):
        ordered = np.triu(np.ones((4, 4)), 1) + .5 * np.eye(4)
        with self.assertRaises(ValueError):
            bt_scores(ordered, np.ones(4) / 4)

    def test_population_deltas_minimize_actual_losses(self):
        mu = np.array([.05, .15, .7, .1])
        for method in ("ipo", "dpo"):
            delta, _ = population_delta(method, P, mu, 1.7)
            for index in range(4):
                e = np.eye(4)[index] * 1e-5
                gradient = (expected_loss(delta + e, P, mu, 1.7, method)
                            - expected_loss(delta - e, P, mu, 1.7, method)) / 2e-5
                self.assertAlmostEqual(float(gradient), 0., places=7)

    def test_bt_gradient_and_hessian(self):
        mu = np.array([.1, .2, .3, .4])
        v = np.array([.2, -.4, .3, -.1])
        _, g, h = bt_objective_gradient_hessian(v, P, mu)
        for index in range(4):
            e = np.eye(4)[index] * 1e-5
            up = bt_objective_gradient_hessian(v + e, P, mu)
            down = bt_objective_gradient_hessian(v - e, P, mu)
            self.assertAlmostEqual(float((up[0] - down[0]) / 2e-5), float(g[index]), places=8)
            np.testing.assert_allclose((up[1] - down[1]) / 2e-5, h[:, index], atol=1e-8)

    def test_first_step_and_same_state_reduce_to_ordinary(self):
        initial = np.array([[-1., -2, -4, -8]])
        current = np.array([[-2., -1, -5, -9]])
        for method in ("ipo", "dpo"):
            ordinary = build_outer_state(initial, current, current, [P], method, .9, .8, 1)
            for nu, kappa in ((.45, 0), (0, .25)):
                modified = build_outer_state(initial, current, current, [P], method, .9, .8, 1, nu, kappa)
                np.testing.assert_allclose(modified["target_logits"], ordinary["target_logits"], atol=1e-12)

    def test_reference_weights_and_no_mutation(self):
        initial = np.zeros((1, 4))
        current = np.array([[1., -2, 3, -.5]])
        previous = -current
        before = current.copy()
        state = build_outer_state(initial, current, previous, [P], "ipo", .9, .8, 1, .45, 0)
        np.testing.assert_allclose(state["reference"], .1 * initial + .45 * current + .45 * previous)
        np.testing.assert_array_equal(current, before)

    def test_sampling_extrapolation_translation(self):
        initial = np.zeros((1, 4))
        current = np.array([[1., -2, 3, -.5]])
        previous = -current
        for method in ("ipo", "dpo"):
            state = build_outer_state(initial, current, previous, [P], method, .9, .8, 1, 0, .25)
            expected = centered(state["reference"] + 1.25 * state["feedback_delta"]
                                - .25 * state["previous_feedback_delta"])
            np.testing.assert_allclose(state["target_logits"], expected, atol=1e-12)
            delta = state["target_logits"][0] - centered(state["effective_reference"])[0]
            for index in range(4):
                e = np.eye(4)[index] * 1e-5
                numerical_gradient = (expected_loss(delta + e, P, state["mu"][0], 1, method)
                                      - expected_loss(delta - e, P, state["mu"][0], 1, method)) / 2e-5
                self.assertAlmostEqual(float(numerical_gradient), 0., places=7)

    def test_ipo_extrapolation_matches_linear_payoff(self):
        initial = np.zeros((1, 4))
        current = np.array([[1., -2, 3, -.5]])
        previous = -current
        state = build_outer_state(initial, current, previous, [P], "ipo", .9, .8, 1.3, 0, .25)
        signed_mu = 1.25 * sampler(current, .8) - .25 * sampler(previous, .8)
        expected = centered(state["reference"] + ((P - .5) @ signed_mu[0])[None, :] / 1.3)
        np.testing.assert_allclose(state["target_logits"], expected, atol=1e-12)

    def test_pair_proposals_are_matched_across_arms(self):
        a = sample_pairs(np.array([[.1, .2, .3, .4], [.4, .3, .2, .1]]), 10, 0, 3)
        b = sample_pairs(np.ones((2, 4)) / 4, 10, 0, 3)
        self.assertEqual([x[:3] for x in a], [x[:3] for x in b])
        self.assertTrue(all(x[3] >= 0 for x in a + b))

    def test_metrics_do_not_confuse_relative_and_panel(self):
        scores = np.array([[-1., -2, -3, -4]])
        metrics = describe_distribution(scores, scores)
        self.assertAlmostEqual(float(metrics["relative_sequence_entropy"][0]), np.log(4))
        self.assertLess(float(metrics["panel_entropy"][0]), np.log(4))

    def test_six_configs_are_valid(self):
        plan = json.loads(PLAN.read_text())
        self.assertEqual(len(plan["runs"]), 6)
        for row in plan["runs"]:
            cfg = resolve_config(PLAN, row["run_id"])
            validate_config(cfg)
            self.assertEqual(cfg["seed"], 0)

    def test_preview_cannot_train(self):
        with patch("sys.argv", ["runner", "--run-id", "ipo_baseline_s0"]):
            with patch("run_cyclic_history.train") as mock_train, contextlib.redirect_stdout(io.StringIO()):
                main()
                mock_train.assert_not_called()

    def test_wrong_entrypoint_rejected(self):
        with patch("sys.argv", ["runner", "--run-id", "ipo_baseline_s0"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(required_scheme="lagged_sampling")

    def test_response_not_silently_truncated(self):
        class Tokenizer:
            eos_token_id = 9
            def __call__(self, text, **kwargs):
                return type("Encoding", (), {"input_ids": [1] * len(text)})()
        encoded = encode_response(Tokenizer(), "abcdefgh", "xy", 6)
        self.assertEqual(encoded["truncated_prompt_tokens"], 5)
        self.assertEqual(encoded["labels"], [-100, -100, -100, 1, 1, 9])
        with self.assertRaises(ValueError):
            encode_response(Tokenizer(), "abc", "abcde", 6)

    def test_panel_candidates_share_exact_prompt_tokens(self):
        class Tokenizer:
            eos_token_id = 99
            def __call__(self, text, **kwargs):
                return type("Encoding", (), {"input_ids": [ord(char) for char in text]})()
        panel = dict(prompt="abcdefgh", responses=["x", "xy", "xyz", "wxyz"])
        encoded = encode_panel(Tokenizer(), panel, 8)
        for row in encoded:
            self.assertEqual(row["input_ids"][:-row["response_tokens"]], [ord(c) for c in "fgh"])
            self.assertEqual(row["truncated_prompt_tokens"], 5)
            self.assertEqual(row["labels"][:3], [-100] * 3)

    def test_bt_solver_across_concentrated_policy_states(self):
        rng = np.random.default_rng(112)
        for scale in (.1, 1, 10, 100):
            for _ in range(25):
                mu = sampler(rng.normal(size=4) * scale, .8)
                scores, info = bt_scores(P, mu)
                self.assertTrue(np.isfinite(scores).all())
                self.assertLess(info["residual"], 1e-10)

    def test_panel_selection_preserves_seed_and_matrix(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "panels.jsonl"
            rows = [dict(prompt_id=i, prompt=f"p{i}", responses=["a", "b", "c", "d"],
                         preference_matrix=P.tolist()) for i in range(6)]
            path.write_text("\n".join(json.dumps(row) for row in rows))
            a, b = load_panels(path, 3, 123), load_panels(path, 3, 123)
            self.assertEqual(a, b)
            with self.assertRaises(ValueError):
                load_panels(path, 7, 123)


if __name__ == "__main__":
    unittest.main(verbosity=2)
