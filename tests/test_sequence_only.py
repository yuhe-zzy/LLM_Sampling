"""Unequal-length regressions for scoring, sampling, ranking and CSV imports."""
import importlib.util
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
FULL = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "peft", "pandas"))
if FULL:
    import numpy as np
    import pandas as pd
    import torch
    import sequence_utils as utils
    import run_preference_oracle_core as core
    import reconstruct_oracle_relative_entropy as reconstruction


class Tokenizer:
    eos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[2 + ord(c) % 5 for c in text])


@unittest.skipUnless(FULL, "Install full CPU dependencies")
class SequenceOnlyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_unequal_length_scores_and_gradients_are_sums(self):
        batch = utils.build_batch(Tokenizer(), ["pp", "pp"], ["a", "abc"], 20, "cpu")
        logits = torch.zeros((*batch["labels"].shape, 8), requires_grad=True)
        scores, counts = utils.sum_logprob_and_count_from_outputs(logits, batch["labels"])
        torch.testing.assert_close(counts, torch.tensor([2, 4]))
        torch.testing.assert_close(scores, -counts.float() * np.log(8))
        for method in ("ipo", "dpo"):
            logits.grad = None
            core.loss_from_delta((scores[0] - scores[1]).unsqueeze(0), 1, method).sum().backward(retain_graph=True)
            # Every response token has the same coefficient, independent of length.
            for row in range(2):
                valid = batch["labels"][row, 1:] != -100
                grad = logits.grad[row, :-1][valid]
                norms = torch.linalg.vector_norm(grad, dim=-1)
                torch.testing.assert_close(norms, norms[0].expand_as(norms))
            torch.testing.assert_close(logits.grad[0].abs().sum() * 2, logits.grad[1].abs().sum())

    def test_batch_api_returns_only_sequence_scores_and_counts(self):
        def model(input_ids, **kwargs):
            return SimpleNamespace(logits=torch.zeros((*input_ids.shape, 8)))
        scores, counts = utils.batch_sequence_logprob(model, Tokenizer(), ["p", "p"], ["a", "abc"], 20, "cpu")
        torch.testing.assert_close(scores, -counts.float() * np.log(8))

    def test_pair_weights_use_sequence_margins(self):
        dataset = utils.PairDataset([
            {"prompt": "p", "chosen": "long", "rejected": "r"},
            {"prompt": "p", "chosen": "short", "rejected": "r"},
        ])
        def scorer(model, tok, prompts, responses, *args):
            values = {"long": (-4., 10), "short": (-3., 2), "r": (-2., 2)}
            return tuple(torch.tensor([values[y][i] for y in responses]) for i in (0, 1))
        with patch.object(utils, "batch_sequence_logprob", side_effect=scorer):
            _, diagnostic, _, _ = utils.build_prompt_aware_training_subset(
                None, None, dataset, utils.build_prompt_to_pair_indices(dataset),
                random.Random(0), 1, 2, 1, 1, 0, 20, "cpu", 2, 0, 1e6)
        np.testing.assert_allclose(diagnostic.margin_sequence_logprob, [-2, -1])
        np.testing.assert_allclose(diagnostic.induced_pair_prob, utils.safe_softmax_np([-2, -1]))

    def test_candidate_ranking_is_not_length_normalized(self):
        with patch.object(utils, "generate_candidate_responses", return_value=["long", "short"]), \
             patch.object(utils, "batch_sequence_logprob", return_value=(torch.tensor([-4., -3.]), torch.tensor([10, 2]))):
            result = utils.build_generated_eval_set(
                torch.nn.Linear(1, 1), None, ["p"], [0], 1, 2, 1, 8, True, .8, .95, 2, 20, 0, "cpu")
        self.assertEqual(result[2], [["short"]])
        self.assertEqual(result[4][0]["sum_logprob"], -3.)
        self.assertNotIn("avg_logprob", result[4][0])

    def test_new_dump_schema_reconstructs_without_average_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            dumps = run / "iter_dumps_test"
            dumps.mkdir()
            for t in (0, 1):
                sums = np.array([-4. + t, -8.])
                ref = np.array([-4., -8.])
                q = utils.safe_softmax_np(sums)[None, :]
                qr = utils.safe_softmax_np(sums - ref)[None, :]
                core.dump_prompt_metrics(
                    str(dumps), t, "transitive", [0], ["p"], [["a", "abc"]], [["test", "test"]],
                    [(0, 2)], 2, sums, ref, np.array([2, 4]), q,
                    np.array([utils.entropy_from_probs(q[0])]), np.array([np.nan]), np.array([0]), None,
                    qr, np.array([utils.entropy_from_probs(qr[0])]), np.array([np.nan]), np.array([0]), None)
                frame = pd.read_csv(dumps / f"iter_{t:04d}_prompt_metrics.csv")
                self.assertFalse(any("avg" in c for c in frame.columns))
                self.assertIn("sequence_prob_0", frame)
            pd.DataFrame([dict(iter=t, loss_type="ipo", alpha=.8, **{"lambda": .5}, beta=1,
                               seed=0, tau=1, oracle_win_rate=.5) for t in (0, 1)]).to_csv(run / "metrics_test.csv", index=False)
            output = reconstruction.reconstruct_run(run, Tokenizer(), 20)
            result = pd.read_csv(output)
            self.assertAlmostEqual(result.prompt_relative_sequence_entropy_mean.iloc[0], np.log(2))
            self.assertLess(result.prompt_relative_sequence_entropy_mean.iloc[1], np.log(2))
            self.assertFalse(any("normalized_entropy" in c for c in result))
            self.assertEqual(result.seed.tolist(), [0, 0])

    def test_saved_sequence_scores_take_priority_over_archived_columns(self):
        row = pd.Series({"sequence_logprob_0": -4., "sequence_logprob_1": -3.,
                         "avg_logprob_0": 999., "avg_logprob_1": 999.})
        np.testing.assert_allclose(reconstruction.sequence_scores(row, 2, np.array([10, 2])), [-4, -3])

    def test_archived_csv_conversion_outputs_only_sequence_entropy(self):
        frame = pd.DataFrame([{"K": 2, "avg_logprob_0": -1., "avg_logprob_1": -2.}])
        counts, reference = np.array([[2., 4.]]), np.array([[-4., -8.]])
        probabilities, mode = reconstruction.prompt_probabilities(frame, counts, reference, 1.)
        np.testing.assert_allclose(probabilities[0], utils.safe_softmax_np([2., 0.]))
        self.assertEqual(mode, "imported_archived_scores_times_token_count")


if __name__ == "__main__":
    unittest.main()
