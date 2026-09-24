"""Optional Torch CPU tests with synthetic tensors and a tiny random model only."""
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from history_math import build_outer_state
from run_cyclic_history import (build_batch, pair_loss, resolve_config, sequence_scores, train_round)


@unittest.skipIf(torch is None, "Torch unavailable; run on server CPU environment")
class TensorTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_scores_ignore_prompt_and_padding_and_use_sum(self):
        logits = torch.randn(2, 6, 13, requires_grad=True)
        labels = torch.tensor([[-100, -100, 3, 4, 5, -100], [-100, 2, 1, -100, -100, -100]])
        result, counts = sequence_scores(logits, labels)
        expected = []
        for row in range(2):
            valid = labels[row, 1:] != -100
            logp = torch.log_softmax(logits[row, :-1][valid].float(), -1)
            expected.append(logp.gather(1, labels[row, 1:][valid, None]).sum())
        torch.testing.assert_close(result, torch.stack(expected))
        self.assertEqual(counts.tolist(), [3, 2])
        result.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_loss_matches_legacy_hard_label_objectives(self):
        delta = torch.tensor([-.2, .3], requires_grad=True)
        p = torch.ones(2)
        torch.testing.assert_close(pair_loss(delta, p, 2, "ipo"), (delta - .25) ** 2)
        torch.testing.assert_close(pair_loss(delta, p, 2, "dpo"), -torch.nn.functional.logsigmoid(2 * delta))

    def test_padding_labels_and_attention(self):
        data = [dict(input_ids=[1, 2, 3], labels=[-100, 2, 3]),
                dict(input_ids=[1, 4], labels=[-100, 4])]
        batch, labels = build_batch(data, 0, "cpu")
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1], [1, 1, 0]])
        self.assertEqual(labels.tolist(), [[-100, 2, 3], [-100, 4, -100]])

    def test_nonfinite_scores_fail(self):
        logits = torch.full((1, 2, 3), float("nan"))
        with self.assertRaises(FloatingPointError):
            sequence_scores(logits, torch.tensor([[-100, 1]]))

    def test_dpo_cpu_microstep_updates_only_lora_and_freezes_reference(self):
        self.check_cpu_microstep("dpo")

    def test_ipo_cpu_microstep_updates_only_lora_and_freezes_reference(self):
        self.check_cpu_microstep("ipo")

    def check_cpu_microstep(self, method):
        from transformers import Qwen2Config, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
        cfg = resolve_config(Path(__file__).with_name("experiment_plan.json"), method + "_reference_s0")
        cfg.update(pairs_per_prompt=3, grad_accum=2, batch_size=2, epochs_per_iter=1,
                   warmup_ratio=0, lr=1e-3)
        tiny = AutoModelForCausalLM.from_config(Qwen2Config(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32))
        model = get_peft_model(tiny, LoraConfig(r=2, lora_alpha=4, lora_dropout=0,
                                               task_type="CAUSAL_LM", target_modules=["q_proj", "v_proj"]))
        fixed_before = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
        train_before = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        encoded = [dict(input_ids=[1, 2, j + 3, 12], labels=[-100, -100, j + 3, 12]) for j in range(8)]
        p = np.array([[.5, 1, 1, 0], [0, .5, 1, 1], [0, 0, .5, 1], [1, 0, 0, .5]])
        matrices = np.stack([p, p])
        scores = np.array([[-2., -3, -4, -5], [-4., -2, -3, -5]])
        state = build_outer_state(scores, scores + .2, scores, matrices, method, .9, .8, 1, .45, 0)
        reference_before = state["effective_reference"].copy()
        result = train_round(model, encoded, matrices, state, cfg, 0, "cpu", 0)
        self.assertEqual(result["train_pairs"], 6)
        self.assertEqual(result["optimizer_steps"], 2)
        np.testing.assert_array_equal(state["effective_reference"], reference_before)
        for name, param in model.named_parameters():
            if name in fixed_before:
                torch.testing.assert_close(param, fixed_before[name], atol=0, rtol=0)
        self.assertTrue(any(not torch.equal(p, train_before[n]) for n, p in model.named_parameters() if n in train_before))


if __name__ == "__main__":
    unittest.main(verbosity=2)
