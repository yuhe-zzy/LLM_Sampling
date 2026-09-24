import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import experiment

FULL = all(importlib.util.find_spec(name) is not None
           for name in ("torch", "transformers", "peft", "pandas"))
if FULL:
    import torch
    import sequence_utils as scoring
    import run_preference_oracle_core as core


class RecipeTests(unittest.TestCase):
    def test_production_metrics_include_plot_identity(self):
        tree = ast.parse((ROOT / "scripts/run_preference_oracle_core.py").read_text())
        row = next(n.value for n in ast.walk(tree) if isinstance(n, ast.AnnAssign)
                   and isinstance(n.target, ast.Name) and n.target.id == "metric_row")
        keys = {k.value for k in row.keys if isinstance(k, ast.Constant)}
        self.assertTrue({"seed", "loss_type", "alpha", "lambda", "beta", "iter",
                         "prompt_relative_sequence_entropy_mean"} <= keys)

    def test_recipe_sizes_and_no_duplicate_ids(self):
        sizes = {"oracle": 16, "nonoracle_transitive": 2, "cyclic_sequence_sum": 2, "cyclic_sampling_sweep": 10}
        for name, size in sizes.items():
            config = experiment.load_config(ROOT / "configs" / (name + ".json"))
            self.assertEqual(len(experiment.enumerate_runs(config)), size)

    def test_oracle_stop80_is_81_states(self):
        config = experiment.load_config(ROOT / "configs/oracle.json")
        for run in experiment.enumerate_runs(config):
            command, _ = experiment.build_command(config, run, "model", "data", "out", "reward")
            self.assertEqual(command[command.index("--iters") + 1], "81")
            self.assertEqual(command[command.index("--oracle_eval_every") + 1], "20")
            self.assertEqual(command[command.index("--oracle_train_pairs") + 1], "1")

    def test_static_disables_reward_loading(self):
        config = experiment.load_config(ROOT / "configs/nonoracle_transitive.json")
        command, _ = experiment.build_command(config, experiment.enumerate_runs(config)[0],
                                               "model", "data", "out", "reward")
        for flag in ("--enable_oracle", "--oracle_train_pairs"):
            self.assertEqual(command[command.index(flag) + 1], "0")
        self.assertNotIn("--oracle_model_path", command)
        self.assertTrue(command[1].endswith("run_ipo.py"))

    def test_cyclic_sweep_uses_standard_core(self):
        config = experiment.load_config(ROOT / "configs/cyclic_sampling_sweep.json")
        runs = experiment.enumerate_runs(config)
        self.assertEqual(runs[0]["parameters"]["beta"], 10)
        self.assertEqual(runs[5]["parameters"]["beta"], 1)
        command, _ = experiment.build_command(config, runs[0], "model", "data", "out", "reward")
        self.assertEqual(Path(command[1]), ROOT / "scripts/run_ipo.py")
        self.assertEqual(command[command.index("--enable_oracle") + 1], "0")
        self.assertEqual(command[command.index("--iters") + 1], "151")

    def test_no_token_average_training_or_scoring_entry_points(self):
        self.assertFalse(any((ROOT / "scripts/legacy").glob("*.py")))
        for folder in (ROOT / "scripts", ROOT / "experiments"):
            for path in folder.rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.Name)):
                        name = getattr(node, "name", getattr(node, "id", ""))
                        self.assertNotIn("avg_logprob", name, str(path))
                        self.assertNotIn("sum_and_avg", name, str(path))

    def test_preview_never_imports_torch_or_creates_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "must_not_exist"
            code = (
                "import sys; sys.path.insert(0, " + repr(str(ROOT / "scripts")) + "); "
                "import experiment; sys.argv=['experiment', '--config', "
                + repr(str(ROOT / "configs/oracle.json")) + ", '--index','0','--output-root',"
                + repr(str(target)) + "]; experiment.main(); "
                "assert 'torch' not in sys.modules"
            )
            result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("PREVIEW ONLY", result.stdout)
            self.assertFalse(target.exists())

    def test_execute_rejects_missing_allocation_before_gpu_import(self):
        env = dict(os.environ)
        env.pop("SLURM_JOB_ID", None)
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/experiment.py"), "--config",
             str(ROOT / "configs/oracle.json"), "--index", "0", "--execute"],
            env=env, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("requires an approved Slurm", result.stderr)

    def test_duplicate_grid_rejected(self):
        config = experiment.load_config(ROOT / "configs/oracle.json")
        config["grid"]["alpha"] = [0.8, 0.8]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            experiment.enumerate_runs(config)

    def test_history_paths_portable_and_shared_controls(self):
        plan = json.loads((ROOT / "experiments/cyclic_history/experiment_plan.json").read_text())
        self.assertEqual(len(plan["runs"]), 6)
        self.assertEqual(sum(r["scheme"] == "ordinary" for r in plan["runs"]), 2)
        self.assertEqual(plan["common"]["seed"], 0)
        for key in ("model_path", "eval_path", "output_root"):
            self.assertFalse(Path(plan["common"][key]).is_absolute())


@unittest.skipUnless(FULL, "Install the Torch/Transformers/PEFT environment for core tests")
class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_all_standard_recipe_flags_parse(self):
        for name in ("oracle", "nonoracle_transitive", "cyclic_sequence_sum", "cyclic_sampling_sweep"):
            config = experiment.load_config(ROOT / "configs" / (name + ".json"))
            for run in experiment.enumerate_runs(config):
                command, _ = experiment.build_command(config, run, "model", "data", "out", "reward")
                parser = argparse.ArgumentParser()
                core.add_common_args(parser, run["method"])
                args = parser.parse_args(command[2:])
                core.validate_args(args)

    def test_sampling_semantics_distinguish_oracle_and_static(self):
        dynamic = core.sampling_metadata(argparse.Namespace(enable_oracle=1, oracle_train_pairs=1, preference_case="transitive"))
        static = core.sampling_metadata(argparse.Namespace(enable_oracle=0, oracle_train_pairs=0, preference_case="cyclic"))
        self.assertEqual(dynamic["lambda_meaning"], "initial_generator_probability")
        self.assertEqual(static["lambda_meaning"], "model_induced_pair_target_weight")
        self.assertEqual(static["training_logprob_reduction"], "sequence_sum")
        self.assertEqual(static["sampling_protocol"], "static_sequence_margin_pair_weights")
        self.assertEqual(dynamic["eval_support_ranking"], "sequence_sum")

    def test_losses_match_positive_pair_objectives(self):
        delta = torch.tensor([-.2, .3], requires_grad=True)
        torch.testing.assert_close(core.dpo_loss_from_delta(delta, 2), -torch.nn.functional.logsigmoid(2*delta))
        torch.testing.assert_close(core.ipo_loss_from_delta(delta, 2), (delta-.25)**2)
        (core.dpo_loss_from_delta(delta, 2).sum() + core.ipo_loss_from_delta(delta, 2).sum()).backward()
        self.assertTrue(torch.isfinite(delta.grad).all())

    def test_sequence_sum_masks_prompt_and_padding(self):
        torch.manual_seed(7)
        logits = torch.randn(2, 6, 11, requires_grad=True)
        labels = torch.tensor([[-100, -100, 3, 4, 5, -100], [-100, 3, 2, -100, -100, -100]])
        scores, counts = core.sum_logprob_and_count_from_outputs(logits, labels)
        expected = []
        for row in range(2):
            valid = labels[row, 1:] != -100
            logp = torch.log_softmax(logits[row, :-1][valid], -1)
            expected.append(logp.gather(1, labels[row, 1:][valid, None]).sum())
        torch.testing.assert_close(scores, torch.stack(expected))
        self.assertEqual(counts.tolist(), [3, 2])
        scores.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_nonfinite_scores_fail_instead_of_uniform(self):
        with self.assertRaises(FloatingPointError):
            core.safe_softmax_np([0., float("nan")])
        with patch.object(scoring, "build_batch", return_value={
            "input_ids": torch.zeros((1, 2), dtype=torch.long),
            "attention_mask": torch.ones((1, 2)), "labels": torch.tensor([[-100, 0]])
        }):
            model = lambda **kwargs: argparse.Namespace(logits=torch.full((1, 2, 2), float("nan")))
            with self.assertRaises(FloatingPointError):
                core.batch_sequence_logprob(model, None, ["p"], ["y"], 8, "cpu")

    def test_reference_uses_sum_and_stays_frozen(self):
        model, initial = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
        with torch.no_grad():
            model.weight.fill_(8)
            initial.weight.fill_(2)
        rows = [{"prompt": "p", "chosen": "c", "rejected": "r"}] * 2
        dataset = core.DynamicOraclePairDataset(rows)

        def scores(net, tok, prompts, responses, max_length, device):
            value = float(net.weight[0, 0].item())
            sums = torch.full((len(prompts),), value)
            return sums, torch.full((len(prompts),), 10)

        with patch.object(core, "batch_sequence_logprob", side_effect=scores):
            frozen = core.freeze_outer_reference_scores(model, initial, None, dataset, .8, 1, 32, "cpu")
        self.assertAlmostEqual(frozen[0]["chosen_reference_score"], 6.8, places=5)
        with torch.no_grad():
            model.weight.fill_(99)
        self.assertAlmostEqual(frozen[0]["chosen_reference_score"], 6.8, places=5)
        self.assertTrue(model.training)

    def test_bad_frozen_reference_lengths_rejected(self):
        with self.assertRaises(ValueError):
            core.FrozenReferenceDataset([{}], [], [])

    def test_nonoracle_entry_rejects_oracle_flag_before_loading(self):
        argv = ["run_ipo", "--model_path", "missing", "--pairs_path", "missing",
                "--eval_prompts_path", "missing", "--preference_case", "transitive",
                "--enable_oracle", "1"]
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            core.run_experiment("ipo", nonoracle=True)

    def test_cyclic_oracle_training_rejected_before_loading(self):
        argv = ["run_dpo_oracle", "--model_path", "missing", "--pairs_path", "missing",
                "--eval_prompts_path", "missing", "--preference_case", "cyclic"]
        with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
            core.run_experiment("dpo")


if __name__ == "__main__":
    unittest.main()
