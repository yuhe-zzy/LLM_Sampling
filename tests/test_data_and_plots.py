import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
HAS_PLOTS = all(importlib.util.find_spec(p) for p in ("pandas", "matplotlib", "numpy"))
if HAS_PLOTS:
    import pandas as pd
    from plot_results import collect, render


class DataTests(unittest.TestCase):
    def test_synthetic_builders_are_runnable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.jsonl"
            rows = []
            for pid in range(3):
                for rid in range(4):
                    rows.append(dict(prompt=f"synthetic prompt {pid}", response=f"answer {rid}",
                                     helpfulness=rid, correctness=rid, coherence=rid,
                                     complexity=rid, verbosity=rid))
            raw.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            for kind in ("", "_cyclic"):
                pairs, panel = root / f"pairs{kind}.jsonl", root / f"panels{kind}.jsonl"
                cmd = [sys.executable, str(ROOT / "scripts" / f"build{kind}_pairs.py"),
                       "--input", str(raw), "--out_pairs", str(pairs),
                       "--out_eval_prompts", str(panel), "--input_format", "response",
                       "--score_fields", "helpfulness,correctness,coherence,complexity,verbosity",
                       "--dedup_responses", "--keep_exact_k", "4", "--eval_prompts", "3", "--seed", "0"]
                if kind:
                    cmd += ["--comparison_mode", "full_tournament", "--order_policy", "score_desc"]
                else:
                    cmd += ["--pair_mode", "all"]
                result = subprocess.run(cmd, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                built = [json.loads(s) for s in pairs.read_text().splitlines()]
                panels = [json.loads(s) for s in panel.read_text().splitlines()]
                self.assertEqual(len(built), 18)
                self.assertEqual(len(panels), 3)
                self.assertTrue(all(r["chosen"] != r["rejected"] for r in built))


@unittest.skipUnless(HAS_PLOTS, "Install plotting dependencies")
class PlotTests(unittest.TestCase):
    def write_metrics(self, path, method="ipo", legacy=False):
        frame = pd.DataFrame({
            "iter": [0, 1, 20], "loss_type": method, "alpha": .8, "lambda": .5,
            "beta": 1, "seed": 0, "outer_reference_frozen": True,
            "prompt_relative_sequence_entropy_mean": [1.5, 1.4, 1.2],
            "oracle_win_rate": [.5, float("nan"), .7],
        })
        if legacy:
            frame = frame.rename(columns={"prompt_relative_sequence_entropy_mean": "prompt_entropy_mean"})
        frame.to_csv(path, index=False)

    def test_plot_outputs_and_partial_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_metrics(root / "metrics_ipo.csv")
            self.write_metrics(root / "metrics_dpo.csv", "dpo")
            render(collect(root), root / "figures")
            self.assertEqual(len(list((root / "figures").glob("*.pdf"))), 4)
            self.assertEqual(len(list((root / "figures").glob("*.png"))), 4)
            summary = pd.read_csv(root / "figures/summary.csv")
            self.assertFalse(summary.iter80_and_wr80_complete.any())

    def test_legacy_entropy_never_substituted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self.write_metrics(path / "metrics_old.csv", legacy=True)
            with self.assertRaisesRegex(ValueError, "not a corrected"):
                collect(path)

    def test_duplicate_attempts_not_concatenated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            self.write_metrics(path / "metrics_one.csv")
            self.write_metrics(path / "metrics_two.csv")
            with self.assertRaisesRegex(ValueError, "Duplicate configuration"):
                collect(path)


if __name__ == "__main__":
    unittest.main()
