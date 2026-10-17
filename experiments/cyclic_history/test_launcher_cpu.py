"""Exercise stage selection with /bin/echo, never a scheduler or model."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(sys.platform != "win32" and shutil.which("bash"), "Linux bash needed")
class LauncherTests(unittest.TestCase):
    def preview(self, stage, index, approval=None):
        env = dict(os.environ, SLURM_JOB_ID="cpu-script-test", SLURM_ARRAY_TASK_ID=str(index),
                   PROJECT_ROOT=str(ROOT), PYTHON="/bin/echo", CUDA_VISIBLE_DEVICES="",
                   APPROVED_CALIBRATED_STAGE=approval if approval is not None else stage)
        return subprocess.run(["bash", str(ROOT / "slurm/cyclic_history_calibrated.sh"), stage],
                              env=env, capture_output=True, text=True, check=False)

    def test_stages_cover_exactly_the_frozen_plan(self):
        ids = []
        for stage, count in (("A", 4), ("B", 4), ("C", 2)):
            for index in range(count):
                result = self.preview(stage, index)
                self.assertEqual(result.returncode, 0, result.stderr)
                arguments = result.stdout.split()
                self.assertIn("--approve-calibrated-micropilot", arguments)
                ids.append(arguments[arguments.index("--run-id") + 1])
        plan = json.loads((ROOT / "experiments/cyclic_history/calibration/2026-09-28/experiment_plan.json").read_text())
        self.assertCountEqual(ids, [run["run_id"] for run in plan["runs"]])

    def test_invalid_stage_index_or_approval_cannot_launch(self):
        for stage, index, approval in (("A", 0, ""), ("B", 0, "A"), ("D", 0, "D"),
                                        ("C", 2, "C"), ("A", -1, "A"), ("A", 4, "A")):
            result = self.preview(stage, index, approval)
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("--execute", result.stdout)


if __name__ == "__main__":
    unittest.main()
