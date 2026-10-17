import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / 'campaigns/cyclic_history_stage_b100_results_20260929'
spec = importlib.util.spec_from_file_location('stage_b_public_plot', REPORT / 'plot_pi_curves.py')
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


class PublicPlotTests(unittest.TestCase):
    def test_public_table_reconstructs_all_probabilities(self):
        data, panels = plot.load_table(REPORT / 'pi_curves/pi_trajectories.csv')
        plot.validate_probabilities(data)
        self.assertEqual(len(data), 6)
        self.assertEqual([p['prompt_id'] for p in panels], [54, 251, 612, 737, 867, 945])
        for run in data.values():
            self.assertEqual(run['pi'].shape, (101, 6, 4))
        self.assertTrue(all(set(p) == {'prompt_id'} for p in panels))

    def test_incomplete_or_duplicate_table_rejected(self):
        rows = (REPORT / 'pi_curves/pi_trajectories.csv').read_text().splitlines(keepends=True)
        for invalid in (rows[:-1], rows + [rows[1]]):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'invalid.csv'
                path.write_text(''.join(invalid))
                with self.assertRaises(ValueError):
                    plot.load_table(path)

    def test_nonfinite_probabilities_rejected(self):
        data, _ = plot.load_table(REPORT / 'pi_curves/pi_trajectories.csv')
        data['ipo', 'ordinary']['pi'][10, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            plot.validate_probabilities(data)


if __name__ == '__main__':
    unittest.main()
