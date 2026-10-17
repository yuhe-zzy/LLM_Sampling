"""Focused checks for the public, text-free 2026-10-13 result reconciliation."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('publisher', ROOT/'scripts/publish_results_20261013.py')
PUBLISHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PUBLISHER)
LEGACY = ROOT/'experiments/legacy_sampling_analysis/snapshot_20260911'


class PublicationTests(unittest.TestCase):
    def test_private_text_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'private.csv'
            for text in ('prompt,response\n12,private\n', 'step,metric\n0,private sentence\n'):
                path.write_text(text, encoding='utf-8')
                with self.assertRaises(AssertionError):
                    PUBLISHER.check_csv(path)

    def test_checkpoint_series_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'public.csv'
            path.write_text('method,series\nipo,0:0.5; 20:0.7\n', encoding='utf-8')
            self.assertEqual(len(PUBLISHER.check_csv(path)), 1)
            with self.assertRaises(AssertionError):
                PUBLISHER.check_csv(path, numeric_only=True)

    def test_conflicting_result_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            first, second = Path(directory)/'a.csv', Path(directory)/'b.csv'
            first.write_text('step,value\n0,1\n', encoding='utf-8')
            second.write_text('step,value\n0,2\n', encoding='utf-8')
            with self.assertRaises(AssertionError):
                PUBLISHER.copy_checked(first, second)
            self.assertIn('0,2', second.read_text())

    def test_import_hashes_and_numeric_data(self):
        manifest = json.loads((ROOT/'docs/result_import_manifest_20261013.json').read_text())
        self.assertEqual(len(manifest['files']), 118)
        for item in manifest['files']:
            path = ROOT/item['path']
            self.assertEqual(PUBLISHER.published_sha(path), item['sha256'])
            if path.suffix == '.csv':
                PUBLISHER.check_csv(path, numeric_only='coordinate' in item)

    def test_legacy_coverage_and_invalid_tail(self):
        with (LEGACY/'metrics/selected_policy.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 10*3*150)
        for row in rows:
            invalid = row['method'] == 'dpo' and float(row['rho']) == 0 and int(row['step']) >= 82
            self.assertEqual(row['valid'] == 'False', invalid)
            if invalid:
                self.assertTrue(all(row[f'p{k}'] == '' for k in range(4)))
            else:
                self.assertAlmostEqual(sum(float(row[f'p{k}']) for k in range(4)), 1, places=8)
        with (LEGACY/'metrics/prompt_diagnostics.csv').open(newline='') as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 5000)

    def test_inventory_hashes(self):
        inventory = json.loads((ROOT/'docs/result_artifacts_20261013.json').read_text())
        self.assertEqual(inventory['file_count'], len(inventory['files']))
        for item in inventory['files']:
            path = ROOT/item['path']
            self.assertIn(path.suffix, ('.csv', '.png', '.pdf'))
            data = path.read_bytes()
            if item['hash_encoding'] == 'lf_normalized':
                data = data.replace(b'\r\n', b'\n')
            self.assertEqual(len(data), item['bytes'])
            self.assertEqual(hashlib.sha256(data).hexdigest(), item['sha256'])


if __name__ == '__main__':
    unittest.main()
