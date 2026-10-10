import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import validate_snapshot as snapshot


class SnapshotTests(unittest.TestCase):
    def test_original_snapshot(self):
        self.assertEqual(snapshot.validate()["complete_to_80"], 12)

    def test_changed_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "snapshot"
            shutil.copytree(snapshot.ROOT, root, ignore=shutil.ignore_patterns("__pycache__"))
            path = root / "oracle_curves_20260911.csv"
            path.write_bytes(path.read_bytes() + b"\n")
            with patch.object(snapshot, "ROOT", root), self.assertRaises(AssertionError):
                snapshot.validate()

    def test_bad_aggregate_fails_even_with_updated_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "snapshot"
            shutil.copytree(snapshot.ROOT, root, ignore=shutil.ignore_patterns("__pycache__"))
            path = root / "oracle_curves_20260911.csv"
            frame = snapshot.pd.read_csv(path)
            frame.loc[0, "oracle_win_rate"] = 0.0
            frame.to_csv(path, index=False)
            manifest_path = root / "publication_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            for entry in manifest["files"]:
                if entry["path"] == path.name:
                    entry.update(bytes=path.stat().st_size,
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with patch.object(snapshot, "ROOT", root), self.assertRaises(AssertionError):
                snapshot.validate()


if __name__ == "__main__":
    unittest.main()
