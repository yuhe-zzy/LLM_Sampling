"""Read-only integrity and numeric validation of the published historical snapshot."""

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ENTROPY = "prompt_relative_sequence_entropy_mean"
CHECKPOINTS = [0, 20, 40, 60, 80]


def load_runs():
    runs = {}
    for path in sorted(ROOT.glob("*_relative_sequence_metrics.csv")):
        match = re.match(r"^(dpo|ipo)_transitive_a([0-9.]+)_l([0-9.]+)_b1_", path.name)
        assert match, path.name
        method, alpha, lam = match.groups()
        key = (method, float(alpha), float(lam))
        frame = pd.read_csv(path)
        frame = frame.loc[frame["iter"].between(0, 80)].sort_values("iter")
        assert not frame.empty and not frame["iter"].duplicated().any(), key
        assert frame["primary_entropy_metric"].eq(ENTROPY).all()
        assert frame["loss_type"].eq(method).all()
        for column, value in [("alpha", key[1]), ("lambda", key[2]), ("beta", 1)]:
            assert frame[column].eq(value).all(), (key, column)
        assert np.isfinite(frame[ENTROPY]).all()
        assert frame[ENTROPY].between(-1e-8, math.log(5) + 1e-8).all()
        wr = frame.dropna(subset=["oracle_win_rate"])
        assert wr["iter"].isin(CHECKPOINTS).all()
        assert wr["oracle_win_rate"].between(0, 1).all()
        raw = pd.read_csv(ROOT / path.name.replace("_relative_sequence_metrics.csv", "_training_metrics.csv"))
        assert raw["outer_reference_frozen"].astype(str).str.lower().eq("true").all()
        assert not raw["iter"].duplicated().any()
        raw_wr = raw.set_index("iter")["oracle_win_rate"]
        np.testing.assert_allclose(wr["oracle_win_rate"], raw_wr.loc[wr["iter"]], rtol=1e-12, atol=1e-12)
        assert key not in runs
        runs[key] = {"frame": frame, "wr": wr, "source": path.name,
                     "last": int(frame["iter"].max()), "complete": 80 in set(wr["iter"])}
    return runs


def validate():
    manifest = json.loads((ROOT / "publication_manifest.json").read_text(encoding="utf-8-sig"))
    entries = manifest["files"]
    names = [entry["path"] for entry in entries]
    assert len(names) == len(set(names)) == 38
    for entry in entries:
        assert Path(entry["path"]).name == entry["path"]
        path = ROOT / entry["path"]
        content = path.read_bytes()
        assert len(content) == entry["bytes"], path.name
        assert hashlib.sha256(content).hexdigest() == entry["sha256"], path.name
        if path.suffix == ".csv":
            columns = set(pd.read_csv(path, nrows=0).columns)
            assert not columns & {"prompt", "response", "text", "chosen", "rejected"}, path.name

    runs = load_runs()
    assert len(runs) == 13
    assert sum(run["complete"] for run in runs.values()) == 12
    for key, run in runs.items():
        assert set(run["frame"]["iter"]) == set(range(run["last"] + 1)), key
        assert set(run["wr"]["iter"]) == {t for t in CHECKPOINTS if t <= run["last"]}, key
        assert run["last"] == (80 if run["complete"] else 65), key

    curves = pd.read_csv(ROOT / "oracle_curves_20260911.csv")
    expected = pd.concat([
        run["frame"].assign(method=key[0], source_file=run["source"])[curves.columns]
        for key, run in runs.items()
    ], ignore_index=True)
    keys = ["method", "alpha", "lambda", "iter"]
    pd.testing.assert_frame_equal(
        curves.sort_values(keys).reset_index(drop=True),
        expected.sort_values(keys).reset_index(drop=True),
        check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12,
    )

    summary = pd.read_csv(ROOT / "oracle_parameter_summary_20260911.csv")
    assert len(summary) == 16
    assert not summary.duplicated(["method", "alpha", "lambda_value"]).any()
    expected_keys = {(m, a, l) for m in ("ipo", "dpo")
                     for a in (.8, .9, .95, .99) for l in (.5, .9)}
    assert set(zip(summary.method, summary.alpha, summary.lambda_value)) == expected_keys
    for row in summary.itertuples(index=False):
        run = runs.get((row.method, row.alpha, row.lambda_value))
        if run is None:
            assert row.status == "pending_rerun"
            assert pd.isna(row.latest_iter)
        else:
            assert row.status == ("complete" if run["complete"] else "running")
            assert row.latest_iter == run["last"]
            assert row.source_file == run["source"]
            for name, value in [
                ("entropy_initial", run["frame"][ENTROPY].iloc[0]),
                ("entropy_latest", run["frame"][ENTROPY].iloc[-1]),
                ("latest_wr_iter", run["wr"]["iter"].iloc[-1]),
                ("latest_wr", run["wr"]["oracle_win_rate"].iloc[-1]),
            ]:
                assert np.isclose(getattr(row, name), value, rtol=1e-12, atol=1e-12)
        for t in CHECKPOINTS:
            values = [] if run is None else run["wr"].loc[run["wr"]["iter"] == t, "oracle_win_rate"]
            actual = getattr(row, f"wr_{t}")
            if len(values):
                assert np.isclose(actual, values.iloc[0], rtol=1e-12, atol=1e-12)
            else:
                assert pd.isna(actual)

    wr_table = pd.read_csv(ROOT / "oracle_winning_rate_checkpoints_20260911.csv")
    pd.testing.assert_frame_equal(wr_table, summary[wr_table.columns], check_dtype=False)
    return {"verified_copied_files": 38, "grid_cells": 16, "curves": 13,
            "complete_to_80": 12, "partial_to_65": 1, "absent": 3,
            "snapshot_date": "2026-09-11", "fresh_server_check": False}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
