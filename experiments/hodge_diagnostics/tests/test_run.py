"""End-to-end smoke test of the runner on synthetic panels."""
import numpy as np
import pandas as pd

from hodge import run


def test_runner_on_synthetic(tmp_path):
    run.main(["synthetic", "--out", str(tmp_path), "--pop-sample", "20", "--null-draws", "20"])
    summary = pd.read_csv(tmp_path / "summary.csv")
    panels = pd.read_csv(tmp_path / "panels.csv")
    assert summary.fact_bounds_hold.all()
    one = panels[panels.panel_id.str.startswith("bt1") & (panels.construction == "single_score_bt")
                 & (panels.link == "logit")]
    assert np.allclose(one.cyclic_share, 0, atol=1e-12)  # single BT annotator, logit link: C = 0
    two = panels[panels.panel_id.str.startswith("bt2") & (panels.construction == "attribute_bt")
                 & (panels.link == "logit")]
    assert (two.cyclic_share > 1e-6).mean() > 0.9  # mixtures are generically cyclic
    assert (tmp_path / "report.md").exists() and (tmp_path / "population.csv").exists()
