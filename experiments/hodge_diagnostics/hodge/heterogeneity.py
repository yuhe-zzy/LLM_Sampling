"""Annotator heterogeneity in nvidia/HelpSteer2 preference pairs.

HelpSteer2 pairs have K = 2, so they carry no cyclic residual themselves. They do
record every annotator's signed preference strength (-3..3), which measures the
premise of prop:cyclic-sources: individual annotators disagree, so the population
preference over a larger candidate set is a mixture rather than one BT model.

    python -m hodge.heterogeneity
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .datasets import _jsonl_gz


def helpsteer2_disagreement(root):
    per_pair = []
    for row in _jsonl_gz(Path(root) / "helpsteer2_preference" / "preference.jsonl.gz"):
        prefs = row.get("all_preferences_unprocessed") or []
        strengths = [p["strength"] for p in prefs if isinstance(p, dict) and p.get("strength") is not None]
        if len(strengths) >= 2:
            per_pair.append((row.get("split"), np.array(strengths, dtype=float)))
    n = len(per_pair)
    signs = [np.sign(s) for _, s in per_pair]
    opposite = np.array([(sg > 0).any() and (sg < 0).any() for sg in signs])
    not_unanimous = np.array([len(set(sg)) > 1 for sg in signs])
    share_2 = np.array([np.mean(np.where(sg > 0, 1.0, np.where(sg < 0, 0.0, 0.5))) for sg in signs])
    spread = np.array([s.std() for _, s in per_pair])
    counts = np.bincount([len(s) for _, s in per_pair])
    return dict(
        pairs_with_2plus_annotators=n,
        annotators_per_pair={int(k): int(v) for k, v in enumerate(counts) if v},
        frac_opposite_directions=float(opposite.mean()),
        frac_not_unanimous_in_direction=float(not_unanimous.mean()),
        frac_vote_share_strictly_between_0_and_1=float(np.mean((share_2 > 0) & (share_2 < 1))),
        median_strength_std=float(np.median(spread)),
        note="strength > 0 prefers response 2; ties (0) count as half votes in the share",
    )


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "outputs" / "helpsteer2_preference"
    out.mkdir(parents=True, exist_ok=True)
    result = helpsteer2_disagreement(root / "data")
    (out / "heterogeneity.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
