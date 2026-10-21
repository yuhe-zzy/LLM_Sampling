"""Loaders that turn public preference datasets into finite panels.

A panel is one prompt (or, for model-level arenas, one pooled comparison set)
with K >= 3 candidates, because K = 2 panels have C = 0 identically. Rating
datasets yield a K x m rating matrix (one column per rated attribute); vote
datasets yield a K x K win-count matrix with ties split as halves.
"""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HELPSTEER_ALL5 = ("helpfulness", "correctness", "coherence", "complexity", "verbosity")
HELPSTEER_QUALITY3 = ("helpfulness", "correctness", "coherence")
ULTRAFEEDBACK_ASPECTS = ("helpfulness", "honesty", "instruction_following", "truthfulness")


@dataclass
class Panel:
    dataset: str
    panel_id: str
    labels: list
    ratings: np.ndarray | None = None
    attributes: tuple | None = None
    wins: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def K(self):
        return len(self.labels)


def _jsonl_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_helpsteer(root, attributes=HELPSTEER_ALL5, min_k=3, splits=("train", "validation")):
    """nvidia/HelpSteer: responses grouped by identical prompt text; duplicates averaged."""
    groups = defaultdict(lambda: defaultdict(list))
    for split in splits:
        for row in _jsonl_gz(Path(root) / "helpsteer" / f"{split}.jsonl.gz"):
            groups[(split, row["prompt"])][row["response"]].append([float(row[a]) for a in attributes])
    panels = []
    for n, ((split, _), responses) in enumerate(groups.items()):
        if len(responses) < min_k:
            continue
        R = np.array([np.mean(v, axis=0) for v in responses.values()])
        panels.append(Panel("helpsteer", f"{split}-{n}", list(range(len(R))), ratings=R,
                            attributes=tuple(attributes), meta=dict(split=split)))
    return panels


def _rating(annotation):
    try:
        return float(annotation["Rating"])
    except (KeyError, TypeError, ValueError):
        return np.nan


def load_ultrafeedback(root, aspects=ULTRAFEEDBACK_ASPECTS, min_k=3):
    """openbmb/UltraFeedback: four completions per instruction, four GPT-4 aspect ratings."""
    import pyarrow.parquet as pq

    panels = []
    for path in sorted((Path(root) / "ultrafeedback").glob("train-*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2000, columns=["source", "completions"]):
            for row in batch.to_pylist():
                completions = row["completions"] or []
                R = np.array([[_rating((c.get("annotations") or {}).get(a)) for a in aspects]
                              for c in completions]).reshape(len(completions), len(aspects))
                if len(completions) < min_k or np.all(np.isnan(R)):
                    continue
                panels.append(Panel("ultrafeedback", f"uf-{len(panels)}", [c["model"] for c in completions],
                                    ratings=R, attributes=tuple(aspects),
                                    meta=dict(source=row["source"],
                                              overall=[c.get("overall_score") for c in completions])))
    return panels


def _add_vote(W, index, a, b, outcome):
    i, j = index[a], index[b]
    if outcome == "a":
        W[i, j] += 1
    elif outcome == "b":
        W[j, i] += 1
    else:
        W[i, j] += 0.5
        W[j, i] += 0.5


def _mt_outcome(winner):
    return {"model_a": "a", "model_b": "b"}.get(winner, "tie")


def load_mt_bench(root, split="human", pooled=False):
    """lmsys/mt_bench_human_judgments: one panel per (question, turn), six models.

    ``pooled=True`` instead returns one model-level panel per turn, pooling all questions.
    """
    import pandas as pd

    df = pd.read_parquet(Path(root) / "mt_bench" / f"{split}.parquet",
                         columns=["question_id", "model_a", "model_b", "winner", "judge", "turn"])
    models = sorted(set(df.model_a) | set(df.model_b))
    index = {m: k for k, m in enumerate(models)}
    keys = ["turn"] if pooled else ["question_id", "turn"]
    panels = []
    for key, group in df.groupby(keys):
        key = key if isinstance(key, tuple) else (key,)
        W = np.zeros((len(models), len(models)))
        for a, b, w in zip(group.model_a, group.model_b, group.winner):
            _add_vote(W, index, a, b, _mt_outcome(w))
        panels.append(Panel(f"mt_bench_{split}" + ("_pooled" if pooled else ""),
                            "-".join(map(str, key)), models, wins=W,
                            meta=dict(votes=len(group), judges=group.judge.nunique())))
    return panels


def load_arena55k(root, min_battles=0):
    """lmarena-ai/arena-human-preference-55k pooled into one model-level panel."""
    import pandas as pd

    df = pd.read_parquet(Path(root) / "arena55k" / "train-0.parquet",
                         columns=["model_a", "model_b", "winner_model_a", "winner_model_b", "winner_tie"])
    counts = pd.concat([df.model_a, df.model_b]).value_counts()
    models = sorted(counts[counts >= min_battles].index)
    index = {m: k for k, m in enumerate(models)}
    W = np.zeros((len(models), len(models)))
    for a, b, wa, wb in zip(df.model_a, df.model_b, df.winner_model_a, df.winner_model_b):
        if a in index and b in index and a != b:
            _add_vote(W, index, a, b, "a" if int(wa) else ("b" if int(wb) else "tie"))
    return [Panel("arena55k", f"models>={min_battles}", models, wins=W, meta=dict(battles=int(W.sum())))]


LOADERS = {
    "helpsteer": lambda root: load_helpsteer(root, HELPSTEER_ALL5),
    "helpsteer_quality3": lambda root: [
        Panel("helpsteer_quality3", p.panel_id, p.labels, p.ratings, p.attributes, meta=p.meta)
        for p in load_helpsteer(root, HELPSTEER_QUALITY3)],
    "ultrafeedback": load_ultrafeedback,
    "mt_bench_human": lambda root: load_mt_bench(root, "human"),
    "mt_bench_gpt4": lambda root: load_mt_bench(root, "gpt4_pair"),
    "mt_bench_human_pooled": lambda root: load_mt_bench(root, "human", pooled=True),
    "arena55k": lambda root: load_arena55k(root, min_battles=200),
}


def load_synthetic(root=None, n=200, seed=0):
    """Controlled panels for smoke tests: single BT annotators (C = 0 under logit),
    two-annotator BT mixtures (C != 0 under logit), and the repo's cyclic tournament
    as ratings-free vote panels."""
    rng = np.random.default_rng(seed)
    panels = []
    for k in range(n):
        K = int(rng.integers(3, 9))
        m = 1 if k % 2 == 0 else 2
        panels.append(Panel("synthetic", f"bt{m}-{k}", list(range(K)), ratings=rng.normal(size=(K, m)) * 1.5,
                            attributes=tuple(f"annotator{j}" for j in range(m)), meta=dict(annotators=m)))
    W = np.zeros((4, 4))
    for w, l in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)]:
        W[w, l] = 10
    panels.append(Panel("synthetic", "repo-cyclic-tournament", list(range(4)), wins=W))
    return panels


LOADERS["synthetic"] = load_synthetic
