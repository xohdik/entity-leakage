"""Adapter: CodeXGLUE Clone-detection-BigCloneBench.

Expected files (from
github.com/microsoft/CodeXGLUE/tree/main/Code-Code/Clone-detection-BigCloneBench/dataset):
    data.jsonl            {"func": ..., "idx": "<function-id>"}
    train.txt / valid.txt / test.txt   lines: "<idx1>\t<idx2>\t<label>"

Example  = a labeled pair (idx1, idx2, label)
Artifact = a function id  ->  hub-reuse sharing: pairs share whenever they
reuse a function. Entity key candidate: function id (sound iff clusters
are function-connected components, which build() computes exactly).
"""
from __future__ import annotations

import os
from typing import Dict, Iterator, Tuple

from ..graph import SharingGraph

SPLIT_FILES = {"train": "train.txt", "valid": "valid.txt", "test": "test.txt"}


def iter_pairs(path: str) -> Iterator[Tuple[str, str, str, int]]:
    """yield (example_id, func_a, func_b, label)."""
    stem = os.path.basename(path).rsplit(".", 1)[0]
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            a, b = parts[0], parts[1]
            label = int(parts[2]) if len(parts) > 2 else -1
            yield (f"{stem}:{i}", a, b, label)


def load(
    dataset_dir: str, treat_valid_as_train: bool = True
) -> tuple[SharingGraph, Dict[str, str], Dict[str, int]]:
    """Returns (sharing graph over ALL splits, published split map, labels).

    treat_valid_as_train: valid is fit-adjacent (model selection); for a
    conservative contamination measurement count it as train.
    """
    records = []
    split: Dict[str, str] = {}
    labels: Dict[str, int] = {}
    for part, fname in SPLIT_FILES.items():
        fpath = os.path.join(dataset_dir, fname)
        if not os.path.exists(fpath):
            continue
        eff = (
            "train"
            if (part == "valid" and treat_valid_as_train)
            else part
        )
        for ex_id, a, b, y in iter_pairs(fpath):
            records.append((ex_id, (a, b)))
            split[ex_id] = eff
            labels[ex_id] = y
    graph = SharingGraph.build(records)
    return graph, split, labels
