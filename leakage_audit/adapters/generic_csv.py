"""Generic adapter: any benchmark reducible to a CSV of
    example_id, artifact_ids (;-separated), label [, split]

Covers patch-assessment datasets (artifact = bug id: e.g. "Chart_1"),
Devign-style function datasets (artifact = commit or project id),
SStuBs (artifact = repo), etc. Write a tiny converter per benchmark,
then everything downstream is uniform.
"""
from __future__ import annotations

import csv
from typing import Dict, Optional

from ..graph import SharingGraph


def load(
    csv_path: str,
    example_col: str = "example_id",
    artifact_col: str = "artifact_ids",
    label_col: str = "label",
    split_col: Optional[str] = "split",
    sep: str = ";",
) -> tuple[SharingGraph, Optional[Dict[str, str]], Dict[str, int]]:
    records = []
    split: Dict[str, str] = {}
    labels: Dict[str, int] = {}
    has_split = False
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ex = row[example_col]
            arts = [a for a in row[artifact_col].split(sep) if a]
            records.append((ex, arts))
            if label_col in row and row[label_col] != "":
                labels[ex] = int(row[label_col])
            if split_col and split_col in row and row[split_col]:
                split[ex] = row[split_col]
                has_split = True
    graph = SharingGraph.build(records)
    return graph, (split if has_split else None), labels
