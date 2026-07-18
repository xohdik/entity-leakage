"""Leakage-safe split generation.

A split is leakage-safe iff every leakage cluster lies entirely in one
fold. We assign whole clusters to folds with greedy size balancing
(LPT scheduling), which keeps fold sizes within one max-cluster of even.
"""
from __future__ import annotations

import random
from typing import Dict, Hashable, List

from .graph import SharingGraph


def cluster_kfold(
    graph: SharingGraph, n_folds: int = 5, seed: int = 42
) -> Dict[Hashable, int]:
    """example_id -> fold index; every cluster confined to one fold."""
    clusters = graph.clusters()
    order = sorted(
        clusters.items(), key=lambda kv: len(kv[1]), reverse=True
    )
    rng = random.Random(seed)
    # shuffle within equal-size runs so seed matters
    i = 0
    shuffled: List[tuple[int, List[Hashable]]] = []
    while i < len(order):
        j = i
        while j < len(order) and len(order[j][1]) == len(order[i][1]):
            j += 1
        block = order[i:j]
        rng.shuffle(block)
        shuffled.extend(block)
        i = j

    fold_sizes = [0] * n_folds
    assignment: Dict[Hashable, int] = {}
    for _, members in shuffled:
        f = min(range(n_folds), key=lambda x: fold_sizes[x])
        for ex in members:
            assignment[ex] = f
        fold_sizes[f] += len(members)
    return assignment


def holdout_from_folds(
    fold_assignment: Dict[Hashable, int], test_fold: int
) -> Dict[Hashable, str]:
    return {
        ex: ("test" if f == test_fold else "train")
        for ex, f in fold_assignment.items()
    }


def verify_leakage_safe(
    graph: SharingGraph, split: Dict[Hashable, str]
) -> bool:
    seen: Dict[int, str] = {}
    for ex, part in split.items():
        cid = graph.example_cluster[ex]
        if cid in seen and seen[cid] != part:
            return False
        seen[cid] = part
    return True
