"""Sharing graph and leakage clusters.

An example x carries a provenance set art(x) of raw-artifact IDs.
Two examples share (x ~ y) iff art(x) & art(y) != {}.
Leakage clusters = connected components of the sharing graph, computed
via union-find over artifact IDs (no explicit edge materialization:
O(sum |art(x)|) instead of O(n^2)).
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Hashable, Iterable, List, Sequence


class _UnionFind:
    __slots__ = ("parent", "rank")

    def __init__(self) -> None:
        self.parent: Dict[Hashable, Hashable] = {}
        self.rank: Dict[Hashable, int] = {}

    def find(self, x: Hashable) -> Hashable:
        parent = self.parent
        if x not in parent:
            parent[x] = x
            self.rank[x] = 0
            return x
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(self, a: Hashable, b: Hashable) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@dataclass
class SharingGraph:
    """Built from (example_id, artifact_ids) records."""

    example_cluster: Dict[Hashable, int] = field(default_factory=dict)
    cluster_sizes: Counter = field(default_factory=Counter)
    n_examples: int = 0
    n_artifacts: int = 0

    @classmethod
    def build(
        cls, records: Iterable[tuple[Hashable, Sequence[Hashable]]]
    ) -> "SharingGraph":
        uf = _UnionFind()
        example_arts: Dict[Hashable, Hashable] = {}
        artifacts = set()
        for ex_id, arts in records:
            arts = list(arts)
            if not arts:
                raise ValueError(f"example {ex_id!r} has empty provenance")
            artifacts.update(arts)
            for a in arts[1:]:
                uf.union(arts[0], a)
            example_arts[ex_id] = arts[0]

        root_to_cid: Dict[Hashable, int] = {}
        g = cls(n_examples=len(example_arts), n_artifacts=len(artifacts))
        for ex_id, a in example_arts.items():
            root = uf.find(a)
            cid = root_to_cid.setdefault(root, len(root_to_cid))
            g.example_cluster[ex_id] = cid
        g.cluster_sizes = Counter(g.example_cluster.values())
        return g

    # ---- stats -------------------------------------------------------

    def cluster_size_distribution(self) -> Counter:
        """size k -> number of clusters of size k."""
        return Counter(self.cluster_sizes.values())

    def clusters(self) -> Dict[int, List[Hashable]]:
        out: Dict[int, List[Hashable]] = defaultdict(list)
        for ex, cid in self.example_cluster.items():
            out[cid].append(ex)
        return dict(out)

    def summary(self) -> dict:
        dist = self.cluster_size_distribution()
        sizes = self.cluster_sizes
        n_clusters = len(sizes)
        singletons = dist.get(1, 0)
        max_k = max(sizes.values()) if sizes else 0
        mean_k = self.n_examples / n_clusters if n_clusters else 0.0
        # example-weighted mean cluster size (size of the cluster a random
        # example belongs to) — the quantity that drives contamination
        ew_mean = (
            sum(k * k for k in sizes.values()) / self.n_examples
            if self.n_examples
            else 0.0
        )
        return {
            "n_examples": self.n_examples,
            "n_artifacts": self.n_artifacts,
            "n_clusters": n_clusters,
            "singleton_clusters": singletons,
            "examples_in_nonsingleton_clusters": self.n_examples
            - singletons,
            "mean_cluster_size": mean_k,
            "example_weighted_mean_cluster_size": ew_mean,
            "max_cluster_size": max_k,
        }
