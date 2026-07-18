"""Theorem 2: inflation prediction.

Mixture model:
    M_observed = rho * M_mem + (1 - rho) * M_gen
    Delta      = M_observed - M_gen = rho * (M_mem - M_gen)

rho    : contaminated test fraction (from metrics.py, predicted or measured)
M_mem  : memorization ceiling — performance achievable on contaminated
         examples by artifact lookup alone (no generalization)
M_gen  : honest generalization performance (measured under entity-level
         split, or solved from the mixture if M_observed is known)

Report Delta as an interval using two ceiling estimators:
  - exact-match probe: label of the training sibling sharing the artifact
    (lower bound on memorization power)
  - kNN probe: nearest-neighbor label in any cheap embedding space
    (upper-ish bound; supply similarity function)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Mapping, Sequence


@dataclass
class InflationEstimate:
    rho: float
    m_mem_low: float
    m_mem_high: float
    m_gen: float

    @property
    def delta_low(self) -> float:
        return self.rho * (self.m_mem_low - self.m_gen)

    @property
    def delta_high(self) -> float:
        return self.rho * (self.m_mem_high - self.m_gen)

    def predicted_observed(self) -> tuple[float, float]:
        return (
            self.rho * self.m_mem_low + (1 - self.rho) * self.m_gen,
            self.rho * self.m_mem_high + (1 - self.rho) * self.m_gen,
        )


def solve_m_gen(m_observed: float, m_mem: float, rho: float) -> float:
    """Solve mixture for M_gen given observed leaky metric."""
    if rho >= 1.0:
        raise ValueError("rho == 1 leaves M_gen unidentifiable")
    return (m_observed - rho * m_mem) / (1.0 - rho)


def exact_match_probe(
    test_ids: Sequence[Hashable],
    labels: Mapping[Hashable, int],
    sibling_of: Mapping[Hashable, Hashable],
) -> float:
    """Accuracy of predicting each contaminated test example's label with
    the label of its training sibling (the shared-artifact neighbor).
    sibling_of: test example -> a chosen training example in same cluster.
    """
    if not test_ids:
        return 0.0
    hit = sum(
        1
        for ex in test_ids
        if ex in sibling_of and labels[sibling_of[ex]] == labels[ex]
    )
    return hit / len(test_ids)


def knn_probe(
    test_ids: Sequence[Hashable],
    train_ids: Sequence[Hashable],
    labels: Mapping[Hashable, int],
    similarity: Callable[[Hashable, Hashable], float],
) -> float:
    """1-NN label-transfer accuracy under an arbitrary similarity.
    O(|test| * |train|): restrict to contaminated test ids and a sampled
    train set for large benchmarks.
    """
    if not test_ids or not train_ids:
        return 0.0
    hit = 0
    for ex in test_ids:
        best = max(train_ids, key=lambda tr: similarity(ex, tr))
        if labels[best] == labels[ex]:
            hit += 1
    return hit / len(test_ids)
