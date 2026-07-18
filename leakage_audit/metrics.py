"""Contamination metrics.

Theorem 1 (expected contamination under example-level random split).
For a leakage cluster of size k and i.i.d. Bernoulli(t) assignment to
test, a test example is *contaminated* iff >=1 of its k-1 siblings is in
train. P(contaminated | in test) = 1 - t^(k-1).

Example-weighted expected contaminated fraction of the test set:
    rho_pred = sum_k  n_k * k * (1 - t^(k-1))  /  sum_k n_k * k
where n_k = number of clusters of size k. (Bernoulli approximation of
the exact hypergeometric split; excellent for N >> max k.)
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Hashable, Mapping

from .graph import SharingGraph


def predicted_contamination(
    cluster_size_dist: Mapping[int, int], test_fraction: float
) -> float:
    t = test_fraction
    num = 0.0
    den = 0.0
    for k, n_k in cluster_size_dist.items():
        num += n_k * k * (1.0 - t ** (k - 1))
        den += n_k * k
    return num / den if den else 0.0


def measured_contamination(
    graph: SharingGraph, split: Mapping[Hashable, str]
) -> dict:
    """Contamination of an *actual* split.

    split: example_id -> 'train' | 'test' (ignore other labels, e.g. 'valid'
    can be mapped to 'train' by the caller if it is fit on).
    Returns test contamination rate = fraction of test examples whose
    leakage cluster also contains >=1 train example.
    """
    cluster_has_train: Dict[int, bool] = {}
    for ex, part in split.items():
        if part != "train":
            continue
        cid = graph.example_cluster.get(ex)
        if cid is not None:
            cluster_has_train[cid] = True

    n_test = 0
    n_contam = 0
    contaminated_ids = []
    for ex, part in split.items():
        if part != "test":
            continue
        cid = graph.example_cluster.get(ex)
        if cid is None:
            continue
        n_test += 1
        if cluster_has_train.get(cid, False):
            n_contam += 1
            contaminated_ids.append(ex)
    return {
        "n_test": n_test,
        "n_contaminated": n_contam,
        "rho_measured": (n_contam / n_test) if n_test else 0.0,
        "contaminated_ids": contaminated_ids,
    }


def contamination_report(
    graph: SharingGraph,
    split: Mapping[Hashable, str] | None,
    test_fraction: float,
) -> dict:
    dist = graph.cluster_size_distribution()
    rep = {
        **graph.summary(),
        "test_fraction": test_fraction,
        "rho_predicted_random_split": predicted_contamination(
            dist, test_fraction
        ),
    }
    if split is not None:
        m = measured_contamination(graph, split)
        m.pop("contaminated_ids")
        rep.update({f"published_split_{k}": v for k, v in m.items()})
    return rep


def cluster_size_hist(graph: SharingGraph, top: int = 15) -> Counter:
    return Counter(dict(graph.cluster_size_distribution().most_common(top)))
