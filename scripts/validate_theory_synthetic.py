"""Monte Carlo validation of Theorem 1 and the Theorem 2 mixture identity.

1. Generate synthetic benchmarks with controlled cluster-size
   distributions (incl. one mimicking CodeNet: ~4.4 pairs/problem).
2. Random example-level splits -> measure contamination, compare to the
   closed-form prediction.
3. Simulate a memorizing classifier -> verify M_obs = rho*M_mem + (1-rho)*M_gen.
"""
import random
import sys

sys.path.insert(0, ".")

from leakage_audit import (  # noqa: E402
    SharingGraph,
    predicted_contamination,
    measured_contamination,
    cluster_kfold,
    holdout_from_folds,
    verify_leakage_safe,
)


def synth(n_clusters, size_sampler, seed=0):
    rng = random.Random(seed)
    records = []
    for c in range(n_clusters):
        k = size_sampler(rng)
        art = f"A{c}"
        for j in range(k):
            records.append((f"E{c}_{j}", [art]))
    return SharingGraph.build(records)


def random_split(graph, t, seed):
    rng = random.Random(seed)
    return {
        ex: ("test" if rng.random() < t else "train")
        for ex in graph.example_cluster
    }


def run_theorem1():
    print("=== Theorem 1: predicted vs measured contamination ===")
    scenarios = {
        "codenet-like (k in 3..6)": lambda r: r.randint(3, 6),
        "hub-heavy (k in 10..40)": lambda r: r.randint(10, 40),
        "mostly singletons (80% k=1)": lambda r: 1
        if r.random() < 0.8
        else r.randint(2, 5),
    }
    t = 0.2
    for name, sampler in scenarios.items():
        g = synth(4000, sampler, seed=1)
        preds = predicted_contamination(g.cluster_size_distribution(), t)
        meas = []
        for s in range(20):
            sp = random_split(g, t, seed=100 + s)
            meas.append(measured_contamination(g, sp)["rho_measured"])
        mean_meas = sum(meas) / len(meas)
        print(
            f"{name:34s} pred={preds:.4f}  measured={mean_meas:.4f}  "
            f"|err|={abs(preds - mean_meas):.4f}"
        )


def run_theorem2():
    print("\n=== Theorem 2: mixture identity on a simulated memorizer ===")
    # Model: on contaminated examples classifier answers via sibling lookup
    # (acc m_mem); on clean examples it generalizes (acc m_gen).
    rng = random.Random(7)
    g = synth(4000, lambda r: r.randint(3, 6), seed=2)
    t = 0.2
    m_mem, m_gen = 0.95, 0.62
    sp = random_split(g, t, seed=3)
    contam = set(measured_contamination(g, sp)["contaminated_ids"])
    test_ids = [e for e, p in sp.items() if p == "test"]
    correct = 0
    for e in test_ids:
        p = m_mem if e in contam else m_gen
        correct += rng.random() < p
    m_obs = correct / len(test_ids)
    rho = len(contam) / len(test_ids)
    m_obs_pred = rho * m_mem + (1 - rho) * m_gen
    print(
        f"rho={rho:.4f}  M_obs(sim)={m_obs:.4f}  "
        f"M_obs(mixture)={m_obs_pred:.4f}  Delta={m_obs - m_gen:+.4f}"
    )


def run_split_safety():
    print("\n=== Leakage-safe cluster k-fold ===")
    g = synth(4000, lambda r: r.randint(3, 6), seed=4)
    folds = cluster_kfold(g, n_folds=5, seed=42)
    sizes = {}
    for f in folds.values():
        sizes[f] = sizes.get(f, 0) + 1
    ok = all(
        verify_leakage_safe(g, holdout_from_folds(folds, tf))
        for tf in range(5)
    )
    print(f"fold sizes: {sorted(sizes.values())}  leakage_safe_all_folds={ok}")
    # and contamination under the safe split is exactly 0
    sp = holdout_from_folds(folds, 0)
    rho = measured_contamination(g, sp)["rho_measured"]
    print(f"contamination under cluster split: {rho:.4f}")


if __name__ == "__main__":
    run_theorem1()
    run_theorem2()
    run_split_safety()
