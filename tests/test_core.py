import sys
sys.path.insert(0, ".")
from leakage_audit import (
    SharingGraph, predicted_contamination, measured_contamination,
    cluster_kfold, holdout_from_folds, verify_leakage_safe, solve_m_gen,
)

def test_clusters_and_soundness():
    g = SharingGraph.build([
        ("p1", ["a1", "w1"]), ("p2", ["a1", "w2"]),  # hub on a1
        ("p3", ["a2", "w3"]),                          # singleton cluster
    ])
    assert g.example_cluster["p1"] == g.example_cluster["p2"]
    assert g.example_cluster["p3"] != g.example_cluster["p1"]
    assert g.summary()["n_clusters"] == 2

def test_theorem1_edge_cases():
    assert predicted_contamination({1: 100}, 0.2) == 0.0       # singletons
    assert abs(predicted_contamination({2: 100}, 0.2) - 0.8) < 1e-12

def test_measured():
    g = SharingGraph.build([("x", ["a"]), ("y", ["a"]), ("z", ["b"])])
    m = measured_contamination(g, {"x": "train", "y": "test", "z": "test"})
    assert m["rho_measured"] == 0.5 and m["contaminated_ids"] == ["y"]

def test_safe_split():
    g = SharingGraph.build([(f"e{i}", [f"a{i//3}"]) for i in range(30)])
    folds = cluster_kfold(g, 5, seed=1)
    for tf in range(5):
        assert verify_leakage_safe(g, holdout_from_folds(folds, tf))

def test_solve_m_gen():
    m_gen = 0.6; rho = 0.9; m_mem = 0.95
    m_obs = rho*m_mem + (1-rho)*m_gen
    assert abs(solve_m_gen(m_obs, m_mem, rho) - m_gen) < 1e-12

for name, fn in list(globals().items()):
    if name.startswith("test_"):
        fn(); print(f"PASS {name}")
