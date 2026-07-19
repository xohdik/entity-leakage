"""Emit published + cluster-safe splits for Devign, with contamination flags.

Outputs (data/devign/):
  split_published.json  {example_id: train|valid|test}
  split_cluster.json    cluster 10-fold: fold0=test, fold1=valid, rest=train
  contaminated_test_ids.json  test ids sharing a commit with published train
"""
import json, sys
sys.path.insert(0, ".")
from leakage_audit.adapters.generic_csv import load
from leakage_audit.metrics import measured_contamination
from leakage_audit.splits import cluster_kfold

graph, split, _ = load("data/devign/devign.csv")
json.dump(split, open("data/devign/split_published.json", "w"))

m = measured_contamination(graph, {k: ("train" if v in ("train", "valid") else v)
                                   for k, v in split.items()})
json.dump(m["contaminated_ids"], open("data/devign/contaminated_test_ids.json", "w"))
print(f"published: rho={m['rho_measured']:.4f} ({m['n_contaminated']}/{m['n_test']})")

folds = cluster_kfold(graph, n_folds=10, seed=42)
cs = {ex: ("test" if f == 0 else "valid" if f == 1 else "train")
      for ex, f in folds.items()}
json.dump(cs, open("data/devign/split_cluster.json", "w"))
from collections import Counter
print("cluster split sizes:", Counter(cs.values()))
