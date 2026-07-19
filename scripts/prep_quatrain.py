"""bugreport_patch.txt -> jsonl + random(patch-level) & cluster(bug-level)
splits + contaminated test ids + feasibility report.

Line format: bug$$summary$$description$$patch_name$$patch_tokens$$label
Lines with field-count != 6 are skipped (counted). Exact-duplicate patch
texts within the same bug are dropped (counted).
"""
import json, random, sys
sys.path.insert(0, ".")
from collections import Counter
from leakage_audit.graph import SharingGraph
from leakage_audit.metrics import contamination_report, measured_contamination
from leakage_audit.splits import cluster_kfold

SRC = "data/patches/Quatrain/data/bugreport_patch.txt"
rows, skipped, dup = [], 0, 0
seen = set()
for i, line in enumerate(open(SRC, encoding="utf-8", errors="replace")):
    parts = line.rstrip("\n").split("$$")
    if len(parts) != 6:
        skipped += 1
        continue
    bug, summ, desc, name, patch, lab = parts
    key = (bug, patch.strip())
    if key in seen:
        dup += 1
        continue
    seen.add(key)
    rows.append({"ex_id": f"qt:{i}:{name}", "bug": bug.strip(),
                 "report": (summ + " " + desc).strip(),
                 "patch": patch.strip(), "label": int(lab)})

print(f"parsed={len(rows)} skipped={skipped} dups_dropped={dup}")
print("label dist:", Counter(r["label"] for r in rows))
print("bugs:", len({r['bug'] for r in rows}))

json.dump(rows, open("data/patches/quatrain_text.json", "w"))

graph = SharingGraph.build([(r["ex_id"], [r["bug"]]) for r in rows])
rep = contamination_report(graph, None, test_fraction=0.2)
print(json.dumps({k: v for k, v in rep.items()}, indent=2, default=str))

# patch-level random split 70/10/20 (field-standard protocol)
rng = random.Random(42)
ids = [r["ex_id"] for r in rows]
rng.shuffle(ids)
n = len(ids)
rand = {}
for j, ex in enumerate(ids):
    rand[ex] = "train" if j < 0.7*n else ("valid" if j < 0.8*n else "test")
json.dump(rand, open("data/patches/qt_split_random.json", "w"))

m = measured_contamination(graph, {k: ("train" if v in ("train","valid") else v)
                                   for k, v in rand.items()})
print(f"random split rho_measured={m['rho_measured']:.4f} "
      f"({m['n_contaminated']}/{m['n_test']})")
json.dump(m["contaminated_ids"], open("data/patches/qt_contaminated_test.json", "w"))

# bug-level cluster split: fold0 test, fold1 valid
folds = cluster_kfold(graph, n_folds=5, seed=42)
clus = {ex: ("test" if f == 0 else "valid" if f == 1 else "train")
        for ex, f in folds.items()}
json.dump(clus, open("data/patches/qt_split_cluster.json", "w"))
print("cluster split sizes:", Counter(clus.values()))
# base rates per split-part
for nm, sp in (("random", rand), ("cluster", clus)):
    t = [r for r in rows if sp[r["ex_id"]] == "test"]
    c = Counter(r["label"] for r in t)
    print(f"{nm} test base: n={len(t)} labels={dict(c)} "
          f"majority={max(c.values())/len(t):.4f}")
