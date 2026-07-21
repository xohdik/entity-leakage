"""Tufano BFP -> generic CSV with shipped split. Artifact = commit SHA
(col 2 of bugfixes.key.csv, aligned by line order with buggy_all/fixed_all).
Split recovered by matching (buggy, fixed) line pairs; ambiguous duplicates
are assigned greedily and counted."""
import csv, sys
from collections import defaultdict

base = sys.argv[1]           # e.g. data/tufano/datasets/50
out = sys.argv[2]

keys = [line.split(",")[1].strip() for line in open(f"{base}/bugfixes.key.csv")]
buggy = [l.rstrip("\n") for l in open(f"{base}/buggy_all.txt")]
fixed = [l.rstrip("\n") for l in open(f"{base}/fixed_all.txt")]
assert len(keys) == len(buggy) == len(fixed)

index = defaultdict(list)          # (buggy,fixed) -> [row ids]
for i, (b, f) in enumerate(zip(buggy, fixed)):
    index[(b, f)].append(i)

split_of = {}
ambiguous = unmatched = 0
for part in ("train", "test", "eval"):
    eff = "valid" if part == "eval" else part
    bl = [l.rstrip("\n") for l in open(f"{base}/{part}/buggy.txt")]
    fl = [l.rstrip("\n") for l in open(f"{base}/{part}/fixed.txt")]
    for b, f in zip(bl, fl):
        cand = index.get((b, f))
        if not cand:
            unmatched += 1
            continue
        i = cand.pop(0)            # greedy: duplicates consumed in order
        if len(cand) >= 1:
            ambiguous += 1
        split_of[i] = eff

with open(out, "w", newline="") as fo:
    w = csv.writer(fo)
    w.writerow(["example_id", "artifact_ids", "label", "split"])
    for i, k in enumerate(keys):
        w.writerow([f"bfp:{i}", k, 1, split_of.get(i, "train")])
print(f"pairs={len(keys)} matched={len(split_of)} "
      f"ambiguous_dups={ambiguous} unmatched={unmatched}")
