"""sstubsLarge.json -> generic CSV. Artifact = fixCommitSHA1 (sibling derivation:
multiple single-statement bugs per fixing commit). Label = bug-pattern present (1)."""
import csv, json, sys
data = json.load(open(sys.argv[1], encoding="utf-8", errors="replace"))
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["example_id", "artifact_ids", "label"])
    for i, r in enumerate(data):
        commit = r.get("fixCommitSHA1") or r.get("fixCommitSha1") or ""
        proj = r.get("projectName", "")
        w.writerow([f"ss:{i}", f"{proj}@{commit}", 1])
print("wrote", len(data))
