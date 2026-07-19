"""devign_meta.json -> generic CSV. Artifact = project@commit_id."""
import csv, json, sys
rows = json.load(open(sys.argv[1]))
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["example_id", "artifact_ids", "label", "split"])
    for r in rows:
        w.writerow([f"devign:{r['id']}", f"{r['project']}@{r['commit_id']}",
                    r["target"], r["split"]])
print("wrote", len(rows))
