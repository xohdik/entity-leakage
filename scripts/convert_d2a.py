"""D2A (claudios/D2A 'code' config) -> generic CSV with shipped split.
Artifact = fixing commit parsed from bug_url; fallback = full bug_url."""
import csv, re, sys
import pandas as pd

frames = []
for part, fname in (("train","d2a_train.parquet"),("valid","d2a_dev.parquet"),
                    ("test","d2a_test.parquet")):
    df = pd.read_parquet(f"data/{fname}")
    df["split"] = part
    frames.append(df)
df = pd.concat(frames)
pat = re.compile(r"(?:commit|;h=|/)([0-9a-f]{40})")
unparsed = 0
with open("data/d2a/d2a.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["example_id", "artifact_ids", "label", "split"])
    for _, r in df.iterrows():
        url = str(r["bug_url"])
        m = pat.search(url)
        art = m.group(1) if m else url
        if not m: unparsed += 1
        w.writerow([f"d2a:{r['id']}", art, int(r["label"]), r["split"]])
print(f"rows={len(df)} unparsed_urls={unparsed}")
print("sample bug_url:", df.iloc[0]["bug_url"])
