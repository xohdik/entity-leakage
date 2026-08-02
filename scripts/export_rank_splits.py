import csv, json, os, sys

def pick(d, keys):
    for k in keys:
        if k in d:
            return k
    return None

TEXT_KEYS  = ['func', 'code', 'function', 'source', 'text', 'patch', 'patch_code']
TEXT2_KEYS = ['bug_report', 'report', 'nl', 'context']
LABEL_KEYS = ['target', 'label', 'y', 'correct']
ID_KEYS    = ['idx', 'id', 'example_id', 'uid', 'name', 'ex_id']

def load_rows(path):
    """Load either a .csv or .json/.jsonl file into a list of dicts."""
    if path.endswith(".csv"):
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            return list(csv.DictReader(f))
    obj = json.load(open(path, encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        # dict of id -> record; fold the key in as an id field if absent
        out = []
        for k, v in obj.items():
            if isinstance(v, dict) and not pick(v, ID_KEYS):
                v = {**v, "ex_id": k}
            out.append(v)
        return out
    sys.exit(f"unrecognized rows format in {path}")

def load_split_map(path):
    obj = json.load(open(path, encoding="utf-8"))
    if not isinstance(obj, dict):
        sys.exit(f"expected an id->split dict in {path}, got {type(obj)}")
    return {str(k): v for k, v in obj.items()}

def load_devign_rows():
    """Exact replica of train_devign.py's own loading path -- NOT devign.csv,
    which turns out to hold only audit metadata (example_id, artifact_ids,
    label, split), no code text. The real function text and the ex_id
    join key both come from the HF dataset directly, concatenated across
    its three native splits, matching train_devign.py line-for-line:
        ds = load_dataset("google/code_x_glue_cc_defect_detection")
        for sp in ["train", "validation", "test"]:
            for r in ds[sp]:
                ex_id = f"devign:{r['id']}"
    """
    import os
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    from datasets import load_dataset
    ds = load_dataset("google/code_x_glue_cc_defect_detection")
    rows = []
    for sp in ["train", "validation", "test"]:
        for r in ds[sp]:
            rows.append({"ex_id": f"devign:{r['id']}", "func": r["func"],
                        "target": int(r["target"])})
    return rows

def export_devign_cell(split_path, out_prefix, out_dir, rows_cache=[]):
    if not rows_cache:
        rows_cache.append(load_devign_rows())
    rows = rows_cache[0]
    print(f"[{out_prefix}] rows={len(rows)} (from HF dataset, matching train_devign.py)")

    smap = load_split_map(split_path)
    buckets = {"train": [], "test": []}
    missing = 0
    for r in rows:
        part = smap.get(r["ex_id"])
        if part is None:
            missing += 1
            continue
        if part in ("train", "valid"):
            buckets["train"].append(r)
        elif part == "test":
            buckets["test"].append(r)
    if missing:
        print(f"   !! {missing}/{len(rows)} rows had no split-map entry (skipped)")
    for part, out_rows in buckets.items():
        out_path = os.path.join(out_dir, f"{part}_{out_prefix}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps({"idx": r["ex_id"], "func": r["func"],
                                   "target": r["target"]}) + "\n")
        print(f"   wrote {out_path}  ({len(out_rows)} rows)")

def export_cell(rows_path, split_path, id_key_hint, out_prefix, out_dir):
    """Generic auto-detecting exporter, used for corpora (like Quatrain) that
    already ship as one file with real text fields -- unlike Devign, which
    needs export_devign_cell() above instead."""
    rows = load_rows(rows_path)
    if not rows:
        sys.exit(f"no rows loaded from {rows_path}")
    r0 = rows[0]
    kt  = pick(r0, TEXT_KEYS)
    kt2 = pick(r0, TEXT2_KEYS)
    kl  = pick(r0, LABEL_KEYS)
    ki  = id_key_hint if id_key_hint in r0 else pick(r0, ID_KEYS)
    if not (kt and kl and ki):
        print(f"!! could not detect schema of {rows_path}; sample row below, "
             f"paste this back and I'll adjust the key lists:")
        print(json.dumps(r0, default=str)[:600])
        sys.exit(1)
    print(f"[{out_prefix}] rows={len(rows)} text_key={kt} text2_key={kt2} "
         f"label_key={kl} id_key={ki}")

    smap = load_split_map(split_path)
    buckets = {"train": [], "test": []}
    missing = 0
    for r in rows:
        rid = str(r.get(ki, ""))
        part = smap.get(rid)
        if part is None:
            missing += 1
            continue
        if part in ("train", "valid"):
            buckets["train"].append(r)
        elif part == "test":
            buckets["test"].append(r)
    if missing:
        print(f"   !! {missing}/{len(rows)} rows had no split-map entry (skipped) "
             f"-- check that id_key={ki!r} actually matches the split map's keys")
    for part, out_rows in buckets.items():
        out_path = os.path.join(out_dir, f"{part}_{out_prefix}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in out_rows:
                rec = {"idx": str(r.get(ki, "")), kt: r.get(kt, "")}
                if kt2:
                    rec[kt2] = r.get(kt2, "")
                rec["target"] = int(float(r.get(kl, 0)))
                f.write(json.dumps(rec) + "\n")
        print(f"   wrote {out_path}  ({len(out_rows)} rows)")

def main():
    devign_dir = "../data/devign"
    patches_dir = "../data/patches"
    export_devign_cell(f"{devign_dir}/split_published.json", "pub", devign_dir)
    export_devign_cell(f"{devign_dir}/split_cluster.json", "clu", devign_dir)

    print("\n=== Patch assessment (Quatrain) ===")
    export_cell(f"{patches_dir}/quatrain_text.json", f"{patches_dir}/qt_split_random.json",
               "ex_id", "rand", patches_dir)
    export_cell(f"{patches_dir}/quatrain_text.json", f"{patches_dir}/qt_split_cluster.json",
               "ex_id", "clu", patches_dir)

if __name__ == "__main__":
    main()