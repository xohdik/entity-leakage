"""Prep BigCloneBench for controlled training: draw a fixed 30K-example pool
with two split labelings over the SAME example identities.

Design note: BCB's shipped train/valid/test files are function-disjoint by
the original authors' construction (that is why the shipped split has
rho=0). Pooling all three files and re-clustering to rediscover safety is
unnecessary and, at subsample scale, collapses back into the same three
giant blocks (observed: one ~60% mega-cluster, two ~20% clusters, ~99.9%
of any pooled subsample) that whole-cluster LPT balancing cannot subdivide.
Instead we subsample directly WITHIN each shipped file to hit target counts
(safe split: exactly the original file boundaries, trivially rho=0) and
separately reshuffle the SAME pooled identities at pair level, ignoring
origin, for the unsafe split (predicted rho~1 by Theorem 1).
"""
import json, os, random, sys, urllib.request
sys.path.insert(0, ".")

from leakage_audit.graph import SharingGraph
from leakage_audit.metrics import measured_contamination, predicted_contamination

SEED = 42
N_TRAIN, N_VALID, N_TEST = 24000, 3000, 3000   # 80/10/10 of 30,000

def load_jsonl_funcs(path):
    funcs = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                o = json.loads(line)
                funcs[o["idx"]] = o["func"]
    return funcs

def load_pairs_by_label(fname, funcs):
    out = {0: [], 1: []}
    with open(f"data/bcb/{fname}") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            id1, id2, y = parts[0], parts[1], int(parts[2])
            if id1 in funcs and id2 in funcs:
                out[y].append((id1, id2, y))
    return out

def stratified_sample(pairs_by_label, n_total, rng):
    n_each = n_total // 2
    a = rng.sample(pairs_by_label[0], min(n_each, len(pairs_by_label[0])))
    b = rng.sample(pairs_by_label[1], min(n_each, len(pairs_by_label[1])))
    out = a + b
    rng.shuffle(out)
    return out

def main():
    os.makedirs("data/bcb", exist_ok=True)
    jsonl_path = "data/bcb/data.jsonl"
    EXPECTED_MIN_BYTES = 15_000_000
    if not os.path.exists(jsonl_path) or os.path.getsize(jsonl_path) < EXPECTED_MIN_BYTES:
        print("downloading data.jsonl ...")
        url = ("https://raw.githubusercontent.com/microsoft/CodeXGLUE/main/"
               "Code-Code/Clone-detection-BigCloneBench/dataset/data.jsonl")
        tmp = jsonl_path + ".tmp"
        urllib.request.urlretrieve(url, tmp)
        if os.path.getsize(tmp) < EXPECTED_MIN_BYTES:
            raise RuntimeError("download incomplete; use wget -c instead")
        os.replace(tmp, jsonl_path)
    funcs = load_jsonl_funcs(jsonl_path)
    print(f"loaded {len(funcs)} function texts")

    train_pairs = load_pairs_by_label("train.txt", funcs)
    valid_pairs = load_pairs_by_label("valid.txt", funcs)
    test_pairs = load_pairs_by_label("test.txt", funcs)
    print(f"origin pool sizes: train={sum(len(v) for v in train_pairs.values())} "
         f"valid={sum(len(v) for v in valid_pairs.values())} "
         f"test={sum(len(v) for v in test_pairs.values())}")

    rng = random.Random(SEED)
    picks = {
        "train": stratified_sample(train_pairs, N_TRAIN, rng),
        "valid": stratified_sample(valid_pairs, N_VALID, rng),
        "test": stratified_sample(test_pairs, N_TEST, rng),
    }
    for origin, ps in picks.items():
        print(f"  drew {len(ps)} from {origin}.txt "
             f"(label0={sum(1 for p in ps if p[2]==0)}, "
             f"label1={sum(1 for p in ps if p[2]==1)})")

    rows = []
    i = 0
    for origin, ps in picks.items():
        for id1, id2, y in ps:
            rows.append({"ex_id": f"bcb:{i}", "func1": funcs[id1], "func2": funcs[id2],
                        "label": y, "art1": id1, "art2": id2, "origin": origin})
            i += 1
    json.dump(rows, open("data/bcb/bcb_pairs.json", "w"))
    print(f"total pool: {len(rows)} examples")

    # --- safe split: the origin label IS the entity-safe partition,
    #     inherited directly from the dataset's own function-disjoint
    #     construction; no clustering needed, and it is safe by inheritance.
    safe_split = {r["ex_id"]: r["origin"] for r in rows}
    json.dump(safe_split, open("data/bcb/bcb_split_safe.json", "w"))

    # --- unsafe split: pure random pair-level reshuffle of the SAME ids ---
    ids = [r["ex_id"] for r in rows]
    rng2 = random.Random(SEED)
    rng2.shuffle(ids)
    n = len(ids)
    unsafe_split = {}
    for j, ex in enumerate(ids):
        unsafe_split[ex] = ("train" if j < 0.8*n else ("valid" if j < 0.9*n else "test"))
    json.dump(unsafe_split, open("data/bcb/bcb_split_unsafe.json", "w"))

    # --- build the full sharing graph once, for both verification and Thm1 ---
    graph = SharingGraph.build([(r["ex_id"], [r["art1"], r["art2"]]) for r in rows])
    print("full pool cluster summary:", graph.summary())

    m_safe = measured_contamination(graph, {k: ("train" if v in ("train","valid") else v)
                                            for k, v in safe_split.items()})
    print(f"SAFE split: rho_measured={m_safe['rho_measured']:.4f} "
         f"({m_safe['n_contaminated']}/{m_safe['n_test']})  [expect exactly 0]")

    n_train_side = sum(1 for v in unsafe_split.values() if v in ("train", "valid"))
    n_test_side = sum(1 for v in unsafe_split.values() if v == "test")
    t_effective = n_test_side / (n_train_side + n_test_side)
    dist = graph.cluster_size_distribution()
    rho_pred = predicted_contamination(dist, t_effective)
    m_unsafe = measured_contamination(graph, {k: ("train" if v in ("train","valid") else v)
                                              for k, v in unsafe_split.items()})
    print(f"UNSAFE split: rho_measured={m_unsafe['rho_measured']:.4f} "
         f"({m_unsafe['n_contaminated']}/{m_unsafe['n_test']})  "
         f"[Theorem-1 predicted {rho_pred:.4f} at effective t={t_effective:.4f}]")
    json.dump(m_unsafe["contaminated_ids"],
             open("data/bcb/bcb_contaminated_test_unsafe.json", "w"))

    from collections import Counter
    print("safe split sizes:  ", Counter(safe_split.values()))
    print("unsafe split sizes:", Counter(unsafe_split.values()))
    print("\nboth protocols share the identical 30,000-example pool; "
         "only fold assignment differs.")

if __name__ == "__main__":
    main()