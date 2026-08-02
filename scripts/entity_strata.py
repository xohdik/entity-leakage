import argparse, csv, glob, json, os, sys, collections
import numpy as np

def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z*z/n
    c = (p + z*z/(2*n)) / den
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / den
    return c - h, c + h

def load_preds(d):
    obj = json.load(open(os.path.join(d, 'preds.json')))
    return {k: (v[0], v[1]) for k, v in obj.items()}

def runs(pattern):
    ds = [d for d in sorted(glob.glob(pattern)) if os.path.isfile(os.path.join(d, 'preds.json'))]
    if not ds: sys.exit(f'nothing matches {pattern!r}')
    return {d: load_preds(d) for d in ds}

def devign_entity_sizes():
    """Commit-cluster size per example from devign.csv (example_id, artifact_ids, ...)."""
    path = '../data/devign/devign.csv'
    rows = list(csv.DictReader(open(path)))
    key_id = 'example_id' if 'example_id' in rows[0] else 'ex_id'
    key_art = 'artifact_ids' if 'artifact_ids' in rows[0] else 'artifacts'
    # union-find over artifact ids
    parent = {}
    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb
    ex_arts = {}
    for r in rows:
        arts = [a for a in str(r[key_art]).split(';') if a]
        ex_arts[r[key_id]] = arts
        for a, b in zip(arts, arts[1:]): union(a, b)
    root_count = collections.Counter()
    ex_root = {}
    for ex, arts in ex_arts.items():
        root = find(arts[0]) if arts else ex
        ex_root[ex] = root
        root_count[root] += 1
    return {ex: root_count[r] for ex, r in ex_root.items()}

def patch_entity_sizes():
    rows = json.load(open('../data/patches/quatrain_text.json'))
    sizes = collections.Counter(r['bug'] for r in rows)
    return {r['ex_id']: sizes[r['bug']] for r in rows}

def report(tag, run_preds, esize):
    print(f'\n== {tag} ==')
    out = {}
    for stratum, cond in [('multi', lambda s: s >= 2), ('singleton', lambda s: s == 1)]:
        per_seed, kpool = [], 0
        n_ref = None
        for d, preds in run_preds.items():
            ids = [i for i in preds if cond(esize.get(str(i), 1))]
            if n_ref is None: n_ref = len(ids)
            k = sum(preds[i][0] == preds[i][1] for i in ids)
            per_seed.append(k / len(ids))
            kpool += k
        n = n_ref
        p = np.mean(per_seed)
        lo, hi = wilson(round(p * n), n)
        out[stratum] = (n, p, per_seed)
        print(f'{stratum:10s} n={n:5d}  per-seed {[f"{v:.3f}" for v in per_seed]}  mean {p:.4f}  Wilson95 [{lo:.3f}, {hi:.3f}]')
    gap = out['multi'][1] - out['singleton'][1]
    gaps = [m - s for m, s in zip(out['multi'][2], out['singleton'][2])]
    print(f'multi - singleton gap: mean {gap:+.4f}  per-seed {[f"{g:+.3f}" for g in gaps]}')
    return gap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('task', choices=['devign', 'patch'])
    ap.add_argument('--leaky', required=True)
    ap.add_argument('--safe', required=True)
    a = ap.parse_args()
    esize = devign_entity_sizes() if a.task == 'devign' else patch_entity_sizes()
    sizes = np.array(list(esize.values()))
    print(f'corpus: {len(esize)} examples, singleton share {np.mean(sizes==1):.3f}')
    gl = report(f'{a.task} LEAKY ({a.leaky})', runs(a.leaky), esize)
    gs = report(f'{a.task} SAFE  ({a.safe})', runs(a.safe), esize)
    
if __name__ == '__main__':
    main()