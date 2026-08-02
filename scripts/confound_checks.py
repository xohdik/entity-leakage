import argparse, csv, glob, json, os, sys
import numpy as np

B = 2000
rng = np.random.default_rng(20260725)

def load_preds(d):
    obj = json.load(open(os.path.join(d, 'preds.json')))
    ids = list(obj.keys())
    return ids, np.array([obj[i][1] for i in ids]), np.array([obj[i][0] for i in ids])

def load_map(path):
    if path.endswith('.csv'):
        m = {}
        for row in csv.DictReader(open(path)):
            k = row.get('example_id') or row.get('id') or row.get('ex_id') or list(row.values())[0]
            v = row.get('stratum') or row.get('sim') or row.get('J') or list(row.values())[1]
            m[str(k)] = v
        return m
    obj = json.load(open(path))
    return {str(k): v for k, v in obj.items()}

def f1(yt, yp):
    tp = ((yp == 1) & (yt == 1)).sum(); fp = ((yp == 1) & (yt == 0)).sum(); fn = ((yp == 0) & (yt == 1)).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)

def acc(yt, yp): return (yt == yp).mean()

def runs(pattern):
    ds = [d for d in sorted(glob.glob(pattern)) if os.path.isfile(os.path.join(d, 'preds.json'))]
    if not ds: sys.exit(f'nothing matches {pattern!r}')
    return [(d, *load_preds(d)) for d in ds]

def boot_metric(runs_, metric):
    """Bootstrap the seed-mean metric by resampling test examples per seed."""
    out = np.empty(B)
    for b in range(B):
        vals = []
        for _, ids, yt, yp in runs_:
            ix = rng.integers(0, len(yt), len(yt))
            vals.append(metric(yt[ix], yp[ix]))
        out[b] = np.mean(vals)
    return out

def ci(a): return np.percentile(a, [2.5, 97.5])

def stratum_of(ids, smap, sizes):
    st = []
    for i in ids:
        v = smap.get(str(i))
        if v is None: sys.exit(f'id {i} missing from strata map')
        if sizes is not None:
            v = 'singleton' if int(sizes.get(str(v), sizes.get(v, 0))) <= 1 else 'hub'
        st.append(str(v).lower())
    return np.array(st)

def cmd_patch(a):
    L, S = runs(a.leaky), runs(a.safe)
    smap = load_map(a.strata)
    sizes = load_map(a.bug_sizes) if a.bug_sizes else None

    for name, metric in [('F1', f1), ('accuracy', acc)]:
        bl, bs = boot_metric(L, metric), boot_metric(S, metric)
        d = bl - bs
        print(f'== {name} ==')
        print(f'leaky  mean {np.mean([metric(y,p) for _,_,y,p in L]):.4f}  CI {ci(bl).round(4)}')
        print(f'safe   mean {np.mean([metric(y,p) for _,_,y,p in S]):.4f}  CI {ci(bs).round(4)}')
        lo, hi = ci(d)
        print(f'Delta  mean {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  excludes 0: {lo > 0 or hi < 0}')

    # composition matching on F1: reweight safe test to leaky singleton/hub mix
    _, idsL, ytL, _ = L[0]
    stL = stratum_of(idsL, smap, sizes)
    pL = {s: (stL == s).mean() for s in ('singleton', 'hub')}
    print(f'\nleaky test composition: {pL}')

    print('== composition-matched Delta (F1), post-stratified ==')
    # per-seed: F1 within each stratum of safe runs, recombined at leaky proportions
    # (F1 is not stratum-decomposable, so we do it by subsample bootstrap instead:)
    d = np.empty(B)
    for b in range(B):
        vl, vs = [], []
        for _, ids, yt, yp in L:
            ix = rng.integers(0, len(yt), len(yt)); vl.append(f1(yt[ix], yp[ix]))
        for _, ids, yt, yp in S:
            st = stratum_of(ids, smap, sizes)
            ix = []
            n = len(yt)
            for s in ('singleton', 'hub'):
                pool = np.flatnonzero(st == s)
                k = int(round(pL[s] * n))
                if len(pool) == 0: sys.exit(f'safe test has no {s} examples')
                ix.append(rng.choice(pool, k, replace=True))
            ix = np.concatenate(ix)
            vs.append(f1(yt[ix], yp[ix]))
        d[b] = np.mean(vl) - np.mean(vs)
    lo, hi = ci(d)
    print(f'Delta_matched mean {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  excludes 0: {lo > 0 or hi < 0}')
    print('(safe test resampled to the leaky singleton/hub mix; leakage effect net of composition)')

def cmd_devign(a):
    L, S = runs(a.leaky), runs(a.safe)
    bl, bs = boot_metric(L, acc), boot_metric(S, acc)
    d = bl - bs; lo, hi = ci(d)
    print(f'Devign accuracy Delta mean {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  excludes 0: {lo > 0 or hi < 0}')

def cmd_dose(a):
    L = runs(a.leaky)
    sim = {k: float(v) for k, v in load_map(a.sim).items()}
    edges = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0001]
    print('bin        n     acc(mean+-sd over seeds)   Wilson95 (pooled)')
    for lo_, hi_ in zip(edges[:-1], edges[1:]):
        accs, npool, kpool = [], 0, 0
        for _, ids, yt, yp in L:
            m = np.array([lo_ <= sim.get(str(i), -1) < hi_ for i in ids])
            if m.sum() == 0: continue
            accs.append(acc(yt[m], yp[m])); npool += int(m.sum()); kpool += int((yt[m] == yp[m]).sum())
        if not accs: continue
        p = kpool / npool; z = 1.96
        den = 1 + z*z/npool; c = (p + z*z/(2*npool)) / den
        h = z*np.sqrt(p*(1-p)/npool + z*z/(4*npool*npool)) / den
        print(f'[{lo_:.1f},{hi_:.1f}) {npool//len(L):5d}  {np.mean(accs):.3f} +- {np.std(accs):.3f}          [{c-h:.3f}, {c+h:.3f}]')

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('patch'); p.add_argument('--leaky', required=True); p.add_argument('--safe', required=True)
    p.add_argument('--strata', required=True); p.add_argument('--bug-sizes')
    d = sub.add_parser('devign'); d.add_argument('--leaky', required=True); d.add_argument('--safe', required=True)
    o = sub.add_parser('dose'); o.add_argument('--leaky', required=True); o.add_argument('--sim', required=True)
    a = ap.parse_args()
    {'patch': cmd_patch, 'devign': cmd_devign, 'dose': cmd_dose}[a.cmd](a)

if __name__ == '__main__':
    main()