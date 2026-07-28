#!/usr/bin/env python3
"""Recompute every headline number in paper 1 from the run artifacts and print
it beside the value the manuscript claims.

Nothing here is taken on trust: each row states the claim, the recomputed
value, the files it was computed from, and PASS/FAIL/CHECK.

  PASS  = matches the manuscript within tolerance
  FAIL  = does not match -> the manuscript is wrong, or this script is; investigate
  CHECK = computed, but the manuscript value was not machine-comparable
          (interval, prose claim); read the two side by side yourself

Run from runs/:
    python ../scripts/verify_paper.py
    python ../scripts/verify_paper.py --section did      # one section only
"""
import argparse, glob, json, os, sys
import numpy as np

rng = np.random.default_rng(20260725)
DATA = '../data'

# ---------------------------------------------------------------- utilities

def load(d):
    """preds.json -> (ids, y_true, y_pred). Format: {ex_id: [pred, true]}."""
    o = json.load(open(os.path.join(d, 'preds.json')))
    ids = list(o.keys())
    return ids, np.array([o[i][1] for i in ids]), np.array([o[i][0] for i in ids])

def dirs(pat):
    out = [d for d in sorted(glob.glob(pat))
           if os.path.isdir(d) and os.path.isfile(os.path.join(d, 'preds.json'))]
    if not out:
        print(f'  !! no runs match {pat!r} - skipping')
    return out

def acc(yt, yp):
    return float((yt == yp).mean())

def f1(yt, yp, pos=1):
    tp = int(((yp == pos) & (yt == pos)).sum())
    fp = int(((yp == pos) & (yt != pos)).sum())
    fn = int(((yp != pos) & (yt == pos)).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)

def majority(yt):
    m = yt.mean()
    return float(max(m, 1 - m))

def idset(path, key=None):
    o = json.load(open(path))
    if isinstance(o, list):
        return set(map(str, o))
    return {str(k) for k, v in o.items() if v}

def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h

ROWS = []
def row(claim, claimed, computed, files, tol=0.002):
    """Register one checked quantity."""
    if isinstance(claimed, (int, float)) and isinstance(computed, (int, float)):
        status = 'PASS' if abs(claimed - computed) <= tol else 'FAIL'
        c_str, k_str = f'{computed:.4f}', f'{claimed:.4f}'
    else:
        status, c_str, k_str = 'CHECK', str(computed), str(claimed)
    ROWS.append((status, claim, k_str, c_str, files))
    mark = {'PASS': 'ok ', 'FAIL': '!! ', 'CHECK': ' ? '}[status]
    print(f'  {mark}{claim:<52s} paper={k_str:<22s} computed={c_str}')

# ---------------------------------------------------------------- sections

def sec_devign_core():
    print('\n== Devign, CodeBERT: exposure, strata, inflation (Tables 7, 8, 11) ==')
    L, S = dirs('pub_w_s*'), dirs('clu_w_s*')
    if not L or not S: return
    cids = idset(f'{DATA}/devign/contaminated_test_ids.json')
    rho, Mc, Mh, Mobs = [], [], [], []
    for d in L:
        ids, yt, yp = load(d)
        fl = np.array([str(i) in cids for i in ids])
        rho.append(fl.mean()); Mobs.append(acc(yt, yp))
        Mc.append(acc(yt[fl], yp[fl])); Mh.append(acc(yt[~fl], yp[~fl]))
    Ms = [acc(*load(d)[1:]) for d in S]
    f = 'runs/pub_w_s*, runs/clu_w_s*, data/devign/contaminated_test_ids.json'
    row('rho (measured contamination, shipped split)', 0.665, float(np.mean(rho)), f)
    row('M_c contaminated stratum accuracy',           0.701, float(np.mean(Mc)),  f)
    row('M_h clean stratum accuracy',                  0.543, float(np.mean(Mh)),  f)
    row('M_safe clustered-protocol accuracy',          0.608, float(np.mean(Ms)),  f)
    row('Delta = M_obs - M_safe',                      0.040, float(np.mean(Mobs) - np.mean(Ms)), f)
    row('beta = M_h - M_safe',                        -0.065, float(np.mean(Mh) - np.mean(Ms)), f)
    per = [a - b for a, b in zip(Mh, Ms)]
    row('beta per-seed all negative',                 'yes', 'yes' if all(x < 0 for x in per) else 'NO', f)
    row('beta per-seed sd',                           'n/a', f'{np.std(per, ddof=1):.4f}', f)

def sec_placebo():
    print('\n== Devign placebo on commit-composition strata (Table 11) ==')
    sizes = devign_entity_sizes()
    if sizes is None: return
    for tag, pat, claims in [
        ('shipped',  'pub_w_s*', dict(multi=0.695, sing=0.547, bm=0.585, bs=0.553)),
        ('clustered','clu_w_s*', dict(multi=0.626, sing=0.569, bm=0.697, bs=0.566))]:
        D = dirs(pat)
        if not D: continue
        am, asg, bm, bs, nm, ns = [], [], None, None, None, None
        for d in D:
            ids, yt, yp = load(d)
            multi = np.array([sizes.get(str(i), 1) >= 2 for i in ids])
            am.append(acc(yt[multi], yp[multi])); asg.append(acc(yt[~multi], yp[~multi]))
            bm, bs = majority(yt[multi]), majority(yt[~multi])
            nm, ns = int(multi.sum()), int((~multi).sum())
        f = f'runs/{pat}, data/devign/devign.csv'
        row(f'{tag}: multi-commit accuracy (n={nm})', claims['multi'], float(np.mean(am)), f)
        row(f'{tag}: singleton accuracy (n={ns})',    claims['sing'],  float(np.mean(asg)), f)
        row(f'{tag}: multi-commit majority rate',     claims['bm'],    bm, f)
        row(f'{tag}: singleton majority rate',        claims['bs'],    bs, f)
        row(f'{tag}: skill multi (acc - majority)',   round(claims['multi']-claims['bm'],3),
            float(np.mean(am)) - bm, f)
        row(f'{tag}: skill singleton',                round(claims['sing']-claims['bs'],3),
            float(np.mean(asg)) - bs, f)
    # majority baseline of the clustered test set as a whole
    D = dirs('clu_w_s*')
    if D:
        _, yt, _ = load(D[0])
        row('clustered test overall majority rate', 0.6146, majority(yt), 'runs/clu_w_s13')

def devign_entity_sizes():
    """Commit-cluster size per example, via union-find over devign.csv artifact ids."""
    path = f'{DATA}/devign/devign.csv'
    if not os.path.isfile(path):
        print(f'  !! {path} not found - placebo section skipped'); return None
    import csv, collections
    rows = list(csv.DictReader(open(path)))
    kid = 'example_id' if 'example_id' in rows[0] else 'ex_id'
    kart = 'artifact_ids' if 'artifact_ids' in rows[0] else 'artifacts'
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
        arts = [a for a in str(r[kart]).split(';') if a]
        ex_arts[r[kid]] = arts
        for a, b in zip(arts, arts[1:]): union(a, b)
    cnt = collections.Counter()
    root = {}
    for ex, arts in ex_arts.items():
        rt = find(arts[0]) if arts else ex
        root[ex] = rt; cnt[rt] += 1
    return {ex: cnt[r] for ex, r in root.items()}

def sec_patch():
    print('\n== Patch assessment, CodeBERT: inflation (Table 6) ==')
    L, S = dirs('qt_rand_w_s*'), dirs('qt_clu_w_s*')
    if not L or not S: return
    lf = [f1(*load(d)[1:]) for d in L]; sf = [f1(*load(d)[1:]) for d in S]
    la = [acc(*load(d)[1:]) for d in L]; sa = [acc(*load(d)[1:]) for d in S]
    f = 'runs/qt_rand_w_s*, runs/qt_clu_w_s*'
    row('patch-level F1',        0.858, float(np.mean(lf)), f)
    row('bug-level F1',          0.650, float(np.mean(sf)), f)
    row('Delta F1',              0.208, float(np.mean(lf) - np.mean(sf)), f, tol=0.003)
    row('patch-level accuracy',  0.950, float(np.mean(la)), f)
    row('bug-level accuracy',    0.846, float(np.mean(sa)), f)
    row('Delta accuracy',        0.104, float(np.mean(la) - np.mean(sa)), f, tol=0.003)
    row('bug-level F1 seed sd',  0.0383, float(np.std(sf, ddof=1)), f, tol=0.004)
    row('patch-level F1 seed sd',0.0030, float(np.std(lf, ddof=1)), f, tol=0.002)

def sec_did():
    print('\n== Difference-in-differences over bug-composition strata (Sec 7.1) ==')
    sp = f'{DATA}/patches/strata.json'
    if not os.path.isfile(sp):
        print('  !! data/patches/strata.json missing. Rebuild:')
        print("     python3 -c \"import json,collections;r=json.load(open('../data/patches/quatrain_text.json'));"
              "s=collections.Counter(x['bug'] for x in r);"
              "json.dump({x['ex_id']:('singleton' if s[x['bug']]==1 else 'hub') for x in r},"
              "open('../data/patches/strata.json','w'))\"")
        return
    st = json.load(open(sp))
    L, S = dirs('qt_rand_w_s*'), dirs('qt_clu_w_s*')
    if not L or not S: return
    def by(pat_dirs, want):
        out = []
        for d in pat_dirs:
            ids, yt, yp = load(d)
            m = np.array([st.get(str(i)) == want for i in ids])
            out.append(f1(yt[m], yp[m]))
        return out
    hl, hs = by(L, 'hub'), by(S, 'hub')
    sl, ss = by(L, 'singleton'), by(S, 'singleton')
    f = 'runs/qt_rand_w_s*, runs/qt_clu_w_s*, data/patches/strata.json'
    row('hub F1, patch-level',       0.597, float(np.mean(hl)), f, tol=0.004)
    row('hub F1, bug-level',         0.290, float(np.mean(hs)), f, tol=0.004)
    row('singleton F1, patch-level', 0.977, float(np.mean(sl)), f, tol=0.004)
    row('singleton F1, bug-level',   0.942, float(np.mean(ss)), f, tol=0.004)
    hub_d = [a - b for a, b in zip(hl, hs)]
    sng_d = [a - b for a, b in zip(sl, ss)]
    did = [a - b for a, b in zip(hub_d, sng_d)]
    row('hub drop',        0.307, float(np.mean(hub_d)), f, tol=0.004)
    row('singleton drop',  0.036, float(np.mean(sng_d)), f, tol=0.004)
    row('DiD = hub drop - singleton drop', 0.271, float(np.mean(did)), f, tol=0.004)
    sd = np.std(did, ddof=1); se = sd / np.sqrt(len(did))
    row('DiD per-seed values', '0.238 to 0.297',
        '[' + ', '.join(f'{d:.3f}' for d in sorted(did)) + ']', f)
    row('DiD 95% CI (z, n=3)', '+0.237 to +0.305',
        f'[{np.mean(did)-1.96*se:+.3f}, {np.mean(did)+1.96*se:+.3f}]', f)
    row('DiD 95% CI (t, n=3)', '+0.196 to +0.346',
        f'[{np.mean(did)-4.303*se:+.3f}, {np.mean(did)+4.303*se:+.3f}]', f)

def sec_dose():
    print('\n== Dose-response bins, Jaccard (Table 13) ==')
    corpus = f'{DATA}/patches/quatrain_text.json'
    split  = f'{DATA}/patches/qt_split_random.json'
    if not (os.path.isfile(corpus) and os.path.isfile(split)):
        print('  !! corpus or split file missing - skipping'); return
    rows = {r['ex_id']: r for r in json.load(open(corpus))}
    smap = json.load(open(split))
    train = {}
    for ex, part in smap.items():
        if part in ('train', 'valid'):
            train.setdefault(rows[ex]['bug'], []).append(set(str(rows[ex]['patch']).split()))
    test = [ex for ex, p in smap.items() if p == 'test']
    sim = {}
    for ex in test:
        sibs = train.get(rows[ex]['bug'], [])
        if not sibs: continue
        t = set(str(rows[ex]['patch']).split())
        sim[ex] = max((len(t & s) / len(t | s) if (t | s) else 0.0) for s in sibs)
    D = dirs('qt_rand_w_s*')
    if not D: return
    preds = {d: json.load(open(f'{d}/preds.json')) for d in D}
    edges = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0001]
    claims = {0.0: (213, 0.742), 0.3: (308, 0.971), 0.5: (516, 0.988),
              0.7: (391, 0.991), 0.9: (62, 1.000)}
    f = 'runs/qt_rand_w_s*, data/patches/quatrain_text.json, qt_split_random.json'
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = [e for e, v in sim.items() if lo <= v < hi]
        if not b: continue
        cn, ca = claims[lo]
        accs, k, n = [], 0, 0
        for d in D:
            p = preds[d]; got = [e for e in b if e in p]
            kk = sum(p[e][0] == p[e][1] for e in got)
            accs.append(kk / len(got)); k += kk; n += len(got)
        row(f'bin [{lo:.1f},{hi:.1f}) n', cn, len(b), f, tol=0.5)
        row(f'bin [{lo:.1f},{hi:.1f}) accuracy', ca, float(np.mean(accs)), f, tol=0.004)
        lo_w, hi_w = wilson(round(np.mean(accs) * len(b)), len(b))
        row(f'bin [{lo:.1f},{hi:.1f}) Wilson95 (distinct n)', 'see Table 13',
            f'[{lo_w:.3f}, {hi_w:.3f}]', f)

def sec_bcb():
    print('\n== BigCloneBench subsample facts (Sec 5.6) ==')
    p = f'{DATA}/bcb/bcb_pairs.json'
    if not os.path.isfile(p):
        print('  !! data/bcb/bcb_pairs.json missing - skipping'); return
    import collections
    rows = json.load(open(p))
    by = collections.Counter((r['origin'], r['label']) for r in rows)
    funcs = set()
    for r in rows: funcs.update((r['art1'], r['art2']))
    f = 'data/bcb/bcb_pairs.json'
    row('pool size',            30000, len(rows), f, tol=0.5)
    row('train label0 / label1', '12000 / 12000',
        f"{by[('train',0)]} / {by[('train',1)]}", f)
    row('valid label0 / label1', '1500 / 1500',
        f"{by[('valid',0)]} / {by[('valid',1)]}", f)
    row('test label0 / label1',  '1500 / 1500',
        f"{by[('test',0)]} / {by[('test',1)]}", f)
    row('distinct functions covered', 9046, len(funcs), f, tol=0.5)

def sec_ux():
    print('\n== UniXcoder replication (Sec 7.4, Appendix C) ==')
    for name, lk, sf in [('Devign', 'ux_pub_s*', 'ux_clu_s*'),
                         ('patch',  'ux_qt_rand_s*', 'ux_qt_clu_s*')]:
        L, S = dirs(lk), dirs(sf)
        if not L or not S: continue
        m = f1 if name == 'patch' else acc
        lv = [m(*load(d)[1:]) for d in L]; sv = [m(*load(d)[1:]) for d in S]
        f = f'runs/{lk}, runs/{sf}'
        metric = 'F1' if name == 'patch' else 'accuracy'
        row(f'UX {name} leaky {metric}',  'see paper', f'{np.mean(lv):.4f}', f)
        row(f'UX {name} safe {metric}',   'see paper', f'{np.mean(sv):.4f}', f)
        row(f'UX {name} Delta',           'see paper', f'{np.mean(lv)-np.mean(sv):+.4f}', f)
        row(f'UX {name} safe per-seed',   'stability check',
            '[' + ', '.join(f'{v:.3f}' for v in sv) + ']', f)

SECTIONS = {'devign': sec_devign_core, 'placebo': sec_placebo, 'patch': sec_patch,
            'did': sec_did, 'dose': sec_dose, 'bcb': sec_bcb, 'ux': sec_ux}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--section', choices=list(SECTIONS) + ['all'], default='all')
    a = ap.parse_args()
    if not os.path.isdir('pub_w_s13') and not os.path.isdir('qt_rand_w_s13'):
        sys.exit('run this from the runs/ directory')
    todo = SECTIONS if a.section == 'all' else {a.section: SECTIONS[a.section]}
    for fn in todo.values():
        try:
            fn()
        except Exception as e:
            print(f'  !! section raised {type(e).__name__}: {e}')

    p = sum(1 for r in ROWS if r[0] == 'PASS')
    fl = [r for r in ROWS if r[0] == 'FAIL']
    c = sum(1 for r in ROWS if r[0] == 'CHECK')
    print('\n' + '=' * 72)
    print(f'PASS {p}   FAIL {len(fl)}   CHECK-BY-EYE {c}')
    if fl:
        print('\nFAILED - the manuscript and the data disagree here:')
        for _, claim, k, cm, files in fl:
            print(f'  {claim}\n     paper={k}  computed={cm}\n     from: {files}')
    else:
        print('\nEvery machine-comparable claim matches the artifacts.')
    print('\nCHECK rows are computed but not auto-compared (intervals, per-seed lists).')
    print('Read those against the manuscript yourself.')

if __name__ == '__main__':
    main()