#!/usr/bin/env python3
"""Multi-seed dose-response for patch assessment, reimplemented to spec with a
verification gate: it must reproduce the published seed-42 bins (first bin
n=213, acc 0.742) under one of the candidate similarity measures before the
multi-seed numbers are trusted.

Similarity: max over same-bug TRAINING patches of measure(test_tokens, train_tokens),
tokens = whitespace split of the 'patch' field. Split fixed by qt_split_random.json
(valid folded into train). Clean test examples (no same-bug training sibling)
reported separately, not binned.

Usage (from runs/):
  python dose_response.py --measure all          # verification pass
  python dose_response.py --measure jaccard      # once locked, final table
"""
import argparse, glob, json, os, sys
import numpy as np

EDGES = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0001]

def toks(s): return set(str(s).split())

def jaccard(a, b):    u = a | b; return len(a & b) / len(u) if u else 0.0
def containment(a, b): return len(a & b) / len(a) if a else 0.0
def dice(a, b):        d = len(a) + len(b); return 2 * len(a & b) / d if d else 0.0
MEASURES = {'jaccard': jaccard, 'containment': containment, 'dice': dice}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', default='../data/patches/quatrain_text.json')
    ap.add_argument('--split', default='../data/patches/qt_split_random.json')
    ap.add_argument('--runs', default='qt_rand_w_s*')
    ap.add_argument('--verify-seed', type=int, default=42)
    ap.add_argument('--measure', default='all', choices=['all'] + list(MEASURES))
    a = ap.parse_args()

    rows = {r['ex_id']: r for r in json.load(open(a.corpus))}
    smap = json.load(open(a.split))
    train_by_bug = {}
    for ex, part in smap.items():
        if part in ('train', 'valid'):
            r = rows[ex]
            train_by_bug.setdefault(r['bug'], []).append(toks(r['patch']))
    test_ids = [ex for ex, part in smap.items() if part == 'test']
    print(f'test={len(test_ids)}  bugs with training patches={len(train_by_bug)}')

    run_dirs = sorted(d for d in glob.glob(a.runs) if os.path.isfile(os.path.join(d, 'preds.json')))
    if not run_dirs: sys.exit(f'no runs match {a.runs!r}')
    preds = {}
    for d in run_dirs:
        obj = json.load(open(os.path.join(d, 'preds.json')))
        preds[d] = {k: (v[0], v[1]) for k, v in obj.items()}

    names = list(MEASURES) if a.measure == 'all' else [a.measure]
    for mname in names:
        mf = MEASURES[mname]
        sim = {}
        n_clean = 0
        for ex in test_ids:
            r = rows[ex]
            sibs = train_by_bug.get(r['bug'], [])
            if not sibs:
                n_clean += 1; continue
            t = toks(r['patch'])
            sim[ex] = max(mf(t, s) for s in sibs)
        print(f'\n#### measure = {mname}  (contaminated binned = {len(sim)}, clean excluded = {n_clean})')

        # verification column first
        vdir = next((d for d in run_dirs if d.endswith(f's{a.verify_seed}')), None)
        header = f'{"bin":10s} {"n":>5s}'
        if vdir: header += f'  {"acc(s"+str(a.verify_seed)+")":>10s}'
        header += f'  {"mean+-sd("+str(len(run_dirs))+" seeds)":>20s}  {"Wilson95(pooled)":>18s}'
        print(header)
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            binned = [ex for ex, v in sim.items() if lo <= v < hi]
            if not binned: 
                print(f'[{lo:.1f},{hi:.1f})  {0:5d}   (empty)'); continue
            accs, kpool, npool = [], 0, 0
            vacc = None
            for d in run_dirs:
                pd = preds[d]
                got = [ex for ex in binned if ex in pd]
                k = sum(pd[ex][0] == pd[ex][1] for ex in got)
                accs.append(k / len(got))
                kpool += k; npool += len(got)
                if d == vdir: vacc = k / len(got)
            p = kpool / npool; z = 1.96
            den = 1 + z*z/npool
            c = (p + z*z/(2*npool)) / den
            h = z*np.sqrt(p*(1-p)/npool + z*z/(4*npool*npool)) / den
            line = f'[{lo:.1f},{hi:.1f})  {len(binned):5d}'
            if vacc is not None: line += f'  {vacc:10.3f}'
            line += f'  {np.mean(accs):8.3f} +- {np.std(accs):.3f}       [{c-h:.3f}, {c+h:.3f}]'
            print(line)
        print('>> VERIFY: does one measure reproduce the published Table-13 seed-42 column')
        print('>> (first bin n=213, acc 0.742)? If yes, lock it; if no measure does, STOP.')

if __name__ == '__main__':
    main()