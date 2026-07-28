#!/usr/bin/env python3
"""Recompute Tables 11 and 12 (eight-architecture comparison) and the run-level
census in Section 7.5, and print each value beside the manuscript's claim.

Same convention as verify_paper.py:
  PASS  = matches within tolerance
  FAIL  = manuscript and data disagree -> investigate
  CHECK = computed, compare by eye

Run from runs/:
    python ../scripts/verify_arch_tables.py
"""
import glob, json, os, re, sys
import numpy as np

TAGS = {'cb': 'CodeBERT-base', 'gcb': 'GraphCodeBERT-base', 'ux': 'UniXcoder-base',
        'ct5': 'CodeT5-base', 'ct5p': 'CodeT5+ 220M', 'plb': 'PLBART-base',
        'csg': 'CodeSage-small', 'rob': 'RoBERTa-base (NL)'}
SEEDS = (13, 21, 42)

# manuscript values: tag -> (acc_leaky, skill_leaky, acc_safe, skill_safe, f1_leaky, f1_safe)
CLAIM_DEVIGN = {
    'cb':  (0.658, +0.118, 0.567, -0.048, 0.555, 0.308),
    'gcb': (0.622, +0.082, 0.595, -0.020, 0.408, 0.356),
    'ux':  (0.626, +0.085, 0.546, -0.069, 0.419, 0.488),
    'ct5': (0.657, +0.116, 0.550, -0.065, 0.610, 0.489),
    'ct5p':(0.657, +0.116, 0.551, -0.064, 0.620, 0.490),
    'plb': (0.623, +0.082, 0.496, -0.119, 0.608, 0.484),
    'csg': (0.645, +0.104, 0.586, -0.029, 0.632, 0.432),
    'rob': (0.620, +0.080, 0.615, +0.000, 0.394, 0.000),
}
CLAIM_PATCH = {
    'cb':  (0.952, +0.134, 0.863, +0.044, 0.872, 0.708),
    'gcb': (0.952, +0.134, 0.870, +0.051, 0.869, 0.706),
    'ux':  (0.951, +0.133, 0.867, +0.048, 0.868, 0.707),
    'ct5': (0.950, +0.132, 0.866, +0.047, 0.861, 0.708),
    'ct5p':(0.947, +0.129, 0.891, +0.072, 0.861, 0.739),
    'plb': (0.946, +0.128, 0.863, +0.044, 0.852, 0.705),
    'csg': (0.942, +0.124, 0.886, +0.067, 0.823, 0.734),
    'rob': (0.946, +0.128, 0.881, +0.062, 0.854, 0.728),
}

ROWS = []
def row(claim, claimed, computed, tol=0.0015):
    if isinstance(claimed, (int, float)) and isinstance(computed, (int, float)):
        ok = abs(claimed - computed) <= tol
        st = 'PASS' if ok else 'FAIL'
        k, c = f'{claimed:+.3f}', f'{computed:+.3f}'
    else:
        st, k, c = 'CHECK', str(claimed), str(computed)
    ROWS.append((st, claim, k, c))
    print(f'  {{"PASS":"ok ","FAIL":"!! ","CHECK":" ? "}}[st] {claim:<44s} paper={k:<10s} computed={c}'
          .replace('{"PASS":"ok ","FAIL":"!! ","CHECK":" ? "}[st]',
                   {'PASS': 'ok ', 'FAIL': '!! ', 'CHECK': ' ? '}[st]))

def load(d):
    o = json.load(open(os.path.join(d, 'preds.json')))
    ids = list(o.keys())
    return np.array([o[i][1] for i in ids]), np.array([o[i][0] for i in ids])

def acc(yt, yp): return float((yt == yp).mean())
def f1(yt, yp):
    tp = int(((yp == 1) & (yt == 1)).sum()); fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1)
def majority(yt): m = yt.mean(); return float(max(m, 1 - m))

def table(task, leaky, safe, claims, tnum):
    print(f'\n== Table {tnum}: {task}, eight architectures ==')
    mj = {}
    for proto in (leaky, safe):
        d = f'rank_cb_{task}_{proto}_s{SEEDS[0]}'
        if not os.path.isdir(d):
            print(f'  !! {d} missing - section skipped'); return None
        mj[proto] = majority(load(d)[0])
    print(f'  majority baselines: {leaky}={mj[leaky]:.4f}  {safe}={mj[safe]:.4f}')
    for tag in CLAIM_DEVIGN:
        vals = {}
        ok = True
        for proto in (leaky, safe):
            a, f = [], []
            for s in SEEDS:
                d = f'rank_{tag}_{task}_{proto}_s{s}'
                if not os.path.isdir(d): ok = False; break
                yt, yp = load(d); a.append(acc(yt, yp)); f.append(f1(yt, yp))
            if not ok: break
            vals[proto] = (float(np.mean(a)), float(np.mean(f)))
        if not ok:
            print(f'  !! {tag}: incomplete runs, skipped'); continue
        cl = claims[tag]
        name = TAGS[tag]
        row(f'{name} acc {leaky}',   cl[0], vals[leaky][0])
        row(f'{name} skill {leaky}', cl[1], vals[leaky][0] - mj[leaky])
        row(f'{name} acc {safe}',    cl[2], vals[safe][0])
        row(f'{name} skill {safe}',  cl[3], vals[safe][0] - mj[safe])
        row(f'{name} F1 {leaky}',    cl[4], vals[leaky][1])
        row(f'{name} F1 {safe}',     cl[5], vals[safe][1])
    return mj

def census(task, leaky, safe, label):
    print(f'\n== Run-level census: {label} ==')
    counts = {'below': {leaky: 0, safe: 0}, 'collapsed': {leaky: 0, safe: 0}, 'total': {leaky: 0, safe: 0}}
    for tag in TAGS:
        for proto in (leaky, safe):
            for s in SEEDS:
                d = f'rank_{tag}_{task}_{proto}_s{s}'
                if not os.path.isdir(d): continue
                yt, yp = load(d)
                counts['total'][proto] += 1
                pr = float((yp == 1).mean())
                collapsed = pr < 0.02 or pr > 0.98
                if collapsed: counts['collapsed'][proto] += 1
                if collapsed or acc(yt, yp) <= majority(yt) + 0.005:
                    counts['below'][proto] += 1
    for proto in (leaky, safe):
        print(f'  {proto:6s}: {counts["below"][proto]}/{counts["total"][proto]} at or below majority; '
              f'{counts["collapsed"][proto]} collapsed (no minority predictions)')
    return counts

def main():
    if not os.path.isdir('rank_cb_devign_pub_s13'):
        sys.exit('run this from the runs/ directory')
    table('devign', 'pub', 'clu', CLAIM_DEVIGN, 11)
    table('qt', 'rand', 'clu', CLAIM_PATCH, 12)
    cd = census('devign', 'pub', 'clu', 'Devign (Section 7.5 prose)')
    cq = census('qt', 'rand', 'clu', 'Patch assessment (Section 7.5 prose)')

    print('\n== Section 7.5 prose claims ==')
    row('Devign shipped runs at/below majority', 3, cd['below']['pub'], tol=0.5)
    row('Devign entity runs at/below majority', 23, cd['below']['clu'], tol=0.5)
    row('Devign runs collapsed (total)', 7,
        cd['collapsed']['pub'] + cd['collapsed']['clu'], tol=0.5)
    row('Devign collapsed, shipped protocol', 3, cd['collapsed']['pub'], tol=0.5)
    row('Devign collapsed, entity-level protocol', 4, cd['collapsed']['clu'], tol=0.5)
    row('Patch runs at/below majority (both protocols)', 0,
        cq['below']['rand'] + cq['below']['clu'], tol=0.5)

    p = sum(1 for r in ROWS if r[0] == 'PASS'); fl = [r for r in ROWS if r[0] == 'FAIL']
    print('\n' + '=' * 70)
    print(f'PASS {p}   FAIL {len(fl)}   CHECK {sum(1 for r in ROWS if r[0]=="CHECK")}')
    if fl:
        print('\nFAILED:')
        for _, claim, k, c in fl:
            print(f'  {claim}: paper={k} computed={c}')
    else:
        print('\nTables 11-12 and the Section 7.5 census reproduce from the run artifacts.')

if __name__ == '__main__':
    main()