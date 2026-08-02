#!/usr/bin/env python3
"""Prop-2 decomposition from runs/<name>/preds.json layout.

Defaults target patch assessment under the unified (warmup) config:
  leaky = qt_rand_w_s*   safe = qt_clu_w_s*
For UniXcoder replication:
  --leaky 'ux_qt_rand_s*' --safe 'ux_qt_clu_s*'

Contamination map: file mapping example id -> 0/1 (or a list/set of
contaminated ids) for the LEAKY test split. Accepts .json or .csv.
If you don't know where it is:  find .. -iname '*contam*' -o -iname '*strat*'

Usage (from runs/):
  python decompose_patch2.py --contam ../data/patches/qt_contaminated_test.json
  python decompose_patch2.py --contam ../data/patches/qt_contaminated_test.json --leaky 'ux_qt_rand_s*' --safe 'ux_qt_clu_s*'
  python decompose_patch2.py --contam ../data/devign/contaminated_test_ids.json --leaky 'pub_w_s*' --safe 'clu_w_s*'
"""
import argparse, glob, json, os, sys, csv
import numpy as np

ID_KEYS   = ['example_id', 'id', 'idx', 'index', 'pair_id', 'uid', 'name']
TRUE_KEYS = ['y_true', 'label', 'labels', 'gold', 'target', 'true']
PRED_KEYS = ['y_pred', 'pred', 'preds', 'prediction', 'predicted']

def pick(d, keys):
    for k in keys:
        if k in d: return k
    return None

def load_preds(path, pred_first=True):
    """Return (ids, y_true, y_pred) from a preds.json of unknown-ish schema.

    pred_first: for the compact {example_id: [v0, v1]} format used by this
    codebase's train_*.py scripts (preds[ex] = (pi, yi)), v0 is the
    prediction and v1 is the true label. If the printed Mobs/accuracy for a
    known seed doesn't match the paper's tables, rerun with --true-first.
    """
    obj = json.load(open(path))

    # compact format: {example_id: [v0, v1]}
    if isinstance(obj, dict):
        vals = list(obj.values())
        if vals and isinstance(vals[0], (list, tuple)) and len(vals[0]) == 2 \
           and not isinstance(vals[0][0], dict):
            ids = list(obj.keys())
            if pred_first:
                yp = [v[0] for v in vals]
                yt = [v[1] for v in vals]
            else:
                yt = [v[0] for v in vals]
                yp = [v[1] for v in vals]
            return ids, yt, yp

    if isinstance(obj, dict):
        kt, kp = pick(obj, TRUE_KEYS), pick(obj, PRED_KEYS)
        ki = pick(obj, ID_KEYS)
        if kt and kp and isinstance(obj[kt], list):          # dict of arrays
            n = len(obj[kt])
            ids = obj[ki] if ki else list(range(n))
            return ids, obj[kt], obj[kp]
        if vals and isinstance(vals[0], dict):                # dict id -> record
            kt, kp = pick(vals[0], TRUE_KEYS), pick(vals[0], PRED_KEYS)
            if kt and kp:
                ids = list(obj.keys())
                return ids, [obj[i][kt] for i in ids], [obj[i][kp] for i in ids]
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):  # list of records
        kt, kp, ki = pick(obj[0], TRUE_KEYS), pick(obj[0], PRED_KEYS), pick(obj[0], ID_KEYS)
        if kt and kp:
            ids = [r[ki] for r in obj] if ki else list(range(len(obj)))
            return ids, [r[kt] for r in obj], [r[kp] for r in obj]
    print(f'!! could not detect schema of {path}; sample below — paste this back:')
    print(json.dumps(obj if not isinstance(obj, list) else obj[:2], default=str)[:800])
    sys.exit(1)

def to_int_pred(p):
    if isinstance(p, list):            # logits/probs
        return int(np.argmax(p))
    if isinstance(p, float) and 0.0 <= p <= 1.0 and p not in (0.0, 1.0):
        return int(p >= 0.5)
    return int(p)

def load_contam(path):
    """Return dict id->0/1 or a set of contaminated ids."""
    if path.endswith('.csv'):
        m = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                ki = pick(row, ID_KEYS)
                kc = pick(row, ['contaminated', 'contam', 'is_contaminated', 'leaky', 'stratum'])
                v = row[kc]
                m[row[ki]] = int(v in ('1', 'True', 'true', 'contaminated', 'contam'))
        return m
    obj = json.load(open(path))
    if isinstance(obj, list):
        return set(str(x) for x in obj)
    if isinstance(obj, dict):
        return {str(k): int(bool(v)) for k, v in obj.items()}
    sys.exit('unrecognized contamination file format')

def contam_flag(cmap, i):
    i = str(i)
    if isinstance(cmap, set):
        return int(i in cmap)
    return cmap.get(i, None)

def metrics(y, p):
    y, p = np.array(y, int), np.array(p, int)
    acc = float((y == p).mean())
    ba = float(np.mean([ (p[y == c] == c).mean() for c in np.unique(y) ]))
    return acc, ba

def collect(pattern, pred_first):
    dirs = sorted(glob.glob(pattern))
    out = []
    for d in dirs:
        f = os.path.join(d, 'preds.json')
        if os.path.isfile(f):
            ids, yt, yp = load_preds(f, pred_first)
            out.append((d, ids, yt, yp))
    if not out:
        sys.exit(f'no preds.json under pattern {pattern!r} — run from runs/ or fix glob')
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--leaky', default='qt_rand_w_s*')
    ap.add_argument('--safe',  default='qt_clu_w_s*')
    ap.add_argument('--contam', required=True)
    ap.add_argument('--true-first', action='store_true',
                    help='compact preds.json is [true, pred] not [pred, true] '
                         '(only matters if printed Mobs disagrees with the paper)')
    a = ap.parse_args()
    pred_first = not a.true_first

    cmap = load_contam(a.contam)
    L, S = collect(a.leaky, pred_first), collect(a.safe, pred_first)

    for mi, name in [(0, 'accuracy'), (1, 'balanced_accuracy')]:
        rhos, Mobs, Mcs, Mhs = [], [], [], []
        for d, ids, yt, yp in L:
            yp = [to_int_pred(x) for x in yp]
            yt = [int(x) for x in yt]
            flags = [contam_flag(cmap, i) for i in ids]
            miss = sum(f is None for f in flags)
            if miss:
                print(f'!! {d}: {miss}/{len(ids)} ids missing from contamination map — id mismatch, stop and check')
                sys.exit(1)
            flags = np.array(flags)
            yt, yp = np.array(yt), np.array(yp)
            rhos.append(flags.mean())
            Mobs.append(metrics(yt, yp)[mi])
            Mcs.append(metrics(yt[flags == 1], yp[flags == 1])[mi])
            Mhs.append(metrics(yt[flags == 0], yp[flags == 0])[mi])
        Msafe = [metrics([int(x) for x in yt], [to_int_pred(x) for x in yp])[mi] for _, _, yt, yp in S]

        rho, mo, mc, mh, ms = map(np.mean, (rhos, Mobs, Mcs, Mhs, Msafe))
        eps, beta = mc - mh, mh - ms
        print(f'== {name} ==  ({len(L)} leaky seeds: {[d for d,_,_,_ in L]}, {len(S)} safe seeds)')
        print(f'rho={rho:.3f}  Mobs={mo:.3f}  Mc={mc:.3f}  Mh={mh:.3f}  Msafe={ms:.3f}')
        print(f'eps = Mc-Mh = {eps:.3f}    beta = Mh-Msafe = {beta:+.3f}')
        print(f'Delta_pred = rho*eps + beta = {rho*eps + beta:+.4f}')
        print(f'Delta_meas = Mobs - Msafe   = {mo - ms:+.4f}')
        print(f'per-seed Mobs: {[f"{v:.3f}" for v in Mobs]}   per-seed Msafe: {[f"{v:.3f}" for v in Msafe]}\n')


if __name__ == '__main__':
    main()