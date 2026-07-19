"""PatchLabelsYe.csv -> generic CSV. Artifact = benchmark-qualified bug ID.

  GenProg_patch_Defects4J_Chart_1_0_129            -> Defects4J_Chart_1
  Arja_patch_Bugs.jar_Commons-Math_a06a1584_0_49_X -> Bugs.jar_Commons-Math_a06a1584
  patch1-math-70_HDRepair_PatchNaturalness         -> Defects4J_Math_70
  ..Bears_<slug>-<buildids>..                      -> Bears_<slug>-<buildids>
Unparsed -> singleton (conservative; under-estimates rho only).
"""
import csv, re, sys
from collections import Counter

PATTERNS = [
    # Bugs.jar: project + commit hash IS the bug identity
    re.compile(r"Bugs\.?jar_(?P<proj>[A-Za-z0-9\-]+)_(?P<hash>[0-9a-f]{6,12})", re.I),
    # Bears: slug with build-id pair
    re.compile(r"Bears_(?P<slug>[A-Za-z0-9\-]+_\d+-\d+)"),
    re.compile(r"Bears[-_](?P<bug>\d+)(?=[_\.\-]|$)"),
    # Defects4J-style: _patch_<ds>_<Proj>_<num>
    re.compile(r"_patch_(?P<ds>[A-Za-z0-9]+)_(?P<proj>[A-Za-z][A-Za-z0-9]*)_(?P<bug>\d+)"),
    # patch1-math-70 style
    re.compile(r"patch\d*-(?P<proj>[A-Za-z]+)-(?P<bug>\d+)", re.I),
    # QuixBugs
    re.compile(r"QuixBugs[-_](?P<slug>[A-Za-z0-9_\-]+)", re.I),
]

BARE_D4J  = re.compile(r"^(?P<proj>[A-Za-z][A-Za-z0-9]*)_(?P<bug>\d+)$")
BARE_BJAR = re.compile(r"^Bugs\.?jar_(?P<hash>[0-9a-f]{6,12})_?$", re.I)

def bug_id(name):
    m = BARE_BJAR.match(name)
    if m:
        return f"Bugs.jar_{m.group('hash')}"
    m = BARE_D4J.match(name)
    if m:
        return f"Defects4J_{m.group('proj').capitalize()}_{m.group('bug')}"
    for p in PATTERNS:
        m = p.search(name)
        if not m:
            continue
        g = m.groupdict()
        if g.get("hash"):
            return f"Bugs.jar_{g['hash']}"
        if g.get("slug"):
            pref = "Bears" if "Bears" in p.pattern else "QuixBugs"
            return f"{pref}_{g['slug']}"
        if g.get("proj") and g.get("bug"):
            ds = g.get("ds") or "Defects4J"
            return f"{ds}_{g['proj'].capitalize()}_{g['bug']}"
        if g.get("bug"):
            return f"Bears_{g['bug']}"
    return None

src, dst = sys.argv[1], sys.argv[2]
unparsed = []
with open(src, newline="") as f, open(dst, "w", newline="") as out:
    r = csv.DictReader(f)
    w = csv.writer(out)
    w.writerow(["example_id", "artifact_ids", "label"])
    for i, row in enumerate(r):
        name = row["Patches"].strip()
        b = bug_id(name)
        if b is None:
            unparsed.append(name)
            b = f"UNPARSED::{name}"
        y = 0 if row["Labels"].strip().lower().startswith("overfit") else 1
        w.writerow([f"q:{i}:{name}", b, y])
print(f"unparsed={len(unparsed)}")
for n in Counter("_".join(x.split("_")[:3]) for x in unparsed).most_common(10):
    print("  pattern:", n)
for n in unparsed[:10]:
    print("  ex:", n)
