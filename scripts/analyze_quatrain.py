import json, glob, statistics as st
res = {p: [json.load(open(f)) for f in sorted(glob.glob(f"runs/qt_{p}_s*/result.json"))]
       for p in ("rand", "clu")}
for p, rs in res.items():
    if not rs: continue
    f1s = [r["test_f1"] for r in rs]; accs = [r["test_acc"] for r in rs]
    print(f"{p}: F1={[f'{x:.4f}' for x in f1s]} mean={st.mean(f1s):.4f} | "
          f"acc mean={st.mean(accs):.4f}")
if res["rand"] and res["clu"]:
    d_f1 = st.mean([r["test_f1"] for r in res["rand"]]) - st.mean([r["test_f1"] for r in res["clu"]])
    d_ac = st.mean([r["test_acc"] for r in res["rand"]]) - st.mean([r["test_acc"] for r in res["clu"]])
    print(f"\nDelta F1 = {d_f1:+.4f}   Delta acc = {d_ac:+.4f}")
    r = res["rand"]
    if "acc_contaminated" in r[0]:
        print(f"rand-split strata: contam acc={st.mean([x['acc_contaminated'] for x in r]):.4f} "
              f"(n={r[0]['n_contam']})  clean acc={st.mean([x['acc_clean'] for x in r]):.4f} "
              f"(n={r[0]['n_clean']})")
