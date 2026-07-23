"""CodeBERT (or --model) on BigCloneBench clone pairs, given a split file.
Mirrors train_quatrain.py exactly; fields are func1/func2 instead of report/patch.

Usage:
  HF_ENDPOINT=https://hf-mirror.com python train_bcb.py \
    --split data/bcb/bcb_split_safe.json --seed 42 --out runs/bcb_safe_s42
"""
import argparse, json, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

class DS(Dataset):
    def __init__(self, rows, tok):
        self.rows, self.tok = rows, tok
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["func1"], r["func2"], truncation="longest_first",
                       max_length=400, padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"][0],
                "attention_mask": enc["attention_mask"][0],
                "label": torch.tensor(r["label"]), "ex_id": r["ex_id"]}

def collate(b):
    return {"input_ids": torch.stack([x["input_ids"] for x in b]),
            "attention_mask": torch.stack([x["attention_mask"] for x in b]),
            "label": torch.stack([x["label"] for x in b]),
            "ex_id": [x["ex_id"] for x in b]}

def evaluate(model, loader, dev):
    model.eval(); preds = {}
    with torch.no_grad():
        for b in loader:
            out = model(input_ids=b["input_ids"].to(dev),
                        attention_mask=b["attention_mask"].to(dev))
            for ex, pi, yi in zip(b["ex_id"],
                                  out.logits.argmax(-1).cpu().tolist(),
                                  b["label"].tolist()):
                preds[ex] = (pi, yi)
    return preds

def f1_acc(pairs):
    tp = sum(1 for p, y in pairs if p == 1 and y == 1)
    fp = sum(1 for p, y in pairs if p == 1 and y == 0)
    fn = sum(1 for p, y in pairs if p == 0 and y == 1)
    acc = sum(1 for p, y in pairs if p == y) / max(len(pairs), 1)
    return acc, 2*tp/max(2*tp+fp+fn, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--model", default="microsoft/codebert-base")
    a = ap.parse_args()
    set_seed(a.seed); dev = "cuda"

    rows = json.load(open("data/bcb/bcb_pairs.json"))
    split = json.load(open(a.split))
    parts = {p: [r for r in rows if split.get(r["ex_id"]) == p]
             for p in ("train", "valid", "test")}
    print({k: len(v) for k, v in parts.items()}, flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        a.model, num_labels=2).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    tl = DataLoader(DS(parts["train"], tok), batch_size=a.bs, shuffle=True,
                    collate_fn=collate, num_workers=2)
    vl = DataLoader(DS(parts["valid"], tok), batch_size=48, collate_fn=collate)
    xl = DataLoader(DS(parts["test"], tok), batch_size=48, collate_fn=collate)

    total = a.epochs * len(tl)
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.SequentialLR(opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 0.01, 1.0, total_iters=warm),
         torch.optim.lr_scheduler.LinearLR(opt, 1.0, 0.0, total_iters=total - warm)],
        milestones=[warm])

    best, best_state = -1, None
    for ep in range(a.epochs):
        model.train()
        for i, b in enumerate(tl):
            out = model(input_ids=b["input_ids"].to(dev),
                        attention_mask=b["attention_mask"].to(dev),
                        labels=b["label"].to(dev))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            if i % 200 == 0:
                print(f"ep{ep} {i}/{len(tl)} loss={out.loss.item():.4f}", flush=True)
        vacc, vf1 = f1_acc(list(evaluate(model, vl, dev).values()))
        print(f"ep{ep} valid acc={vacc:.4f} f1={vf1:.4f}", flush=True)
        if vacc > best:
            best = vacc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    preds = evaluate(model, xl, dev)
    acc, f1 = f1_acc(list(preds.values()))
    res = {"split": a.split, "seed": a.seed, "test_acc": acc, "test_f1": f1,
          "valid_best_acc": best}

    try:
        contam = set(json.load(open("data/bcb/bcb_contaminated_test_unsafe.json")))
        c = [v for k, v in preds.items() if k in contam]
        g = [v for k, v in preds.items() if k not in contam]
        if c: res["acc_contaminated"] = f1_acc(c)[0]; res["n_contam"] = len(c)
        if g: res["acc_clean"] = f1_acc(g)[0]; res["n_clean"] = len(g)
    except FileNotFoundError:
        pass

    import os
    os.makedirs(a.out, exist_ok=True)
    json.dump(res, open(f"{a.out}/result.json", "w"), indent=2)
    json.dump({k: v for k, v in preds.items()}, open(f"{a.out}/preds.json", "w"))
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()