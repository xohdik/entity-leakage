#!/usr/bin/env python3
"""Uniform-head fine-tuning driver for the rank-stability study (paper 2).

Only the backbone varies: every model gets the same mean-pool + linear head,
same hyperparameters, same splits, same seeds. P40-safe: fp32, cudnn off.

Run name convention (analysis depends on it):
  rank_{model_tag}_{task}_{proto}_s{seed}     e.g. rank_gcb_devign_pub_s13

Usage:
  python train_rank.py --model microsoft/graphcodebert-base --tag gcb \
      --task devign --proto pub --seed 13 \
      --train_file ../data/devign/train_pub.jsonl --test_file ../data/devign/test_pub.jsonl \
      --out ../runs
If your data field names differ from the auto-detected ones, pass
--text-key/--text2-key/--label-key/--id-key explicitly.
"""
import argparse, json, os, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, AutoConfig

torch.backends.cudnn.enabled = False  # Pascal / P40

TEXT_KEYS  = ['func', 'code', 'function', 'source', 'text', 'patch', 'patch_code']
TEXT2_KEYS = ['bug_report', 'report', 'nl', 'context']
LABEL_KEYS = ['target', 'label', 'y', 'correct']
ID_KEYS    = ['idx', 'id', 'example_id', 'uid', 'name']

def pick(d, keys, override):
    if override: return override
    for k in keys:
        if k in d: return k
    return None

def load_jsonl(path):
    rows = []
    with open(path) as f:
        first = f.read(1); f.seek(0)
        if first == '[':
            rows = json.load(f)
        else:
            for line in f:
                line = line.strip()
                if line: rows.append(json.loads(line))
    return rows

class DS(Dataset):
    def __init__(self, rows, tok, kt, kt2, kl, ki, max_len):
        self.rows, self.tok = rows, tok
        self.kt, self.kt2, self.kl, self.ki, self.max_len = kt, kt2, kl, ki, max_len
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        a = str(r[self.kt])
        b = str(r[self.kt2]) if self.kt2 and self.kt2 in r else None
        enc = self.tok(b, a, truncation=True, max_length=self.max_len,
                       padding='max_length', return_tensors='pt') if b else \
              self.tok(a, truncation=True, max_length=self.max_len,
                       padding='max_length', return_tensors='pt')
        ex_id = str(r[self.ki]) if self.ki and self.ki in r else str(i)
        return {k: v.squeeze(0) for k, v in enc.items()}, int(r[self.kl]), ex_id

class UniformHead(nn.Module):
    """Backbone encoder + mean-pool + linear. Identical for every model."""
    def __init__(self, name, n_labels=2):
        super().__init__()
        cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
        # encoder-decoder checkpoints (codet5, plbart): use encoder only
        self.backbone = AutoModel.from_pretrained(name, trust_remote_code=True)
        if hasattr(self.backbone, 'encoder') and hasattr(self.backbone, 'decoder'):
            self.backbone = self.backbone.encoder
        h = getattr(cfg, 'hidden_size', None) or getattr(cfg, 'd_model')
        self.head = nn.Linear(h, n_labels)
    def forward(self, **enc):
        enc = {k: v for k, v in enc.items() if k in ('input_ids', 'attention_mask')}
        out = self.backbone(**enc).last_hidden_state
        mask = enc['attention_mask'].unsqueeze(-1).float()
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        return self.head(pooled)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--tag', required=True)
    ap.add_argument('--task', required=True);  ap.add_argument('--proto', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--train_file', required=True); ap.add_argument('--test_file', required=True)
    ap.add_argument('--out', default='../runs')
    ap.add_argument('--epochs', type=int, default=5); ap.add_argument('--bs', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-5); ap.add_argument('--max_len', type=int, default=512)
    ap.add_argument('--text-key'); ap.add_argument('--text2-key')
    ap.add_argument('--label-key'); ap.add_argument('--id-key')
    a = ap.parse_args()

    set_seed(a.seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_rows, test_rows = load_jsonl(a.train_file), load_jsonl(a.test_file)
    r0 = train_rows[0]
    kt  = pick(r0, TEXT_KEYS, a.text_key);  kl = pick(r0, LABEL_KEYS, a.label_key)
    kt2 = pick(r0, TEXT2_KEYS, a.text2_key); ki = pick(r0, ID_KEYS, a.id_key)
    assert kt and kl, f'could not detect text/label keys in {list(r0)} - pass --text-key/--label-key'
    print(f'[keys] text={kt} text2={kt2} label={kl} id={ki}')

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = UniformHead(a.model).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    lossf = nn.CrossEntropyLoss()

    tr = DataLoader(DS(train_rows, tok, kt, kt2, kl, ki, a.max_len), batch_size=a.bs,
                    shuffle=True, num_workers=2, drop_last=False)
    te = DataLoader(DS(test_rows, tok, kt, kt2, kl, ki, a.max_len), batch_size=a.bs,
                    shuffle=False, num_workers=2)

    for ep in range(a.epochs):
        model.train(); tot = 0.0
        for enc, y, _ in tr:
            enc = {k: v.to(dev) for k, v in enc.items()}; y = y.to(dev)
            opt.zero_grad(); loss = lossf(model(**enc), y)
            loss.backward(); opt.step(); tot += loss.item()
        print(f'[ep {ep+1}/{a.epochs}] loss {tot/len(tr):.4f}', flush=True)

    model.eval(); preds = {}
    with torch.no_grad():
        for enc, y, ids in te:
            enc = {k: v.to(dev) for k, v in enc.items()}
            p = model(**enc).argmax(-1).cpu().tolist()
            for i, pi, yi in zip(ids, p, y.tolist()):
                preds[i] = [int(pi), int(yi)]   # [pred, true] - matches paper-1 format

    name = f'rank_{a.tag}_{a.task}_{a.proto}_s{a.seed}'
    d = os.path.join(a.out, name); os.makedirs(d, exist_ok=True)
    json.dump(preds, open(os.path.join(d, 'preds.json'), 'w'))
    yt = np.array([v[1] for v in preds.values()]); yp = np.array([v[0] for v in preds.values()])
    acc = float((yt == yp).mean())
    tp = int(((yp == 1) & (yt == 1)).sum()); fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    json.dump({'run': name, 'model': a.model, 'seed': a.seed, 'n_test': len(preds),
               'accuracy': acc, 'f1_pos': f1}, open(os.path.join(d, 'result.json'), 'w'), indent=2)
    print(f'[done] {name}  acc={acc:.4f}  f1={f1:.4f}')

if __name__ == '__main__':
    main()