#!/bin/bash
cd /data/workzone/entity-leakage
while pgrep -f train_quatrain.py > /dev/null; do sleep 60; done
for s in 42 1 7 13 21; do
  HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 python scripts/train_devign.py \
    --split data/devign/split_published.json --seed $s --out runs/pub_w_s$s > runs/pub_w_s$s.log 2>&1 &
  HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=1 python scripts/train_devign.py \
    --split data/devign/split_cluster.json --seed $s --out runs/clu_w_s$s > runs/clu_w_s$s.log 2>&1 &
  wait
done
echo RERUN-ALL DONE
