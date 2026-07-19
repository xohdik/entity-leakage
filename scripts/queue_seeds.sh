#!/bin/bash
cd /data/workzone/entity-leakage
wait  # let the seed-42 pair finish if launched from same shell
for s in 1 7; do
  HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 python scripts/train_devign.py \
    --split data/devign/split_published.json --seed $s --out runs/pub_s$s > runs/pub_s$s.log 2>&1 &
  HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=1 python scripts/train_devign.py \
    --split data/devign/split_cluster.json --seed $s --out runs/clu_s$s > runs/clu_s$s.log 2>&1 &
  wait
done
echo ALL DONE
