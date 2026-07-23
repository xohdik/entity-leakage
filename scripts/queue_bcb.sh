#!/bin/bash
cd /data/workzone/entity-leakage
export HF_ENDPOINT=https://hf-mirror.com
for s in 42 13 21; do
  CUDA_VISIBLE_DEVICES=0 python scripts/train_bcb.py \
    --split data/bcb/bcb_split_safe.json --seed $s --out runs/bcb_safe_s$s > runs/bcb_safe_s$s.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python scripts/train_bcb.py \
    --split data/bcb/bcb_split_unsafe.json --seed $s --out runs/bcb_unsafe_s$s > runs/bcb_unsafe_s$s.log 2>&1 &
  wait
done
echo BCB QUEUE DONE