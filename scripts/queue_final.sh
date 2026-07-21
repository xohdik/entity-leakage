#!/bin/bash
cd /data/workzone/entity-leakage
UX="microsoft/unixcoder-base"
export HF_ENDPOINT=https://hf-mirror.com
# (a) patch-assessment config-consistency: s13/s21 under warmup
for s in 13 21; do
  CUDA_VISIBLE_DEVICES=0 python scripts/train_quatrain.py --split data/patches/qt_split_random.json  --seed $s --out runs/qt_rand_w_s$s > runs/qt_rand_w_s$s.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python scripts/train_quatrain.py --split data/patches/qt_split_cluster.json --seed $s --out runs/qt_clu_w_s$s  > runs/qt_clu_w_s$s.log 2>&1 &
  wait
done
# (b) UniXcoder: patch, both protocols, 3 seeds
for s in 42 13 21; do
  CUDA_VISIBLE_DEVICES=0 python scripts/train_quatrain.py --model $UX --split data/patches/qt_split_random.json  --seed $s --out runs/ux_qt_rand_s$s > runs/ux_qt_rand_s$s.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python scripts/train_quatrain.py --model $UX --split data/patches/qt_split_cluster.json --seed $s --out runs/ux_qt_clu_s$s  > runs/ux_qt_clu_s$s.log 2>&1 &
  wait
done
# (c) UniXcoder: devign, both protocols, 3 seeds
for s in 42 13 21; do
  CUDA_VISIBLE_DEVICES=0 python scripts/train_devign.py --model $UX --split data/devign/split_published.json --seed $s --out runs/ux_pub_s$s > runs/ux_pub_s$s.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 python scripts/train_devign.py --model $UX --split data/devign/split_cluster.json  --seed $s --out runs/ux_clu_s$s  > runs/ux_clu_s$s.log 2>&1 &
  wait
done
echo FINAL QUEUE DONE
