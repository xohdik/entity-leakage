#!/bin/bash
cd /data/workzone/entity-leakage
while pgrep -f train_devign.py > /dev/null; do sleep 60; done
for s in 42 13 21; do
  HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 python scripts/train_quatrain.py \
    --split data/patches/qt_split_random.json --seed $s --out runs/qt_rand_s$s > runs/qt_rand_s$s.log 2>&1 &
  HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=1 python scripts/train_quatrain.py \
    --split data/patches/qt_split_cluster.json --seed $s --out runs/qt_clu_s$s > runs/qt_clu_s$s.log 2>&1 &
  wait
done
echo QUATRAIN DONE
