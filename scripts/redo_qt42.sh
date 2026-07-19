#!/bin/bash
cd /data/workzone/entity-leakage
while pgrep -f "train_quatrain.py|train_devign.py|rerun_bad" > /dev/null; do sleep 120; done
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 python scripts/train_quatrain.py \
  --split data/patches/qt_split_random.json --seed 42 --out runs/qt_rand_w_s42 > runs/qt_rand_w_s42.log 2>&1 &
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=1 python scripts/train_quatrain.py \
  --split data/patches/qt_split_cluster.json --seed 42 --out runs/qt_clu_w_s42 > runs/qt_clu_w_s42.log 2>&1 &
wait
echo QT42 REDO DONE
