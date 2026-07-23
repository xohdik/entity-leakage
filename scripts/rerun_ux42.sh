#!/bin/bash
cd /data/workzone/entity-leakage
while pgrep -f "train_quatrain.py|train_devign.py|queue_final" > /dev/null; do sleep 120; done
HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=0 python scripts/train_quatrain.py \
  --model microsoft/unixcoder-base --split data/patches/qt_split_random.json \
  --seed 42 --out runs/ux_qt_rand_s42 > runs/ux_qt_rand_s42.log 2>&1
echo UX42 DONE
