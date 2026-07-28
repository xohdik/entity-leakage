#!/bin/bash
# Two-GPU queue for the rank-stability matrix. Run from scripts/.
#   bash queue_rank.sh gen    -> writes rank_jobs.txt (edit paths there if needed)
#   bash queue_rank.sh run    -> two workers, one per P40, resume-safe
set -u
JOBS=rank_jobs.txt
OUT=../runs

MODELS=(
  "microsoft/codebert-base cb"
  "microsoft/graphcodebert-base gcb"
  "microsoft/unixcoder-base ux"
  "Salesforce/codet5-base ct5"
  "Salesforce/codet5p-220m ct5p"
  "uclanlp/plbart-base plb"
  "roberta-base rob"
  "codesage/codesage-small csg"
)
SEEDS=(13 21 42)

# task proto train_file test_file   <-- EDIT these four paths to your actual files
CELLS=(
  "devign pub  ../data/devign/train_pub.jsonl  ../data/devign/test_pub.jsonl"
  "devign clu  ../data/devign/train_clu.jsonl  ../data/devign/test_clu.jsonl"
  "qt     rand ../data/patches/train_rand.jsonl ../data/patches/test_rand.jsonl"
  "qt     clu  ../data/patches/train_clu.jsonl  ../data/patches/test_clu.jsonl"
)

if [ "${1:-}" = "gen" ]; then
  : > $JOBS
  for m in "${MODELS[@]}"; do set -- $m; MODEL=$1; TAG=$2
    for c in "${CELLS[@]}"; do set -- $c; TASK=$1; PROTO=$2; TR=$3; TE=$4
      for S in "${SEEDS[@]}"; do
        echo "python train_rank.py --model $MODEL --tag $TAG --task $TASK --proto $PROTO --seed $S --train_file $TR --test_file $TE --out $OUT" >> $JOBS
      done
    done
  done
  wc -l $JOBS; exit 0
fi

if [ "${1:-}" = "run" ]; then
  worker() {  # $1 = gpu id
    local n=0
    while IFS= read -r cmd; do
      n=$((n+1))
      # round-robin: gpu0 takes odd lines, gpu1 even
      [ $((n % 2)) -ne "$1" ] && continue
      # resume-safe: skip if preds.json already exists
      name=$(echo "$cmd" | sed -E 's/.*--tag ([^ ]+).*--task ([^ ]+).*--proto ([^ ]+).*--seed ([^ ]+).*/rank_\1_\2_\3_s\4/')
      if [ -f "$OUT/$name/preds.json" ]; then echo "[skip] $name"; continue; fi
      echo "[gpu$1] $name"
      CUDA_VISIBLE_DEVICES=$1 $cmd > "$OUT/${name}.log" 2>&1
    done < $JOBS
  }
  worker 0 & worker 1 &
  wait
  echo "queue complete"
fi