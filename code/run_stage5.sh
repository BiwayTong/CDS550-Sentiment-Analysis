#!/usr/bin/env bash
# CDS550 stage 5: all 5 folds of Qwen3-14B-Base + LoRA (same recipe and split as the 8B run),
# then the pre-registered combination with DeBERTa and Qwen3-8B and the exact-match backfill.
# Run from anywhere; it works in the package root (the folder that holds code/, data/, ensemble-*/).
# The copy run on the rented GPU also packed the results into zips (cloud-only step, omitted here).
#   nohup bash code/run_stage5.sh > stage5.log 2>&1 &
# Interrupted? Run the same command again: finished folds are skipped.
# Local model folder?  MODEL=/path/Qwen/Qwen3-14B-Base bash code/run_stage5.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=python
[ -f ./env.sh ] && source ./env.sh
export HF_HUB_OFFLINE=1
MODEL="${MODEL:-Qwen/Qwen3-14B-Base}"
echo "stage 5 started $(date '+%F %T') with $MODEL"

"$PY" code/run_cv.py --trainer llm --preset qwen3-14b --model "$MODEL" --data-dir data --folds 5 \
    --output runs/cv-qwen3-14b --name qwen3-14b --eval-every 0.5

"$PY" code/assemble_cv.py --data-dir data --runs runs/cv-qwen3-14b --output ensemble-qwen3-14b

"$PY" code/ensemble_cv.py --data-dir data --output ensemble-stage5 \
    --oof deberta=ensemble-deberta qwen3=ensemble-qwen3 qwen3_14b=ensemble-qwen3-14b

"$PY" code/apply_exact_match.py --train data/train.csv --test data/test.csv \
    --submission ensemble-stage5/submission.csv --output ensemble-stage5/submission_exact.csv

echo "stage 5 finished $(date '+%F %T')"
