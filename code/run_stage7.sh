#!/usr/bin/env bash
# CDS550 stage 7: all 5 folds of Qwen3.8-27B + LoRA (same recipe, split and prompt as stages 4-6),
# then the pre-registered combination with DeBERTa, Qwen3-8B, Qwen3-14B and Mistral-24B and the
# exact-match backfill.
# This is the script that ran on the rented GPU, adapted to the layout of this package: it runs from the
# package root (the folder that holds code/, data/, runs/ and ensemble-*/), the cloud package's env.sh
# (download cache settings) is used only if present, and the cloud-only step that zipped the results
# for download is left out.
#   MODEL=/path/to/Qwen3.8-27B nohup bash code/run_stage7.sh > stage7.log 2>&1 &
# Interrupted? Run the same command again: finished folds are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=python
[ -f ./env.sh ] && source ./env.sh
export HF_HUB_OFFLINE=1
MODEL="${MODEL:-Qwen/Qwen3.8-27B}"
SEED="${SEED:-42}"   # training seed; all five folds of the submitted run used 42
echo "stage 7 started $(date '+%F %T') with $MODEL, seed $SEED"

"$PY" code/run_cv.py --trainer llm --preset qwen3.8-27b --model "$MODEL" --data-dir data --folds 5 \
    --output runs/cv-qwen38-27b --name qwen38-27b --eval-every 0.5 --seed "$SEED"

"$PY" code/assemble_cv.py --data-dir data --runs runs/cv-qwen38-27b --output ensemble-qwen38-27b

"$PY" code/ensemble_cv.py --data-dir data --output ensemble-stage7 \
    --oof deberta=ensemble-deberta qwen3=ensemble-qwen3 qwen3_14b=ensemble-qwen3-14b \
          mistral_24b=ensemble-mistral-24b qwen38_27b=ensemble-qwen38-27b

"$PY" code/apply_exact_match.py --train data/train.csv --test data/test.csv \
    --submission ensemble-stage7/submission.csv --output ensemble-stage7/submission_exact.csv

echo "rule (fixed before the run): replace D by E = ensemble-stage7/submission_exact.csv only if the"
echo "estimated accuracy of the 'chosen' option above is at least 97.384% (D's OOF 97.304% + 0.08 points)."
echo "stage 7 finished $(date '+%F %T')"
