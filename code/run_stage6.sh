#!/usr/bin/env bash
# CDS550 stage 6: all 5 folds of Mistral-Small-24B-Base-2501 + LoRA (same recipe and split as the
# Qwen3 runs), then the pre-registered combination with DeBERTa, Qwen3-8B and Qwen3-14B and the
# exact-match backfill.
# This is the script that ran on the rented GPU, adapted to the layout of this package: it runs from the
# package root (the folder that holds code/, data/, runs/ and ensemble-*/), the cloud package's env.sh
# (download cache settings) is used only if present, and the cloud-only step that zipped the results
# for download is left out.
#   nohup bash code/run_stage6.sh > stage6.log 2>&1 &
# Interrupted? Run the same command again: finished folds are skipped.
# Local model folder?  MODEL=/path/to/Mistral-Small-24B-Base-2501 bash code/run_stage6.sh
# Fold 3 diverged with training seed 42 (report Section 6.9) and was re-trained on its own with
#   python code/run_cv.py --trainer llm --preset mistral-24b --model "$MODEL" --data-dir data --folds 5 \
#       --output runs/cv-mistral-24b --name mistral-24b --eval-every 0.5 --seed 43 --only 3
# before this script was started again for fold 4 and the combination.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=python
[ -f ./env.sh ] && source ./env.sh
export HF_HUB_OFFLINE=1
MODEL="${MODEL:-mistralai/Mistral-Small-24B-Base-2501}"
echo "stage 6 started $(date '+%F %T') with $MODEL"

"$PY" code/run_cv.py --trainer llm --preset mistral-24b --model "$MODEL" --data-dir data --folds 5 \
    --output runs/cv-mistral-24b --name mistral-24b --eval-every 0.5

"$PY" code/assemble_cv.py --data-dir data --runs runs/cv-mistral-24b --output ensemble-mistral-24b

"$PY" code/ensemble_cv.py --data-dir data --output ensemble-stage6 \
    --oof deberta=ensemble-deberta qwen3=ensemble-qwen3 qwen3_14b=ensemble-qwen3-14b mistral_24b=ensemble-mistral-24b

"$PY" code/apply_exact_match.py --train data/train.csv --test data/test.csv \
    --submission ensemble-stage6/submission.csv --output ensemble-stage6/submission_exact.csv

echo "rule (fixed before the run): replace C by D = ensemble-stage6/submission_exact.csv only if the"
echo "estimated accuracy of the 'chosen' option above is at least 97.254% (C's OOF 97.204% + 0.05 points)."
echo "stage 6 finished $(date '+%F %T')"
