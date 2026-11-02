# CDS550 Individual Project — Sentiment Analysis of IMDB Movie Reviews


| File | What it is |
| --- | --- |
| [`Project_Report.pdf`](docs/Project_Report.pdf) ([Word source](docs/Project_Report.docx)) | Project report (anonymised version), 41 pages: introduction, problem, dataset analysis, methodology, implementation, results, discussion, references. |
| `code/` | Python source code; the experiment folders contain saved evidence and submitted predictions. |

The project has seven stages; each one was motivated by an analysis of the previous one (report Section 1).

## Public release contents

This public repository excludes all submission files, per-review test predictions, test margins/scores and test-derived pseudo-labels. These outputs must be generated locally by running the training and prediction pipelines. File paths in the results table and report describe the original experiments; they are not downloadable submission files.

Source code, tests, aggregate metrics, figures and training-set validation/OOF evidence remain available. The report describes the original experiment package, which contained more artifacts than this public release.

## Repository layout

- `code/`: Python training, evaluation and ensemble code.
- `tests/`: CPU unit tests and optional reference comparisons.
- `docs/`: report with identifying information and personal metadata removed.
- `results/`, `validation/`, `runs/` and `ensemble*/`: saved metrics, figures and training-set validation evidence.
- `requirements*.txt`: CPU and GPU dependencies.

Run the commands below from the repository root. Place the course's `train.csv`, `test.csv` and `sample_submission.csv` in `data/`; datasets and trained weights are not included.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

See the stage-specific instructions below to reproduce training and saved predictions. The signed submission cover sheet is excluded from this repository.

## Kaggle submissions (public leaderboard ≈ 10,000 test reviews)

| Stage | Submission (file) | Validation estimate | Public |
| --- | --- | --- | --- |
| 1 | NB-LR, word 1–3-grams, NumPy only (`results/submission.csv`) | 91.78 % (5-fold CV) | 0.91730 |
| 2 | DeBERTa-v3-large, seed 42 (`runs/deberta/submission.csv`) | 96.72 % (2,500 hold-out) | 0.96880 |
| 2 | Mean of DeBERTa and NB-LR (`ensemble/submission.csv`) | 96.82 % (CV in hold-out) | 0.96910 |
| 2 | Mean of 4 transformer runs and NB-LR (`ensemble2/submission.csv`) | 96.90 % (CV in hold-out) | 0.97050 |
| 3 | DeBERTa, mean of 5 fold models (`ensemble-deberta/submission.csv` = `ensemble-final/submission.csv`) | 96.73 % (25,000 OOF) | 0.97150 |
| 3 | ELECTRA-large, mean of 5 fold models (`ensemble-electra/submission.csv`) | 96.41 % (OOF) | 0.96940 |
| 3 | DeBERTa fresh re-run + backfill (`ensemble-retrain-control/submission_exact.csv`) | 96.77 % (OOF) | 0.97010 |
| 4 | Pilot: DeBERTa 5-fold + Qwen3 fold 0, s-mean (`ensemble-pilot/submission.csv`) | 96.72 % (fold 0) | 0.97320 |
| 4 | Qwen3, mean of 5 fold models (`ensemble-qwen3/submission.csv`; uploaded twice, the second time as the identical `ensemble-stage4/submission.csv`) | 97.02 % (OOF) | 0.97360 |
| 4 | A: Qwen3 + backfill (`ensemble-stage4/submission_exact.csv`; Stage 4 rule choice) | 97.02 % (OOF) | 0.97380 |
| 4 | B: stack of DeBERTa and Qwen3 + backfill (`ensemble-stage4-stack/submission_exact.csv`; final selection until Stage 7) | 97.07 % (OOF, CV) | 0.97450 |
| 5 | C: s-mean of DeBERTa, Qwen3-8B and Qwen3-14B + backfill (`ensemble-stage5/submission_exact.csv`; Stage 5 rule choice) | 97.20 % (OOF) | 0.97560 |
| 6 | **Final D**: s-mean of DeBERTa, Qwen3-8B, Qwen3-14B and Mistral-24B + backfill (`ensemble-stage6/submission_exact.csv`) | 97.30 % (OOF) | **0.97620** |
| 7 | **Final E**: s-mean of the four families and Qwen3.8-27B + backfill (`ensemble-stage7/submission_exact.csv`) | 97.34 % (OOF) | **0.97650** |

OOF = out-of-fold: every training review is predicted by the fold model that did not see it. The two final
selections for the private leaderboard (the other ≈ 60 % of the test reviews) are D and E (report Section 6.11).
Each later file replaced the earlier rule choice under a rule fixed before its stage ran: C replaced A because
the Stage 5 choice estimated above A's 97.02 % OOF; D replaced C because the Stage 6 choice estimated at least
97.254 % (C's 97.204 % + 0.05 points). The Stage 7 rule kept D, because E's estimate (97.344 %) was below the
97.384 % fixed before the run (D's 97.304 % + 0.08 points). E then took B's place as the second selection. That
change was the student's own decision after E's public score was known; on the OOF rows E is right on 111
reviews where B is wrong and wrong on 45 where B is right (report Section 6.11).

## Code and retained experiment evidence

```
code/
  run_pipeline.py            Stage 1 driver: --stage eda | cv | final | all
  sentiment/                 Stage 1 package (NumPy only): sparse.py, features.py, models.py, evaluation.py, plots.py
  finetune_transformer.py    Stage 2-3: DeBERTa-v3 / RoBERTa / ELECTRA fine-tuning (GPU)
  ensemble.py                Stage 2: combinations inside the 2,500-review hold-out (NumPy only)
  analyze_stage2.py          Stage 2: DeBERTa vs NB-LR analysis, report Table 3 and Figures 4a, 10, 11
  analyze_ensemble.py        Stage 2: multi-run analysis, report Table 4 and Figure 12
  pseudo_label.py            Stage 3: self-training experiment (not used by any final submission)
  run_cv.py                  Stage 3-7: runs all folds of a trainer, resumable (GPU)
  assemble_cv.py             Stage 3-7: pools the folds -> 25,000-row OOF estimate + averaged submission
  ensemble_cv.py             Stage 3-7: combines model families on the OOF rows (pre-registered rule)
  finetune_llm.py            Stage 4-7: Qwen3-8B/14B-Base, Mistral-Small-24B-Base, Qwen3.8-27B + LoRA classifier (GPU; presets qwen3-8b, qwen3-14b, mistral-24b, qwen3.8-27b)
  run_stage5.sh              Stage 5: the script run on the rented GPU (5 folds -> combination -> backfill)
  run_stage6.sh              Stage 6: the same for Mistral-Small-24B-Base
  run_stage7.sh              Stage 7: the same for Qwen3.8-27B
  compare_runs.py            Stage 4: paired comparison of two runs on one fold (the pilot)
  apply_exact_match.py       Stage 4: exact-match backfill of test reviews that duplicate a training review
  analyze_stage34.py         Stage 3-7: the numbers, tables and figures of report Sections 4.8-4.13, 5.5-5.8, 6.5-6.11, 7.3-7.5
  stage34_plots.py           figures for analyze_stage34.py
  predict_windows.py         optional re-scoring of a saved transformer with overlapping windows (not used by any submission)
tests/                       141 unit tests in 7 files (NumPy only; 7 optional scikit-learn comparisons)
validation/                  Stage 1 optimisation checks and final-refit evidence
results/                     Stage 1 outputs (cv_results.csv, cv_summary.json, eda.json, figures/, ...)
  stage2/                    Stage 2 summaries and figures
  stage34/                   stage34_summary.json and the Stage 3-7 figures (Figures 4b, 13-15)
  stage5/logs/, stage6/logs/, stage7/logs/   smoke-test and full logs of the three rented-GPU runs
runs/deberta, runs/deberta-large-s1, runs/deberta-large-s2, runs/roberta-large   Stage 2 runs
runs/deberta-pl              Stage 3 self-training run
runs/cv-deberta, runs/cv-electra, runs/retrain-cv-control, runs/cv-qwen3          Stage 3-4 fold folders
runs/cv-qwen3-14b           Stage 5 fold folders
runs/cv-mistral-24b         Stage 6 fold folders (fold 3: seed-43 re-run; the diverged seed-42 log is kept in its train_log.txt)
runs/cv-qwen38-27b          Stage 7 fold folders (all folds seed 42)
runs/retrain-screen-*        Stage 3 hyper-parameter screening on the 2,500-review hold-out
  every run folder: run_info.json (command, config, environment, data hashes, history), train_log.txt,
  holdout_predictions.csv (training-set validation only); cv-qwen3*/*/adapter/ and cv-mistral-24b/*/adapter/ and cv-qwen38-27b/*/adapter/ hold the adapter config and metadata
ensemble/, ensemble2/        Stage 2 combinations
ensemble-deberta/, ensemble-electra/, ensemble-retrain-control/, ensemble-qwen3/, ensemble-qwen3-14b/,
ensemble-mistral-24b/, ensemble-qwen38-27b/
                             pooled families: oof_predictions.csv, cv_summary.json (test and submission outputs excluded)
ensemble-final/              Stage 3 combination of DeBERTa and ELECTRA (rule kept DeBERTa alone)
ensemble-pilot/              Stage 4 pilot (fold 0)
ensemble-stage4/             Stage 4 rule choice (= Qwen3) + backfill: file A
ensemble-stage4-stack/       Stage 4 stack + backfill: file B (final selection until Stage 7)
ensemble-stage5/             Stage 5 rule choice (s-mean of three families) + backfill: file C
ensemble-stage6/             Stage 6 rule choice (s-mean of four families) + backfill: final D
ensemble-stage7/             Stage 7 rule choice (s-mean of five families) + backfill: final E
pseudo/                      Stage 3 metadata (test-derived pseudo-label CSV excluded)
data/                        NOT included in the repository: place train.csv, test.csv and sample_submission.csv here
requirements.txt             Stage 1 (numpy, pandas, matplotlib)
requirements-gpu.txt         Stage 2-3 (transformers, sentencepiece, ...; install the CUDA build of PyTorch first)
requirements-llm.txt         Stage 4-6 (adds peft 0.18.1 and accelerate; pins transformers 4.57.6)
requirements-stage7.txt      Stage 7 (transformers >= 5.2, peft >= 0.18, flash-linear-attention / fla-core 0.4.2)
requirements-dev.txt         optional reference tests (scipy, scikit-learn)
```

Model weights are not included: the DeBERTa checkpoints are 1.4–1.7 GB each, and the LoRA adapters are about
175 MB (Qwen3-8B), 250 MB (Qwen3-14B), 360 MB (Mistral-24B) and 470 MB (Qwen3.8-27B) per fold. The base models
(16 GB, 28 GB, 47 GB and 56 GB) are downloaded from Hugging Face. Model weights and per-review test outputs are excluded from this public release. Reproducing test predictions requires running the training/prediction stages locally.

## Assemble Stage 4-7 outputs after generating local predictions

The following commands require locally generated validation and test predictions. This public repository does not contain test predictions. First run the corresponding training/prediction stages; once those inputs exist, the combination steps require only numpy and pandas:

```bash
pip install -r requirements.txt
python code/assemble_cv.py --data-dir data --runs runs/cv-deberta --output ensemble-deberta
python code/assemble_cv.py --data-dir data --runs runs/cv-qwen3 --output ensemble-qwen3
python code/assemble_cv.py --data-dir data --runs runs/cv-qwen3-14b --output ensemble-qwen3-14b
python code/assemble_cv.py --data-dir data --runs runs/cv-mistral-24b --output ensemble-mistral-24b
python code/assemble_cv.py --data-dir data --runs runs/cv-qwen38-27b --output ensemble-qwen38-27b
python code/ensemble_cv.py --data-dir data --oof deberta=ensemble-deberta qwen3=ensemble-qwen3 --output ensemble-stage4
python code/ensemble_cv.py --data-dir data --oof deberta=ensemble-deberta qwen3=ensemble-qwen3 \
    --output ensemble-stage4-stack --choice stack
python code/apply_exact_match.py --train data/train.csv --test data/test.csv \
    --submission ensemble-stage4/submission.csv --output ensemble-stage4/submission_exact.csv              # file A
python code/apply_exact_match.py --train data/train.csv --test data/test.csv \
    --submission ensemble-stage4-stack/submission.csv --output ensemble-stage4-stack/submission_exact.csv  # file B
python code/ensemble_cv.py --data-dir data --output ensemble-stage5 \
    --oof deberta=ensemble-deberta qwen3=ensemble-qwen3 qwen3_14b=ensemble-qwen3-14b
python code/apply_exact_match.py --train data/train.csv --test data/test.csv \
    --submission ensemble-stage5/submission.csv --output ensemble-stage5/submission_exact.csv              # file C
python code/ensemble_cv.py --data-dir data --output ensemble-stage6 \
    --oof deberta=ensemble-deberta qwen3=ensemble-qwen3 qwen3_14b=ensemble-qwen3-14b mistral_24b=ensemble-mistral-24b
python code/apply_exact_match.py --train data/train.csv --test data/test.csv \
    --submission ensemble-stage6/submission.csv --output ensemble-stage6/submission_exact.csv              # final D
python code/ensemble_cv.py --data-dir data --output ensemble-stage7 \
    --oof deberta=ensemble-deberta qwen3=ensemble-qwen3 qwen3_14b=ensemble-qwen3-14b mistral_24b=ensemble-mistral-24b \
          qwen38_27b=ensemble-qwen38-27b
python code/apply_exact_match.py --train data/train.csv --test data/test.csv \
    --submission ensemble-stage7/submission.csv --output ensemble-stage7/submission_exact.csv              # final E
python code/analyze_stage34.py --data-dir data --output results/stage34                                     # report numbers
```

On Windows use `py -3.12 code\...`. These commands were re-run on Linux (Python 3.10): they reproduce all 25,000
labels of every Stage 3 and Stage 4 file listed above, and the files A, B, C, D and E byte for byte
(SHA-256 `17f55fc207e8…` for A, `24bc9a952909…` for B, `ccc45b404d63…` for C, `4efa15e47644…` for D,
`9ac9d596470d…` for E). The CSV files written on Windows have CRLF line endings,
so an intermediate file regenerated on macOS/Linux can have a different SHA-256 with identical rows.

## Reproduce Stage 4 (GPU, about 5.8 hours on one 24 GB GPU)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128      # CUDA build of PyTorch first
pip install -r requirements-llm.txt
# 2-minute end-to-end checks (small model, then the real one)
python code/finetune_llm.py --data-dir data --folds 5 --fold 0 --preset qwen3-0.6b --output runs/_smoke_llm --smoke
python code/finetune_llm.py --data-dir data --folds 5 --fold 0 --preset qwen3-8b --output runs/_smoke_llm8b --smoke
# all five folds (resumable; finished folds are skipped)
python code/run_cv.py --trainer llm --preset qwen3-8b --data-dir data --folds 5 --output runs/cv-qwen3 --name qwen3 --eval-every 0.5
# re-create one fold's predictions from its saved adapter
python code/finetune_llm.py --data-dir data --folds 5 --fold 0 --output runs/cv-qwen3/qwen3-f0 --predict-only
```

`finetune_llm.py` accepts only base (pre-trained) checkpoints and refuses instruction/chat, sentiment, embedding
and re-ranking models; the one exception is the Stage 7 model (see below). It checks the zero-shot classifier at update 0, probes GPU memory before training, saves
the adapter, frees the model, re-loads it from disk and predicts with the re-loaded copy (reload check in
`run_info.json → fingerprint`). Hyper-parameters are fixed in the `qwen3-8b` preset: LoRA r = 16, α = 32 on all
seven projections, learning rate 1e-4 (head 2e-4), 1 epoch, 16 reviews per update, 1,024 tokens (head 256 + tail).

## Reproduce Stage 5 (GPU with at least 96 GB, about 4.6 hours)

Stage 5 ran on a rented NVIDIA RTX PRO 6000 (96 GB, AutoDL). Without gradient checkpointing the 14B model peaks
at 85.2 GB; each fold took about 55 minutes.

```bash
pip install -r requirements-llm.txt
python code/finetune_llm.py --data-dir data --folds 5 --fold 0 --preset qwen3-14b --output runs/_smoke_llm14b --smoke
bash code/run_stage5.sh        # 5 folds, combination, backfill (resumable)
```

The `qwen3-14b` preset uses the same hyper-parameters, split and prompt format as `qwen3-8b`.

## Reproduce Stage 6 (GPU with at least 80 GB, about 7.4 hours)

Stage 6 ran on a rented NVIDIA H800 PCIe (80 GB, AutoDL) with gradient checkpointing on; training peaked at
54.8 GB and each fold took about 89 minutes. The seed-42 run of fold 3 diverged; the divergence guard in
`finetune_llm.py` refused to save it and the fold was re-trained with `--seed 43` (report Section 6.9).

```bash
pip install -r requirements-llm.txt
python code/finetune_llm.py --data-dir data --folds 5 --fold 0 --preset mistral-24b --output runs/_smoke_mistral24b --smoke
# to repeat the submitted run exactly, train fold 3 with seed 43 first; the script then skips it
python code/run_cv.py --trainer llm --preset mistral-24b --data-dir data --folds 5 --output runs/cv-mistral-24b \
    --name mistral-24b --eval-every 0.5 --seed 43 --only 3
bash code/run_stage6.sh        # 5 folds, combination, backfill (resumable)
```

The `mistral-24b` preset uses the same hyper-parameters, split and prompt format as the Qwen3 presets.

## Reproduce Stage 7 (Linux GPU with at least 80 GB, about 11.5 hours)

Stage 7 ran on a rented NVIDIA RTX PRO 6000 (96 GB, AutoDL) with PyTorch 2.8.0 (CUDA 12.8), transformers 5.22.0,
peft 0.21.2 and flash-linear-attention 0.4.2, with gradient checkpointing on. Training peaked at about 56 GB;
each fold took about 137 minutes (95 min training, 40 min prediction). Qwen3.8-27B is only released post-trained,
so the `qwen3.8-27b` preset is the one exception to the base-checkpoint rule (report Section 4.13). Newer
flash-linear-attention releases need a newer triton than torch 2.8 ships; install the pinned version with `--no-deps`.

```bash
pip install -r requirements-stage7.txt
pip install --no-deps fla-core==0.4.2 flash-linear-attention==0.4.2
python code/finetune_llm.py --data-dir data --folds 5 --fold 0 --preset qwen3.8-27b --model /path/to/Qwen3.8-27B \
    --output runs/_smoke_qwen38 --smoke
MODEL=/path/to/Qwen3.8-27B bash code/run_stage7.sh     # 5 folds, combination, backfill (resumable)
```

## Reproduce Stage 3 (GPU)

```bash
pip install -r requirements-gpu.txt
python code/run_cv.py --data-dir data --folds 5 --output runs/cv-deberta --name deberta          # ~2.4 h
python code/run_cv.py --data-dir data --folds 5 --output runs/cv-electra --name electra \
    --model google/electra-large-discriminator                                                  # ~0.8 h
python code/assemble_cv.py --data-dir data --runs runs/cv-deberta --output ensemble-deberta
python code/assemble_cv.py --data-dir data --runs runs/cv-electra --output ensemble-electra
python code/ensemble_cv.py --data-dir data --oof deberta=ensemble-deberta electra=ensemble-electra --output ensemble-final
```

The fresh re-run (`runs/retrain-*`, `ensemble-retrain-control`) used the same commands with `--fresh-start
--verify-fresh --eval-every 0.25` and an output folder of its own; the exact command of every fold is in its
`run_info.json`. All model families share one stratified 5-fold partition (split seed 42), whose fold 0
contains the 2,500-review Stage 2 hold-out, so their OOF predictions can be compared review by review.

## Reproduce Stage 2 (GPU)

```bash
python code/finetune_transformer.py --data-dir data --output runs/deberta --smoke   # 2-5 min check
python code/finetune_transformer.py --data-dir data --output runs/deberta
python code/finetune_transformer.py --data-dir data --output runs/deberta-large-s1 --seed 1
python code/finetune_transformer.py --data-dir data --output runs/deberta-large-s2 --seed 2
python code/finetune_transformer.py --data-dir data --output runs/roberta-large --preset roberta-large
python code/ensemble.py --data-dir data --runs runs/deberta --nblr-results results --output ensemble
python code/ensemble.py --data-dir data --runs runs/deberta runs/deberta-large-s1 runs/deberta-large-s2 runs/roberta-large \
    --nblr-results results --output ensemble2
python code/finetune_transformer.py --data-dir data --output runs/deberta --predict-only  # reproduce from the checkpoint
```

`--predict-only` re-loads `runs/deberta/best_model` and regenerated a `submission.csv` byte-identical to the
file uploaded to Kaggle (SHA-256 `cc4f338f…`). Re-training from scratch is seeded, but GPU kernels are not
bit-deterministic, so a re-trained model matches within a few tenths of a point rather than exactly (the
Stage 3 re-run is a measured example: report Section 6.5).

## Reproduce Stage 1 (CPU, about 15 minutes)

```bash
pip install -r requirements.txt
python code/run_pipeline.py --data-dir data --output results_rerun                 # EDA + 5-fold CV of 42 models + final fit
```

To repeat only the final refit (about 1 minute), give it the saved CV result, because `--stage final` reads the
chosen configuration from `<output>/cv_summary.json`:

```bash
mkdir -p results_rerun && cp results/cv_summary.json results_rerun/
python code/run_pipeline.py --data-dir data --output results_rerun --stage final   # refit + submission.csv only
```

The pipeline is deterministic (seed 42, convex models); re-running produces an identical `submission.csv`
(SHA-256 recorded in `results/final_model.json`, also after the code optimisation documented in
`validation/OPTIMIZATION_REPORT.md`).

## Tests

```bash
python -m unittest discover -s tests -v            # NumPy only; no GPU needed
pip install -r requirements-dev.txt                # optional: 7 reference comparisons against scikit-learn/SciPy
```

The tests cover the Stage 1 numerics, the Stage 2 data handling and ensemble maths, the K-fold partition, OOF
assembly and McNemar direction, the combiners (a family averaged with itself must not change), batching and
truncation, model-name checks, the `run_cv.py` command lines and the backfill. The GPU code paths are not
unit-tested; they were checked with the `--smoke` runs.

## Licences

All libraries and models carry OSI-approved licences that allow commercial use (Kaggle rule 6.c): numpy, pandas,
PyTorch (BSD-3-Clause); transformers, peft, accelerate, sentencepiece (Apache-2.0); DeBERTa-v3-large and
RoBERTa-large (MIT); ELECTRA-large, Qwen3-8B-Base, Qwen3-14B-Base, Mistral-Small-24B-Base-2501 and Qwen3.8-27B (Apache-2.0); flash-linear-attention (MIT). Dataset: Maas et al. (2011), Large Movie
Review Dataset.

## Authorship

AI coding assistants were used as pair programmers and editors: Claude (Anthropic) helped draft the code of all
seven stages, the instructions for running it on the student's GPU and the rented cloud GPUs, the figures and the report; OpenAI Codex
helped optimise the Stage 1 code and add unit tests. The student decided which experiments to run and which
files to submit, ran every training job (on their own computer and, for Stages 5–7, GPUs they rented), uploaded all submissions to Kaggle, checked the
reported numbers against the saved result files, and reviewed the final results.
