# Source optimization validation

Checked on 12 October 2026. The regenerated 25,000-row prediction file is byte-identical
to the original Kaggle submission. The model, selected hyperparameters, cleaning, vocabulary
ordering, TF-IDF formula and optimization stopping rules are preserved.

## Changes

- `NgramCountVectorizer.fit_transform`: one extraction pass per document, storing compact
  candidate column IDs and float32 counts until DF pruning. Temporary state is released.
- EDA: one 1–3-gram vocabulary fit gives all three cumulative sizes.
- Sparse matrix: unused `copy`, `rows` and `vstack` were removed.
- LR: one local float64 conversion is shared by all objective/gradient products during a fit.
- Confusion matrix: vectorized counting plus same-length 1D binary-label validation.
- Ten figure functions moved to `sentiment/plots.py`; `--no-plots` writes numeric outputs
  without importing matplotlib. The default still generates the same figures.
- Numerical tests and optional sklearn reference comparisons added.

## Checks completed

- **17 core tests passed; 7 optional sklearn checks skipped.** See `unit_tests.txt`.
- Full 25,000-review feature extraction: feature names, document frequencies and all CSR
  arrays have exact original/optimized hash equality (921,043 columns, 10,392,267 entries).
  Ninety additional parameter/edge-case comparisons passed.
- Sparse/model changes: 189 exact original/optimized comparisons covered products,
  coefficients, loss traces, scores and predictions on small fixtures.
- Full EDA matches the saved original within 1e-12; only review-length standard deviation
  differs by about 3e-14 with the current library versions. Vocabulary sizes are exact.
- Quick 5-fold CV on 500 real reviews exercises all four model families and cache recovery;
  original/optimized numerical outputs and resumed outputs agree. See `pipeline_review.json`.
- All ten extracted plotting blocks retain their original ASTs after parameter substitutions.
- Final selected NB-LR refit: 921,043 features, 33 L-BFGS iterations,
  12,785 positive and 12,215 negative predictions.
  The complete numeric final stage finished in 27.08 seconds.

SHA-256 of the original and regenerated `submission.csv`:

```
6cd29733998ab0d22f5b27ea1b76257200cb3e56c3fd33db07f0857e82c42732
```

## Follow-up check (12 October 2026, matplotlib available)

- Full 42-candidate × 5-fold CV re-run with the optimized code: every per-fold accuracy, CV mean/std
  and the best model's out-of-fold scores are exactly equal (max abs. difference 0.0) to `results/`.
  Selected model unchanged: `nblr_3g_C1_b0.75`.
- EDA, CV and final stages re-run **with plots**: all ten PNG figures are byte-identical to
  `results/figures/`, and `eda.json` is identical.
- Final stage regenerated `submission.csv` with the same SHA-256 as above.
- Evidence: `full_cv_rerun.json`. The seven optional sklearn/SciPy tests are still skipped here.

## Local performance measurement

On the full training set, a single same-runtime comparison measured vectorization at
13.581 → 11.894 seconds
(12.4% less time).
Peak process RSS was 740 → 827 MB.
The one-pass method holds unpruned sparse counts temporarily, so speed comes with a modest
memory increase. This measurement is not a guarantee of the same speedup on another machine.
The suggestion's estimated 40%/30% improvements were not used as measured results.

## Limits and reference compatibility

The project's default sublinear TF is `log1p(count)`, while sklearn's is `1 + log(count)`.
The optional reference test explicitly preweights by log1p and then applies sklearn IDF/L2
normalization with sublinear TF disabled. Linear TF is compared directly with equal settings.
No sklearn-equivalence claim is made until those optional tests actually execute.

SciPy/sklearn and matplotlib are absent from the available runtime. Network installation
did not execute because automatic approval review failed with an unsupported-model service
error. The seven sklearn reference tests remain unverified here (rendering was verified later; see the follow-up check). Numeric EDA
and the full final prediction were run through `--no-plots` with real data, not mocked results.
The existing study report/results were retained; the full 42-candidate CV was later repeated (see the follow-up check).

Original source and ZIP backups are under `work/optimization_before/` outside the package.
The packaged source hashes are in `verification.json`.
