#!/usr/bin/env python3
"""Stack fine-tuned transformer runs with the NB-LR model (NumPy + pandas only).

Every transformer run folder (from ``finetune_transformer.py``) contains
``holdout_predictions.csv`` (2,500 training reviews it never trained on) and
``test_predictions.csv``. For NB-LR we use its out-of-fold scores on the same
hold-out reviews (``results/oof_scores_best.npy``; each made by a CV model that
never saw the review) and its test scores (``results/test_scores.csv``).

A small L2-regularised logistic regression is fitted on the hold-out margins of
all models (one weight per model + bias). Its accuracy is estimated with
repeated stratified 10-fold CV *inside* the hold-out and compared with each
single model and with a plain average of standardised margins; the best option
by CV is written to submission.csv. No test labels are used anywhere.

Usage (from the ``submission`` folder)::

    python code/ensemble.py --data-dir data --runs runs/deberta-v3-large runs/deberta-v3-large-seed2 \
        --nblr-results results --output ensemble
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ("negative", "positive")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fit_logreg(X: np.ndarray, y: np.ndarray, lam: float = 1e-2, iters: int = 50) -> np.ndarray:
    """Newton's method for L2 logistic regression; returns [bias, w...]. X is (n, k), y in {0,1}."""
    A = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(A.shape[1])
    reg = lam * len(X) * np.eye(A.shape[1])
    reg[0, 0] = 0.0
    for _ in range(iters):
        p = 0.5 * (1 + np.tanh(0.5 * (A @ w)))
        g = A.T @ (p - y) + reg @ w
        H = (A * (p * (1 - p))[:, None]).T @ A + reg + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def predict_margin(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return w[0] + X @ w[1:]


def stratified_folds(y: np.ndarray, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), dtype=int)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % k
    return fold


def cv_accuracy(X: np.ndarray, y: np.ndarray, method: str, k: int = 10, repeats: int = 5) -> float:
    """Mean accuracy of a combiner, estimated by repeated stratified k-fold inside the hold-out."""
    accs = []
    for r in range(repeats):
        fold = stratified_folds(y, k, seed=100 + r)
        correct = 0
        for f in range(k):
            tr, va = fold != f, fold == f
            if method == "stack":
                m = predict_margin(fit_logreg(X[tr], y[tr]), X[va])
            else:  # "mean": average of margins standardised with training-fold statistics
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
                m = ((X[va] - mu) / sd).mean(1)
            correct += int(np.sum((m > 0) == y[va]))
        accs.append(correct / len(y))
    return float(np.mean(accs))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="transformer run folders")
    ap.add_argument("--nblr-results", default="results", help="folder with oof_scores_best.npy and test_scores.csv")
    ap.add_argument("--no-nblr", action="store_true", help="do not include NB-LR")
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    data = Path(args.data_dir)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(data / "train.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    label_of = dict(zip(train["id"], (train["sentiment"] == LABELS[1]).astype(int)))

    names, hold_cols, test_cols = [], [], []
    hold_ids = None
    for run in args.runs:
        run = Path(run)
        h = pd.read_csv(run / "holdout_predictions.csv").sort_values("id")
        t = pd.read_csv(run / "test_predictions.csv").set_index("id")
        if hold_ids is None:
            hold_ids = h["id"].values
        elif not np.array_equal(hold_ids, h["id"].values):
            sys.exit(f"{run}: different hold-out reviews; all runs must use --split-seed 42 and the same --holdout-frac")
        if len(t) != len(sample) or set(t.index) != set(sample["id"]):
            sys.exit(f"{run}: test_predictions.csv does not cover all {len(sample):,} test ids (smoke run?)")
        names.append(run.name)
        hold_cols.append(h["margin"].values)
        test_cols.append(t.loc[sample["id"].values, "margin"].values)

    if not args.no_nblr:
        res = Path(args.nblr_results)
        oof = np.load(res / "oof_scores_best.npy")
        if len(oof) != len(train):
            sys.exit("oof_scores_best.npy does not match train.csv")
        oof_by_id = dict(zip(train["id"].values, oof))
        ts = pd.read_csv(res / "test_scores.csv").set_index("id").loc[sample["id"].values, "score"].values
        # the final NB-LR refit (25k reviews) produces slightly larger scores than the 20k-review CV models;
        # match the test-score spread to the out-of-fold spread so the stacking weight transfers
        ts = ts * (np.std(oof) / np.std(ts))
        names.append("nblr")
        hold_cols.append(np.array([oof_by_id[i] for i in hold_ids]))
        test_cols.append(ts)

    X_hold = np.column_stack(hold_cols)
    X_test = np.column_stack(test_cols)
    y = np.array([label_of[i] for i in hold_ids])
    print(f"hold-out reviews: {len(y):,}; models: {', '.join(names)}")

    singles = {n: float(np.mean((X_hold[:, j] > 0) == y)) for j, n in enumerate(names)}
    for n, a in singles.items():
        print(f"  single {n:<32s} hold-out accuracy {a:.4%}")
    corr = np.corrcoef(X_hold.T) if X_hold.shape[1] > 1 else np.ones((1, 1))

    options = {}
    best_single = max(singles, key=singles.get)
    options[f"single:{best_single}"] = singles[best_single]
    if X_hold.shape[1] > 1:
        options["mean"] = cv_accuracy(X_hold, y, "mean")
        options["stack"] = cv_accuracy(X_hold, y, "stack")
    for k, v in options.items():
        print(f"  option {k:<32s} estimated accuracy {v:.4%}")
    # prefer simpler options unless the stack is clearly better (> 0.05 points)
    choice = max(options, key=lambda k: options[k] - (0.0005 if k == "stack" else 0.0))
    print(f"chosen: {choice}")

    if choice == "stack":
        w = fit_logreg(X_hold, y)
        m_test = predict_margin(w, X_test)
        weights = {"bias": float(w[0]), **{n: float(v) for n, v in zip(names, w[1:])}}
    elif choice == "mean":
        mu, sd = X_hold.mean(0), X_hold.std(0) + 1e-12
        m_test = ((X_test - mu) / sd).mean(1)
        weights = {n: 1.0 / len(names) for n in names}
    else:
        j = names.index(best_single)
        m_test = X_test[:, j]
        weights = {best_single: 1.0}

    sub = pd.DataFrame({"id": sample["id"].values, "sentiment": np.where(m_test > 0, LABELS[1], LABELS[0])})
    assert sub["id"].is_unique and set(sub["sentiment"]) <= set(LABELS)
    sub.to_csv(out / "submission.csv", index=False)
    pd.DataFrame({"id": sample["id"].values, "margin": m_test}).to_csv(out / "test_margins.csv", index=False)
    info = {
        "runs": [str(r) for r in args.runs], "models": names, "n_holdout": int(len(y)),
        "single_holdout_accuracy": singles, "options_estimated_accuracy": options, "choice": choice,
        "weights": weights, "margin_correlation": np.round(corr, 4).tolist(),
        "test_positive_share": float(np.mean(m_test > 0)),
        "submission_sha256": sha256_file(out / "submission.csv"),
    }
    (out / "ensemble_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"wrote {out / 'submission.csv'} (positive share {info['test_positive_share']:.4f}, "
          f"sha256 {info['submission_sha256'][:16]}...)")


if __name__ == "__main__":
    main()
