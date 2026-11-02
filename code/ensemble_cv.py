#!/usr/bin/env python3
"""Combine several cross-validated model families on the full 25,000-row OOF set.

``ensemble.py`` fits its combination weights on 2,500 hold-out reviews. That is too few:
with 2,500 rows the default equal-weight average beats a greedily tuned one when both are
checked on held-out data, because the weights are fitted to noise. Here the weights are
fitted on the out-of-fold predictions of every training review instead, and — because the
OOF vector of one model family is *not* independent of another family's when the two use
the same reviews — the estimate of the combination itself is obtained with repeated
stratified 10-fold CV over all 25,000 rows, exactly as ``ensemble.py`` does inside the
small hold-out but now with ten times the data and a standard error around 0.1 points.

Combiners (all scored on the 25,000 OOF rows):

* ``single:<name>`` — the best family alone;
* ``s-mean``  — mean of the margins after dividing each family's margins by its OOF standard
  deviation. Nothing is subtracted, so every family keeps its own decision threshold at 0, and
  averaging a family with itself reproduces it exactly. No labels are used;
* ``stack``   — L2 logistic regression on the margins (weights and a bias fitted on labels, scored
  by repeated 10-fold CV);
* shown for comparison only, never chosen automatically: ``z-mean`` (the stage-2 combiner, which
  also subtracts each family's mean margin and therefore moves its threshold), ``mean`` (raw
  margins) and ``rank-mean`` (forces exactly 50 % positive predictions).

Automatic choice, fixed before seeing any stage-4 result: ``s-mean`` if it beats the best single
family on the OOF rows, otherwise that single family; the fitted ``stack`` only if it beats the
choice by more than one standard error (``--allow-weights``: by any margin). ``--choice`` forces
any combiner, including the comparison-only ones.

Whatever is chosen is applied to the test margins with the *same* formula and the same OOF
statistics (fixed 2026-10-18: the earlier version fell back to the single best family whenever
``z-mean`` or ``rank-mean`` won, and applied the standardised mean when the raw ``mean`` won).

Inputs
    A folder per model family (as produced by ``code/run_cv.py`` + ``code/assemble_cv.py``):
    each must contain ``oof_predictions.csv`` (id, label, margin) and ``test_predictions.csv``
    (id, margin), the latter being the average of the K fold models.

    python code/ensemble_cv.py --oof deberta=ensemble-deberta qwen3=ensemble-qwen3 --output ensemble-stage4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ensemble import fit_logreg, predict_margin, cv_accuracy, sha256_file  # noqa: E402
from assemble_cv import mcnemar_exact  # noqa: E402
import finetune_transformer as ft  # noqa: E402

LABELS = ("negative", "positive")
INFO_ONLY = ("z-mean", "mean", "rank-mean")  # printed for comparison, never chosen automatically


def load_family(path: Path, name: str, train: pd.DataFrame, sample: pd.DataFrame):
    """OOF margins for every training review plus test margins for one model family."""
    oof_path = path / "oof_predictions.csv"
    if not oof_path.exists():
        sys.exit(f"{path}: no oof_predictions.csv. Write it with code/assemble_cv.py, or point --oof "
                 "at the folder that holds it.")
    oof = pd.read_csv(oof_path)
    need = {"id", "margin"} - set(oof.columns)
    if need:
        sys.exit(f"{oof_path} is missing {sorted(need)}")
    if len(oof) != len(train) or set(oof["id"]) != set(train["id"]):
        sys.exit(f"{oof_path} does not cover all {len(train):,} training reviews exactly once")
    oof = oof.set_index("id").loc[train["id"].values]
    if "label" in oof.columns:
        expect = (train["sentiment"].values == LABELS[1]).astype(int)
        if not np.array_equal(oof["label"].values.astype(int), expect):
            sys.exit(f"{oof_path}: labels disagree with train.csv — wrong dataset?")

    test_path = path / "test_predictions.csv"
    if not test_path.exists():
        sys.exit(f"{path}: no test_predictions.csv")
    test = pd.read_csv(test_path).set_index("id")
    if set(test.index) != set(sample["id"]):
        sys.exit(f"{test_path} does not cover all {len(sample):,} test ids")
    test = test.loc[sample["id"].values, "margin"].values
    return name, oof["margin"].values, test


def rank_mean(X: np.ndarray) -> np.ndarray:
    """Mean of per-column average ranks, centred at its median (so > 0 means the top half)."""
    R = np.column_stack([pd.Series(X[:, j]).rank().values for j in range(X.shape[1])]).mean(1)
    return R - np.median(R)


def combine(choice: str, X: np.ndarray, X_new: np.ndarray, y: np.ndarray, names: list, lam: float):
    """Margins of combiner ``choice`` on ``X_new``; statistics/weights come from the OOF matrix ``X``.

    Returns (margins, weights-description). Only ``stack`` looks at the labels ``y``.
    """
    if choice.startswith("single:"):
        nm = choice.split(":", 1)[1]
        j = names.index(nm)
        return X_new[:, j].astype(float), {nm: 1.0}
    if choice == "s-mean":
        sd = X.std(0) + 1e-12
        return (X_new / sd).mean(1), {nm: float(1.0 / len(names) / s) for nm, s in zip(names, sd)}
    if choice == "z-mean":
        mu, sd = X.mean(0), X.std(0) + 1e-12
        return ((X_new - mu) / sd).mean(1), {nm: 1.0 / len(names) for nm in names}
    if choice == "mean":
        return X_new.mean(1), {nm: 1.0 / len(names) for nm in names}
    if choice == "rank-mean":
        return rank_mean(X_new), {nm: 1.0 / len(names) for nm in names}
    if choice == "stack":
        w = fit_logreg(X, y, lam=lam)
        return predict_margin(w, X_new), {"bias": float(w[0]), **{nm: float(v) for nm, v in zip(names, w[1:])}}
    raise ValueError(f"unknown combiner {choice!r}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--oof", nargs="+", required=True,
                   help="one or more NAME=FOLDER pairs holding oof_predictions.csv + test_predictions.csv")
    p.add_argument("--output", required=True)
    p.add_argument("--repeats", type=int, default=10, help="repeats of the 10-fold CV used to score combiners")
    p.add_argument("--lam", type=float, default=1e-2, help="L2 strength of the logistic stack")
    p.add_argument("--allow-weights", action="store_true",
                   help="allow the fitted stack to be chosen even when it does not clearly beat the average")
    p.add_argument("--choice", help="force a combiner (s-mean, stack, z-mean, mean, rank-mean or single:<name>) "
                                    "instead of the automatic choice; the estimates are still printed")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    for spec in args.oof:  # check the command line before reading 50,000 reviews
        if "=" not in spec:
            sys.exit(f"--oof expects NAME=FOLDER, got {spec!r}")
    specs = [spec.split("=", 1) for spec in args.oof]
    if len({nm for nm, _ in specs}) != len(specs):
        sys.exit(f"family names must be unique: {[nm for nm, _ in specs]}")
    data = Path(args.data_dir)
    train = pd.read_csv(data / "train.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    y = (train["sentiment"].values == LABELS[1]).astype(np.int64)

    names, oof_cols, test_cols = [], [], []
    for name, folder in specs:
        n, o, t = load_family(Path(folder), name, train, sample)
        names.append(n)
        oof_cols.append(o)
        test_cols.append(t)

    X = np.column_stack(oof_cols)
    X_test = np.column_stack(test_cols)
    n = len(y)
    print(f"OOF rows: {n:,} (one prediction per training review); models: {', '.join(names)}")

    singles = {nm: float(np.mean((X[:, j] > 0) == y)) for j, nm in enumerate(names)}
    for nm, a in singles.items():
        print(f"  single {nm:<24s} OOF accuracy {a:.4%}")
    corr = np.corrcoef(X.T) if X.shape[1] > 1 else np.ones((1, 1))
    off = corr[~np.eye(len(names), dtype=bool)] if len(names) > 1 else np.array([1.0])
    print(f"  OOF margin correlation: min {off.min():.4f}, mean {off.mean():.4f}")

    options = {}
    best_single = max(singles, key=singles.get)
    options[f"single:{best_single}"] = singles[best_single]
    if X.shape[1] > 1:
        # s-mean uses only label-free statistics (the OOF standard deviations), so scoring it on the
        # same OOF rows is unbiased; the stack fits labels and is therefore scored by repeated CV.
        options["s-mean"] = float(np.mean((combine("s-mean", X, X, y, names, args.lam)[0] > 0) == y))
        options["stack"] = cv_accuracy(X, y, "stack", repeats=args.repeats)
        # comparison only: "z-mean" is the stage-2 combiner (standardised with the training part of
        # each CV split, which also re-centres every family), "mean" uses raw margins, "rank-mean"
        # forces a 50/50 split of the predictions
        options["z-mean"] = cv_accuracy(X, y, "mean", repeats=args.repeats)
        options["mean"] = float(np.mean((X.mean(1) > 0) == y))
        options["rank-mean"] = float(np.mean((rank_mean(X) > 0) == y))
    for k, v in sorted(options.items(), key=lambda kv: -kv[1]):
        tag = "   (comparison only)" if k in INFO_ONLY else ""
        print(f"  option {k:<24s} estimated accuracy {v:.4%}{tag}")

    # paired comparison of the label-free combiners with the best single family on the same rows
    best_correct = (X[:, names.index(best_single)] > 0) == (y == 1)
    paired = {}
    if X.shape[1] > 1:
        for k in ("s-mean",) + INFO_ONLY:
            m_oof, _ = combine(k, X, X, y, names, args.lam)
            mc = mcnemar_exact((m_oof > 0) == (y == 1), best_correct)
            paired[k] = mc
            print(f"  {k:<10s} vs single:{best_single}: fixes {mc['a_only']} / breaks {mc['b_only']} "
                  f"OOF reviews, McNemar p = {mc['p_value']:.3g}")

    # Pre-registered rule: s-mean if it beats the best single family, else that family; the fitted
    # stack only if it beats the choice by more than one standard error (any margin with --allow-weights).
    base_acc = options.get("s-mean", options[f"single:{best_single}"])
    se = float(np.sqrt(base_acc * (1 - base_acc) / n))
    choice = f"single:{best_single}"
    if "s-mean" in options and options["s-mean"] > options[choice]:
        choice = "s-mean"
    if "stack" in options:
        gain = options["stack"] - options[choice]
        if gain > (0.0 if args.allow_weights else se):
            choice = "stack"
        elif gain > 0:
            print(f"  stack beats {choice} by only {gain:.4%} (< 1 SE = {se:.4%}) -> keeping {choice}")
    if args.choice:
        valid = set(options) | {f"single:{nm}" for nm in names}
        if args.choice not in valid:
            sys.exit(f"--choice {args.choice!r} is not one of {sorted(valid)}")
        print(f"  automatic choice would be {choice}; forced by --choice: {args.choice}")
        choice = args.choice
    print(f"chosen: {choice}")

    m_test, weights = combine(choice, X, X_test, y, names, args.lam)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    sub = pd.DataFrame({"id": sample["id"].values,
                        "sentiment": np.where(m_test > 0, LABELS[1], LABELS[0])})
    assert sub["id"].is_unique and len(sub) == len(sample) and set(sub["sentiment"]) <= set(LABELS)
    sub.to_csv(out / "submission.csv", index=False)
    pd.DataFrame({"id": sample["id"].values, "margin": m_test}).to_csv(out / "test_margins.csv", index=False)
    single_test = X_test[:, names.index(best_single)] > 0
    changed = int(np.sum((m_test > 0) != single_test))
    info = {
        "models": names, "folders": args.oof, "oof_rows": n, "choice": choice,
        "single_oof_accuracy": singles, "options_estimated_accuracy": options,
        "paired_vs_best_single": paired, "standard_error": se, "weights": weights,
        "oof_margin_correlation": np.round(corr, 4).tolist(),
        "test_positive_share": float(np.mean(m_test > 0)),
        "test_predictions_changed_vs_best_single": changed,
        "submission_sha256": sha256_file(out / "submission.csv"),
    }
    (out / "ensemble_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"test: {changed:,} of {len(m_test):,} predictions differ from single:{best_single}")
    print(f"wrote {out / 'submission.csv'} (positive share {info['test_positive_share']:.4f}, "
          f"sha256 {info['submission_sha256'][:16]}...)")


if __name__ == "__main__":
    main()
