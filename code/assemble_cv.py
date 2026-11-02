#!/usr/bin/env python3
"""Combine the folds written by ``code/run_cv.py`` into one out-of-fold estimate.

Reads ``<runs>/<name>-f0 ... -f{K-1}`` (each a folder produced by
``finetune_transformer.py --folds K --fold j``), and produces

* ``oof_predictions.csv`` — one row per training review: ``id,label,margin,p_positive,fold``.
  Every review is predicted by the model that did *not* train on it, so accuracy over all
  25,000 rows is an honest estimate with a standard error three times smaller than the
  single 2,500-row hold-out.
* ``test_predictions.csv`` — the mean of the K fold models' margins per test review, and
  ``submission.csv`` — the same averaged into labels. Averaging K models that each saw
  80 % of the data is the cheap half of bagging.
* ``cv_summary.json`` — per-fold accuracy, the pooled OOF accuracy with a Wilson interval,
  accuracy by review length, and McNemar tests against the stage-2 hold-out predictions
  when those are available.

Sanity checks are run before anything is written: the folds must partition the training
rows (every ``id`` predicted exactly once), and fold 0 must cover the same reviews as the
stage-2 hold-out, otherwise the numbers cannot be compared with the report.

    python code/assemble_cv.py --data-dir data --runs runs/cv-deberta --output ensemble-cv
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_transformer as ft  # noqa: E402  (numpy/pandas only)

LABELS = ("negative", "positive")


# --------------------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score interval for a binomial proportion (better than normal at 97 %)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def mcnemar_exact(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    """Two-sided exact McNemar test for two classifiers scored on the same reviews.

    Only the reviews where the two disagree carry information, so on a small hold-out the
    test is weak by construction — which is exactly why the pooled OOF estimate is used.
    """
    a_correct = np.asarray(a_correct, dtype=bool)
    b_correct = np.asarray(b_correct, dtype=bool)
    b_only = int(np.sum(~a_correct & b_correct))   # a wrong, b right
    a_only = int(np.sum(a_correct & ~b_correct))   # a right, b wrong
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "n_discordant": 0, "p_value": 1.0}
    p = 2.0 * sum(math.comb(n, i) for i in range(0, min(a_only, b_only) + 1)) / 2.0 ** n
    return {"a_only": a_only, "b_only": b_only, "n_discordant": n, "p_value": float(min(1.0, p))}


# --------------------------------------------------------------------------- loading
def load_fold(d: Path, fold: int) -> pd.DataFrame:
    v = pd.read_csv(d / "holdout_predictions.csv")
    t = pd.read_csv(d / "test_predictions.csv")
    missing = {"id", "label", "margin"} - set(v.columns)
    if missing:
        sys.exit(f"{d / 'holdout_predictions.csv'} is missing {sorted(missing)}")
    info = {}
    info_path = d / "run_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
    v = v.assign(fold=fold)
    return v, t, info


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data", help="folder with train.csv, test.csv, sample_submission.csv")
    p.add_argument("--runs", required=True, help="root folder holding <name>-f0 ... (see run_cv.py)")
    p.add_argument("--name", default=None,
                   help="fold folder prefix passed to run_cv.py (e.g. 'deberta' for deberta-f0 ...). "
                        "Omit it to discover the prefix from the folder automatically.")
    p.add_argument("--folds", type=int, help="number of folds (default: discover from the folder)")
    p.add_argument("--output", required=True, help="folder for oof_predictions.csv, submission.csv, cv_summary.json")
    p.add_argument("--stage2-holdout", default="runs/deberta/holdout_predictions.csv",
                   help="stage-2 hold-out predictions, for the McNemar comparison (optional)")
    p.add_argument("--no-submission", action="store_true", help="do not write submission.csv")
    return p.parse_args(argv)


def discover_prefix(root: Path, name=None) -> tuple:
    """Find the fold-folder prefix ('deberta' for deberta-f0 ...) and the fold numbers present.

    ``run_cv.py --name deberta`` writes ``deberta-f0``, so the prefix cannot be assumed
    to be 'cv'. When ``--name`` is given it is used as the prefix; otherwise every
    ``<prefix>-f<number>`` folder under ``root`` is collected and the prefixes are required
    to be unique, so a folder holding two different model families is rejected instead of
    silently mixing them.
    """
    found = {}
    for p in root.iterdir() if root.is_dir() else []:
        if not p.is_dir() or "-f" not in p.name:
            continue
        prefix, _, tail = p.name.rpartition("-f")
        if prefix and tail.isdigit():
            found.setdefault(prefix, set()).add(int(tail))
    if name is not None:
        if name not in found:
            sys.exit(f"{root}: no '{name}-f*' folders. Contents: "
                     f"{sorted(p.name for p in root.iterdir()) if root.is_dir() else 'missing folder'}")
        return name, sorted(found[name])
    if not found:
        sys.exit(f"{root}: no '<prefix>-f<number>' fold folders found; run code/run_cv.py first.")
    if len(found) > 1:
        sys.exit(f"{root}: fold folders from more than one run ({', '.join(sorted(found))}). "
                 "Pass --name to choose one, or assemble them into separate folders first.")
    prefix, nums = next(iter(found.items()))
    return prefix, sorted(nums)


def main(argv=None) -> None:
    args = parse_args(argv)
    root = Path(args.runs)
    args.name, discovered = discover_prefix(root, args.name)
    if args.folds:
        folds = list(range(args.folds))
    else:
        folds = discovered
    if not folds:
        sys.exit(f"no {args.name}-f* folders under {root}; run code/run_cv.py first")
    missing = [f for f in folds if not (root / f"{args.name}-f{f}" / "holdout_predictions.csv").exists()]
    if missing:
        sys.exit(f"folds {missing} are missing from {root}; finish them with code/run_cv.py")
    print(f"fold folders: {args.name}-f{{{', '.join(str(f) for f in folds)}}}")

    data = Path(args.data_dir)
    train = pd.read_csv(data / "train.csv")
    test = pd.read_csv(data / "test.csv")
    y_all = (train["sentiment"].values == LABELS[1]).astype(np.int64)
    row_of = {int(i): k for k, i in enumerate(train["id"].values)}

    parts, test_margins, fold_info = [], [], []
    for f in folds:
        d = root / f"{args.name}-f{f}"
        v, t, info = load_fold(d, f)
        v = v.assign(row=v["id"].map(row_of))
        if v["row"].isna().any():
            sys.exit(f"{d}/holdout_predictions.csv contains ids that are not in train.csv")
        v["row"] = v["row"].astype(int)
        parts.append(v)
        test_margins.append(t.set_index("id").loc[test["id"].values, "margin"].values)
        acc = float(np.mean((v["margin"].values > 0) == v["label"].values))
        conf = (info.get("config") or {}).get("model") or info.get("preset") or "?"
        fold_info.append({"fold": f, "n_val": int(len(v)), "accuracy": acc,
                          "epochs": (info.get("config") or {}).get("epochs"),
                          "minutes": info.get("total_minutes"), "model": conf})
        print(f"fold {f}: n={len(v):,} accuracy {acc:.4%} ({conf}"
              + (f", {info['total_minutes']:.1f} min)" if info.get("total_minutes") else ")"))

    oof = pd.concat(parts, ignore_index=True)
    # ---- sanity: the folds must partition the training rows
    dup = oof["row"].duplicated()
    if dup.any():
        bad = oof.loc[dup, "row"].values[:5]
        sys.exit(f"folds overlap: rows {bad} are validated more than once. Did you pass the same "
                 "--split-seed and --folds to every fold?")
    if len(oof) != len(train):
        sys.exit(f"folds cover {len(oof):,} of {len(train):,} training reviews; "
                 f"run the remaining folds before assembling.")
    oof = oof.sort_values("row").reset_index(drop=True)
    if not np.array_equal(oof["label"].values, y_all):
        sys.exit("label mismatch between the fold files and train.csv — mixing datasets?")
    oof["fold"] = oof["fold"].astype(int)

    # ---- pooled out-of-fold accuracy
    correct = (oof["margin"].values > 0) == (y_all == 1)
    k, n = int(correct.sum()), len(correct)
    acc = k / n
    lo, hi = wilson(k, n)
    print(f"\npooled out-of-fold accuracy: {acc:.4%}  ({k:,}/{n:,}, 95% CI {lo:.4%}–{hi:.4%}, "
          f"SE ±{math.sqrt(acc * (1 - acc) / n):.4%})")
    per_fold = np.array([f["accuracy"] for f in fold_info])
    print(f"per-fold accuracy: mean {per_fold.mean():.4%} ± {per_fold.std(ddof=1):.4%} (sd), "
          f"min {per_fold.min():.4%}, max {per_fold.max():.4%}")

    summary = {
        "folds": folds, "n_folds": len(folds), "n_oof": int(n),
        "oof_accuracy": acc, "oof_correct": k,
        "oof_wilson_95": [lo, hi], "oof_se": math.sqrt(acc * (1 - acc) / n),
        "per_fold": fold_info,
        "per_fold_mean": float(per_fold.mean()), "per_fold_sd": float(per_fold.std(ddof=1)),
        "model": fold_info[0]["model"],
    }

    # ---- comparison with the stage-2 single hold-out
    s2 = Path(args.stage2_holdout)
    if s2.exists():
        h = pd.read_csv(s2)
        pos = h["id"].map({int(i): k for k, i in enumerate(oof["id"].values)})
        if pos.notna().all():
            h = h.assign(oof_pos=pos.astype(int)).sort_values("oof_pos")
            a = (oof["margin"].values[h["oof_pos"].values] > 0) == (y_all[h["oof_pos"].values] == 1)
            b = (h["margin"].values > 0) == (h["label"].values == 1)
            n_ov = len(h)
            ov_acc = float(a.mean())
            l2, h2 = wilson(int(a.sum()), n_ov)
            mc = mcnemar_exact(a, b)  # a = fold model, b = stage-2 model
            print(f"\nstage-2 hold-out ({n_ov:,} rows):     {b.mean():.4%}   "
                  f"[{s2.parent.name}]")
            print(f"fold 0 of the CV split, same rows: {ov_acc:.4%}  (95% CI {l2:.4%}–{h2:.4%})")
            print(f"McNemar (fold vs stage 2): {mc['a_only']} fixed / {mc['b_only']} broken, "
                  f"p = {mc['p_value']:.3g} — "
                  + ("a real difference" if mc["p_value"] < 0.05 else "not distinguishable from noise"))
            summary["stage2_holdout"] = {
                "file": str(s2), "n": int(n_ov), "accuracy": float(b.mean()),
                "fold_accuracy_on_same_rows": ov_acc, "wilson_95": [l2, h2], "mcnemar": mc}
        else:
            print(f"\n(not comparing with {s2}: its ids are not all in the fold validation sets)")

    # ---- accuracy by review length (does the picture change from stage 2?)
    words = train["review"].str.split().str.len().values[oof["row"].values]
    buckets = [("<100", 0, 100), ("100-199", 100, 200), ("200-299", 200, 300),
               ("300-499", 300, 500), (">=500", 500, 10 ** 9)]
    by_len = []
    for name, a_, b_ in buckets:
        m = (words >= a_) & (words < b_)
        if not m.sum():
            continue
        a_l, b_l = wilson(int(correct[m].sum()), int(m.sum()))
        by_len.append({"bucket": name, "n": int(m.sum()), "accuracy": float(correct[m].mean()),
                       "wilson_95": [a_l, b_l]})
        print(f"  length {name:<8} n={m.sum():6,}  accuracy {correct[m].mean():.4%} "
              f"(95% CI {a_l:.3%}–{b_l:.3%})")
    summary["accuracy_by_length"] = by_len

    # ---- averaged test predictions
    T = np.column_stack(test_margins)
    if T.shape[1] != len(folds):
        sys.exit("test_predictions.csv rows do not line up across folds; re-run the folds")
    mean_margin = T.mean(1)
    test_share = float(np.mean(mean_margin > 0))
    agreement = float(np.mean(np.all(T > 0, 1) | np.all(T < 0, 1)))
    print(f"\nfold models on test: averaged positive share {test_share:.4f}; "
          f"all {len(folds)} folds agree on {agreement:.2%} of the 25,000 reviews")
    print(f"margin correlation between fold models: min {np.corrcoef(T.T).min():.4f}")
    summary["test"] = {"positive_share": test_share, "unanimous_share": agreement,
                       "mean_fold_correlation": float(np.corrcoef(T.T).mean()),
                       "min_fold_correlation": float(np.corrcoef(T.T).min())}

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    oof_out = oof[["id", "label", "margin", "fold"]].copy()
    oof_out["p_positive"] = ft.sigmoid(oof_out["margin"].values)
    oof_out.to_csv(out / "oof_predictions.csv", index=False)
    print(f"\nwrote {out / 'oof_predictions.csv'} ({len(oof_out):,} rows)")

    pd.DataFrame({"id": test["id"].values, "p_positive": ft.sigmoid(mean_margin),
                  "margin": mean_margin}).to_csv(out / "test_predictions.csv", index=False)
    if not args.no_submission:
        sub = ft.write_submission(test["id"].values, mean_margin,
                                  data / "sample_submission.csv", out / "submission.csv")
        summary["submission"] = sub
        print(f"wrote {out / 'submission.csv'} sha256={sub['sha256']} "
              f"({sub['positive']:,} positive / {sub['negative']:,} negative)")
    (out / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out / 'cv_summary.json'}")


if __name__ == "__main__":
    main()
