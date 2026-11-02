#!/usr/bin/env python3
"""Compare two models on the reviews they both validated, and optionally write their s-mean ensemble.

Built for the stage-4 pilot gate: DeBERTa fold 0 (``runs/cv-deberta/deberta-f0``) against the
LLM fold 0 (``runs/cv-qwen3/qwen3-f0``). Both were validated on the same 5,000 training reviews
(fold 0 of the same 5-fold split, seed 42), so the comparison is *paired*: only the reviews on
which the two disagree carry information, which makes it far more sensitive than comparing two
accuracies with their separate confidence intervals.

Printed: accuracy of each model (Wilson 95 % interval), McNemar test, margin correlation, the
"both wrong" count (a lower bound on the label-noise floor of this pair), the accuracy of the
label-free s-mean combination (plus z-mean and rank-mean for comparison), accuracy by review
length, accuracy on reviews that state an explicit "N/10" rating, and duplicate reviews shared by
train and test.

s-mean: each model's margins are divided by the standard deviation of its own validation margins
(no labels are used) and the two are averaged. Nothing is subtracted, so each model keeps its own
decision threshold at 0 and averaging a model with itself reproduces it exactly. (z-mean, the
stage-2 combiner, also subtracts the mean margin, which moves the threshold; on the stage-2
hold-out that moved 6 of 2,500 decisions, the same size as the pilot gate itself.)

With ``--output`` it also writes the s-mean ensemble of the two models' *test* margins:
``submission.csv``, ``test_margins.csv`` and ``compare_info.json``. The intended use is

    py -3.12 code/compare_runs.py --data-dir data --a runs/cv-deberta/deberta-f0 \\
        --a-test ensemble-deberta/test_predictions.csv --b runs/cv-qwen3/qwen3-f0 \\
        --names deberta qwen3 --output ensemble-pilot

i.e. the 5-fold DeBERTa average (the 0.97150 file) plus the single LLM fold-0 model.
Test labels are never used. NumPy + pandas only.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_transformer as ft  # noqa: E402  (numpy/pandas only)
from assemble_cv import mcnemar_exact, wilson  # noqa: E402

LABELS = ("negative", "positive")
RATING_RE = re.compile(r"(?<![\d.])(10|[0-9])(?:\.\d+)?\s*(?:/|out\s+of)\s*10(?!\d)", re.I)
LENGTH_BUCKETS = (("<100", 0, 100), ("100-199", 100, 200), ("200-299", 200, 300),
                  ("300-499", 300, 500), (">=500", 500, 10 ** 9))


# --------------------------------------------------------------------------- helpers
def load_preds(path: Path) -> pd.DataFrame:
    """Validation predictions (id, margin[, label]) from a run folder or a CSV file."""
    f = path / "holdout_predictions.csv" if path.is_dir() else path
    if not f.exists():
        sys.exit(f"{f} not found")
    df = pd.read_csv(f)
    need = {"id", "margin"} - set(df.columns)
    if need:
        sys.exit(f"{f} is missing {sorted(need)}")
    if df["id"].duplicated().any():
        sys.exit(f"{f} contains duplicate ids")
    return df


def load_test(path, sample_ids: np.ndarray) -> np.ndarray:
    df = pd.read_csv(path)
    if "margin" not in df.columns or "id" not in df.columns:
        sys.exit(f"{path} needs columns id, margin")
    if set(df["id"]) != set(sample_ids):
        sys.exit(f"{path} does not cover exactly the {len(sample_ids):,} test ids")
    return df.set_index("id").loc[sample_ids, "margin"].values.astype(float)


def zstats(m: np.ndarray) -> tuple:
    return float(np.mean(m)), float(np.std(m) + 1e-12)


def zmean(margins: list, stats: list) -> np.ndarray:
    """Centre and scale (moves each model's threshold); kept for comparison with stage 2."""
    return np.mean([(m - mu) / sd for m, (mu, sd) in zip(margins, stats)], axis=0)


def smean(margins: list, stats: list) -> np.ndarray:
    """Scale only: every model keeps its own decision threshold at 0."""
    return np.mean([m / sd for m, (mu, sd) in zip(margins, stats)], axis=0)


def rank_mean(margins: list) -> np.ndarray:
    R = np.mean([pd.Series(m).rank().values for m in margins], axis=0)
    return R - np.median(R)


def explicit_rating(text: str):
    """'low' (all stated ratings <= 4/10), 'high' (all >= 7/10), 'mixed', 'mid' or None."""
    vals = [int(v) for v in RATING_RE.findall(str(text))]
    if not vals:
        return None
    low = any(v <= 4 for v in vals)
    high = any(v >= 7 for v in vals)
    if low and high:
        return "mixed"
    if low:
        return "low"
    if high:
        return "high"
    return "mid"


def norm_text(text: str) -> str:
    return ft.clean_review(text).lower()


def acc_line(name: str, correct: np.ndarray) -> str:
    k, n = int(correct.sum()), len(correct)
    lo, hi = wilson(k, n)
    return f"{name:<22s} accuracy {k / max(1, n):.4%}  ({k:,}/{n:,}, 95% CI {lo:.2%}-{hi:.2%})"


def mc_line(label: str, a_correct: np.ndarray, b_correct: np.ndarray) -> tuple:
    mc = mcnemar_exact(a_correct, b_correct)
    verdict = "a real difference" if mc["p_value"] < 0.05 else "not distinguishable from noise"
    return (f"{label}: fixes {mc['a_only']} / breaks {mc['b_only']} reviews, "
            f"McNemar p = {mc['p_value']:.3g} ({verdict})"), mc


# --------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--a", required=True, help="reference run folder (or holdout/oof CSV), e.g. runs/cv-deberta/deberta-f0")
    p.add_argument("--b", required=True, help="new run folder (or CSV), e.g. runs/cv-qwen3/qwen3-f0")
    p.add_argument("--a-test", help="test margins of A (default: A/test_predictions.csv)")
    p.add_argument("--b-test", help="test margins of B (default: B/test_predictions.csv)")
    p.add_argument("--names", nargs=2, default=("A", "B"), metavar=("NAME_A", "NAME_B"))
    p.add_argument("--output", help="folder for the s-mean ensemble of the test margins (optional)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    na, nb = args.names
    data = Path(args.data_dir)
    train = pd.read_csv(data / "train.csv")
    test = pd.read_csv(data / "test.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    y_of = dict(zip(train["id"].values, (train["sentiment"].values == LABELS[1]).astype(int)))
    row_of = {int(i): k for k, i in enumerate(train["id"].values)}

    A, B = load_preds(Path(args.a)), load_preds(Path(args.b))
    common = np.intersect1d(A["id"].values, B["id"].values)
    if len(common) == 0:
        sys.exit("the two runs share no validation reviews (different folds or split seeds?)")
    missing = [int(i) for i in common if int(i) not in row_of]
    if missing:
        sys.exit(f"{len(missing)} validation ids (e.g. {missing[:5]}) are not in train.csv")
    A = A.set_index("id").loc[common]
    B = B.set_index("id").loc[common]
    y = np.array([y_of[int(i)] for i in common])
    for nm, df in ((na, A), (nb, B)):
        if "label" in df.columns and not np.array_equal(df["label"].values.astype(int), y):
            sys.exit(f"{nm}: labels in the prediction file disagree with train.csv")
    mA, mB = A["margin"].values.astype(float), B["margin"].values.astype(float)
    cA, cB = (mA > 0) == (y == 1), (mB > 0) == (y == 1)
    print(f"common validation reviews: {len(common):,} ({na}: {len(A):,} rows used, {nb}: {len(B):,})")
    print("  " + acc_line(na, cA))
    print("  " + acc_line(nb, cB))
    line, mc_ba = mc_line(f"{nb} vs {na}", cB, cA)
    print("  " + line)
    pear = float(np.corrcoef(mA, mB)[0, 1])
    spear = float(np.corrcoef(pd.Series(mA).rank(), pd.Series(mB).rank())[0, 1])
    both_wrong = int(np.sum(~cA & ~cB))
    print(f"  margin correlation {pear:.4f} (rank {spear:.4f}); both wrong on {both_wrong} reviews "
          f"({both_wrong / len(y):.2%}); oracle of the two {np.mean(cA | cB):.4%}")
    print(f"  of {int((~cA).sum())} {na} errors, {nb} gets {int((~cA & cB).sum())} right")

    stats = [zstats(mA), zstats(mB)]
    mS = smean([mA, mB], stats)
    mZ = zmean([mA, mB], stats)
    mR = rank_mean([mA, mB])
    cS, cZ, cR = (mS > 0) == (y == 1), (mZ > 0) == (y == 1), (mR > 0) == (y == 1)
    print("  " + acc_line(f"s-mean({na}+{nb})", cS))
    print("  " + acc_line(f"z-mean({na}+{nb})", cZ) + "   (comparison only)")
    print("  " + acc_line(f"rank-mean({na}+{nb})", cR) + "   (comparison only)")
    line_sa, mc_sa = mc_line(f"s-mean vs {na}", cS, cA)
    line_sb, mc_sb = mc_line(f"s-mean vs {nb}", cS, cB)
    print("  " + line_sa)
    print("  " + line_sb)
    gain_b = 100 * (cB.mean() - cA.mean())
    gain_s = 100 * (cS.mean() - cA.mean())
    print(f"\nGATE numbers: {nb} - {na} = {gain_b:+.2f} points; s-mean - {na} = {gain_s:+.2f} points "
          f"(one review = {100 / len(y):.3f} points on these {len(y):,} rows)")

    # ---- by review length
    rows = np.array([row_of[int(i)] for i in common])
    words = train["review"].str.split().str.len().values[rows]
    by_len = []
    print("\n  length      n      " + f"{na:>9s} {nb:>9s}  s-mean")
    for name, lo, hi in LENGTH_BUCKETS:
        m = (words >= lo) & (words < hi)
        if not m.any():
            continue
        by_len.append({"bucket": name, "n": int(m.sum()), na: float(cA[m].mean()), nb: float(cB[m].mean()),
                       "s-mean": float(cS[m].mean())})
        print(f"  {name:<8s} {int(m.sum()):6,}   {cA[m].mean():9.2%} {cB[m].mean():9.2%}  {cS[m].mean():7.2%}")

    # ---- explicit ratings ("7/10", "3 out of 10")
    kinds = np.array([explicit_rating(t) or "none" for t in train["review"].values[rows]])
    has = kinds != "none"
    rule_pred = np.where(kinds == "high", 1, np.where(kinds == "low", 0, -1))
    rule_ok = (rule_pred >= 0)
    rating = {"n_with_rating": int(has.sum()), "n_rule_applies": int(rule_ok.sum())}
    if has.any():
        rule_acc = float(np.mean(rule_pred[rule_ok] == y[rule_ok])) if rule_ok.any() else float("nan")
        rating.update({na: float(cA[has].mean()), nb: float(cB[has].mean()), "s-mean": float(cS[has].mean()),
                       "rule_accuracy_where_it_applies": rule_acc,
                       "s_errors_rule_would_fix": int(np.sum(rule_ok & ~cS & (rule_pred == y))),
                       "s_correct_rule_would_break": int(np.sum(rule_ok & cS & (rule_pred != y)))})
        print(f"\n  reviews stating an N/10 rating: {int(has.sum())} ({has.mean():.1%}); accuracy {na} "
              f"{cA[has].mean():.2%}, {nb} {cB[has].mean():.2%}, s-mean {cS[has].mean():.2%}")
        print(f"  rule 'high -> positive, low -> negative' applies to {int(rule_ok.sum())}: accuracy {rule_acc:.2%}; "
              f"it would fix {rating['s_errors_rule_would_fix']} s-mean errors and break "
              f"{rating['s_correct_rule_would_break']} correct ones")

    # ---- duplicates shared by train and test (diagnostic only; nothing is overridden)
    train_keys = [norm_text(t) for t in train["review"].values]
    labels_of_key: dict = {}
    for k_, lab in zip(train_keys, (train["sentiment"].values == LABELS[1]).astype(int)):
        labels_of_key.setdefault(k_, []).append(int(lab))
    dup_groups = [v for v in labels_of_key.values() if len(v) > 1]
    dup_consistent = sum(1 for v in dup_groups if len(set(v)) == 1)
    test_keys = [norm_text(t) for t in test["review"].values]
    twin = [labels_of_key.get(k_) for k_ in test_keys]          # list of train labels, or None
    has_twin = np.array([t is not None for t in twin], dtype=bool)
    dups = {"train_duplicate_groups": len(dup_groups), "train_groups_same_label": dup_consistent,
            "test_reviews_with_identical_train_review": int(has_twin.sum())}
    print(f"\n  duplicates: {len(dup_groups)} groups of identical reviews inside train.csv "
          f"({dup_consistent} with one label); {int(has_twin.sum())} test reviews have an identical train review")

    # ---- the ensemble of the test margins
    info = {"a": args.a, "b": args.b, "names": [na, nb], "n_common": int(len(common)), "ensemble": "s-mean",
            "accuracy": {na: float(cA.mean()), nb: float(cB.mean()), "s-mean": float(cS.mean()),
                         "z-mean (comparison)": float(cZ.mean()), "rank-mean (comparison)": float(cR.mean())},
            "mcnemar": {f"{nb}_vs_{na}": mc_ba, f"s-mean_vs_{na}": mc_sa, f"s-mean_vs_{nb}": mc_sb},
            "margin_correlation": pear, "rank_correlation": spear, "both_wrong": both_wrong,
            "by_length": by_len, "explicit_rating": rating, "duplicates": dups,
            "validation_margin_mean_sd": {na: stats[0], nb: stats[1]}}
    if args.output:
        sample_ids = sample["id"].values
        ta_path = Path(args.a_test) if args.a_test else Path(args.a) / "test_predictions.csv"
        tb_path = Path(args.b_test) if args.b_test else Path(args.b) / "test_predictions.csv"
        for p_ in (ta_path, tb_path):
            if not p_.exists():
                sys.exit(f"{p_} not found; pass --a-test / --b-test")
        tA, tB = load_test(ta_path, sample_ids), load_test(tb_path, sample_ids)
        tS = smean([tA, tB], stats)
        agree = float(np.mean((tA > 0) == (tB > 0)))
        changed = int(np.sum((tS > 0) != (tA > 0)))
        twin_of_id = dict(zip(test["id"].values, twin))
        disagree_twin = 0
        for i, s_ in zip(sample_ids, tS):
            labs = twin_of_id.get(i)
            if labs is not None and len(set(labs)) == 1 and int(s_ > 0) != labs[0]:
                disagree_twin += 1
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        sub = ft.write_submission(sample_ids, tS, data / "sample_submission.csv", out / "submission.csv")
        pd.DataFrame({"id": sample_ids, "margin": tS}).to_csv(out / "test_margins.csv", index=False)
        info.update({"a_test": str(ta_path), "b_test": str(tb_path), "test_agreement": agree,
                     "test_changed_vs_a": changed, "test_disagrees_with_identical_train_review": disagree_twin,
                     "submission": sub})
        print(f"\n  test: {na} and {nb} agree on {agree:.2%} of {len(sample_ids):,}; the s-mean changes "
              f"{changed} of {na}'s predictions; {disagree_twin} s-mean predictions disagree with the label of "
              f"an identical train review")
        print(f"wrote {out / 'submission.csv'} sha256={sub['sha256']} "
              f"({sub['positive']:,} positive / {sub['negative']:,} negative)")
        (out / "compare_info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
        print(f"wrote {out / 'compare_info.json'}")


if __name__ == "__main__":
    main()
