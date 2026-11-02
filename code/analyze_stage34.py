#!/usr/bin/env python3
"""Stage 3-7 analysis: the numbers, tables and figures of report Sections 6.5-6.11.

Everything is computed from saved prediction files, so this script needs no GPU and no torch:

  results/oof_scores_best.npy, results/test_scores.csv   NB-LR (Stage 1) out-of-fold / test scores
  ensemble-<family>/oof_predictions.csv                   one out-of-fold margin per training review
  ensemble-<family>/test_predictions.csv                  mean test margin of the K fold models
  runs/<cv-folder>/<name>-f<j>/run_info.json              per-fold training record
  ensemble-stage4*/, ensemble-stage5/6/7/  ensemble_info.json, *.audit.json   ensemble_cv.py / apply_exact_match.py records
  the files uploaded to Kaggle (table KAGGLE below)

Outputs: results/stage34/stage34_summary.json and results/stage34/figures/*.png

    python code/analyze_stage34.py --data-dir data --output results/stage34
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_transformer as ft  # noqa: E402  (numpy/pandas only at import time)
from assemble_cv import wilson  # noqa: E402
from ensemble import fit_logreg, predict_margin  # noqa: E402

LABELS = ("negative", "positive")
LENGTH_BUCKETS = (("<100", 0, 100), ("100-199", 100, 200), ("200-299", 200, 300),
                  ("300-499", 300, 500), (">=500", 500, 10 ** 9))

# Model families with 25,000 out-of-fold predictions (folder written by assemble_cv.py) and the
# folder of their per-fold runs (written by run_cv.py).
FAMILIES = {
    "deberta": {"label": "DeBERTa-v3-large, 5-fold", "oof": "ensemble-deberta", "runs": "runs/cv-deberta", "prefix": "deberta"},
    "electra": {"label": "ELECTRA-large, 5-fold", "oof": "ensemble-electra", "runs": "runs/cv-electra", "prefix": "electra"},
    "deberta_rerun": {"label": "DeBERTa-v3-large, 5-fold re-run", "oof": "ensemble-retrain-control",
                      "runs": "runs/retrain-cv-control", "prefix": "deberta"},
    "qwen3": {"label": "Qwen3-8B-Base + LoRA, 5-fold", "oof": "ensemble-qwen3", "runs": "runs/cv-qwen3", "prefix": "qwen3"},
    "qwen3_14b": {"label": "Qwen3-14B-Base + LoRA, 5-fold (Stage 5)", "oof": "ensemble-qwen3-14b",
                  "runs": "runs/cv-qwen3-14b", "prefix": "qwen3-14b"},
    "mistral_24b": {"label": "Mistral-Small-24B-Base + LoRA, 5-fold (Stage 6)", "oof": "ensemble-mistral-24b",
                    "runs": "runs/cv-mistral-24b", "prefix": "mistral-24b"},
    "qwen38_27b": {"label": "Qwen3.8-27B (post-trained) + LoRA, 5-fold (Stage 7)", "oof": "ensemble-qwen38-27b",
                   "runs": "runs/cv-qwen38-27b", "prefix": "qwen38-27b"},
}

# Every file uploaded to Kaggle, in upload order, with the public score Kaggle reported
# (about 10,000 test reviews). The public scores are typed in from the Kaggle page; everything
# else in a row is recomputed from the file itself.
KAGGLE = [
    ("nblr", "NB-LR, uni/bi/trigrams (Stage 1)", "results/submission.csv", 0.91730),
    ("deberta_s42", "DeBERTa-v3-large, seed 42 (Stage 2)", "runs/deberta/submission.csv", 0.96880),
    ("ens_deb_nblr", "Mean of DeBERTa and NB-LR (Stage 2)", "ensemble/submission.csv", 0.96910),
    ("ensemble2", "Mean of 4 transformer runs and NB-LR (Stage 2)", "ensemble2/submission.csv", 0.97050),
    ("deberta_5fold", "DeBERTa-v3-large, mean of 5 fold models (Stage 3)", "ensemble-final/submission.csv", 0.97150),
    ("electra_5fold", "ELECTRA-large, mean of 5 fold models (Stage 3)", "ensemble-electra/submission.csv", 0.96940),
    ("rerun_exact", "DeBERTa 5-fold re-run + exact-match backfill (Stage 3)", "ensemble-retrain-control/submission_exact.csv", 0.97010),
    ("pilot", "DeBERTa 5-fold + Qwen3 fold 0, s-mean (Stage 4 pilot)", "ensemble-pilot/submission.csv", 0.97320),
    ("qwen3_5fold", "Qwen3-8B + LoRA, mean of 5 fold models (Stage 4)", "ensemble-qwen3/submission.csv", 0.97360),
    ("stage4", "Stage 4 rule choice (= Qwen3 5-fold)", "ensemble-stage4/submission.csv", 0.97360),
    ("stage4_exact", "Qwen3 5-fold + exact-match backfill (final A)", "ensemble-stage4/submission_exact.csv", 0.97380),
    ("stack_exact", "Stack DeBERTa + Qwen3-8B + exact-match backfill (final)", "ensemble-stage4-stack/submission_exact.csv", 0.97450),
    ("stage5_exact", "s-mean DeBERTa + Qwen3-8B + Qwen3-14B + exact-match backfill (Stage 5, final)",
     "ensemble-stage5/submission_exact.csv", 0.97560),
    ("stage6_exact", "s-mean DeBERTa + Qwen3-8B + Qwen3-14B + Mistral-24B + exact-match backfill (Stage 6, final)",
     "ensemble-stage6/submission_exact.csv", 0.97620),
    ("stage7_exact", "s-mean of the four families + Qwen3.8-27B + exact-match backfill (Stage 7, final)",
     "ensemble-stage7/submission_exact.csv", 0.97650),
]
# Stage 4 selected stage4_exact + stack_exact; the Stage 5 rule (fixed before the run: replace stage4_exact
# if the Stage 5 choice estimates more than its 97.024% OOF) swapped stage4_exact for stage5_exact; the
# Stage 6 rule (fixed before the run: replace stage5_exact if the Stage 6 choice estimates at least
# 97.254% = C's 97.204% OOF + 0.05 points) swapped stage5_exact for stage6_exact. The Stage 7 rule (fixed
# before the run: replace stage6_exact only if the Stage 7 choice estimates at least 97.384% = D's 97.304%
# OOF + 0.08 points) was NOT met (s-mean 97.344%), so stage6_exact stays. The other slot, held by
# stack_exact, was then given to stage7_exact by the student, because E is ahead of B on out-of-fold
# accuracy (97.344% vs 97.080%) as well as on the public board; this second step was not pre-registered.
FINAL_SELECTION = ("stage6_exact", "stage7_exact")
STAGE6_SELECTION = ("stack_exact", "stage6_exact")
STAGE7_THRESHOLD = 0.97384
STAGE5_SELECTION = ("stack_exact", "stage5_exact")
STAGE6_THRESHOLD = 0.97254
STAGE4_SELECTION = ("stage4_exact", "stack_exact")


# --------------------------------------------------------------------------- helpers
def r4(x: float) -> float:
    return float(round(float(x), 6))


def mcnemar_exact(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    """Two-sided exact McNemar test, as in assemble_cv.py but in exact rational arithmetic.

    assemble_cv.mcnemar_exact divides by the float 2**n, which overflows for more than 1,023
    discordant reviews (e.g. Qwen3 vs NB-LR on 25,000 rows); Fraction keeps it exact.
    """
    a_correct = np.asarray(a_correct, dtype=bool)
    b_correct = np.asarray(b_correct, dtype=bool)
    a_only = int(np.sum(a_correct & ~b_correct))   # a right, b wrong
    b_only = int(np.sum(~a_correct & b_correct))   # a wrong, b right
    n = a_only + b_only
    if n == 0:
        return {"a_only": 0, "b_only": 0, "n_discordant": 0, "p_value": 1.0}
    tail = sum(math.comb(n, i) for i in range(0, min(a_only, b_only) + 1))
    p = float(min(Fraction(1), Fraction(2 * tail, 2 ** n)))
    return {"a_only": a_only, "b_only": b_only, "n_discordant": n, "p_value": p}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(pd.Series(a).rank().values, pd.Series(b).rank().values)[0, 1])


def acc_block(correct: np.ndarray) -> dict:
    k, n = int(np.sum(correct)), int(len(correct))
    lo, hi = wilson(k, n)
    return {"n": n, "correct": k, "accuracy": r4(k / n), "wilson_95": [r4(lo), r4(hi)],
            "se": r4(np.sqrt((k / n) * (1 - k / n) / n))}


def binary_metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    tp = int(np.sum(pred & (y == 1))); tn = int(np.sum(~pred & (y == 0)))
    fp = int(np.sum(pred & (y == 0))); fn = int(np.sum(~pred & (y == 1)))
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "precision_positive": r4(prec), "recall_positive": r4(rec),
            "f1_positive": r4(2 * prec * rec / max(1e-12, prec + rec)), "specificity": r4(tn / max(1, tn + fp))}


def by_length(correct: np.ndarray, words: np.ndarray) -> list:
    out = []
    for name, lo, hi in LENGTH_BUCKETS:
        m = (words >= lo) & (words < hi)
        out.append({"bucket": name, **acc_block(correct[m])})
    return out


def by_confidence(correct: np.ndarray, margin: np.ndarray) -> list:
    a = np.abs(margin)
    q = np.digitize(a, np.quantile(a, [0.2, 0.4, 0.6, 0.8]))
    n_err = int(np.sum(~correct))
    return [{"quintile": k + 1, "accuracy": r4(np.mean(correct[q == k])),
             "share_of_errors": r4(np.sum(~correct[q == k]) / max(1, n_err))} for k in range(5)]


def pair(name_a: str, ma: np.ndarray, name_b: str, mb: np.ndarray, y: np.ndarray) -> dict:
    ca, cb = (ma > 0) == (y == 1), (mb > 0) == (y == 1)
    mc = mcnemar_exact(ca, cb)
    return {"a": name_a, "b": name_b, "pearson": r4(np.corrcoef(ma, mb)[0, 1]), "spearman": r4(spearman(ma, mb)),
            "both_right": int(np.sum(ca & cb)), "a_only_right": mc["a_only"], "b_only_right": mc["b_only"],
            "both_wrong": int(np.sum(~ca & ~cb)), "mcnemar_p": float(f"{mc['p_value']:.4g}"),
            "label_agreement": r4(np.mean((ma > 0) == (mb > 0))), "oracle_accuracy": r4(np.mean(ca | cb))}


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_oof(folder: Path, train_ids: np.ndarray) -> pd.DataFrame:
    oof = pd.read_csv(folder / "oof_predictions.csv").set_index("id").loc[train_ids]
    return oof


def load_test_margin(folder: Path, sample_ids: np.ndarray) -> np.ndarray:
    return pd.read_csv(folder / "test_predictions.csv").set_index("id").loc[sample_ids, "margin"].values.astype(float)


def read_labels(path: Path, sample_ids: np.ndarray) -> np.ndarray:
    sub = pd.read_csv(path).set_index("id").loc[sample_ids, "sentiment"].values
    assert set(sub) <= set(LABELS), path
    return sub == LABELS[1]


def fold_records(root: Path, fam: dict) -> list:
    recs = []
    for j in range(5):
        d = root / fam["runs"] / f"{fam['prefix']}-f{j}"
        info = read_json(d / "run_info.json")
        hist = info.get("history", [])
        rec = {"fold": j, "accuracy": info.get("holdout_accuracy"), "auc": info.get("holdout_auc"),
               "total_minutes": info.get("total_minutes"), "timings": info.get("timings", {}),
               "history": [{k: h.get(k) for k in ("epoch", "holdout_accuracy", "holdout_auc", "check_accuracy",
                                                   "check_auc", "phase") if h.get(k) is not None} for h in hist],
               "submission_sha256": info.get("submission", {}).get("sha256")}
        if "fingerprint" in info:
            rec["fingerprint"] = info["fingerprint"]
        accs = [h["holdout_accuracy"] for h in hist if h.get("holdout_accuracy") is not None]
        if accs:
            rec["last_eval_accuracy"] = accs[-1]
            rec["best_eval_accuracy"] = max(accs)
        recs.append(rec)
    return recs


# --------------------------------------------------------------------------- main analysis
def analyse(root: Path, data: Path) -> tuple:
    train = pd.read_csv(data / "train.csv")
    test = pd.read_csv(data / "test.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    y = (train["sentiment"].values == LABELS[1]).astype(int)
    train_ids, sample_ids = train["id"].values, sample["id"].values
    words = train["review"].str.split().str.len().values
    n = len(y)

    # ---- out-of-fold margins of every family (+ NB-LR from stage 1)
    M, T, FOLD = {}, {}, {}
    nb_oof = np.load(root / "results" / "oof_scores_best.npy")
    assert len(nb_oof) == n
    M["nblr"] = nb_oof.astype(float)
    T["nblr"] = pd.read_csv(root / "results" / "test_scores.csv").set_index("id").loc[sample_ids, "score"].values
    for key, fam in FAMILIES.items():
        oof = load_oof(root / fam["oof"], train_ids)
        assert np.array_equal(oof["label"].values.astype(int), y), key
        M[key] = oof["margin"].values.astype(float)
        FOLD[key] = oof["fold"].values.astype(int)
        T[key] = load_test_margin(root / fam["oof"], sample_ids)
    # all families use the same 5-fold partition (split seed 42)
    for key in FOLD:
        assert np.array_equal(FOLD[key], FOLD["deberta"]), key
    fold = FOLD["deberta"]

    fam_stats = {}
    for key in ["nblr", *FAMILIES]:
        correct = (M[key] > 0) == (y == 1)
        st = {"label": FAMILIES[key]["label"] if key in FAMILIES else "NB-LR (Stage 1), 5-fold",
              "oof": acc_block(correct), "metrics": binary_metrics(M[key] > 0, y),
              "by_length": by_length(correct, words), "by_confidence": by_confidence(correct, M[key]),
              "test_positive_share": r4(np.mean(T[key] > 0))}
        if key in FAMILIES:
            st["per_fold"] = [r4(np.mean(correct[fold == j])) for j in range(5)]
            st["per_fold_sd"] = r4(np.std(st["per_fold"], ddof=1))
            st["folds"] = fold_records(root, FAMILIES[key])
            st["cv_summary"] = read_json(root / FAMILIES[key]["oof"] / "cv_summary.json")
        fam_stats[key] = st

    pairs = {}
    for a, b in [("qwen3", "deberta"), ("qwen3", "nblr"), ("deberta", "nblr"), ("deberta", "electra"),
                 ("deberta", "deberta_rerun"), ("qwen3", "electra"), ("qwen3", "deberta_rerun")]:
        pairs[f"{a}_vs_{b}"] = pair(a, M[a], b, M[b], y)
    # per-fold paired comparison Qwen3 vs DeBERTa (the same 5,000 reviews in every fold)
    per_fold_pair = []
    for j in range(5):
        m = fold == j
        p = pair("qwen3", M["qwen3"][m], "deberta", M["deberta"][m], y[m])
        per_fold_pair.append({"fold": j, "qwen3": r4(np.mean((M["qwen3"][m] > 0) == y[m])),
                              "deberta": r4(np.mean((M["deberta"][m] > 0) == y[m])),
                              "qwen3_only_right": p["a_only_right"], "deberta_only_right": p["b_only_right"],
                              "mcnemar_p": p["mcnemar_p"]})

    # ---- combining DeBERTa and Qwen3 (reproduces ensemble_cv.py on the same OOF matrix)
    X = np.column_stack([M["deberta"], M["qwen3"]])
    Xt = np.column_stack([T["deberta"], T["qwen3"]])
    w = fit_logreg(X, y, lam=1e-2)
    sd = X.std(0)
    stack_oof = predict_margin(w, X)
    stack_test = predict_margin(w, Xt)
    s_mean_oof = (X / sd).mean(1)
    stack_info = read_json(root / "ensemble-stage4-stack" / "ensemble_info.json")
    rule_info = read_json(root / "ensemble-stage4" / "ensemble_info.json")
    combo = {
        "options_estimated_accuracy": stack_info["options_estimated_accuracy"],
        "paired_vs_qwen3": stack_info["paired_vs_best_single"],
        "standard_error": stack_info["standard_error"],
        "rule_choice": rule_info["choice"],
        "stack_weights_raw": {"bias": r4(w[0]), "deberta": r4(w[1]), "qwen3": r4(w[2])},
        "stack_weights_recorded": stack_info["weights"],
        "stack_weights_per_sd": {"deberta": r4(w[1] * sd[0]), "qwen3": r4(w[2] * sd[1])},
        "oof_margin_sd": {"deberta": r4(sd[0]), "qwen3": r4(sd[1])},
        "stack_in_sample_oof_accuracy": r4(np.mean((stack_oof > 0) == (y == 1))),
        "stack_vs_qwen3_oof": pair("stack", stack_oof, "qwen3", M["qwen3"], y),
        "s_mean_vs_qwen3_oof": pair("s-mean", s_mean_oof, "qwen3", M["qwen3"], y),
        "stack_test_changed_vs_qwen3": int(np.sum((stack_test > 0) != (T["qwen3"] > 0))),
        "stack_test_changed_recorded": stack_info["test_predictions_changed_vs_best_single"],
        "stack_test_positive_share": r4(np.mean(stack_test > 0)),
    }

    # ---- exact duplicates and the backfill
    tr_text, te_text = train["review"].values, test["review"].values
    groups = train.groupby("review")["sentiment"].agg(["size", "nunique"])
    dup = groups[groups["size"] > 1]
    label_of = dict(zip(tr_text, train["sentiment"].values))
    twin = np.array([t in label_of for t in te_text])
    norm_train = {ft.clean_review(t).lower() for t in tr_text}
    twin_norm = np.array([ft.clean_review(t).lower() in norm_train for t in te_text])
    twin_label = np.array([label_of.get(t) == LABELS[1] if t in label_of else False for t in te_text])
    dups = {"train_duplicate_groups": int(len(dup)), "train_groups_with_one_label": int(np.sum(dup["nunique"] == 1)),
            "train_rows_repeating_an_earlier_row": int(train["review"].duplicated().sum()),
            "test_rows_with_identical_train_review": int(twin.sum()),
            "distinct_texts_shared_by_train_and_test": int(len(set(tr_text) & set(te_text))),
            "test_rows_with_identical_train_review_after_cleaning": int(twin_norm.sum())}
    # how often each model disagreed with the train label of an identical review
    disagree = {}
    for key in ["nblr", "deberta", "electra", "deberta_rerun", "qwen3", "qwen3_14b", "mistral_24b", "qwen38_27b"]:
        disagree[key] = int(np.sum((T[key][twin] > 0) != twin_label[twin]))
    disagree["stack"] = int(np.sum((stack_test[twin] > 0) != twin_label[twin]))
    dups["test_twins_where_model_disagrees_with_train_label"] = disagree
    for tag, folder in (("final_A", "ensemble-stage4"), ("final_B", "ensemble-stage4-stack"),
                        ("rerun", "ensemble-retrain-control"), ("stage5", "ensemble-stage5"),
                        ("stage6", "ensemble-stage6"), ("stage7", "ensemble-stage7")):
        a = read_json(root / folder / "submission_exact.audit.json")
        dups[f"backfill_{tag}"] = {"n_changed": a["n_changed"], "changed": a["changed"],
                                   "output_sha256": a["output_sha256"]}

    # ---- Kaggle submissions
    labels = {}
    kag = []
    for key, desc, rel, public in KAGGLE:
        path = root / rel
        lab = read_labels(path, sample_ids)
        labels[key] = lab
        kag.append({"key": key, "description": desc, "file": rel, "public": public,
                    "sha256": ft.sha256_file(path), "positive": int(lab.sum()), "negative": int((~lab).sum()),
                    "final_selection": key in FINAL_SELECTION})
    diffs = {}
    for a, b in [("stage4_exact", "stack_exact"), ("qwen3_5fold", "deberta_5fold"), ("deberta_5fold", "ensemble2"),
                 ("stack_exact", "deberta_5fold"), ("qwen3_5fold", "stage4"), ("stage4", "stage4_exact"),
                 ("pilot", "qwen3_5fold"), ("rerun_exact", "deberta_5fold"), ("deberta_5fold", "electra_5fold"),
                 ("stack_exact", "nblr"), ("stage5_exact", "stack_exact"), ("stage5_exact", "stage4_exact"),
                 ("stage6_exact", "stage5_exact"), ("stage6_exact", "stack_exact"),
                 ("stage7_exact", "stage6_exact"), ("stage7_exact", "stack_exact")]:
        diffs[f"{a}_vs_{b}"] = int(np.sum(labels[a] != labels[b]))

    # ---- self-training (warm-started pseudo-labelling) record
    pl = read_json(root / "pseudo" / "pseudo.json")
    pl_run = read_json(root / "runs" / "deberta-pl" / "run_info.json")
    s2 = read_json(root / "runs" / "deberta" / "run_info.json")
    self_training = {"n_test_agreeing": pl["n_agreeing"], "n_pseudo": pl["n_kept"], "kept_fraction": pl["kept_fraction"],
                     "pseudo_weight": pl_run["pseudo_weight"], "lr": pl_run["config"]["lr"], "epochs": pl_run["config"]["epochs"],
                     "history": pl_run["history"], "train_minutes": pl_run["timings"]["train_minutes"],
                     "submission_sha256": pl_run["submission"]["sha256"],
                     "identical_to_round1": pl_run["submission"]["sha256"] == s2["submission"]["sha256"]}

    # ---- fresh re-run screening on the 2,500-review hold-out
    screens = []
    for name in ("retrain-screen-control", "retrain-screen-lr7e6", "retrain-screen-lr15e6", "retrain-screen-head256"):
        info = read_json(root / "runs" / name / "run_info.json")
        screens.append({"run": name, "lr": info["config"]["lr"], "epochs": info["config"]["epochs"],
                        "head": info.get("head"), "holdout_accuracy": info["holdout_accuracy"],
                        "zero_step_accuracy": info["history"][0]["holdout_accuracy"] if info.get("history") else None})

    # ---- examples for the discussion: confident disagreements between Qwen3 and DeBERTa
    def excerpt(i: int) -> dict:
        t = ft.clean_review(tr_text[i])
        return {"id": int(train_ids[i]), "label": LABELS[y[i]], "words": int(words[i]),
                "qwen3_margin": round(float(M["qwen3"][i]), 2), "deberta_margin": round(float(M["deberta"][i]), 2),
                "start": t[:500], "end": t[-300:]}
    cq, cd = (M["qwen3"] > 0) == (y == 1), (M["deberta"] > 0) == (y == 1)
    q_fix = np.flatnonzero(cq & ~cd)
    q_fix = q_fix[np.argsort(-(np.abs(M["deberta"][q_fix]) + np.abs(M["qwen3"][q_fix])))]
    d_fix = np.flatnonzero(~cq & cd)
    d_fix = d_fix[np.argsort(-(np.abs(M["deberta"][d_fix]) + np.abs(M["qwen3"][d_fix])))]
    both = np.flatnonzero(~cq & ~cd)
    both = both[np.argsort(-(np.abs(M["deberta"][both]) + np.abs(M["qwen3"][both])))]
    examples = {"qwen3_right_deberta_wrong": [excerpt(i) for i in q_fix[:15]],
                "deberta_right_qwen3_wrong": [excerpt(i) for i in d_fix[:10]],
                "both_wrong_confident": [excerpt(i) for i in both[:15]],
                "both_wrong_total": int(len(both))}

    # ---- Stage 5: Qwen3-14B and the three-family combination (reproduces ensemble_cv.py)
    for a, b in [("qwen3_14b", "qwen3"), ("qwen3_14b", "deberta")]:
        pairs[f"{a}_vs_{b}"] = pair(a, M[a], b, M[b], y)
    per_fold_14b = []
    for j in range(5):
        m = fold == j
        p = pair("qwen3_14b", M["qwen3_14b"][m], "qwen3", M["qwen3"][m], y[m])
        per_fold_14b.append({"fold": j, "qwen3_14b": r4(np.mean((M["qwen3_14b"][m] > 0) == y[m])),
                             "qwen3": r4(np.mean((M["qwen3"][m] > 0) == y[m])),
                             "q14_only_right": p["a_only_right"], "q8_only_right": p["b_only_right"],
                             "mcnemar_p": p["mcnemar_p"]})
    X3 = np.column_stack([M["deberta"], M["qwen3"], M["qwen3_14b"]])
    X3t = np.column_stack([T["deberta"], T["qwen3"], T["qwen3_14b"]])
    sd3 = X3.std(0)
    s3_oof, s3_test = (X3 / sd3).mean(1), (X3t / sd3).mean(1)
    # final B's stack, cross-fitted on the shared folds so that it can be compared row by row with C
    stack_cf = np.zeros(n)
    for j in range(5):
        m = fold == j
        stack_cf[m] = predict_margin(fit_logreg(X[~m], y[~m], lam=1e-2), X[m])
    s5_info = read_json(root / "ensemble-stage5" / "ensemble_info.json")
    w3 = fit_logreg(X3, y, lam=1e-2)
    stage5 = {
        "options_estimated_accuracy": s5_info["options_estimated_accuracy"],
        "paired_vs_qwen3_14b": s5_info["paired_vs_best_single"],
        "standard_error": s5_info["standard_error"], "rule_choice": s5_info["choice"],
        "rule_threshold_stage4_oof": fam_stats["qwen3"]["oof"]["accuracy"],
        "oof_margin_sd": {k: r4(v) for k, v in zip(("deberta", "qwen3", "qwen3_14b"), sd3)},
        "stack3_weights_per_sd": {k: r4(v * sd) for k, v, sd in zip(("deberta", "qwen3", "qwen3_14b"), w3[1:], sd3)},
        "s_mean3_oof_accuracy": r4(np.mean((s3_oof > 0) == (y == 1))),
        # recomputed s-mean labels differ from the uploaded file only on the backfilled rows
        "s_mean3_test_rows_differing_from_upload": int(np.sum((s3_test > 0) != labels["stage5_exact"])),
        "stack2_crossfit_oof_accuracy": r4(np.mean((stack_cf > 0) == (y == 1))),
        "C_vs_B_oof": pair("s-mean3", s3_oof, "stack2_crossfit", stack_cf, y),
        "C_vs_qwen3_oof": pair("s-mean3", s3_oof, "qwen3", M["qwen3"], y),
        "C_vs_qwen3_14b_oof": pair("s-mean3", s3_oof, "qwen3_14b", M["qwen3_14b"], y),
        "test_changed_vs_qwen3_14b": s5_info["test_predictions_changed_vs_best_single"],
        "test_positive_share": r4(np.mean(s3_test > 0)),
        "qwen3_14b_per_fold_vs_qwen3": per_fold_14b,
        "fold_minutes": {k: [r["total_minutes"] for r in fam_stats[k]["folds"]] for k in ("qwen3", "qwen3_14b")},
    }

    # ---- Stage 6: Mistral-Small-24B and the four-family combination (reproduces ensemble_cv.py)
    for a, b in [("mistral_24b", "qwen3_14b"), ("mistral_24b", "qwen3"), ("mistral_24b", "deberta")]:
        pairs[f"{a}_vs_{b}"] = pair(a, M[a], b, M[b], y)
    per_fold_24b = []
    for j in range(5):
        m = fold == j
        p = pair("mistral_24b", M["mistral_24b"][m], "qwen3_14b", M["qwen3_14b"][m], y[m])
        per_fold_24b.append({"fold": j, "mistral_24b": r4(np.mean((M["mistral_24b"][m] > 0) == y[m])),
                             "qwen3_14b": r4(np.mean((M["qwen3_14b"][m] > 0) == y[m])),
                             "m24_only_right": p["a_only_right"], "q14_only_right": p["b_only_right"],
                             "mcnemar_p": p["mcnemar_p"]})
    k4 = ("deberta", "qwen3", "qwen3_14b", "mistral_24b")
    X4 = np.column_stack([M[k] for k in k4])
    X4t = np.column_stack([T[k] for k in k4])
    sd4 = X4.std(0)
    s4_oof, s4_test = (X4 / sd4).mean(1), (X4t / sd4).mean(1)
    s6_info = read_json(root / "ensemble-stage6" / "ensemble_info.json")
    # the fold-3 run that diverged (seed 42) and its replacement (seed 43): both are in train_log.txt
    f3 = root / "runs" / "cv-mistral-24b" / "mistral-24b-f3"
    f3_info = read_json(f3 / "run_info.json")
    f3_log = [ln.strip() for ln in (f3 / "train_log.txt").read_text(encoding="utf-8").splitlines()
              if "VALIDATION accuracy" in ln or "command:" in ln]
    stage6 = {
        "options_estimated_accuracy": s6_info["options_estimated_accuracy"],
        "paired_vs_mistral_24b": s6_info["paired_vs_best_single"],
        "standard_error": s6_info["standard_error"], "rule_choice": s6_info["choice"],
        "rule_threshold": STAGE6_THRESHOLD,
        "rule_met": bool(s6_info["options_estimated_accuracy"][s6_info["choice"]] >= STAGE6_THRESHOLD),
        "oof_margin_sd": {k: r4(v) for k, v in zip(k4, sd4)},
        "oof_margin_correlation": s6_info["oof_margin_correlation"],
        "s_mean4_oof_accuracy": r4(np.mean((s4_oof > 0) == (y == 1))),
        "s_mean4_test_rows_differing_from_upload": int(np.sum((s4_test > 0) != labels["stage6_exact"])),
        "D_vs_C_oof": pair("s-mean4", s4_oof, "s-mean3", s3_oof, y),
        "D_vs_B_oof": pair("s-mean4", s4_oof, "stack2_crossfit", stack_cf, y),
        "D_vs_mistral_24b_oof": pair("s-mean4", s4_oof, "mistral_24b", M["mistral_24b"], y),
        "test_changed_vs_mistral_24b": s6_info["test_predictions_changed_vs_best_single"],
        "test_positive_share": r4(np.mean(s4_test > 0)),
        "mistral_24b_per_fold_vs_qwen3_14b": per_fold_24b,
        "fold_minutes": [r["total_minutes"] for r in fam_stats["mistral_24b"]["folds"]],
        "gpu": read_json(f3 / "run_info.json")["environment"].get("gpu"),
        "seeds": [read_json(root / "runs" / "cv-mistral-24b" / f"mistral-24b-f{j}" / "run_info.json")
                  .get("seed") for j in range(5)],
        "fold3_seed": f3_info.get("seed"),
        "fold3_log": f3_log,
    }

    # ---- Stage 7: Qwen3.8-27B (post-trained) and the five-family combination (reproduces ensemble_cv.py)
    for a, b in [("qwen38_27b", "mistral_24b"), ("qwen38_27b", "qwen3_14b"), ("qwen38_27b", "qwen3"),
                 ("qwen38_27b", "deberta")]:
        pairs[f"{a}_vs_{b}"] = pair(a, M[a], b, M[b], y)
    per_fold_27b = []
    for j in range(5):
        m = fold == j
        p = pair("qwen38_27b", M["qwen38_27b"][m], "mistral_24b", M["mistral_24b"][m], y[m])
        per_fold_27b.append({"fold": j, "qwen38_27b": r4(np.mean((M["qwen38_27b"][m] > 0) == y[m])),
                             "mistral_24b": r4(np.mean((M["mistral_24b"][m] > 0) == y[m])),
                             "q27_only_right": p["a_only_right"], "m24_only_right": p["b_only_right"],
                             "mcnemar_p": p["mcnemar_p"]})
    k5 = (*k4, "qwen38_27b")
    X5 = np.column_stack([M[k] for k in k5])
    X5t = np.column_stack([T[k] for k in k5])
    sd5 = X5.std(0)
    s5_oof, s5_test = (X5 / sd5).mean(1), (X5t / sd5).mean(1)
    s7_info = read_json(root / "ensemble-stage7" / "ensemble_info.json")
    q27 = [read_json(root / "runs" / "cv-qwen38-27b" / f"qwen38-27b-f{j}" / "run_info.json") for j in range(5)]
    stage7 = {
        "options_estimated_accuracy": s7_info["options_estimated_accuracy"],
        "paired_vs_qwen38_27b": s7_info["paired_vs_best_single"],
        "standard_error": s7_info["standard_error"], "rule_choice": s7_info["choice"],
        "rule_threshold": STAGE7_THRESHOLD,
        "rule_met": bool(s7_info["options_estimated_accuracy"][s7_info["choice"]] >= STAGE7_THRESHOLD),
        "oof_margin_sd": {k: r4(v) for k, v in zip(k5, sd5)},
        "oof_margin_correlation": s7_info["oof_margin_correlation"],
        "s_mean5_oof_accuracy": r4(np.mean((s5_oof > 0) == (y == 1))),
        "s_mean5_test_rows_differing_from_upload": int(np.sum((s5_test > 0) != labels["stage7_exact"])),
        "E_vs_D_oof": pair("s-mean5", s5_oof, "s-mean4", s4_oof, y),
        "E_vs_B_oof": pair("s-mean5", s5_oof, "stack2_crossfit", stack_cf, y),
        "E_vs_qwen38_27b_oof": pair("s-mean5", s5_oof, "qwen38_27b", M["qwen38_27b"], y),
        "test_changed_vs_qwen38_27b": s7_info["test_predictions_changed_vs_best_single"],
        "test_positive_share": r4(np.mean(s5_test > 0)),
        "qwen38_27b_per_fold_vs_mistral_24b": per_fold_27b,
        "fold_minutes": [r["total_minutes"] for r in fam_stats["qwen38_27b"]["folds"]],
        "fold_timings": [r["timings"] for r in fam_stats["qwen38_27b"]["folds"]],
        "zero_shot_check": [r["history"][0] for r in q27],
        "mid_epoch_accuracy": [next(h["holdout_accuracy"] for h in r["history"] if h.get("epoch") == 0.5) for r in q27],
        "environment": q27[0]["environment"],
        "seeds": [r.get("seed") for r in q27],
        "preset": q27[0].get("preset"),
    }

    summary = {
        "n_train": n, "families": fam_stats, "pairs": pairs, "qwen3_vs_deberta_per_fold": per_fold_pair,
        "combination": combo, "stage5": stage5, "stage6": stage6, "stage7": stage7,
        "stage4_selection": list(STAGE4_SELECTION), "stage5_selection": list(STAGE5_SELECTION),
        "stage6_selection": list(STAGE6_SELECTION), "duplicates": dups, "kaggle": kag, "kaggle_label_differences": diffs,
        "self_training": self_training, "rerun_screening": screens, "examples": examples,
        "final_selection": list(FINAL_SELECTION),
    }
    arrays = {"M": M, "T": T, "y": y, "words": words, "fold": fold}
    return summary, arrays


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--root", default=".", help="the submission folder (holds results/, runs/, ensemble-*/)")
    p.add_argument("--output", default="results/stage34")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    root, out = Path(args.root), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary, arrays = analyse(root, Path(args.data_dir))
    (out / "stage34_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out / 'stage34_summary.json'}")
    if not args.no_plots:
        import stage34_plots  # noqa: E402  (matplotlib only when figures are requested)
        stage34_plots.make_all(summary, arrays, out / "figures")


if __name__ == "__main__":
    main()
