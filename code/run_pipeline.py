#!/usr/bin/env python3
"""End-to-end pipeline for the CDS550 individual project (IMDB sentiment).

Stages (all deterministic, seed = 42):

1. ``eda``      - dataset statistics and figures.
2. ``cv``       - stratified 5-fold cross-validation of every candidate model
                  (naive Bayes, TF-IDF logistic regression, NB-LR) and feature
                  configuration; writes ``cv_results.csv`` and figures.
3. ``final``    - refits the best configuration on all 25,000 training reviews,
                  predicts the 25,000 test reviews and writes ``submission.csv``
                  in the Kaggle format, plus analysis artefacts.

Usage::

    python run_pipeline.py --data-dir ../data --output ../results
    python run_pipeline.py --data-dir ../data --output ../results --stage final   # only refit + predict

Only numpy, pandas and matplotlib are required; the learning algorithms live in
the ``sentiment`` package next to this file. Use ``--no-plots`` to write numeric
results without importing matplotlib.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sentiment.evaluation import accuracy, classification_metrics, stratified_kfold  # noqa: E402
from sentiment.features import NgramCountVectorizer, TfidfTransformer, tokenize  # noqa: E402
from sentiment.models import LogisticRegression, MultinomialNB, NBLogisticRegression  # noqa: E402

from sentiment.plots import (  # noqa: E402
    plot_class_balance, plot_confusion_matrix, plot_error_analysis,
    plot_learning_curve, plot_model_comparison, plot_nblr_grid,
    plot_polar_ngrams, plot_review_lengths, plot_test_scores, plot_top_features,
)

SEED = 42
LABELS = {"negative": 0, "positive": 1}
INV_LABELS = {v: k for k, v in LABELS.items()}


# ----------------------------------------------------------------------------
# Candidate models
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    name: str          # short label used in tables
    family: str        # "nb", "nb_binary", "lr_tfidf", "nblr"
    ngram_max: int     # 2 -> unigrams+bigrams, 3 -> up to trigrams
    C: float = 1.0
    beta: float = 1.0
    alpha: float = 1.0

    def describe(self) -> str:
        grams = {1: "uni", 2: "uni+bi", 3: "uni+bi+tri"}[self.ngram_max]
        if self.family == "nb":
            return f"Multinomial NB (counts, {grams}, alpha={self.alpha})"
        if self.family == "nb_binary":
            return f"Multinomial NB (binary, {grams}, alpha={self.alpha})"
        if self.family == "lr_tfidf":
            return f"TF-IDF + Logistic Regression ({grams}, C={self.C})"
        return f"NB-LR ({grams}, C={self.C}, beta={self.beta})"


def build_candidates(quick: bool = False) -> list[Candidate]:
    cands: list[Candidate] = []
    for ng in ([2] if quick else [1, 2, 3]):
        cands.append(Candidate(f"nb_counts_{ng}g", "nb", ng))
        cands.append(Candidate(f"nb_binary_{ng}g", "nb_binary", ng))
        for C in ([16.0] if quick else [4.0, 16.0, 64.0]):
            cands.append(Candidate(f"lr_tfidf_{ng}g_C{C:g}", "lr_tfidf", ng, C=C))
        for C in ([0.3] if quick else [0.1, 0.3, 1.0]):
            for beta in ([0.5] if quick else [0.25, 0.5, 0.75]):
                cands.append(Candidate(f"nblr_{ng}g_C{C:g}_b{beta:g}", "nblr", ng, C=C, beta=beta))
    return cands


class FittedModel:
    """Bundles the vectoriser, optional TF-IDF transformer and classifier."""

    def __init__(self, cand: Candidate):
        self.cand = cand
        self.vectorizer = NgramCountVectorizer(ngram_range=(1, cand.ngram_max), min_df=2, max_df=0.95)
        self.tfidf: TfidfTransformer | None = None
        self.model = None

    def fit(self, texts, y, counts=None):
        X = self.vectorizer.fit_transform(texts) if counts is None else counts
        self._fit_from_counts(X, y)
        return self

    def _fit_from_counts(self, X, y):
        c = self.cand
        if c.family == "nb":
            self.model = MultinomialNB(alpha=c.alpha).fit(X, y)
        elif c.family == "nb_binary":
            self.model = MultinomialNB(alpha=c.alpha).fit(X.binarize(), y)
        elif c.family == "lr_tfidf":
            self.tfidf = TfidfTransformer()
            self.model = LogisticRegression(C=c.C).fit(self.tfidf.fit_transform(X), y)
        elif c.family == "nblr":
            self.model = NBLogisticRegression(C=c.C, beta=c.beta, alpha=c.alpha).fit(X, y)
        else:
            raise ValueError(c.family)

    def decision_function(self, texts=None, counts=None):
        X = self.vectorizer.transform(texts) if counts is None else counts
        c = self.cand
        if c.family == "nb_binary":
            X = X.binarize()
        elif c.family == "lr_tfidf":
            X = self.tfidf.transform(X)
        return self.model.decision_function(X)

    def predict(self, texts=None, counts=None):
        return (self.decision_function(texts, counts) > 0).astype(np.int64)


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_data(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    assert list(train.columns) == ["id", "review", "sentiment"], train.columns
    assert list(test.columns) == ["id", "review"], test.columns
    assert set(train.sentiment.unique()) == set(LABELS)
    assert (sample.id.values == test.id.values).all(), "sample_submission ids differ from test ids"
    return train, test, sample


# ----------------------------------------------------------------------------
# Stage 1: exploratory data analysis
# ----------------------------------------------------------------------------
def run_eda(train: pd.DataFrame, test: pd.DataFrame, out: Path, make_plots: bool = True) -> dict:
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    tr_words = train.review.str.split().str.len()
    te_words = test.review.str.split().str.len()
    tr_chars = train.review.str.len()
    stats = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "class_counts": train.sentiment.value_counts().to_dict(),
        "class_balance_positive_share": float((train.sentiment == "positive").mean()),
        "missing_values_train": int(train.isna().sum().sum()),
        "missing_values_test": int(test.isna().sum().sum()),
        "duplicate_reviews_train": int(train.review.duplicated().sum()),
        "duplicate_reviews_test": int(test.review.duplicated().sum()),
        "train_test_text_overlap": int(len(set(train.review) & set(test.review))),
        "conflicting_duplicate_labels_train": int(
            (train.groupby("review").sentiment.nunique() > 1).sum()),
        "share_with_html_br_train": float(train.review.str.contains("<br />").mean()),
        "words_per_review_train": {k: float(v) for k, v in tr_words.describe().items()},
        "words_per_review_test": {k: float(v) for k, v in te_words.describe().items()},
        "chars_per_review_train": {k: float(v) for k, v in tr_chars.describe().items()},
        "mean_words_by_class": train.assign(w=tr_words).groupby("sentiment").w.mean().round(2).to_dict(),
        "median_words_by_class": train.assign(w=tr_words).groupby("sentiment").w.median().to_dict(),
        "id_range_train": [int(train.id.min()), int(train.id.max())],
        "id_range_test": [int(test.id.min()), int(test.id.max())],
    }

    if make_plots:
        plot_class_balance(train.sentiment.value_counts().reindex(["negative", "positive"]), fig_dir)
        plot_review_lengths(train.sentiment, tr_words, te_words, fig_dir)

    # Figure 3: most polar unigrams/bigrams by NB log-count ratio
    vec = NgramCountVectorizer(ngram_range=(1, 2), min_df=20, max_df=0.95)
    X = vec.fit_transform(train.review.values).binarize()
    y = train.sentiment.map(LABELS).values
    p = X.column_sums(y == 1) + 1.0
    q = X.column_sums(y == 0) + 1.0
    r = np.log(p / p.sum()) - np.log(q / q.sum())
    order = np.argsort(r)
    top_neg = [(vec.feature_names_[i], float(r[i]), int(vec.document_frequency_[i])) for i in order[:20]]
    top_pos = [(vec.feature_names_[i], float(r[i]), int(vec.document_frequency_[i])) for i in order[::-1][:20]]
    stats["most_negative_ngrams"] = top_neg
    stats["most_positive_ngrams"] = top_pos
    stats["vocabulary_size_min_df20_uni_bi"] = int(len(vec.feature_names_))

    if make_plots:
        plot_polar_ngrams(top_neg, top_pos, fig_dir)

    # Vocabulary growth statistics for the report
    tokens_per_doc = np.array([len(tokenize(t)) for t in train.review.values])
    stats["tokens_per_review_after_cleaning"] = {
        "mean": float(tokens_per_doc.mean()), "median": float(np.median(tokens_per_doc)),
        "min": int(tokens_per_doc.min()), "max": int(tokens_per_doc.max())}
    # Pruning depends on each term's DF, so a single 1..3-gram fit gives
    # the same cumulative vocabulary sizes as three separate fits.
    v = NgramCountVectorizer(ngram_range=(1, 3), min_df=2, max_df=0.95).fit(train.review.values)
    sizes_by_order = np.bincount([term.count(" ") for term in v.feature_names_], minlength=3)
    for ng, size in enumerate(np.cumsum(sizes_by_order), start=1):
        stats[f"vocabulary_size_min_df2_ngram_1_{ng}"] = int(size)

    with open(out / "eda.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    log(f"EDA done: {stats['class_counts']}, mean words {stats['words_per_review_train']['mean']:.1f}")
    return stats


# ----------------------------------------------------------------------------
# Stage 2: cross-validated model comparison
# ----------------------------------------------------------------------------
def run_cv(train: pd.DataFrame, out: Path, quick: bool, time_budget: float | None = None,
           make_plots: bool = True) -> tuple[pd.DataFrame, dict]:
    start_time = time.time()
    texts = train.review.values
    y = train.sentiment.map(LABELS).values
    folds = stratified_kfold(y, n_splits=5, seed=SEED)
    cands = build_candidates(quick)
    log(f"Cross-validating {len(cands)} candidates on {len(folds)} folds")

    # Every (fold, n-gram order) block is check-pointed to ``cache/`` so a long
    # run can be interrupted and resumed; finished blocks are loaded, not recomputed.
    cache_dir = out / "cache"
    cache_dir.mkdir(exist_ok=True)
    fold_acc = {c.name: [] for c in cands}
    oof_scores = {c.name: np.zeros(len(y)) for c in cands}
    fit_seconds = {c.name: 0.0 for c in cands}
    n_features_by_ngram: dict[int, list[int]] = {}
    for k, (tr_idx, va_idx) in enumerate(folds):
        for ng in sorted({c.ngram_max for c in cands}):
            block = [c for c in cands if c.ngram_max == ng]
            cache_file = cache_dir / f"cv_fold{k}_ng{ng}{'_quick' if quick else ''}.npz"
            if cache_file.exists():
                cached = np.load(cache_file, allow_pickle=True)
                if all(c.name in cached.files for c in block):
                    for c in block:
                        oof_scores[c.name][va_idx] = cached[c.name]
                        fold_acc[c.name].append(accuracy(y[va_idx], (cached[c.name] > 0).astype(int)))
                        fit_seconds[c.name] += float(cached["__fit_seconds__"].item().get(c.name, 0.0))
                    n_features_by_ngram.setdefault(ng, []).append(int(cached["__n_features__"]))
                    log(f"  fold {k + 1}, n-grams 1..{ng}: loaded from cache")
                    continue
            if time_budget is not None and time.time() - start_time > time_budget:
                log(f"Time budget reached (fold {k + 1}, n-grams 1..{ng} pending); re-run the same command to continue.")
                sys.exit(0)
            # The vocabulary is rebuilt inside every fold from the training part
            # only, so the validation reviews never influence feature selection.
            t0 = time.time()
            vec = NgramCountVectorizer(ngram_range=(1, ng), min_df=2, max_df=0.95)
            Xtr = vec.fit_transform(texts[tr_idx])
            Xva = vec.transform(texts[va_idx])
            n_features_by_ngram.setdefault(ng, []).append(Xtr.shape[1])
            log(f"  fold {k + 1}: n-grams 1..{ng} -> {Xtr.shape[1]:,} features ({time.time() - t0:.1f}s)")
            block_scores: dict[str, np.ndarray] = {}
            block_fit: dict[str, float] = {}
            for c in block:
                fm = FittedModel(c)
                fm.vectorizer = vec
                t0 = time.time()
                fm._fit_from_counts(Xtr, y[tr_idx])
                block_fit[c.name] = time.time() - t0
                fit_seconds[c.name] += block_fit[c.name]
                s = fm.decision_function(counts=Xva)
                block_scores[c.name] = s
                oof_scores[c.name][va_idx] = s
                fold_acc[c.name].append(accuracy(y[va_idx], (s > 0).astype(int)))
            np.savez(cache_file, __fit_seconds__=np.array(block_fit, dtype=object),
                     __n_features__=np.int64(Xtr.shape[1]), **block_scores)
            best_in_block = max(block, key=lambda c: fold_acc[c.name][-1])
            log(f"  fold {k + 1}, n-grams 1..{ng} done ({len(block)} models); best {best_in_block.name} = {fold_acc[best_in_block.name][-1]:.4f}")

    rows = []
    for c in cands:
        accs = np.array(fold_acc[c.name])
        oof_pred = (oof_scores[c.name] > 0).astype(int)
        rows.append({**asdict(c), "description": c.describe(), "cv_mean_accuracy": accs.mean(),
                     "cv_std_accuracy": accs.std(ddof=1), "oof_accuracy": accuracy(y, oof_pred),
                     **{f"fold{i + 1}": a for i, a in enumerate(accs)},
                     "mean_fit_seconds_per_fold": fit_seconds[c.name] / len(folds)})
    results = pd.DataFrame(rows).sort_values("cv_mean_accuracy", ascending=False).reset_index(drop=True)
    results.to_csv(out / "cv_results.csv", index=False)
    best = results.iloc[0]
    log(f"Best candidate: {best['name']} ({best['description']}) CV accuracy {best['cv_mean_accuracy']:.4f}")

    # Figure: best configuration of each model family
    fam_order = ["nb", "nb_binary", "lr_tfidf", "nblr"]
    best_per_family = results.sort_values("cv_mean_accuracy", ascending=False).groupby("family").head(1)
    best_per_family = best_per_family.set_index("family").loc[[f for f in fam_order if f in set(results.family)]]
    if make_plots:
        plot_model_comparison(best_per_family, out)
        plot_nblr_grid(results, out)

    # Out-of-fold diagnostics for the best candidate
    best_name = best["name"]
    oof = oof_scores[best_name]
    oof_pred = (oof > 0).astype(int)
    metrics = classification_metrics(y, oof_pred)
    words = train.review.str.split().str.len().values
    length_bins = [0, 100, 150, 200, 300, 500, 10_000]
    length_labels = ["<100", "100-149", "150-199", "200-299", "300-499", "500+"]
    acc_by_len = []
    for lo, hi, lab in zip(length_bins[:-1], length_bins[1:], length_labels):
        m = (words >= lo) & (words < hi)
        acc_by_len.append({"bucket": lab, "n": int(m.sum()), "accuracy": accuracy(y[m], oof_pred[m])})
    # Confidence vs accuracy
    conf = np.abs(oof)
    q = np.quantile(conf, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    acc_by_conf = []
    for lo, hi in zip(q[:-1], q[1:]):
        m = (conf >= lo) & (conf <= hi)
        acc_by_conf.append({"score_range": [float(lo), float(hi)], "n": int(m.sum()), "accuracy": accuracy(y[m], oof_pred[m])})
    err = train.assign(score=oof, pred=oof_pred, true=y)
    err = err[err.pred != err.true].assign(conf=lambda d: d.score.abs()).sort_values("conf", ascending=False)
    examples = [{"id": int(r.id), "true": INV_LABELS[int(r.true)], "pred": INV_LABELS[int(r.pred)],
                 "score": float(r.score), "excerpt": r.review[:400]} for r in err.head(8).itertuples()]
    np.save(out / "oof_scores_best.npy", oof)

    if make_plots:
        plot_confusion_matrix(metrics, out)
        plot_error_analysis(acc_by_len, acc_by_conf, out)

    # Learning curve for the best candidate (single stratified split, growing training size)
    best_cand = next(c for c in cands if c.name == best_name)
    tr_idx, va_idx = folds[0]
    rng = np.random.RandomState(SEED)
    shuffled = tr_idx.copy(); rng.shuffle(shuffled)
    curve_file = cache_dir / f"learning_curve_{best_name}.json"
    curve = json.load(open(curve_file)) if curve_file.exists() else []
    done_sizes = {c["train_size"] for c in curve}
    for frac in ([0.25, 1.0] if quick else [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]):
        n = int(len(shuffled) * frac)
        if n in done_sizes:
            continue
        if time_budget is not None and time.time() - start_time > time_budget:
            log("Time budget reached during the learning curve; re-run the same command to continue.")
            sys.exit(0)
        sub = shuffled[:n]
        fm = FittedModel(best_cand).fit(texts[sub], y[sub])
        curve.append({"train_size": int(n), "valid_accuracy": accuracy(y[va_idx], fm.predict(texts[va_idx])),
                      "train_accuracy": accuracy(y[sub], fm.predict(texts[sub]))})
        json.dump(curve, open(curve_file, "w"))
        log(f"  learning curve: n={n:,} valid acc {curve[-1]['valid_accuracy']:.4f}")
    curve.sort(key=lambda c: c["train_size"])
    if make_plots:
        plot_learning_curve(curve, best_cand.describe(), out)

    summary = {
        "n_candidates": len(cands), "n_folds": len(folds), "best_candidate": asdict(best_cand),
        "best_description": best_cand.describe(),
        "best_cv_mean_accuracy": float(best["cv_mean_accuracy"]), "best_cv_std_accuracy": float(best["cv_std_accuracy"]),
        "best_fold_accuracies": [float(best[f"fold{i + 1}"]) for i in range(len(folds))],
        "best_oof_metrics": metrics, "accuracy_by_length": acc_by_len, "accuracy_by_confidence": acc_by_conf,
        "hardest_misclassified_examples": examples, "learning_curve": curve,
        "mean_features_by_ngram": {str(k): float(np.mean(v)) for k, v in sorted(n_features_by_ngram.items())},
        "best_per_family": best_per_family.reset_index()[["family", "name", "description", "cv_mean_accuracy", "cv_std_accuracy"]].to_dict("records"),
    }
    with open(out / "cv_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    return results, summary


# ----------------------------------------------------------------------------
# Stage 3: final fit and Kaggle submission
# ----------------------------------------------------------------------------
def run_final(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame, out: Path, cand: Candidate,
              make_plots: bool = True) -> dict:
    y = train.sentiment.map(LABELS).values
    t0 = time.time()
    fm = FittedModel(cand).fit(train.review.values, y)
    fit_s = time.time() - t0
    train_acc = accuracy(y, fm.predict(train.review.values))
    t0 = time.time()
    scores = fm.decision_function(test.review.values)
    pred_s = time.time() - t0
    pred = (scores > 0).astype(int)

    submission = pd.DataFrame({"id": test.id.values, "sentiment": [INV_LABELS[int(p)] for p in pred]})
    assert len(submission) == 25_000 and submission.id.is_unique
    assert set(submission.sentiment) <= {"positive", "negative"}
    assert (submission.id.values == sample.id.values).all()
    submission.to_csv(out / "submission.csv", index=False)
    pd.DataFrame({"id": test.id.values, "score": scores, "p_positive": 1 / (1 + np.exp(-scores))}).to_csv(
        out / "test_scores.csv", index=False)

    # Most influential features of the final model
    names = fm.vectorizer.feature_names_
    if cand.family in ("nb", "nb_binary"):
        w = fm.model.feature_log_prob_[1] - fm.model.feature_log_prob_[0]
    elif cand.family == "nblr":
        w = fm.model.coef_ * fm.model.r_  # weight applied to the binary presence feature
    else:
        w = fm.model.coef_
    order = np.argsort(w)
    top_features = {"negative": [(names[i], float(w[i])) for i in order[:25]],
                    "positive": [(names[i], float(w[i])) for i in order[::-1][:25]]}
    if make_plots:
        plot_top_features(top_features, out)
        plot_test_scores(scores, out)

    info = {
        "candidate": asdict(cand), "description": cand.describe(),
        "n_features": int(len(names)), "train_accuracy_resubstitution": train_acc,
        "fit_seconds": fit_s, "predict_seconds": pred_s,
        "test_predicted_positive_share": float(pred.mean()),
        "test_predictions": {"positive": int(pred.sum()), "negative": int(len(pred) - pred.sum())},
        "submission_sha256": sha256(out / "submission.csv"),
        "top_features": top_features,
        "lbfgs_iterations": len(getattr(getattr(fm.model, "lr_", fm.model), "loss_trace_", [])) - 1,
    }
    with open(out / "final_model.json", "w") as fh:
        json.dump(info, fh, indent=2)
    log(f"Final model fitted in {fit_s:.1f}s, {info['n_features']:,} features; "
        f"test positive share {info['test_predicted_positive_share']:.4f}; submission.csv written")
    return info


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, required=True, help="folder with train.csv, test.csv, sample_submission.csv")
    ap.add_argument("--output", type=Path, required=True, help="folder for results, figures and submission.csv")
    ap.add_argument("--stage", choices=["all", "eda", "cv", "final"], default="all")
    ap.add_argument("--no-plots", action="store_true",
                    help="write numeric results only; matplotlib is not needed")
    ap.add_argument("--quick", action="store_true", help="small candidate grid (smoke test)")
    ap.add_argument("--time-budget-min", type=float, default=None,
                    help="stop after the current CV fold once this many minutes have elapsed; "
                         "re-running the same command resumes from the cached folds")
    args = ap.parse_args()

    out: Path = args.output
    (out / "figures").mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    started = time.time()
    train, test, sample = load_data(args.data_dir)
    log(f"Loaded {len(train):,} training and {len(test):,} test reviews")

    if args.stage == "eda" or (args.stage == "all" and not (out / "eda.json").exists()):
        run_eda(train, test, out, make_plots=not args.no_plots)
    elif args.stage == "all":
        log("EDA already present in output folder; skipping (use --stage eda to redo)")
    cv_summary = None
    if args.stage in ("all", "cv"):
        budget = args.time_budget_min * 60 if args.time_budget_min else None
        _, cv_summary = run_cv(train, out, args.quick, time_budget=budget, make_plots=not args.no_plots)
    if args.stage in ("all", "final"):
        if cv_summary is None:
            with open(out / "cv_summary.json") as fh:
                cv_summary = json.load(fh)
        cand = Candidate(**cv_summary["best_candidate"])
        run_final(train, test, sample, out, cand, make_plots=not args.no_plots)

    env = {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
        "seed": SEED, "stage": args.stage, "quick": args.quick, "plots": not args.no_plots,
        "train_sha256": sha256(args.data_dir / "train.csv"), "test_sha256": sha256(args.data_dir / "test.csv"),
        "total_seconds": time.time() - started, "command": sys.argv,
    }
    with open(out / "run_environment.json", "w") as fh:
        json.dump(env, fh, indent=2)
    log(f"Finished in {env['total_seconds'] / 60:.1f} min")


if __name__ == "__main__":
    main()
