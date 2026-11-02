"""Figures for the sentiment study, separated from experiment logic.

Matplotlib is loaded only when a figure is requested.
"""
from __future__ import annotations

import numpy as np


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_class_balance(counts, fig_dir):
    plt = _pyplot()
    # Figure 1: class balance
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    bars = ax.bar(counts.index, counts.values, color=["#c0504d", "#4f81bd"])
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 100, f"{int(b.get_height()):,}", ha="center", fontsize=9)
    ax.set_ylabel("Number of training reviews")
    ax.set_title("Class balance (train.csv)")
    ax.set_ylim(0, counts.max() * 1.12)
    fig.tight_layout(); fig.savefig(fig_dir / "class_balance.png", dpi=160); plt.close(fig)


def plot_review_lengths(labels, tr_words, te_words, fig_dir):
    plt = _pyplot()
    # Figure 2: review length distribution by class (log x)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    bins = np.logspace(np.log10(5), np.log10(2500), 50)
    for label, color in (("negative", "#c0504d"), ("positive", "#4f81bd")):
        ax.hist(tr_words[labels == label], bins=bins, alpha=0.55, label=label, color=color)
    ax.hist(te_words, bins=bins, histtype="step", color="black", lw=1.0, label="test (unlabelled)")
    ax.set_xscale("log"); ax.set_xlabel("Words per review (log scale)"); ax.set_ylabel("Reviews")
    ax.set_title("Review length distribution"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "review_length.png", dpi=160); plt.close(fig)


def plot_polar_ngrams(top_neg, top_pos, fig_dir):
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4), sharex=False)
    for ax, items, title, color in ((axes[0], top_neg, "Most negative n-grams", "#c0504d"),
                                    (axes[1], top_pos, "Most positive n-grams", "#4f81bd")):
        names = [t for t, _, _ in items][::-1]
        vals = [abs(v) for _, v, _ in items][::-1]
        ax.barh(names, vals, color=color)
        ax.set_title(title, fontsize=10); ax.set_xlabel("|log-count ratio|", fontsize=9)
        ax.tick_params(labelsize=8)
    fig.tight_layout(); fig.savefig(fig_dir / "polar_ngrams.png", dpi=160); plt.close(fig)


def plot_model_comparison(best_per_family, out):
    plt = _pyplot()
    fam_label = {"nb": "Naive Bayes\n(counts)", "nb_binary": "Naive Bayes\n(binary)",
                 "lr_tfidf": "TF-IDF +\nLogistic Reg.", "nblr": "NB-LR"}
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    xs = np.arange(len(best_per_family))
    ax.bar(xs, best_per_family.cv_mean_accuracy, yerr=best_per_family.cv_std_accuracy, capsize=4,
           color=["#9bbb59", "#8064a2", "#f79646", "#4f81bd"][: len(xs)])
    for x, v in zip(xs, best_per_family.cv_mean_accuracy):
        ax.text(x, v + 0.004, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels([fam_label[f] for f in best_per_family.index], fontsize=9)
    ax.set_ylim(0.84, max(0.95, best_per_family.cv_mean_accuracy.max() + 0.02))
    ax.set_ylabel("5-fold CV accuracy"); ax.set_title("Best configuration per model family")
    fig.tight_layout(); fig.savefig(out / "figures" / "model_comparison.png", dpi=160); plt.close(fig)


def plot_nblr_grid(results, out):
    plt = _pyplot()
    # Figure: NB-LR hyper-parameter grid (uni+bi)
    grid = results[(results.family == "nblr")]
    if len(grid) > 1:
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
        for ng, marker in ((1, "^"), (2, "o"), (3, "s")):
            g = grid[grid.ngram_max == ng]
            for beta, color in ((0.25, "#c0504d"), (0.5, "#4f81bd"), (0.75, "#9bbb59"), (1.0, "#8064a2")):
                gg = g[g.beta == beta].sort_values("C")
                if len(gg):
                    ax.plot(gg.C, gg.cv_mean_accuracy, marker=marker, color=color, lw=1,
                            label=f"n-grams 1..{ng}, beta={beta:g}")
        ax.set_xscale("log"); ax.set_xlabel("C (inverse L2 regularisation)"); ax.set_ylabel("5-fold CV accuracy")
        ax.set_title("NB-LR hyper-parameter sensitivity"); ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); fig.savefig(out / "figures" / "nblr_grid.png", dpi=160); plt.close(fig)


def plot_confusion_matrix(metrics, out):
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    cm = np.array(metrics["confusion_matrix"])
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_xticklabels(["negative", "positive"]); ax.set_yticklabels(["negative", "positive"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Out-of-fold confusion matrix")
    fig.tight_layout(); fig.savefig(out / "figures" / "confusion_matrix.png", dpi=160); plt.close(fig)


def plot_error_analysis(acc_by_len, acc_by_conf, out):
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    axes[0].bar([a["bucket"] for a in acc_by_len], [a["accuracy"] for a in acc_by_len], color="#4f81bd")
    axes[0].set_ylim(0.8, 1.0); axes[0].set_xlabel("Words per review"); axes[0].set_ylabel("OOF accuracy")
    axes[0].set_title("Accuracy by review length"); axes[0].tick_params(axis="x", labelsize=8)
    axes[1].bar([f"Q{i + 1}" for i in range(len(acc_by_conf))], [a["accuracy"] for a in acc_by_conf], color="#f79646")
    axes[1].set_ylim(0.6, 1.02); axes[1].set_xlabel("Confidence quintile (|score|, low -> high)")
    axes[1].set_title("Accuracy by model confidence")
    fig.tight_layout(); fig.savefig(out / "figures" / "error_analysis.png", dpi=160); plt.close(fig)


def plot_learning_curve(curve, description, out):
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.plot([c["train_size"] for c in curve], [c["valid_accuracy"] for c in curve], marker="o", label="validation")
    ax.plot([c["train_size"] for c in curve], [c["train_accuracy"] for c in curve], marker="s", ls="--", label="training")
    ax.set_xlabel("Training reviews"); ax.set_ylabel("Accuracy"); ax.set_title(f"Learning curve ({description})", fontsize=9)
    ax.legend(); fig.tight_layout(); fig.savefig(out / "figures" / "learning_curve.png", dpi=160); plt.close(fig)


def plot_top_features(top_features, out):
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.6))
    for ax, key, color in ((axes[0], "negative", "#c0504d"), (axes[1], "positive", "#4f81bd")):
        items = top_features[key][:20][::-1]
        ax.barh([n for n, _ in items], [abs(v) for _, v in items], color=color)
        ax.set_title(f"Strongest {key} features (final model)", fontsize=10); ax.tick_params(labelsize=8)
        ax.set_xlabel("|effective weight|", fontsize=9)
    fig.tight_layout(); fig.savefig(out / "figures" / "top_features.png", dpi=160); plt.close(fig)


def plot_test_scores(scores, out):
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.hist(scores, bins=80, color="#4f81bd")
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("Decision score on test reviews"); ax.set_ylabel("Reviews"); ax.set_title("Test-set score distribution")
    fig.tight_layout(); fig.savefig(out / "figures" / "test_score_distribution.png", dpi=160); plt.close(fig)
