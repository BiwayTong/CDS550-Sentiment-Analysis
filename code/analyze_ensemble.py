#!/usr/bin/env python3
"""Analysis of the multi-run ensemble (several transformer runs + NB-LR). NumPy/pandas/matplotlib only.

Reads every ``runs/<name>/`` folder (from finetune_transformer.py), NB-LR's out-of-fold and test
scores and the ensemble folder written by ensemble.py, and writes
``results/stage2/ensemble2_summary.json`` and ``results/stage2/figures/ensemble_models.png``.

Besides per-run statistics it reports an *ablation*: the cross-validated accuracy of the mean
combiner for every subset of models. This table is descriptive only. The submitted ensemble was
fixed in advance as "all models", because picking the best of ~26 subsets on the same 2,500
reviews would over-fit the hold-out.

    python code/analyze_ensemble.py --data-dir data \
        --runs runs/deberta runs/deberta-large-s1 runs/deberta-large-s2 runs/roberta-large \
        --nblr-results results --ensemble ensemble2 --previous-ensemble ensemble --output results/stage2
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import ensemble as ens

LABELS = ("negative", "positive")


def wilson(k: int, n: int, z: float = 1.96) -> list:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b vs c discordant pairs."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mean_combiner(X_fit: np.ndarray, X_apply: np.ndarray) -> np.ndarray:
    """Average of margins standardised with the statistics of X_fit (as in ensemble.py)."""
    mu, sd = X_fit.mean(0), X_fit.std(0) + 1e-12
    return ((X_apply - mu) / sd).mean(1)


def load(args):
    data = Path(args.data_dir)
    train = pd.read_csv(data / "train.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    label_of = dict(zip(train["id"], (train["sentiment"] == LABELS[1]).astype(int)))
    names, H, T, info = [], [], [], {}
    hold_ids = None
    for run in map(Path, args.runs):
        h = pd.read_csv(run / "holdout_predictions.csv").sort_values("id")
        if hold_ids is None:
            hold_ids = h["id"].values
        assert np.array_equal(hold_ids, h["id"].values), f"{run}: different hold-out reviews"
        t = pd.read_csv(run / "test_predictions.csv").set_index("id").loc[sample["id"].values, "margin"].values
        names.append(run.name); H.append(h["margin"].values); T.append(t)
        info[run.name] = json.loads((run / "run_info.json").read_text(encoding="utf-8"))
    res = Path(args.nblr_results)
    oof = np.load(res / "oof_scores_best.npy")
    by_id = dict(zip(train["id"].values, oof))
    ts = pd.read_csv(res / "test_scores.csv").set_index("id").loc[sample["id"].values, "score"].values
    names.append("nblr"); H.append(np.array([by_id[i] for i in hold_ids])); T.append(ts * np.std(oof) / np.std(ts))
    y = np.array([label_of[i] for i in hold_ids])
    return names, np.column_stack(H), np.column_stack(T), y, info, sample


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--nblr-results", default="results")
    ap.add_argument("--ensemble", required=True, help="folder written by ensemble.py for these runs")
    ap.add_argument("--previous-ensemble", help="optional earlier ensemble folder to compare test predictions with")
    ap.add_argument("--output", required=True)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    names, X, Xt, y, info, sample = load(args)
    n_hold = len(y)
    pred = X > 0
    wrong = pred != y[:, None]

    runs = {}
    for j, nme in enumerate(names):
        acc = float(np.mean(~wrong[:, j]))
        row = {"holdout_accuracy": acc, "errors": int(wrong[:, j].sum()),
               "holdout_accuracy_95ci": wilson(int((~wrong[:, j]).sum()), n_hold),
               "test_positive_share": float(np.mean(Xt[:, j] > 0))}
        if nme in info:
            ri = info[nme]
            hist = ri.get("history", [])
            best = max(hist, key=lambda r: r["holdout_accuracy"]) if hist else None
            row.update({"model": ri["config"]["model"], "seed": ri["seed"],
                        "best_epoch": best["epoch"] if best else None,
                        "best_eval_during_training": best["holdout_accuracy"] if best else None,
                        "history": hist, "train_minutes": ri.get("timings", {}).get("train_minutes"),
                        "predict_minutes": ri.get("timings", {}).get("predict_minutes"),
                        "submission_sha256": (ri.get("submission") or {}).get("sha256")})
        else:
            row.update({"model": "NB-LR (Stage 1, out-of-fold)"})
        runs[nme] = row

    # ensemble of all models (mean combiner) on the hold-out, in sample (no fitted weights besides scaling)
    m_all = mean_combiner(X, X)
    ens_wrong = (m_all > 0) != y
    best_single = max((n for n in names), key=lambda n: runs[n]["holdout_accuracy"])
    js = names.index(best_single)
    b = int(np.sum(wrong[:, js] & ~ens_wrong))   # ensemble fixes
    c = int(np.sum(~wrong[:, js] & ens_wrong))   # ensemble breaks

    tr_idx = [j for j, n in enumerate(names) if n != "nblr"]
    agreement = {
        "errors_of_ensemble_in_sample": int(ens_wrong.sum()),
        "ensemble_vs_best_single": {"best_single": best_single, "fixed_by_ensemble": b, "broken_by_ensemble": c,
                                    "mcnemar_exact_p": mcnemar_exact(b, c)},
        "wrong_in_all_transformer_runs": int(wrong[:, tr_idx].all(1).sum()),
        "wrong_in_all_models": int(wrong.all(1).sum()),
        "wrong_in_at_least_one_model": int(wrong.any(1).sum()),
    }

    # ablation over every subset with >= 2 models
    ablation = []
    for k in range(2, len(names) + 1):
        for subset in itertools.combinations(range(len(names)), k):
            Xs = X[:, subset]
            ablation.append({"models": [names[j] for j in subset],
                             "mean_cv": ens.cv_accuracy(Xs, y, "mean"),
                             "stack_cv": ens.cv_accuracy(Xs, y, "stack")})
    named = {
        "all_models": names,
        "transformers_only": [names[j] for j in tr_idx],
        "deberta_runs_only": [n for n in names if n.startswith("deberta")],
        "deberta_seed42_plus_nblr": ["deberta", "nblr"],
    }
    lookup = {tuple(a["models"]): a for a in ablation}
    named_rows = {k: lookup[tuple(v)] for k, v in named.items() if tuple(v) in lookup}
    mean_cvs = [a["mean_cv"] for a in ablation]

    ens_info = json.loads((Path(args.ensemble) / "ensemble_info.json").read_text(encoding="utf-8"))
    test = {"positive_share": float(np.mean(mean_combiner(X, Xt) > 0))}
    m_test = mean_combiner(X, Xt)
    test["differs_from_best_single"] = int(np.sum((m_test > 0) != (Xt[:, js] > 0)))
    if args.previous_ensemble:
        prev = pd.read_csv(Path(args.previous_ensemble) / "test_margins.csv").set_index("id")
        prev = prev.loc[sample["id"].values, "margin"].values
        test["differs_from_previous_ensemble"] = int(np.sum((m_test > 0) != (prev > 0)))

    corr = np.corrcoef(X.T)
    summary = {
        "n_holdout": int(n_hold), "models": names, "runs": runs,
        "margin_correlation": np.round(corr, 4).tolist(),
        "ensemble": {"choice": ens_info["choice"], "options_estimated_accuracy": ens_info["options_estimated_accuracy"],
                     "weights": ens_info["weights"], "submission_sha256": ens_info["submission_sha256"],
                     "in_sample_mean_accuracy": float(np.mean(~ens_wrong))},
        "agreement": agreement,
        "ablation_named": named_rows,
        "ablation_mean_cv_range": [float(min(mean_cvs)), float(max(mean_cvs))],
        "ablation_n_subsets": len(ablation),
        "ablation": ablation,
        "test": test,
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ensemble2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out / 'ensemble2_summary.json'}")
    for nme in names:
        print(f"  {nme:<20s} {runs[nme]['holdout_accuracy']:.4%}")
    print(f"  ensemble (mean, CV) {ens_info['options_estimated_accuracy']['mean']:.4%}; "
          f"subset range {min(mean_cvs):.4%} - {max(mean_cvs):.4%}")

    if args.no_plots:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pretty = {"deberta": "DeBERTa\nseed 42", "deberta-large-s1": "DeBERTa\nseed 1", "deberta-large-s2": "DeBERTa\nseed 2",
              "roberta-large": "RoBERTa\nseed 42", "nblr": "NB-LR"}
    labels = [pretty.get(n, n) for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [1.45, 1]})
    bars = [(pretty.get(n, n), runs[n]["holdout_accuracy"], runs[n]["holdout_accuracy_95ci"],
             "#C55A11" if n == "nblr" else "#2E5597") for n in names]
    prev_mean = named_rows.get("deberta_seed42_plus_nblr", {}).get("mean_cv")
    if prev_mean is not None:
        bars.append(("Mean of\nDeBERTa 42\n+ NB-LR", prev_mean, None, "#7F7F7F"))
    bars.append(("Mean of\nall five", ens_info["options_estimated_accuracy"]["mean"], None, "#548235"))
    xs = np.arange(len(bars))
    for x, (lab, acc, ci, col) in zip(xs, bars):
        ax[0].bar(x, 100 * acc, color=col, width=0.7)
        if ci:
            ax[0].errorbar(x, 100 * acc, yerr=[[100 * (acc - ci[0])], [100 * (ci[1] - acc)]], color="black", capsize=3, lw=1)
        ax[0].text(x, 100 * acc + 0.12 if not ci else 100 * ci[1] + 0.12, f"{100 * acc:.2f}", ha="center", fontsize=7.5)
    ax[0].set_xticks(xs); ax[0].set_xticklabels([b[0] for b in bars], fontsize=7.5)
    ax[0].set_ylim(90.5, 98.6); ax[0].set_ylabel("hold-out accuracy (%)")
    ax[0].set_title("Single models (95% CI) and mean ensembles", fontsize=10)
    ax[0].grid(axis="y", alpha=0.3)
    im = ax[1].imshow(corr, vmin=0.75, vmax=1.0, cmap="Blues")
    for i in range(len(names)):
        for j in range(len(names)):
            ax[1].text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                       color="white" if corr[i, j] > 0.93 else "black")
    short = [l.replace("\n", " ") for l in labels]
    ax[1].set_xticks(range(len(names))); ax[1].set_xticklabels(short, rotation=35, ha="right", fontsize=7.5)
    ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels(short, fontsize=7.5)
    ax[1].set_title("Correlation of hold-out margins", fontsize=10)
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig_dir = out / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "ensemble_models.png", dpi=150); plt.close(fig)
    print(f"wrote {fig_dir / 'ensemble_models.png'}")


if __name__ == "__main__":
    main()
