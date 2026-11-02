#!/usr/bin/env python3
"""Analysis of the stage-2 (transformer) run against the NB-LR baseline. NumPy/pandas/matplotlib only.

Reads ``runs/<name>/run_info.json`` and ``holdout_predictions.csv`` (from finetune_transformer.py),
``ensemble/ensemble_info.json`` (from ensemble.py) and NB-LR's out-of-fold scores, and writes
``results/stage2/stage2_summary.json`` plus two figures used in the report.

    python code/analyze_stage2.py --data-dir data --run runs/deberta --ensemble ensemble \
        --nblr-results results --output results/stage2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    tp = int(np.sum(pred & (y == 1))); fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1))); tn = int(np.sum(~pred & (y == 0)))
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    return {"accuracy": (tp + tn) / len(y), "precision_positive": prec, "recall_positive": rec,
            "f1_positive": 2 * prec * rec / max(1e-12, prec + rec), "confusion_matrix": [[tn, fp], [fn, tp]]}


def wilson(k: int, n: int, z: float = 1.96) -> list:
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--ensemble", required=True)
    ap.add_argument("--nblr-results", default="results")
    ap.add_argument("--output", required=True)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(Path(args.data_dir) / "train.csv").set_index("id")
    info = json.loads((Path(args.run) / "run_info.json").read_text(encoding="utf-8"))
    ens = json.loads((Path(args.ensemble) / "ensemble_info.json").read_text(encoding="utf-8"))
    h = pd.read_csv(Path(args.run) / "holdout_predictions.csv").sort_values("id").reset_index(drop=True)
    oof = np.load(Path(args.nblr_results) / "oof_scores_best.npy")
    pos = {i: k for k, i in enumerate(train.index.values)}
    h["nblr"] = oof[[pos[i] for i in h["id"]]]
    y = h["label"].values.astype(int)
    d = h["margin"].values > 0
    nb = h["nblr"].values > 0
    X = np.column_stack([h["margin"].values, h["nblr"].values])
    ens_in_sample = ((X - X.mean(0)) / X.std(0)).mean(1) > 0  # same rule as ensemble.py "mean"

    words = train.loc[h["id"], "review"].str.split().str.len().values
    bins = [(0, 128), (128, 256), (256, 512), (512, 10 ** 6)]
    by_len = []
    for lo, hi in bins:
        m = (words >= lo) & (words < hi)
        by_len.append({"words": f"{lo}-{hi - 1}" if hi < 10 ** 6 else f"{lo}+", "n": int(m.sum()),
                       "deberta": float(np.mean(d[m] == y[m])), "nblr": float(np.mean(nb[m] == y[m]))})
    a = np.abs(h["margin"].values)
    qb = np.digitize(a, np.quantile(a, [0.2, 0.4, 0.6, 0.8]))
    by_conf = [{"quintile": k + 1, "accuracy": float(np.mean(d[qb == k] == y[qb == k]))} for k in range(5)]
    err = h[d != y].assign(conf=lambda x: x["margin"].abs()).sort_values("conf", ascending=False)
    hardest = [{"id": int(r.id), "label": "positive" if r.label == 1 else "negative", "margin": round(float(r.margin), 2),
                "nblr_score": round(float(r.nblr), 2), "text_start": train.loc[int(r.id), "review"][:300]}
               for r in err.head(10).itertuples()]
    k_d = int(np.sum(d == y))
    summary = {
        "n_holdout": int(len(y)),
        "deberta": metrics(d, y), "nblr": metrics(nb, y), "ensemble_mean_in_sample": metrics(ens_in_sample, y),
        "deberta_accuracy_95ci": wilson(k_d, len(y)),
        "agreement": {"both_correct": int(np.sum((d == y) & (nb == y))), "deberta_only": int(np.sum((d == y) & (nb != y))),
                      "nblr_only": int(np.sum((d != y) & (nb == y))), "both_wrong": int(np.sum((d != y) & (nb != y)))},
        "margin_correlation": float(np.corrcoef(X.T)[0, 1]),
        "accuracy_by_length": by_len, "deberta_accuracy_by_confidence": by_conf,
        "deberta_errors_in_lowest_confidence_quintile": int(np.sum((d != y) & (qb == 0))),
        "deberta_errors": int(np.sum(d != y)),
        "training_history": info.get("history", []), "run": {k: info[k] for k in ("config", "max_len", "head", "seed",
                            "n_train", "n_holdout", "environment", "timings") if k in info},
        "ensemble": {k: ens[k] for k in ("options_estimated_accuracy", "choice", "weights", "submission_sha256")},
        "deberta_submission": info.get("submission"),
        "hardest_deberta_errors": hardest,
    }
    (out / "stage2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("agreement", "deberta_errors", "margin_correlation")}, indent=1))

    if args.no_plots:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = out / "figures"; fig_dir.mkdir(exist_ok=True)

    hist = summary["training_history"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot([r["epoch"] for r in hist], [100 * r["holdout_accuracy"] for r in hist], "o-", color="#2E5597", label="DeBERTa-v3-large")
    ax[0].axhline(100 * summary["nblr"]["accuracy"], color="#C55A11", ls="--", label="NB-LR (out-of-fold)")
    best = max(hist, key=lambda r: r["holdout_accuracy"])
    ax[0].annotate(f"selected\n{100 * best['holdout_accuracy']:.2f}%", (best["epoch"], 100 * best["holdout_accuracy"]),
                   xytext=(-60, -38), textcoords="offset points", arrowprops=dict(arrowstyle="->"))
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("hold-out accuracy (%)"); ax[0].set_ylim(90, 98)
    ax[0].set_title("Hold-out accuracy during fine-tuning"); ax[0].legend(loc="lower right", fontsize=8); ax[0].grid(alpha=0.3)
    x = np.arange(len(by_len)); w = 0.38
    ax[1].bar(x - w / 2, [100 * r["nblr"] for r in by_len], w, color="#C55A11", label="NB-LR")
    ax[1].bar(x + w / 2, [100 * r["deberta"] for r in by_len], w, color="#2E5597", label="DeBERTa-v3-large")
    ax[1].set_xticks(x); ax[1].set_xticklabels([f"{r['words']}\n(n={r['n']})" for r in by_len], fontsize=8)
    ax[1].set_xlabel("words per review"); ax[1].set_ylim(85, 100); ax[1].set_ylabel("hold-out accuracy (%)")
    ax[1].set_title("Accuracy by review length"); ax[1].legend(fontsize=8); ax[1].grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "transformer_training_and_length.png", dpi=150); plt.close(fig)

    ag = summary["agreement"]
    M = np.array([[ag["both_correct"], ag["deberta_only"]], [ag["nblr_only"], ag["both_wrong"]]])
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.imshow(np.log1p(M), cmap="Blues", vmin=0, vmax=np.log1p(M.max()) * 1.15)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{M[i, j]:,}\n({100 * M[i, j] / M.sum():.1f}%)", ha="center", va="center", fontsize=11,
                    color="white" if M[i, j] > 1000 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["NB-LR correct", "NB-LR wrong"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["DeBERTa\ncorrect", "DeBERTa\nwrong"])
    ax.set_title("Agreement on the 2,500 hold-out reviews", fontsize=10)
    fig.tight_layout(); fig.savefig(fig_dir / "transformer_vs_nblr_agreement.png", dpi=150); plt.close(fig)
    plot_architecture(fig_dir, plt)


def plot_architecture(fig_dir: Path, plt) -> None:
    """Schematic of DeBERTa-v3-large as used for classification (no data needed)."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 102); ax.set_ylim(0, 62); ax.axis("off")

    def box(x, y, w, h, title, sub="", fc="#EAF0FA", ec="#2E5597", tsize=9.5, ssize=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
        if sub:
            ax.text(x + w / 2, y + h - 1.7, title, ha="center", va="center", fontsize=tsize, weight="bold")
            ax.text(x + w / 2, (y + h - 3.2) / 2 + y / 2, sub, ha="center", va="center", fontsize=ssize,
                    color="#333333", linespacing=1.45)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tsize, weight="bold")

    def arrow(x0, y0, x1, y1, color="#444444"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3))

    # left column: data flow (top to bottom)
    X, W = 2, 52
    rows = [
        (54.2, "Review text", "\"My main problem with the film is that it goes on too long.\nOther than that, it's pretty good.\"", "#FFFFFF"),
        (46.0, "1. Sub-word tokenisation (SentencePiece, 128k pieces)",
         "[CLS] ▁My ▁main ▁problem … ▁pretty ▁good . [SEP]   (≤ 512 tokens; head + tail)", "#EAF0FA"),
        (37.0, "2. Token embeddings", "one 1,024-number vector per token (no absolute positions added)", "#EAF0FA"),
        (23.5, "3. 24 × Transformer layer", "every token vector is rewritten using all other tokens\n(details on the right)", "#D9E2F3"),
        (14.0, "4. Classification head (new, trained here)",
         "[CLS] vector → dense + GELU → linear 1,024 → 2 → logits z_neg, z_pos", "#FBE5D6"),
        (5.0, "5. Prediction", "P(positive) = σ(z_pos − z_neg);   label = positive if z_pos > z_neg", "#FFFFFF"),
    ]
    heights = {54.2: 6.4, 46.0: 5.6, 37.0: 5.6, 23.5: 9.6, 14.0: 5.6, 5.0: 5.6}
    for y, t, s, fc in rows:
        box(X, y, W, heights[y], t, s, fc=fc, ec="#C55A11" if fc == "#FBE5D6" else "#2E5597")
    for (y0, *_), (y1, *_) in zip(rows[:-1], rows[1:]):
        arrow(X + W / 2, y0 - 0.5, X + W / 2, y1 + heights[y1] + 0.5)
    ax.text(X + W + 1.4, 26.0, "×24", fontsize=13, weight="bold", color="#2E5597", va="center")

    # right column: one transformer layer (bottom to top, as data flows)
    RX, RW = 63, 35
    ax.add_patch(FancyBboxPatch((RX - 1.5, 3.5), RW + 3, 55, boxstyle="round,pad=0.3,rounding_size=1.5",
                                fc="#F7F9FD", ec="#2E5597", lw=1.0, ls="--"))
    ax.text(RX + RW / 2, 57.0, "One transformer layer", ha="center", fontsize=10.5, weight="bold", color="#2E5597")
    box(RX, 5.5, RW, 4.2, "input: token vectors h₁ … h_L", tsize=8.5, fc="#FFFFFF")
    box(RX, 13.0, RW, 16.0, "Disentangled self-attention",
        "16 heads × 64 dims; score of token i for token j =\n"
        "content ↔ content    q_i · k_j\n"
        "content → position   q_i · k_pos(i−j)\n"
        "position → content   q_pos(j−i) · k_j\n"
        "weights = softmax over j;  output = Σ_j a_ij v_j", tsize=9.5, ssize=8)
    box(RX, 31.5, RW, 4.0, "Add & LayerNorm", tsize=8.5, fc="#FFFFFF")
    box(RX, 38.5, RW, 7.5, "Feed-forward network", "1,024 → 4,096 → 1,024 (GELU), per token")
    box(RX, 49.0, RW, 4.0, "Add & LayerNorm", tsize=8.5, fc="#FFFFFF")
    for y0, y1 in [(9.7, 13.0), (29.0, 31.5), (35.5, 38.5), (46.0, 49.0)]:
        arrow(RX + RW / 2, y0 + 0.4, RX + RW / 2, y1 - 0.4)
    # residual connections, routed outside the boxes on the right
    ex = RX + RW + 1.6
    for y0, y1 in [(7.6, 33.5), (36.5, 51.0)]:
        ax.plot([RX + RW + 0.4, ex, ex], [y0, y0, y1], color="#888888", lw=1)
        ax.annotate("", xy=(RX + RW + 0.4, y1), xytext=(ex, y1), arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1))
    ax.text(ex + 0.9, 20.5, "residual", fontsize=7, color="#666666", rotation=90, va="center")
    # link left layer box to right panel
    ax.plot([X + W + 0.6, RX - 1.6], [28.3, 30.5], color="#2E5597", lw=0.8, ls=":")
    fig.tight_layout(); fig.savefig(fig_dir / "deberta_architecture.png", dpi=170); plt.close(fig)


if __name__ == "__main__":
    main()
