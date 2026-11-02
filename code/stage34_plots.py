"""Figures for report Sections 4.9 and 6.5-6.11 (imported by analyze_stage34.py; matplotlib only)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

BLUE, LIGHT_BLUE, ORANGE, GREY, GREEN, PURPLE = "#2E5597", "#8FAADC", "#C55A11", "#808080", "#548235", "#7030A0"
PINK = "#C0307A"
TEAL = "#1B7F79"
GOLD = "#B8860B"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def make_all(summary: dict, arrays: dict, fig_dir: Path) -> None:
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    models_figure(summary, fig_dir, plt)
    length_agreement_figure(summary, fig_dir, plt)
    kaggle_figure(summary, fig_dir, plt)
    architecture_figure(fig_dir, plt)
    print(f"wrote 4 figures to {fig_dir}")


# --------------------------------------------------------------------------- Figure: model families
def models_figure(s: dict, fig_dir: Path, plt) -> None:
    F, C = s["families"], s["combination"]["options_estimated_accuracy"]
    bars = [
        ("ELECTRA\n5 folds", F["electra"]["oof"], BLUE),
        ("DeBERTa\n5 folds", F["deberta"]["oof"], BLUE),
        ("DeBERTa\nre-run", F["deberta_rerun"]["oof"], BLUE),
        ("Qwen3\n8B", F["qwen3"]["oof"], PURPLE),
        ("Qwen3\n14B", F["qwen3_14b"]["oof"], PINK),
        ("Mistral\n24B", F["mistral_24b"]["oof"], TEAL),
        ("Qwen3.8\n27B", F["qwen38_27b"]["oof"], GOLD),
        ("stack\n2 (B)", {"accuracy": C["stack"], "wilson_95": None}, GREEN),
        ("s-mean\n3 (C)", {"accuracy": s["stage5"]["options_estimated_accuracy"]["s-mean"], "wilson_95": None}, GREY),
        ("s-mean\n4 (D)", {"accuracy": s["stage6"]["options_estimated_accuracy"]["s-mean"], "wilson_95": None}, GREY),
        ("s-mean\n5 (E)", {"accuracy": s["stage7"]["options_estimated_accuracy"]["s-mean"], "wilson_95": None}, GREY),
    ]
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.9), gridspec_kw={"width_ratios": [1.9, 1]})
    x = np.arange(len(bars))
    for i, (lab, b, col) in enumerate(bars):
        acc = 100 * b["accuracy"]
        ax[0].bar(i, acc - 95.5, bottom=95.5, color=col, width=0.68)
        if b.get("wilson_95"):
            lo, hi = (100 * v for v in b["wilson_95"])
            ax[0].errorbar(i, acc, yerr=[[acc - lo], [hi - acc]], color="black", capsize=3, lw=1)
            ax[0].text(i, hi + 0.04, f"{acc:.2f}", ha="center", va="bottom", fontsize=8)
        else:
            ax[0].text(i, acc + 0.04, f"{acc:.2f}", ha="center", va="bottom", fontsize=8)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([b[0] for b in bars], fontsize=6.8)
    ax[0].set_ylim(95.5, 97.75)
    ax[0].set_ylabel("out-of-fold accuracy (%)", fontsize=8.5)
    ax[0].set_title("Out-of-fold accuracy (95% CI); LLMs fine-tuned with LoRA; combinations: estimates", fontsize=9)
    ax[0].grid(axis="y", alpha=0.3)

    folds = np.arange(5)
    for key, col, mk, lab in (("electra", LIGHT_BLUE, "s", "ELECTRA-large"), ("deberta", BLUE, "o", "DeBERTa-v3-large"),
                              ("qwen3", PURPLE, "^", "Qwen3-8B + LoRA"), ("qwen3_14b", PINK, "D", "Qwen3-14B + LoRA"),
                              ("mistral_24b", TEAL, "v", "Mistral-24B + LoRA"), ("qwen38_27b", GOLD, "P", "Qwen3.8-27B + LoRA")):
        ax[1].plot(folds, [100 * a for a in F[key]["per_fold"]], marker=mk, color=col, label=lab, lw=1.5)
    ax[1].set_xticks(folds)
    ax[1].set_xticklabels([f"fold {j}" for j in folds], fontsize=8)
    ax[1].set_ylim(95.5, 97.75)
    ax[1].set_ylabel("validation accuracy (%)", fontsize=8.5)
    ax[1].set_title("The same 5,000 reviews per fold", fontsize=9.5)
    ax[1].legend(fontsize=6.8, loc="lower right")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "stage34_models.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- Figure: length + agreement
def length_agreement_figure(s: dict, fig_dir: Path, plt) -> None:
    F, P = s["families"], s["pairs"]["qwen3_vs_deberta"]
    fig, ax = plt.subplots(1, 2, figsize=(10.4, 3.7), gridspec_kw={"width_ratios": [1.55, 1]})
    bl = F["deberta"]["by_length"]
    x = np.arange(len(bl))
    w = 0.17
    for off, key, col, lab in ((-2 * w, "deberta", BLUE, "DeBERTa-v3-large (5 folds)"),
                               (-1 * w, "qwen3", PURPLE, "Qwen3-8B + LoRA (5 folds)"),
                               (0, "qwen3_14b", PINK, "Qwen3-14B + LoRA (5 folds)"),
                               (1 * w, "mistral_24b", TEAL, "Mistral-24B + LoRA (5 folds)"),
                               (2 * w, "qwen38_27b", GOLD, "Qwen3.8-27B + LoRA (5 folds)")):
        vals = [100 * b["accuracy"] for b in F[key]["by_length"]]
        ax[0].bar(x + off, [v - 95 for v in vals], w, bottom=95, color=col, label=lab)
        for xi, v in zip(x + off, vals):
            ax[0].text(xi, v + 0.05, f"{v:.1f}", ha="center", va="bottom", fontsize=4.6)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f"{b['bucket']}\n(n={b['n']:,})" for b in bl], fontsize=7.5)
    ax[0].set_xlabel("words per review", fontsize=8.5)
    ax[0].set_ylim(95, 99)
    ax[0].set_ylabel("out-of-fold accuracy (%)", fontsize=8.5)
    ax[0].set_title("Accuracy by review length (25,000 training reviews)", fontsize=9.5)
    ax[0].legend(fontsize=6.8, loc="upper left")
    ax[0].grid(axis="y", alpha=0.3)

    # rows: Qwen3 correct / wrong; columns: DeBERTa correct / wrong
    M = np.array([[P["both_right"], P["a_only_right"]], [P["b_only_right"], P["both_wrong"]]])
    ax[1].imshow(np.log1p(M), cmap="Blues", vmin=0, vmax=np.log1p(M.max()) * 1.15)
    for i in range(2):
        for j in range(2):
            ax[1].text(j, i, f"{M[i, j]:,}\n({100 * M[i, j] / M.sum():.1f}%)", ha="center", va="center", fontsize=10,
                       color="white" if M[i, j] > 5000 else "black")
    ax[1].set_xticks([0, 1])
    ax[1].set_xticklabels(["DeBERTa correct", "DeBERTa wrong"], fontsize=8.5)
    ax[1].set_yticks([0, 1])
    ax[1].set_yticklabels(["Qwen3\ncorrect", "Qwen3\nwrong"], fontsize=8.5)
    ax[1].set_title("Agreement on 25,000 out-of-fold reviews", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(fig_dir / "qwen3_length_agreement.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- Figure: Kaggle submissions
SHORT = {"nblr": "NB-LR", "deberta_s42": "DeBERTa\nseed 42", "ens_deb_nblr": "DeBERTa\n+ NB-LR",
         "ensemble2": "ensemble2", "deberta_5fold": "DeBERTa\n5-fold", "electra_5fold": "ELECTRA\n5-fold",
         "rerun_exact": "DeBERTa\nre-run+EM", "pilot": "pilot", "qwen3_5fold": "Qwen3\n5-fold",
         "stage4": "rule\nchoice", "stage4_exact": "Qwen3\n8B+EM", "stack_exact": "stack\n+EM (B)",
         "stage5_exact": "3-way\n+EM (C)", "stage6_exact": "final\n4-way+EM (D)",
         "stage7_exact": "final\n5-way+EM (E)"}
STAGE_OF = {"nblr": 1, "deberta_s42": 2, "ens_deb_nblr": 2, "ensemble2": 2, "deberta_5fold": 3, "electra_5fold": 3,
            "rerun_exact": 3, "pilot": 4, "qwen3_5fold": 4, "stage4": 4, "stage4_exact": 4, "stack_exact": 4, "stage5_exact": 5,
            "stage6_exact": 6, "stage7_exact": 7}
STAGE_COL = {1: ORANGE, 2: BLUE, 3: "#2F8F9D", 4: PURPLE, 5: PINK, 6: TEAL, 7: GOLD}


def kaggle_figure(s: dict, fig_dir: Path, plt) -> None:
    # "stage4" is the same file as "qwen3_5fold" (uploaded twice, same score): plotted once
    K = [k for k in s["kaggle"] if k["key"] != "stage4"]
    keys = [k["key"] for k in K]
    pub = np.array([100 * k["public"] for k in K])
    F = s["families"]
    val = {"nblr": F["nblr"]["oof"]["accuracy"], "deberta_s42": 0.9672, "ens_deb_nblr": 0.9682, "ensemble2": 0.9690,
           "deberta_5fold": F["deberta"]["oof"]["accuracy"], "electra_5fold": F["electra"]["oof"]["accuracy"],
           "rerun_exact": F["deberta_rerun"]["oof"]["accuracy"], "qwen3_5fold": F["qwen3"]["oof"]["accuracy"],
           "stage4": F["qwen3"]["oof"]["accuracy"], "stage4_exact": F["qwen3"]["oof"]["accuracy"],
           "stack_exact": s["combination"]["options_estimated_accuracy"]["stack"],
           "stage5_exact": s["stage5"]["options_estimated_accuracy"]["s-mean"],
           "stage6_exact": s["stage6"]["options_estimated_accuracy"]["s-mean"],
           "stage7_exact": s["stage7"]["options_estimated_accuracy"]["s-mean"]}
    fig, (top, bot) = plt.subplots(2, 1, figsize=(10.4, 4.3), sharex=True, gridspec_kw={"height_ratios": [4.2, 1]})
    x = np.arange(len(K))
    for a in (top, bot):
        a.plot(x, pub, color="#BBBBBB", lw=1, zorder=1)
        for i, k in enumerate(keys):
            col = STAGE_COL[STAGE_OF[k]]
            final = k in s["final_selection"]
            a.scatter(i, pub[i], s=95 if final else 45, marker="*" if final else "o", color=col, zorder=3,
                      edgecolor="black" if final else col, linewidth=0.6)
            if k in val:
                a.scatter(i, 100 * val[k], s=30, marker="o", facecolor="white", edgecolor=GREY, zorder=2, linewidth=1)
    for i, k in enumerate(keys):
        if pub[i] > 96:
            top.text(i, pub[i] + 0.07, f"{pub[i]:.2f}", ha="center", va="bottom", fontsize=7.2)
    bot.text(0.25, pub[0], f"{pub[0]:.2f}", ha="left", va="center", fontsize=7.2)
    top.set_ylim(96.25, 97.85)
    bot.set_ylim(91.5, 92.0)
    top.spines["bottom"].set_visible(False)
    bot.spines["top"].set_visible(False)
    top.tick_params(bottom=False)
    d = 0.006
    kw = dict(transform=top.transAxes, color="black", clip_on=False, lw=0.8)
    top.plot((-d, +d), (-d * 4, +d * 4), **kw)
    top.plot((1 - d, 1 + d), (-d * 4, +d * 4), **kw)
    kw.update(transform=bot.transAxes)
    bot.plot((-d, +d), (1 - d * 16, 1 + d * 16), **kw)
    bot.plot((1 - d, 1 + d), (1 - d * 16, 1 + d * 16), **kw)
    bot.set_xticks(x)
    bot.set_xticklabels([SHORT[k] for k in keys], fontsize=6.0)
    top.set_ylabel("accuracy (%)", fontsize=8.5)
    top.grid(axis="y", alpha=0.3)
    bot.grid(axis="y", alpha=0.3)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", color=STAGE_COL[t], label=f"Stage {t} public score") for t in (1, 2, 3, 4, 5, 6, 7)]
    handles += [Line2D([], [], marker="*", ls="", color=PURPLE, markeredgecolor="black", markersize=10, label="final selection"),
                Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=GREY, label="validation estimate")]
    top.legend(handles=handles, fontsize=6.8, loc="lower right", ncol=3)
    top.set_title("Public leaderboard (≈ 10,000 test reviews) of every submitted file, by project stage", fontsize=9.5)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.06)
    fig.savefig(fig_dir / "kaggle_progress.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- Figure: Qwen3 + LoRA schematic
def architecture_figure(fig_dir: Path, plt) -> None:
    """Schematic of the Qwen3-8B-Base + LoRA classifier (no data needed)."""
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(10, 6.6))
    ax.set_xlim(0, 102)
    ax.set_ylim(0, 66)
    ax.axis("off")

    def box(x, y, w, h, title, sub="", fc="#EAF0FA", ec=BLUE, tsize=9.3, ssize=7.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=1.2))
        if sub:
            ax.text(x + w / 2, y + h - 1.6, title, ha="center", va="center", fontsize=tsize, weight="bold")
            ax.text(x + w / 2, (y + h - 3.0) / 2 + y / 2, sub, ha="center", va="center", fontsize=ssize,
                    color="#333333", linespacing=1.45)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=tsize, weight="bold")

    def arrow(x0, y0, x1, y1, color="#444444"):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3))

    TRAIN_FC, TRAIN_EC = "#FBE5D6", ORANGE
    X, W = 1.5, 53
    rows = [
        (58.0, 6.6, "Review text", "\"My main problem with the film is that it goes on too long.\nOther than that, it's pretty good.\"", "#FFFFFF", BLUE),
        (48.6, 7.2, "1. Prompt + byte-level BPE tokens (151,669-token vocabulary)",
         "Movie review:⏎ [review] ⏎⏎Question: Overall, is this review of the movie\npositive or negative?⏎Answer:      (≤ 1,024 tokens; head 256 + tail)", "#EAF0FA", BLUE),
        (40.0, 6.0, "2. Token embeddings (frozen)", "one 4,096-number vector per token", "#EAF0FA", BLUE),
        (25.5, 11.5, "3. 36 × decoder layer (frozen + LoRA)",
         "causal attention: each token sees only the tokens before it,\nso the last token (\"Answer:\") has read the whole review\nand the question (details on the right)", "#D9E2F3", BLUE),
        (14.5, 7.4, "4. Classification head (2 × 4,096, trained here)",
         "takes the last token's vector; initialised with the model's own\noutput rows for \" negative\" and \" positive\" (= zero-shot answer)", TRAIN_FC, TRAIN_EC),
        (4.5, 6.0, "5. Prediction", "margin m = z_pos − z_neg;   P(positive) = σ(m)", "#FFFFFF", BLUE),
    ]
    for y, h, t, sub, fc, ec in rows:
        box(X, y, W, h, t, sub, fc=fc, ec=ec)
    for (y0, *_), (y1, h1, *_) in zip(rows[:-1], rows[1:]):
        arrow(X + W / 2, y0 - 0.5, X + W / 2, y1 + h1 + 0.5)
    ax.text(X + W + 1.2, 31.2, "×36", fontsize=13, weight="bold", color=BLUE, va="center")

    # right: one decoder layer, bottom to top (pre-normalisation)
    RX, RW = 64, 34
    ax.add_patch(FancyBboxPatch((RX - 1.6, 2.6), RW + 3.2, 60.0, boxstyle="round,pad=0.3,rounding_size=1.5",
                                fc="#F7F9FD", ec=BLUE, lw=1.0, ls="--"))
    ax.text(RX + RW / 2, 61.2, "One decoder layer", ha="center", fontsize=10.5, weight="bold", color=BLUE)
    box(RX, 4.3, RW, 3.6, "input: token vectors h₁ … h_L", tsize=8.2, fc="#FFFFFF")
    box(RX, 10.6, RW, 3.2, "RMSNorm", tsize=8.2, fc="#FFFFFF")
    box(RX, 16.6, RW, 13.0, "Grouped-query self-attention",
        "32 query heads, 8 key/value heads, 128 dims each\nRoPE positions, QK-Norm, causal mask\n"
        "a_ij = softmax over j ≤ i of q_i · k_j / √128\n ", fc="#EAF0FA", ssize=7.6)
    ax.text(RX + RW / 2, 18.4, "+ LoRA on the q, k, v and o projections", ha="center", va="center",
            fontsize=7.6, color=ORANGE, weight="bold")
    box(RX, 32.3, RW, 3.2, "add residual", tsize=8.2, fc="#FFFFFF")
    box(RX, 38.2, RW, 3.2, "RMSNorm", tsize=8.2, fc="#FFFFFF")
    box(RX, 44.2, RW, 9.0, "SwiGLU feed-forward network",
        "4,096 → 12,288 → 4,096, per token\n ", fc="#EAF0FA", ssize=7.6)
    ax.text(RX + RW / 2, 46.0, "+ LoRA on the gate, up and down projections", ha="center", va="center",
            fontsize=7.6, color=ORANGE, weight="bold")
    box(RX, 55.9, RW, 3.2, "add residual", tsize=8.2, fc="#FFFFFF")
    for y0, y1 in [(7.9, 10.6), (13.8, 16.6), (29.6, 32.3), (35.5, 38.2), (41.4, 44.2), (53.2, 55.9)]:
        arrow(RX + RW / 2, y0 + 0.4, RX + RW / 2, y1 - 0.4)
    ex = RX + RW + 1.7
    for y0, y1 in [(6.1, 33.9), (33.9, 57.5)]:
        ax.plot([RX + RW + 0.4, ex, ex], [y0, y0, y1], color="#888888", lw=1)
        ax.annotate("", xy=(RX + RW + 0.4, y1), xytext=(ex, y1), arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1))
    ax.text(ex + 0.9, 20.0, "residual", fontsize=7, color="#666666", rotation=90, va="center")
    ax.plot([X + W + 0.6, RX - 1.7], [33.0, 35.0], color=BLUE, lw=0.8, ls=":")

    # what LoRA does inside every adapted projection
    fig.text(0.5, 0.012, "LoRA in each adapted projection:  W x  →  W x + (α / r) · B A x,   W frozen;  "
             "A (r × d_in) and B (d_out × r) trained, B starts at 0;  r = 16, α = 32",
             fontsize=7.8, ha="center", color=ORANGE, weight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(fig_dir / "qwen3_architecture.png", dpi=170)
    plt.close(fig)
