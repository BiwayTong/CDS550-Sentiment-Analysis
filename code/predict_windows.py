#!/usr/bin/env python3
"""Score an already fine-tuned transformer with longer inputs (no re-training).

``finetune_transformer.py`` reads at most ``max_len = 512`` tokens per review using the
head+tail strategy of Sun et al. (2019): the first ``head`` and the last ``max_len - head - 2``
tokens. About 12 % of the IMDB reviews are longer than that, so their middle is never seen by
the model. This script re-uses the saved checkpoint and changes only how the text is *fed* to
it, which costs minutes instead of a re-training run:

    head_tail   the baseline strategy, used as a control (should reproduce the run folder)
    tta         test-time augmentation: mean of head+tail, first 512 tokens, last 512 tokens
    windows     every review is covered by overlapping 512-token windows, mean of their margins
    windows_tta windows plus the head+tail view

The output folder has exactly the layout of a training run (holdout_predictions.csv,
test_predictions.csv, submission.csv, run_info.json), so it can be passed to ``ensemble.py``
like any other run and compared on the same 2,500 hold-out reviews.

    python code/predict_windows.py --data-dir data --run runs/deberta \\
        --baseline-run runs/deberta --strategy windows --output runs/deberta-win
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_transformer as ft  # noqa: E402  (imports numpy/pandas only)

LABELS = ("negative", "positive")


def wilson(k: int, n: int, z: float = 1.96) -> list:
    """Wilson score interval for k successes out of n (used for the 95 % CI of the accuracy)."""
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


# --------------------------------------------------------------------------- helpers (no torch)
def window_segments(ids: np.ndarray, window: int, stride: int, max_windows: int = 8) -> list:
    """Cover ``ids`` with overlapping windows of ``window`` tokens.

    The first window starts at 0 and the last one ends at the end of the review, so no token is
    skipped. If a review would need more than ``max_windows`` windows, evenly spread ones
    (including both ends) are kept, which bounds the cost per review.
    """
    ids = np.asarray(ids)
    n = len(ids)
    if n <= window:
        return [ids.copy()]
    starts = list(range(0, n - window + 1, max(1, stride)))
    if starts[-1] != n - window:
        starts.append(n - window)
    if len(starts) > max_windows:
        pick = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = [starts[i] for i in dict.fromkeys(pick.tolist())]
    return [ids[s:s + window] for s in starts]


def review_segments(ids, strategy: str, max_body: int, head: int, window: int, stride: int) -> list:
    """Token arrays that should be scored for one review, one per view of the text."""
    ids = np.asarray(ids)
    if strategy == "head_tail":
        return [np.asarray(ft.head_tail(ids, max_body, head))]
    if strategy == "tta":
        return [np.asarray(ft.head_tail(ids, max_body, head)), ids[:max_body], ids[-max_body:]]
    segs = window_segments(ids, window, stride)
    if strategy == "windows_tta":
        segs = [np.asarray(ft.head_tail(ids, max_body, head))] + segs
    return segs


def add_special(seg: np.ndarray, cls_id: int, sep_id: int) -> np.ndarray:
    return np.concatenate([[cls_id], seg, [sep_id]]).astype(np.int64)


def encode_all(texts, tokenizer, strategy, max_body, head, window, stride, cls_id, sep_id, log=None):
    """Tokenise every text and return (docs, lengths) with one entry per *segment*, plus owners."""
    docs, owner = [], []
    for s in range(0, len(texts), 2000):
        chunk = [ft.clean_review(t) for t in texts[s:s + 2000]]
        for k, ids in enumerate(tokenizer(chunk, add_special_tokens=False)["input_ids"]):
            for seg in review_segments(ids, strategy, max_body, head, window, stride):
                docs.append(add_special(seg, cls_id, sep_id))
                owner.append(s + k)
    lengths = np.array([len(d) for d in docs])
    if log:
        log(f"tokenised {len(texts):,} reviews into {len(docs):,} segments "
            f"({len(docs) / max(1, len(texts)):.2f} per review), mean length {lengths.mean():.0f} tokens")
    return docs, lengths, np.array(owner)


def aggregate(owner: np.ndarray, margins: np.ndarray, n_reviews: int) -> np.ndarray:
    """Average the margins of all segments that belong to the same review."""
    total = np.bincount(owner, weights=margins, minlength=n_reviews)
    count = np.bincount(owner, minlength=n_reviews)
    return total / np.maximum(count, 1)


# --------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--run", required=True, help="run folder containing best_model/")
    p.add_argument("--baseline-run", help="optional run folder to compare the hold-out accuracy with")
    p.add_argument("--output", required=True, help="new run folder to write (created if missing)")
    p.add_argument("--strategy", default="windows", choices=["head_tail", "tta", "windows", "windows_tta"])
    p.add_argument("--window", type=int, default=510, help="tokens per window (max_len - 2)")
    p.add_argument("--stride", type=int, default=255, help="step between windows (half a window by default)")
    p.add_argument("--max-windows", type=int, default=8, help="cap on the windows scored per review")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--head", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--holdout-frac", type=float, default=0.10)
    p.add_argument("--split-seed", type=int, default=42, help="must match the training run")
    p.add_argument("--limit", type=int, help="DEBUG: only score the first N test reviews")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    transformers.logging.set_verbosity_error()
    t_start = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    log = ft.Logger(out / "predict_log.txt")
    log("=" * 70)
    log("command: " + " ".join(sys.argv))
    log(f"strategy={args.strategy} window={args.window} stride={args.stride} "
        f"max_windows={args.max_windows} max_len={args.max_len} head={args.head}")

    if not (Path(args.run) / "best_model").exists():
        sys.exit(f"{Path(args.run) / 'best_model'} not found; train first or fix --run.")
    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None)
                                                       and torch.backends.mps.is_available() else "cpu")
    log(f"device={device} torch={torch.__version__} transformers={transformers.__version__}")

    data = Path(args.data_dir)
    train = pd.read_csv(data / "train.csv")
    test = pd.read_csv(data / "test.csv")
    sample = pd.read_csv(data / "sample_submission.csv")
    y_all = (train["sentiment"].values == LABELS[1]).astype(np.int64)
    hold_mask = ft.stratified_holdout(y_all, args.holdout_frac, args.split_seed)
    hold_idx = np.flatnonzero(hold_mask)
    test_idx = np.arange(len(test)) if not args.limit else np.arange(min(args.limit, len(test)))
    log(f"holdout={len(hold_idx):,} test={len(test_idx):,} (split-seed {args.split_seed})")

    model_dir = Path(args.run) / "best_model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    if cls_id is None or sep_id is None or pad_id is None:
        sys.exit("Tokenizer lacks CLS/SEP/PAD tokens; use a BERT/RoBERTa/DeBERTa-style model.")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    max_body = args.max_len - 2
    window = min(args.window, max_body)

    amp_dtype = None
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def autocast():
        if amp_dtype is None:
            return torch.autocast(device_type="cpu", enabled=False)
        return torch.autocast(device_type="cuda", dtype=amp_dtype)

    @torch.no_grad()
    def score(docs, lengths, owner, n_reviews):
        margins = np.zeros(len(docs), dtype=np.float64)
        for b in ft.sorted_batches(lengths, np.arange(len(docs)), args.batch_size):
            ids, mask = ft.collate([docs[j] for j in b], pad_id)
            with autocast():
                logits = model(input_ids=torch.from_numpy(ids).to(device),
                               attention_mask=torch.from_numpy(mask).to(device)).logits.float()
            m = (logits[:, 1] - logits[:, 0]).cpu().numpy()
            for j, v in zip(b, m):
                margins[int(j)] = v
        return aggregate(owner, margins, n_reviews)

    t0 = time.time()
    h_docs, h_len, h_own = encode_all(train["review"].values[hold_idx].tolist(), tokenizer,
                                      args.strategy, max_body, args.head, window, args.stride,
                                      cls_id, sep_id, log)
    m_hold = score(h_docs, h_len, h_own, len(hold_idx))
    acc = float(np.mean((m_hold > 0) == y_all[hold_idx]))
    log(f"hold-out accuracy {acc:.4%} ({np.sum((m_hold > 0) != y_all[hold_idx])} errors of {len(hold_idx):,})")
    del h_docs, h_len, h_own
    if device == "cuda":
        torch.cuda.empty_cache()

    t_docs, t_len, t_own = encode_all(test["review"].values[test_idx].tolist(), tokenizer,
                                      args.strategy, max_body, args.head, window, args.stride,
                                      cls_id, sep_id, log)
    m_test = score(t_docs, t_len, t_own, len(test_idx))
    log(f"scored in {(time.time() - t0) / 60:.1f} min; test positive share {np.mean(m_test > 0):.4f}")

    # same file layout as a training run, so ensemble.py can consume this folder directly
    pd.DataFrame({"id": train["id"].values[hold_idx], "label": y_all[hold_idx],
                  "p_positive": ft.sigmoid(m_hold), "margin": m_hold}).to_csv(
        out / "holdout_predictions.csv", index=False)
    test_ids = test["id"].values[test_idx]
    pd.DataFrame({"id": test_ids, "p_positive": ft.sigmoid(m_test), "margin": m_test}).to_csv(
        out / "test_predictions.csv", index=False)
    sub_info = ft.write_submission(test_ids, m_test, data / "sample_submission.csv", out / "submission.csv")
    log(f"wrote {out / 'submission.csv'} sha256={sub_info['sha256']}")

    compare = None
    if args.baseline_run:
        hb = pd.read_csv(Path(args.baseline_run) / "holdout_predictions.csv")
        ids_here = train["id"].values[hold_idx]
        if set(hb["id"]) == set(ids_here):     # same hold-out reviews, any row order
            base_m = hb.set_index("id").loc[ids_here, "margin"].values
            base_acc = float(np.mean((base_m > 0) == y_all[hold_idx]))
            wrong_b = (base_m > 0) != y_all[hold_idx]
            wrong_n = (m_hold > 0) != y_all[hold_idx]
            compare = {"baseline_accuracy": base_acc, "accuracy": acc,
                       "fixed": int(np.sum(wrong_b & ~wrong_n)), "broken": int(np.sum(~wrong_b & wrong_n)),
                       "holdout_accuracy_95ci": wilson(int(np.sum(~wrong_n)), len(wrong_n))}
            tb = pd.read_csv(Path(args.baseline_run) / "test_predictions.csv").set_index("id")
            n_diff = int(np.sum((m_test > 0) != (tb.loc[test_ids, "margin"].values > 0))) if set(test_ids) <= set(tb.index) else None
            log(f"vs baseline {Path(args.baseline_run).name}: {base_acc:.4%} -> {acc:.4%} "
                f"(fixed {compare['fixed']}, broken {compare['broken']}); "
                + (f"test labels differ on {n_diff} reviews" if n_diff is not None
                   else "baseline has no test predictions for these ids"))
        else:
            log("baseline run uses a different hold-out split; comparison skipped")

    (out / "run_info.json").write_text(json.dumps({
        "mode": "predict-only", "strategy": args.strategy, "command": " ".join(sys.argv),
        "source_run": str(args.run), "window": window, "stride": args.stride,
        "max_windows": args.max_windows, "max_len": args.max_len, "head": args.head,
        "split_seed": args.split_seed, "holdout_frac": args.holdout_frac,
        "n_holdout": int(len(hold_idx)), "holdout_accuracy": acc, "comparison_to_baseline": compare,
        "test_positive_share": float(np.mean(m_test > 0)), "submission": sub_info,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "transformers": transformers.__version__,
                        "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None},
        "total_minutes": round((time.time() - t_start) / 60, 2),
    }, indent=2), encoding="utf-8")
    log(f"done in {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
