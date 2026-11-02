#!/usr/bin/env python3
"""Fine-tune a pre-trained transformer (DeBERTa-v3 / RoBERTa) for IMDB sentiment.

CDS550 individual project, stage 2 (GPU). Only the 25,000 labelled reviews in
train.csv are used for training. A stratified 10 % hold-out (seed 42) is kept
aside for checkpoint selection and for fitting the ensemble weights in
``ensemble.py``; test labels are never used.

Only *general-purpose* pre-trained checkpoints are allowed (e.g.
microsoft/deberta-v3-large). Checkpoints that were already fine-tuned on IMDB or
another sentiment dataset are refused, because they may have been trained on
labels of the reviews in our test split.

Long reviews are truncated with the head+tail strategy of Sun et al. (2019):
the first ``--head`` tokens and the last ``max_len - head - 2`` tokens are kept,
because opinions concentrate at the start and the end of a review.

Typical use (run from the ``submission`` folder)::

    # 2-5 minute end-to-end check on a few hundred reviews (no submission written)
    python code/finetune_transformer.py --data-dir data --output runs/deberta-v3-large --smoke
    # full run; preset chosen automatically from the GPU memory
    python code/finetune_transformer.py --data-dir data --output runs/deberta-v3-large
    # one fold of a 10-fold split (fold 0 validates on the same 2,500 reviews as the
    # single hold-out above, so the two numbers are directly comparable)
    python code/finetune_transformer.py --data-dir data --folds 10 --fold 0 \\
        --output runs/deberta-f0
    # regenerate test predictions from the saved checkpoint (reproducibility check)
    python code/finetune_transformer.py --data-dir data --output runs/deberta-v3-large --predict-only

Outputs in ``--output``: best_model/ (weights + tokenizer), holdout_predictions.csv,
test_predictions.csv, submission.csv, run_info.json, train_log.txt.

``--folds K --fold j`` runs a single fold of a stratified K-fold split. The folds are
cut from the same per-class shuffles as the plain hold-out, so fold 0 of 10 is exactly
the 2,500-review hold-out used in stage 2. Run every fold (``code/run_cv.py`` does this),
concatenate the validation predictions and you get an out-of-fold estimate over all
25,000 reviews — three times finer than the single hold-out — plus K models whose test
predictions can be averaged.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import platform
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

LABELS = ("negative", "positive")
TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")
FORBIDDEN_MODEL_RE = re.compile(r"imdb|sst|sentiment|emotion|polarity|yelp|amazon", re.I)

PRESETS = {
    # name: model, peak lr, epochs, micro-batch, gradient accumulation, gradient checkpointing
    "large": dict(model="microsoft/deberta-v3-large", lr=1e-5, epochs=2, batch_size=8, grad_accum=2, grad_ckpt=False),
    "large-lowmem": dict(model="microsoft/deberta-v3-large", lr=1e-5, epochs=2, batch_size=4, grad_accum=4, grad_ckpt=True),
    "base": dict(model="microsoft/deberta-v3-base", lr=2e-5, epochs=3, batch_size=16, grad_accum=1, grad_ckpt=False),
    "base-lowmem": dict(model="microsoft/deberta-v3-base", lr=2e-5, epochs=3, batch_size=8, grad_accum=2, grad_ckpt=True),
    "roberta-large": dict(model="roberta-large", lr=1e-5, epochs=2, batch_size=8, grad_accum=2, grad_ckpt=False),
}


# --------------------------------------------------------------------------- helpers (no torch)
def clean_review(text: str) -> str:
    """Remove HTML tags (<br />) first, then unescape entities, collapse whitespace.

    Case and punctuation are kept: the tokenizer of a cased pre-trained model uses them.
    """
    text = TAG_RE.sub(" ", str(text))
    text = html.unescape(text)
    return WS_RE.sub(" ", text).strip()


def head_tail(ids, max_body: int, head: int) -> list:
    """Keep the first ``head`` and the last ``max_body - head`` tokens of a long sequence."""
    ids = list(ids)
    if len(ids) <= max_body:
        return ids
    head = min(head, max_body)
    tail = max_body - head
    return ids[:head] + (ids[-tail:] if tail > 0 else [])


def _class_order(y: np.ndarray, seed: int) -> dict:
    """For each class, the index order produced by one shared RNG (one shuffle *per class*)."""
    rng = np.random.default_rng(seed)
    order = {}
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        order[c] = idx
    return order


def stratified_holdout(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Boolean mask selecting ``frac`` of each class (deterministic for a seed)."""
    hold = np.zeros(len(y), dtype=bool)
    for c, idx in _class_order(y, seed).items():
        hold[idx[: int(round(frac * len(idx)))]] = True
    return hold


def kfold_indices(y: np.ndarray, folds: int, fold: int, seed: int):
    """Train/validation indices for fold ``fold`` of ``folds`` stratified folds.

    The partition is built from the *same* per-class shuffles as
    :func:`stratified_holdout`, so ``fold=0`` of ``folds=5`` selects exactly the first
    20 % of each class — i.e. the 2,500 reviews that the stage-2 hold-out used. That
    makes every fold's validation score directly comparable with the numbers already in
    the report, and lets the 25,000-row out-of-fold vector be lined up review by review.
    """
    if folds < 2:
        raise ValueError("folds must be >= 2")
    if not 0 <= fold < folds:
        raise ValueError(f"fold must be in 0..{folds - 1}")
    val = np.zeros(len(y), dtype=bool)
    for c, idx in _class_order(y, seed).items():
        size = len(idx)
        bounds = [round(k * size / folds) for k in range(folds + 1)]
        val[idx[bounds[fold]:bounds[fold + 1]]] = True
    return np.flatnonzero(~val), np.flatnonzero(val)


def length_grouped_batches(lengths: np.ndarray, idx: np.ndarray, batch_size: int,
                           rng: np.random.Generator, group: int = 50) -> list:
    """Random batches of similar length (less padding -> faster), in random order."""
    idx = rng.permutation(np.asarray(idx))
    chunk = batch_size * group
    batches = []
    for s in range(0, len(idx), chunk):
        c = idx[s:s + chunk]
        c = c[np.argsort(-lengths[c], kind="stable")]
        batches.extend(c[i:i + batch_size] for i in range(0, len(c), batch_size))
    order = rng.permutation(len(batches))
    return [batches[i] for i in order]


def sorted_batches(lengths: np.ndarray, idx, batch_size: int) -> list:
    idx = np.asarray(idx)
    idx = idx[np.argsort(-lengths[idx], kind="stable")]
    return [idx[i:i + batch_size] for i in range(0, len(idx), batch_size)]


def collate(docs: list, pad_id: int):
    """Right-pad a list of 1-D int arrays; returns (input_ids, attention_mask) as int64 arrays."""
    width = max(len(d) for d in docs)
    ids = np.full((len(docs), width), pad_id, dtype=np.int64)
    mask = np.zeros((len(docs), width), dtype=np.int64)
    for i, d in enumerate(docs):
        ids[i, :len(d)] = d
        mask[i, :len(d)] = 1
    return ids, mask


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_submission(test_ids: np.ndarray, margin: np.ndarray, sample_path: Path, out_path: Path) -> dict:
    """Write id,sentiment in sample_submission order and validate the format."""
    sample = pd.read_csv(sample_path)
    pred = pd.Series(np.where(margin > 0, LABELS[1], LABELS[0]), index=test_ids)
    if set(sample["id"]) != set(test_ids):
        raise ValueError("test ids do not match sample_submission.csv")
    sub = pd.DataFrame({"id": sample["id"].values, "sentiment": pred.loc[sample["id"].values].values})
    assert sub["id"].is_unique and len(sub) == len(sample)
    assert set(sub["sentiment"]) <= set(LABELS)
    sub.to_csv(out_path, index=False)
    return {"rows": int(len(sub)), "positive": int((sub.sentiment == "positive").sum()),
            "negative": int((sub.sentiment == "negative").sum()), "sha256": sha256_file(out_path)}


class Logger:
    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


# --------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="folder with train.csv, test.csv, sample_submission.csv")
    p.add_argument("--output", required=True, help="run folder (created if missing)")
    p.add_argument("--preset", default="auto", choices=["auto", *PRESETS], help="hyper-parameter preset")
    p.add_argument("--model", help="override the Hugging Face model name of the preset")
    p.add_argument("--lr", type=float)
    p.add_argument("--epochs", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--grad-accum", type=int)
    p.add_argument("--grad-ckpt", action="store_true", help="force gradient checkpointing (saves memory)")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--head", type=int, default=128, help="tokens kept from the start of long reviews")
    p.add_argument("--holdout-frac", type=float, default=0.10)
    p.add_argument("--folds", type=int, default=0,
                   help="K-fold mode: number of stratified folds (0 = classic single hold-out)")
    p.add_argument("--fold", type=int, default=0, help="K-fold mode: which fold to run (0-based)")
    p.add_argument("--seed", type=int, default=42, help="seed of the training run (shuffling, classifier head)")
    p.add_argument("--split-seed", type=int, default=42, help="seed of the hold-out split (keep 42 for ensembling)")
    p.add_argument("--warmup", type=float, default=0.10, help="fraction of updates with linear warm-up")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--eval-every", type=float, default=0.5, help="evaluate the hold-out every N epochs")
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end check, writes to OUTPUT/smoke")
    p.add_argument("--predict-only", action="store_true", help="reload OUTPUT/best_model and predict again")
    p.add_argument("--fresh-start", action="store_true",
                   help="start from the named general pre-trained checkpoint and refuse an existing best_model")
    p.add_argument("--verify-fresh", action="store_true",
                   help="evaluate the newly initialised model at update 0 before training")
    p.add_argument("--allow-cpu", action="store_true", help="allow running without a GPU (very slow)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    transformers.logging.set_verbosity_error()
    t_start = time.time()

    out = Path(args.output) / ("smoke" if args.smoke else "")
    out_preexisting = out.exists() and any(out.iterdir())
    if args.fresh_start and out_preexisting:
        sys.exit(f"refusing to reuse non-empty run folder {out}; choose a new --output folder")
    out.mkdir(parents=True, exist_ok=True)
    log = Logger(out / "train_log.txt")
    log("=" * 70)
    log("command: " + " ".join(sys.argv))
    if args.fresh_start and args.predict_only:
        sys.exit("--fresh-start cannot be combined with --predict-only")
    if args.verify_fresh and args.predict_only:
        sys.exit("--verify-fresh requires a training run, not --predict-only")
    if args.fresh_start:
        args.verify_fresh = True

    # ---------------- device and preset
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device, gpu_name, vram_gb = "mps", "Apple MPS", float("nan")
    else:
        device, gpu_name, vram_gb = "cpu", "none", float("nan")
        if not args.allow_cpu:
            sys.exit("No GPU found (torch.cuda.is_available() is False). Install the CUDA build of PyTorch, "
                     "or pass --allow-cpu to run anyway (many hours).")
    if args.preset == "auto":
        if device == "cuda":
            preset = ("large" if vram_gb >= 22 else "large-lowmem" if vram_gb >= 14
                      else "base" if vram_gb >= 10 else "base-lowmem")
        else:
            preset = "base-lowmem"
    else:
        preset = args.preset
    cfg = dict(PRESETS[preset])
    for key in ("model", "lr", "epochs", "batch_size", "grad_accum"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.grad_ckpt:
        cfg["grad_ckpt"] = True
    if FORBIDDEN_MODEL_RE.search(cfg["model"]):
        sys.exit(f"Refusing model '{cfg['model']}': it looks like a checkpoint already fine-tuned for sentiment "
                 "(possibly on IMDB test reviews). Use a general pre-trained model such as microsoft/deberta-v3-large.")
    if args.fresh_start:
        source = str(cfg["model"])
        if Path(source).exists() or re.match(r"^(?:[A-Za-z]:[\\/]|[\\/]|\.\.?[\\/])", source):
            sys.exit(f"--fresh-start requires a general Hugging Face model identifier, got local path '{source}'")
    if device == "cuda":
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        amp_dtype = None  # fp32 on MPS / CPU
    log(f"device={device} ({gpu_name}, {vram_gb:.1f} GB) torch={torch.__version__} "
        f"transformers={transformers.__version__} preset={preset} amp={amp_dtype}")
    log("config: " + json.dumps(cfg))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # ---------------- data
    data = Path(args.data_dir)
    train = pd.read_csv(data / "train.csv")
    test = pd.read_csv(data / "test.csv")
    y_all = (train["sentiment"].values == LABELS[1]).astype(np.int64)
    if args.folds:
        train_idx, hold_idx = kfold_indices(y_all, args.folds, args.fold, args.split_seed)
        split_desc = {"mode": "kfold", "folds": args.folds, "fold": args.fold}
    else:
        hold_mask = stratified_holdout(y_all, args.holdout_frac, args.split_seed)
        train_idx = np.flatnonzero(~hold_mask)
        hold_idx = np.flatnonzero(hold_mask)
        split_desc = {"mode": "holdout", "holdout_frac": args.holdout_frac}
    test_idx = np.arange(len(test))
    if args.smoke:
        rng = np.random.default_rng(0)
        train_idx = np.sort(rng.choice(train_idx, 512, replace=False))
        hold_idx = np.sort(rng.choice(hold_idx, 256, replace=False))
        test_idx = np.arange(128)
        args.max_len = min(args.max_len, 256)
        cfg["epochs"] = 1
        args.eval_every = 1.0
    log(f"train={len(train_idx):,} holdout={len(hold_idx):,} test={len(test_idx):,} "
        f"max_len={args.max_len} head={args.head} split={split_desc}")

    model_dir = out / "best_model"
    if args.fresh_start and model_dir.exists() and any(model_dir.iterdir()):
        sys.exit(f"refusing to reuse {model_dir}; choose a new --output folder for a fresh run")
    if args.predict_only and not model_dir.exists():
        sys.exit(f"{model_dir} not found; run training first (same --output).")
    tok_source = str(model_dir) if args.predict_only else cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(tok_source)
    cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    if cls_id is None or sep_id is None or pad_id is None:
        sys.exit("Tokenizer lacks CLS/SEP/PAD tokens; use a BERT/RoBERTa/DeBERTa-style model.")
    max_body = args.max_len - 2

    def encode(texts) -> list:
        docs = []
        for s in range(0, len(texts), 2000):
            chunk = [clean_review(t) for t in texts[s:s + 2000]]
            for ids in tokenizer(chunk, add_special_tokens=False)["input_ids"]:
                docs.append(np.asarray([cls_id] + head_tail(ids, max_body, args.head) + [sep_id], dtype=np.int64))
        return docs

    t0 = time.time()
    tr_rows = np.union1d(train_idx, hold_idx)
    train_docs = [None] * len(train)
    for i, d in zip(tr_rows, encode(train["review"].values[tr_rows].tolist())):
        train_docs[i] = d
    test_docs = encode(test["review"].values[test_idx].tolist())
    train_len = np.array([0 if d is None else len(d) for d in train_docs])
    test_len = np.array([len(d) for d in test_docs])
    n_trunc = int(np.sum(train_len[tr_rows] >= args.max_len))
    log(f"tokenised in {time.time() - t0:.0f}s; mean length {train_len[tr_rows].mean():.0f} tokens, "
        f"{n_trunc:,} of {len(tr_rows):,} training reviews truncated (head+tail)")

    def autocast():
        if amp_dtype is None:
            return torch.autocast(device_type="cpu", enabled=False)
        return torch.autocast(device_type="cuda", dtype=amp_dtype)

    @torch.no_grad()
    def predict(model, docs, lengths, idx, batch_size):
        model.eval()
        margin = np.zeros(len(idx), dtype=np.float64)
        pos = {int(j): k for k, j in enumerate(idx)}
        for b in sorted_batches(lengths, idx, batch_size):
            ids, mask = collate([docs[j] for j in b], pad_id)
            with autocast():
                logits = model(input_ids=torch.from_numpy(ids).to(device),
                               attention_mask=torch.from_numpy(mask).to(device)).logits.float()
            m = (logits[:, 1] - logits[:, 0]).cpu().numpy()
            for j, v in zip(b, m):
                margin[pos[int(j)]] = v
        model.train()
        return margin

    def save_holdout(margin, path):
        pd.DataFrame({"id": train["id"].values[hold_idx], "label": y_all[hold_idx],
                      "p_positive": sigmoid(margin), "margin": margin}).to_csv(path, index=False)

    history = []
    timings = {}
    if not args.predict_only:
        # ---------------- model and memory probe
        model = AutoModelForSequenceClassification.from_pretrained(cfg["model"], num_labels=2).to(device)

        def enable_ckpt():
            try:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                model.gradient_checkpointing_enable()

        if cfg["grad_ckpt"]:
            enable_ckpt()
        model.train()
        if device == "cuda":
            while True:
                try:
                    probe = torch.full((cfg["batch_size"], args.max_len), cls_id, dtype=torch.long, device=device)
                    with autocast():
                        logits = model(input_ids=probe, attention_mask=torch.ones_like(probe)).logits
                    loss = torch.nn.functional.cross_entropy(
                        logits.float(), torch.zeros(len(probe), dtype=torch.long, device=device))
                    loss.backward()
                    model.zero_grad(set_to_none=True)
                    del probe, logits, loss
                    torch.cuda.empty_cache()
                    break
                except torch.cuda.OutOfMemoryError:
                    model.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    if not cfg["grad_ckpt"]:
                        cfg["grad_ckpt"] = True
                        enable_ckpt()
                        log("out of memory in probe -> enabling gradient checkpointing")
                    elif cfg["batch_size"] > 1:
                        cfg["batch_size"] //= 2
                        cfg["grad_accum"] *= 2
                        log(f"out of memory in probe -> batch {cfg['batch_size']} x accumulation {cfg['grad_accum']}")
                    else:
                        sys.exit("Out of memory even with batch 1; try --preset base-lowmem or --max-len 384.")
            log(f"memory probe ok: batch {cfg['batch_size']} x accumulation {cfg['grad_accum']}, "
                f"grad_ckpt={cfg['grad_ckpt']}, peak {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GB")
        eval_bs = int(min(64, max(8, 2 * cfg["batch_size"])))

        best_acc, upd, last_eval = -1.0, 0, -1
        if args.verify_fresh:
            m0 = predict(model, train_docs, train_len, hold_idx, eval_bs)
            acc0 = float(np.mean((m0 > 0) == y_all[hold_idx]))
            history.append({"update": 0, "epoch": 0.0, "holdout_accuracy": acc0,
                            "phase": "fresh_initialisation"})
            log(f"  FRESH-START baseline: hold-out accuracy {acc0:.4%} at update 0")
            if acc0 > best_acc:
                best_acc = acc0
                model.save_pretrained(model_dir)
                tokenizer.save_pretrained(model_dir)
                save_holdout(m0, out / "holdout_predictions.csv")
                log(f"  initial baseline saved -> {model_dir}")

        # ---------------- optimiser and schedule
        no_decay = ("bias", "layernorm", "layer_norm", "LayerNorm")
        groups = [
            {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
             "weight_decay": args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
             "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(groups, lr=cfg["lr"], betas=(0.9, 0.98), eps=1e-6)
        shuffle_rng = np.random.default_rng(args.seed)
        n_batches = len(length_grouped_batches(train_len, train_idx, cfg["batch_size"], np.random.default_rng(0)))
        upd_per_epoch = math.ceil(n_batches / cfg["grad_accum"])
        n_epochs = int(math.ceil(cfg["epochs"]))
        total_upd = int(round(upd_per_epoch * cfg["epochs"]))
        warm = max(1, int(args.warmup * total_upd))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda u: min((u + 1) / warm, max(0.0, (total_upd - u) / max(1, total_upd - warm))))
        scaler = None
        if amp_dtype == torch.float16:
            try:
                scaler = torch.amp.GradScaler("cuda")
            except (AttributeError, TypeError):
                scaler = torch.cuda.amp.GradScaler()
        eval_every_upd = max(1, int(round(upd_per_epoch * args.eval_every)))
        log(f"{upd_per_epoch} updates/epoch, {total_upd} updates total, warm-up {warm}, eval every {eval_every_upd}")

        run_loss, run_n, t_train = 0.0, 0, time.time()
        done = False
        for epoch in range(n_epochs):
            batches = length_grouped_batches(train_len, train_idx, cfg["batch_size"], shuffle_rng)
            for bi, b in enumerate(batches):
                ids, mask = collate([train_docs[j] for j in b], pad_id)
                yb = torch.from_numpy(y_all[b]).to(device)
                with autocast():
                    logits = model(input_ids=torch.from_numpy(ids).to(device),
                                   attention_mask=torch.from_numpy(mask).to(device)).logits
                loss = torch.nn.functional.cross_entropy(logits.float(), yb)
                if not torch.isfinite(loss):
                    log("warning: non-finite loss, batch skipped")
                    opt.zero_grad(set_to_none=True)
                    continue
                (scaler.scale(loss / cfg["grad_accum"]) if scaler else loss / cfg["grad_accum"]).backward()
                run_loss += float(loss.detach())
                run_n += 1
                if (bi + 1) % cfg["grad_accum"] != 0 and bi + 1 != len(batches):
                    continue
                if scaler:
                    scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                upd += 1
                if upd % 50 == 0:
                    el = time.time() - t_train
                    log(f"epoch {epoch + 1} update {upd}/{total_upd} loss {run_loss / max(1, run_n):.4f} "
                        f"lr {sched.get_last_lr()[0]:.2e} elapsed {el / 60:.1f} min ETA {el / upd * (total_upd - upd) / 60:.1f} min")
                    run_loss, run_n = 0.0, 0
                end_of_epoch = bi + 1 == len(batches)
                if upd % eval_every_upd == 0 or end_of_epoch or upd >= total_upd:
                    if upd != last_eval:
                        last_eval = upd
                        m = predict(model, train_docs, train_len, hold_idx, eval_bs)
                        acc = float(np.mean((m > 0) == y_all[hold_idx]))
                        history.append({"update": upd, "epoch": round(upd / upd_per_epoch, 3), "holdout_accuracy": acc})
                        log(f"  HOLD-OUT accuracy {acc:.4%} at update {upd} (epoch {upd / upd_per_epoch:.2f})")
                        if acc > best_acc:
                            best_acc = acc
                            model.save_pretrained(model_dir)
                            tokenizer.save_pretrained(model_dir)
                            save_holdout(m, out / "holdout_predictions.csv")
                            log(f"  new best -> saved {model_dir}")
                        if not args.smoke and acc < 0.6 and upd >= upd_per_epoch // 2:
                            sys.exit("Training diverged (hold-out accuracy < 60%). Re-run with another --seed "
                                     "or a lower --lr (e.g. 7e-6).")
                if upd >= total_upd:
                    done = True
                    break
            if done:
                break
        timings["train_minutes"] = round((time.time() - t_train) / 60, 2)
        log(f"training finished in {timings['train_minutes']} min; best hold-out accuracy {best_acc:.4%}")
        del model, opt
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---------------- reload the best checkpoint and predict
    if not model_dir.exists():
        sys.exit(f"{model_dir} not found; run training first.")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    eval_bs = 32
    t0 = time.time()
    m_hold = predict(model, train_docs, train_len, hold_idx, eval_bs)
    hold_acc = float(np.mean((m_hold > 0) == y_all[hold_idx]))
    save_holdout(m_hold, out / "holdout_predictions.csv")
    m_test = predict(model, test_docs, test_len, np.arange(len(test_docs)), eval_bs)
    timings["predict_minutes"] = round((time.time() - t0) / 60, 2)
    test_ids = test["id"].values[test_idx]
    pd.DataFrame({"id": test_ids, "p_positive": sigmoid(m_test), "margin": m_test}).to_csv(
        out / "test_predictions.csv", index=False)
    log(f"best checkpoint: hold-out accuracy {hold_acc:.4%}; test positive share {np.mean(m_test > 0):.4f}")

    sub_info = None
    if not args.smoke:
        sub_info = write_submission(test_ids, m_test, data / "sample_submission.csv", out / "submission.csv")
        log(f"wrote {out / 'submission.csv'} sha256={sub_info['sha256']}")
        if args.folds:
            log("note: in --folds mode this is the *single-fold* prediction. The submission to use is the "
                "average of the folds, written by code/run_cv.py or code/assemble_cv.py.")
    else:
        log("SMOKE TEST OK - the pipeline runs end to end. Now start the full run without --smoke.")

    info_path = out / "run_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if (args.predict_only and info_path.exists()) else {}
    info.update({
        "mode": "predict-only" if args.predict_only else ("smoke" if args.smoke else "train"),
        "command": " ".join(sys.argv), "preset": preset, "config": cfg,
        "fresh_start": bool(args.fresh_start and not args.predict_only),
        "verify_fresh": bool(args.verify_fresh),
        "init_checkpoint": str(model_dir) if args.predict_only else str(cfg["model"]),
        "best_checkpoint": str(model_dir),
        "max_len": args.max_len, "head": args.head, "seed": args.seed, "split_seed": args.split_seed,
        "holdout_frac": args.holdout_frac, "n_train": int(len(train_idx)), "n_holdout": int(len(hold_idx)),
        "folds": args.folds, "fold": args.fold if args.folds else None, "split": split_desc,
        "holdout_accuracy": hold_acc, "test_positive_share": float(np.mean(m_test > 0)),
        "submission": sub_info,        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "transformers": transformers.__version__,
                        "device": device, "gpu": gpu_name, "vram_gb": None if math.isnan(vram_gb) else round(vram_gb, 1),
                        "amp": str(amp_dtype)},
        "data_sha256": {f: sha256_file(data / f) for f in ("train.csv", "test.csv")},
        "total_minutes": round((time.time() - t_start) / 60, 2),
    })
    if history:
        info["history"] = history
    info.setdefault("timings", {}).update(timings)
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    log(f"done in {info['total_minutes']} min -> {out}")


if __name__ == "__main__":
    main()
