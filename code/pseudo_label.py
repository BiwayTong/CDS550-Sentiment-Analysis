#!/usr/bin/env python3
"""Second training round with pseudo-labels (self-training) — no test labels are used.

Round 1 gives four fine-tuned models. On the 25,000 test reviews they agree almost everywhere;
where all of them agree *and* are confident, the review text is almost certainly labelled
correctly by the model. Those reviews are added to the training set with the model's own label
as a soft target, and a new model is fine-tuned on 22,500 labelled + the pseudo-labelled
reviews. The held-out 2,500 reviews stay untouched: their labels are never used for training
and continue to give an honest estimate, so the pseudo-labels can be judged.

Two safeguards: only reviews where every model agrees and the averaged margin is beyond a
threshold are used (``--min-margin``), and their loss is down-weighted (``--weight``) so that a
mistake costs less than a human-labelled review.

    python code/pseudo_label.py --data-dir data --runs runs/deberta runs/deberta-large-s1 \\
        runs/deberta-large-s2 runs/roberta-large --output pseudo/pseudo.csv
    python code/pseudo_label.py --data-dir data --stage train --pseudo pseudo/pseudo.csv \\
        --base runs/deberta --output runs/deberta-pl --preset large-lowmem

By default the student is loaded from the general pre-trained checkpoint in the preset. The
``--base`` path supplies the tokenizer and split context; it does not warm-start the weights.
Use ``--init-from-base`` only for an explicitly intended continuation experiment.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_transformer as ft  # noqa: E402

LABELS = ("negative", "positive")


def read_column(path: Path, ids: np.ndarray, column: str) -> np.ndarray:
    if not path.exists():
        sys.exit(f"{path} not found.\n"
                 f"  Run this script from the 'submission' folder (the one that contains data/ and "
                 f"runs/), and check that the run folder holds test_predictions.csv.\n"
                 f"  {path.parent} contains: "
                 f"{sorted(p.name for p in path.parent.iterdir()) if path.parent.is_dir() else 'nothing (folder missing)'}")
    d = pd.read_csv(path).set_index("id")
    missing = len(set(ids) - set(d.index))
    if missing:
        sys.exit(f"{path} is missing {missing} of the {len(ids):,} ids it should cover "
                 f"(and has a '{column}' column). Check that the run used the same "
                 f"--split-seed/--holdout-frac as the other runs.")
    return d.loc[ids, column].values


def conf_rule(P: np.ndarray, keep_frac: float, min_prob: float):
    """Decide which reviews get a pseudo-label, and return (keep, label, mean_p, threshold).

    Reviews are kept in two steps. First every model must agree on the label. Second, within each
    predicted class, the most confident ``keep_frac`` of that class is kept. Selecting per class
    rather than with one global probability cut matters: the models are systematically more
    confident on negative reviews (79 % of predicted negatives have p < 0.001, but only 60 % of
    predicted positives have p > 0.999), so a single global threshold silently drops positives and
    hands the round-2 model a training set that is 43 % positive instead of 50 %.
    """
    mean_p = P.mean(1)
    agree = (P >= 0.5).all(1) | (P < 0.5).all(1)
    label = (mean_p >= 0.5).astype(int)
    keep = np.zeros(len(P), dtype=bool)
    thr = {}
    for cls in (0, 1):
        side = agree & (label == cls) & ((mean_p >= min_prob) if cls else (mean_p <= 1 - min_prob))
        idx = np.flatnonzero(side)
        if len(idx) == 0:
            thr[cls] = None
            continue
        if keep_frac >= 1.0:
            keep[idx] = True
            thr[cls] = float(mean_p[idx].min() if cls else mean_p[idx].max())
            continue
        cut = np.quantile(mean_p[idx], 1 - keep_frac if cls else keep_frac)
        sel = idx[mean_p[idx] >= cut] if cls else idx[mean_p[idx] <= cut]
        keep[sel] = True
        thr[cls] = float(cut)
    return keep, label, mean_p, agree, thr


def calibrate(keep_frac: float, min_prob: float, holdout_frac: float, split_seed: int,
              data: Path, runs) -> None:
    """Measure the pseudo-label accuracy on the hold-out, where the true labels are known.

    A pseudo-label is only useful if it is more accurate than the labels the model already had.
    The same rule is applied to the 2,500 hold-out reviews and scored against the real labels, so
    the number below is an honest estimate of how good the added labels are.
    """
    train = pd.read_csv(data / "train.csv")
    y_all = (train["sentiment"].values == LABELS[1]).astype(np.int64)
    hold_mask = ft.stratified_holdout(y_all, holdout_frac, split_seed)
    hold_ids = train["id"].values[hold_mask]
    y = y_all[hold_mask]
    P = np.column_stack([read_column(Path(r) / "holdout_predictions.csv", hold_ids, "p_positive")
                         for r in runs])
    keep, label, mean_p, agree, _ = conf_rule(P, keep_frac, min_prob)
    print(f"calibration on the {len(y):,}-review hold-out (all models agree on {agree.mean():.1%})")
    print(f"  {'keep_frac':>9} {'min p':>7} {'kept':>7} {'kept %':>7} {'pos share':>10} "
          f"{'pseudo-label accuracy':>22}")
    for kf in (0.3, 0.5, keep_frac, 0.9):
        k, lab, _, _, _ = conf_rule(P, kf, 0.5)
        if k.sum() == 0:
            continue
        mark = "  <- chosen" if (abs(kf - keep_frac) < 1e-9 and abs(min_prob - 0.5) < 1e-9) else ""
        print(f"  {kf:>9.2f} {0.5:>7.2f} {int(k.sum()):>7,} {k.mean():>6.1%} "
              f"{np.mean(lab[k] == 1):>10.4f} {np.mean(lab[k] == y[k]):>21.4%}{mark}")
    if min_prob > 0.5:
        print(f"  ... plus a floor of p >= {min_prob:.3f} on every model:")
        for kf in (0.5, 0.7, 0.9):
            k, lab, _, _, _ = conf_rule(P, kf, min_prob)
            if k.sum() == 0:
                print(f"  {kf:>9.2f} {min_prob:>7.3f}         0      0.0%          -"
                      f"                     -   (nothing passes)")
                continue
            mark = "  <- chosen" if abs(kf - keep_frac) < 1e-9 else ""
            print(f"  {kf:>9.2f} {min_prob:>7.3f} {int(k.sum()):>7,} {k.mean():>6.1%} "
                  f"{np.mean(lab[k] == 1):>10.4f} {np.mean(lab[k] == y[k]):>21.4%}{mark}")
    if keep.sum():
        print(f"chosen rule keeps {keep.mean():.1%} of the hold-out at "
              f"{np.mean(label[keep] == y[keep]):.4%} accuracy (positive share "
              f"{np.mean(label[keep] == 1):.4f})")
    print("  (a pseudo-label that is right ~99.7 % of the time is worth adding at a loss weight of "
          "0.3-0.5; below ~98 % it risks teaching the model its own mistakes)")


def scatter_margin(margin: np.ndarray, batch, values) -> None:
    """Write one margin per review into ``margin``.

    ``batch`` holds *positions* into the document list that was just predicted, not ids of
    rows in train.csv. Keeping the two apart matters here: this script predicts a compact
    list of hold-out documents (index 0..n-1), while the hold-out itself is a scatter of row
    numbers through the 25,000-row file. Mapping a position back through the row-number
    dictionary raises KeyError — which is what used to abort training at the first
    evaluation, after the first epoch had already been paid for.
    """
    for j, v in zip(batch, values):
        margin[int(j)] = v


def require(path: Path, what: str) -> Path:
    """Fail with a readable message instead of a bare FileNotFoundError from pandas."""
    if not path.exists():
        where = Path.cwd()
        sys.exit(f"{what} not found: {path}\n"
                 f"  Current folder: {where}\n"
                 f"  Run the command from the 'submission' folder, i.e. the folder that contains "
                 f"data\\train.csv and runs\\. If your folder has another name or is on another "
                 f"drive, adjust --data-dir / --runs accordingly.")
    return path


def build_pseudo(args) -> None:
    data = Path(args.data_dir)
    test = pd.read_csv(require(data / "test.csv", "--data-dir file"))
    ids = test["id"].values
    runs = [Path(r) for r in args.runs]
    missing = [str(r) for r in runs if not (r / "test_predictions.csv").exists()]
    if missing:
        sys.exit("these run folders have no test_predictions.csv: "
                 + ", ".join(missing)
                 + "\n  Each of the four stage-2 runs must be present, e.g. runs\\deberta\\test_predictions.csv.")
    if args.calibrate:
        calibrate(args.keep_frac, args.min_prob, args.holdout_frac, args.split_seed, data, runs)
    P = np.column_stack([read_column(r / "test_predictions.csv", ids, "p_positive") for r in runs])
    keep, label, mean_p, agree, thr = conf_rule(P, args.keep_frac, args.min_prob)
    conf = np.abs(0.5 - mean_p) * 2  # 0 = a coin flip, 1 = every model is certain
    print(f"models: {len(runs)}; all models agree on {agree.sum():,} test reviews ({agree.mean():.1%})")
    if thr[0] is None or thr[1] is None:
        print("thresholds: " + ", ".join(
            f"{'negative' if c == 0 else 'positive'} p {'<=' if c == 0 else '>='} "
            f"{'none kept' if thr[c] is None else format(thr[c], '.4f')}" for c in (0, 1)))
    else:
        print(f"kept {keep.sum():,} of {len(ids):,} reviews ({keep.mean():.1%}); thresholds "
              f"p <= {thr[0]:.4f} for negative, p >= {thr[1]:.4f} for positive")
    if not keep.any():
        print("  nothing was kept -- lower --min-prob or raise --keep-frac, "
              "otherwise round 2 would train on the labelled data only")
    if keep.any():
        print(f"kept positive share {label[keep].mean():.4f} (the rule keeps each class separately, "
              f"so this stays near 0.5)")
        for r, col in zip(runs, P.T):
            wrong = (col[keep] >= 0.5) != (label[keep] == 1)
            print(f"  {r.name:<20s} disagrees with {wrong.sum():,} kept pseudo-labels "
                  f"({wrong.mean():.2%} of them)")
    pseudo = pd.DataFrame({"id": ids[keep], "label": label[keep], "mean_p": mean_p[keep],
                           "confidence": conf[keep]})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pseudo.to_csv(out, index=False)
    (out.with_suffix(".json")).write_text(json.dumps({
        "runs": [str(r) for r in runs], "n_test": int(len(ids)),
        "n_agreeing": int(agree.sum()), "n_kept": int(keep.sum()),
        "kept_fraction": float(keep.mean()), "keep_frac": args.keep_frac,
        "min_prob": args.min_prob,
        "threshold_negative": thr[0], "threshold_positive": thr[1],
        "kept_positive_share": float(label[keep].mean()) if keep.any() else None,
        "model_disagreement_with_pseudo": [int(((col[keep] >= 0.5) != (label[keep] == 1)).sum())
                                           for col in P.T],
    }, indent=2), encoding="utf-8")
    print(f"wrote {out}")


def train_pseudo(args) -> None:
    import torch
    import transformers
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    transformers.logging.set_verbosity_error()
    t_start = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    log = ft.Logger(out / "train_log.txt")
    log("=" * 70)
    log("command: " + " ".join(sys.argv))

    cfg = dict(ft.PRESETS[args.preset] if args.preset != "auto" else ft.PRESETS["large-lowmem"])
    for key in ("lr", "epochs", "batch_size", "grad_accum"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.grad_ckpt:
        cfg["grad_ckpt"] = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not args.allow_cpu:
        sys.exit("No GPU found; pass --allow-cpu to run anyway (many hours).")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3 if device == "cuda" else float("nan")
    amp_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device == "cuda" else None)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    log(f"device={device} ({torch.cuda.get_device_name(0) if device == 'cuda' else '-'}, "
        f"{vram:.1f} GB) torch={torch.__version__} preset={args.preset} amp={amp_dtype}")
    log("config: " + json.dumps(cfg))

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    data = Path(args.data_dir)
    train = pd.read_csv(data / "train.csv")
    test = pd.read_csv(data / "test.csv")
    y_all = (train["sentiment"].values == LABELS[1]).astype(np.int64)
    hold_mask = ft.stratified_holdout(y_all, args.holdout_frac, args.split_seed)
    train_idx = np.flatnonzero(~hold_mask)
    hold_idx = np.flatnonzero(hold_mask)

    pseudo = pd.read_csv(args.pseudo)
    test_by_id = dict(zip(test["id"].values, test["review"].values))
    p_texts = [test_by_id[i] for i in pseudo["id"].values]
    p_y = pseudo["label"].values.astype(np.int64)
    log(f"train={len(train_idx):,} holdout={len(hold_idx):,} pseudo={len(pseudo):,} "
        f"(weight {args.weight}, positive share of pseudo {p_y.mean():.4f})")
    if len(pseudo) == 0:
        sys.exit("no pseudo-labelled reviews; re-run the --stage pseudo step with a lower threshold")

    model_dir = out / "best_model"
    if model_dir.exists() and any(model_dir.iterdir()):
        sys.exit(f"refusing to reuse {model_dir}; choose a new --output folder")
    base_dir = Path(args.base) / "best_model" if args.init_from_base and args.base else None
    if args.init_from_base and base_dir is None:
        sys.exit("--init-from-base requires --base RUN_FOLDER")
    if base_dir is not None and not base_dir.exists():
        sys.exit(f"base checkpoint not found: {base_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(base_dir) if args.init_from_base else cfg["model"])
    cls_id, sep_id, pad_id = tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id
    max_body = args.max_len - 2

    def encode(texts):
        docs = []
        for s in range(0, len(texts), 2000):
            chunk = [ft.clean_review(t) for t in texts[s:s + 2000]]
            for ids in tokenizer(chunk, add_special_tokens=False)["input_ids"]:
                docs.append(np.asarray([cls_id] + ft.head_tail(ids, max_body, args.head) + [sep_id],
                                       dtype=np.int64))
        return docs

    texts = train["review"].values[train_idx].tolist() + p_texts
    y = np.concatenate([y_all[train_idx], p_y])
    w = np.concatenate([np.ones(len(train_idx)), np.full(len(p_y), args.weight)])
    log(f"tokenising {len(texts):,} reviews (train + pseudo)...")
    docs = encode(texts)
    lens = np.array([len(d) for d in docs])
    hold_docs = encode(train["review"].values[hold_idx].tolist())
    hold_lens = np.array([len(d) for d in hold_docs])
    log(f"mean length {lens.mean():.0f} tokens; {int(np.sum(lens >= args.max_len)):,} truncated")

    init_source = str(base_dir) if args.init_from_base else cfg["model"]
    model = AutoModelForSequenceClassification.from_pretrained(init_source).to(device)
    if args.init_from_base:
        log(f"initialised from the round-1 checkpoint {base_dir}")
    else:
        log(f"fresh student initialised from the general checkpoint {cfg['model']}")
    if cfg["grad_ckpt"]:
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    model.train()

    def autocast():
        if amp_dtype is None:
            return torch.autocast(device_type="cpu", enabled=False)
        return torch.autocast(device_type="cuda", dtype=amp_dtype)

    if device == "cuda":
        while True:
            try:
                probe = torch.full((cfg["batch_size"], args.max_len), cls_id, dtype=torch.long, device=device)
                with autocast():
                    lg = model(input_ids=probe, attention_mask=torch.ones_like(probe)).logits
                torch.nn.functional.cross_entropy(lg.float(), torch.zeros(len(probe), dtype=torch.long,
                                                                         device=device)).backward()
                model.zero_grad(set_to_none=True); del probe, lg
                torch.cuda.empty_cache(); break
            except torch.cuda.OutOfMemoryError:
                model.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                if not cfg["grad_ckpt"]:
                    cfg["grad_ckpt"] = True
                    try:
                        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    except TypeError:
                        model.gradient_checkpointing_enable()
                    log("out of memory -> gradient checkpointing enabled")
                elif cfg["batch_size"] > 1:
                    cfg["batch_size"] //= 2; cfg["grad_accum"] *= 2
                    log(f"out of memory -> batch {cfg['batch_size']} x accumulation {cfg['grad_accum']}")
                else:
                    sys.exit("out of memory even with batch 1")
        log(f"memory probe ok: batch {cfg['batch_size']} x accumulation {cfg['grad_accum']}, "
            f"peak {torch.cuda.max_memory_allocated() / 1024 ** 3:.1f} GB")

    # Evaluation batch size, resolved from cfg rather than the raw argument: the batch-size
    # flag is an optional override and stays None when the preset supplies the value (and the
    # memory probe above may have halved it), which made every in-training evaluation crash.
    eval_bs = int(min(64, max(8, 2 * cfg["batch_size"])))

    no_decay = ("bias", "layernorm", "layer_norm", "LayerNorm")
    groups = [{"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
               "weight_decay": args.weight_decay},
              {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
               "weight_decay": 0.0}]
    opt = torch.optim.AdamW(groups, lr=cfg["lr"], betas=(0.9, 0.98), eps=1e-6)
    rng = np.random.default_rng(args.seed)
    n_batches = len(ft.length_grouped_batches(lens, np.arange(len(docs)), cfg["batch_size"],
                                              np.random.default_rng(0)))
    upd_per_epoch = math.ceil(n_batches / cfg["grad_accum"])
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
    eval_every = max(1, int(round(upd_per_epoch * args.eval_every)))
    log(f"{upd_per_epoch} updates/epoch, {total_upd} total, eval every {eval_every}")

    @torch.no_grad()
    def predict_hold(model):
        model.eval()
        # one margin per hold-out document, in the order of hold_docs (not train.csv row order)
        margin = np.zeros(len(hold_docs), dtype=np.float64)
        for b in ft.sorted_batches(hold_lens, np.arange(len(hold_docs)), eval_bs):
            ids, mask = ft.collate([hold_docs[j] for j in b], pad_id)
            with autocast():
                lg = model(input_ids=torch.from_numpy(ids).to(device),
                           attention_mask=torch.from_numpy(mask).to(device)).logits.float()
            scatter_margin(margin, b, (lg[:, 1] - lg[:, 0]).cpu().numpy())
        model.train()
        return margin

    history, best_acc, upd, last_eval = [], -1.0, 0, -1

    def evaluate(tag: str) -> None:
        """Score the hold-out and keep the checkpoint if it is the best so far.

        Called once before training and then every ``--eval-every`` epochs. Evaluating first
        means a broken evaluation costs a minute instead of the whole run: this script is
        started from the round-1 checkpoint, so the baseline is what round 1 already achieved
        and round 2 has to beat it to be kept.
        """
        nonlocal best_acc, last_eval
        last_eval = upd
        try:
            m = predict_hold(model)
        except Exception:  # never lose a finished epoch to a scoring bug
            log(f"WARNING: hold-out evaluation failed at update {upd}; training continues. "
                f"Traceback:\n{traceback.format_exc()}")
            return
        acc = float(np.mean((m > 0) == y_all[hold_idx]))
        history.append({"update": upd, "epoch": round(upd / upd_per_epoch, 3), "holdout_accuracy": acc})
        log(f"  HOLD-OUT accuracy {acc:.4%} at update {upd} (epoch {upd / upd_per_epoch:.2f}){tag}")
        if acc > best_acc:
            best_acc = acc
            model.save_pretrained(model_dir); tokenizer.save_pretrained(model_dir)
            pd.DataFrame({"id": train["id"].values[hold_idx], "label": y_all[hold_idx],
                          "p_positive": ft.sigmoid(m), "margin": m}).to_csv(
                out / "holdout_predictions.csv", index=False)
            log(f"  new best -> saved {model_dir}")

    run_loss, run_n, run_w, t_train = 0.0, 0, 0.0, time.time()
    log("baseline evaluation (round-1 weights, before any round-2 update):")
    evaluate("  <- baseline, round 2 must beat this")
    done = False
    for epoch in range(int(math.ceil(cfg["epochs"]))):
        batches = ft.length_grouped_batches(lens, np.arange(len(docs)), cfg["batch_size"], rng)
        for bi, b in enumerate(batches):
            ids, mask = ft.collate([docs[j] for j in b], pad_id)
            yb = torch.from_numpy(y[b]).to(device)
            wb = torch.from_numpy(w[b]).float().to(device)
            with autocast():
                lg = model(input_ids=torch.from_numpy(ids).to(device),
                           attention_mask=torch.from_numpy(mask).to(device)).logits
            loss = torch.nn.functional.cross_entropy(lg.float(), yb, reduction="none")
            loss = (loss * wb).sum() / wb.sum()
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); continue
            (scaler.scale(loss / cfg["grad_accum"]) if scaler else loss / cfg["grad_accum"]).backward()
            run_loss += float(loss.detach()); run_n += 1; run_w += float(wb.sum())
            if (bi + 1) % cfg["grad_accum"] != 0 and bi + 1 != len(batches):
                continue
            if scaler:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scaler:
                scaler.step(opt); scaler.update()
            else:
                opt.step()
            sched.step(); opt.zero_grad(set_to_none=True); upd += 1
            if upd % 50 == 0:
                el = time.time() - t_train
                log(f"epoch {epoch + 1} update {upd}/{total_upd} loss {run_loss / max(1, run_n):.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e} elapsed {el / 60:.1f} min "
                    f"ETA {el / upd * (total_upd - upd) / 60:.1f} min")
                run_loss, run_n = 0.0, 0
            if (upd % eval_every == 0 or upd >= total_upd) and upd != last_eval:
                evaluate("")
            if upd >= total_upd:
                done = True; break
        if done:
            break
    train_minutes = round((time.time() - t_train) / 60, 2)
    log(f"training finished in {train_minutes} min; best hold-out accuracy {best_acc:.4%}")
    if not model_dir.exists():
        # No evaluation ever succeeded, so nothing was checkpointed. Keep the weights anyway
        # rather than crashing at the reload below, and say so loudly.
        log("WARNING: no hold-out evaluation succeeded during training; saving the final weights "
            "so the run still produces predictions. The hold-out accuracy below is the final epoch's.")
        model.save_pretrained(model_dir); tokenizer.save_pretrained(model_dir)
    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()

    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    test_docs = encode(test["review"].values.tolist())
    test_lens = np.array([len(d) for d in test_docs])

    @torch.no_grad()
    def predict_all(docs, lens, bs):
        model.eval()
        margin = np.zeros(len(docs), dtype=np.float64)
        for b in ft.sorted_batches(lens, np.arange(len(docs)), bs):
            ids, mask = ft.collate([docs[j] for j in b], pad_id)
            with autocast():
                lg = model(input_ids=torch.from_numpy(ids).to(device),
                           attention_mask=torch.from_numpy(mask).to(device)).logits.float()
            for j, v in zip(b, (lg[:, 1] - lg[:, 0]).cpu().numpy()):
                margin[int(j)] = v
        return margin

    m_hold = predict_all(hold_docs, hold_lens, eval_bs)
    hold_acc = float(np.mean((m_hold > 0) == y_all[hold_idx]))
    m_test = predict_all(test_docs, test_lens, eval_bs)
    test_ids = test["id"].values
    pd.DataFrame({"id": train["id"].values[hold_idx], "label": y_all[hold_idx],
                  "p_positive": ft.sigmoid(m_hold), "margin": m_hold}).to_csv(
        out / "holdout_predictions.csv", index=False)
    pd.DataFrame({"id": test_ids, "p_positive": ft.sigmoid(m_test), "margin": m_test}).to_csv(
        out / "test_predictions.csv", index=False)
    sub_info = ft.write_submission(test_ids, m_test, data / "sample_submission.csv", out / "submission.csv")
    log(f"best checkpoint: hold-out accuracy {hold_acc:.4%}; test positive share {np.mean(m_test > 0):.4f}")
    (out / "run_info.json").write_text(json.dumps({
        "mode": "pseudo-label", "command": " ".join(sys.argv), "preset": args.preset, "config": cfg,
        "base_run": args.base, "init_from_base": bool(args.init_from_base),
        "init_checkpoint": str(base_dir) if args.init_from_base else str(cfg["model"]),
        "init_type": "local_finetuned" if args.init_from_base else "general_pretrained",
        "pseudo_csv": args.pseudo,
        "n_pseudo": int(len(pseudo)), "pseudo_weight": args.weight,
        "pseudo_positive_share": float(p_y.mean()),
        "max_len": args.max_len, "head": args.head, "seed": args.seed, "split_seed": args.split_seed,
        "holdout_frac": args.holdout_frac, "n_train": int(len(train_idx)), "n_holdout": int(len(hold_idx)),
        "holdout_accuracy": hold_acc, "test_positive_share": float(np.mean(m_test > 0)),
        "history": history, "submission": sub_info,
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__,
                        "transformers": transformers.__version__, "device": device,
                        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None},
        "timings": {"train_minutes": train_minutes},
        "total_minutes": round((time.time() - t_start) / 60, 2),
    }, indent=2), encoding="utf-8")
    log(f"done in {(time.time() - t_start) / 60:.1f} min -> {out}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--stage", default="pseudo", choices=["pseudo", "train"])
    p.add_argument("--runs", nargs="+", help="stage pseudo: run folders that provide test_predictions.csv")
    p.add_argument("--output", required=True)
    p.add_argument("--keep-frac", type=float, default=0.7,
                   help="stage pseudo: keep the most confident share of each predicted class")
    p.add_argument("--min-prob", type=float, default=0.5,
                   help="stage pseudo: floor on every model's probability of the chosen label")
    p.add_argument("--quantile", type=float, default=None, help="(removed; kept for old command lines)")
    p.add_argument("--min-margin", type=float, default=None, help="(removed; kept for old command lines)")
    p.add_argument("--no-calibrate", dest="calibrate", action="store_false",
                   help="skip the hold-out measurement of the pseudo-label accuracy")
    p.add_argument("--pseudo", help="stage train: csv written by the pseudo step")
    p.add_argument("--base", help="stage train: round-1 run whose tokenizer (and optionally weights) to start from")
    p.add_argument("--init-from-base", action="store_true",
                   help="stage train: warm-start from the round-1 weights instead of the pre-trained ones")
    p.add_argument("--preset", default="auto")
    p.add_argument("--weight", type=float, default=0.5, help="loss weight of a pseudo-labelled review")
    p.add_argument("--lr", type=float); p.add_argument("--epochs", type=float)
    p.add_argument("--batch-size", type=int); p.add_argument("--grad-accum", type=int)
    p.add_argument("--grad-ckpt", action="store_true")
    p.add_argument("--max-len", type=int, default=512); p.add_argument("--head", type=int, default=128)
    p.add_argument("--holdout-frac", type=float, default=0.10)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--warmup", type=float, default=0.10); p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--eval-every", type=float, default=0.5)
    p.add_argument("--allow-cpu", action="store_true")
    args = p.parse_args(argv)
    if args.stage == "pseudo" and not args.runs:
        p.error("--runs is required for --stage pseudo")
    if args.stage == "train" and not args.pseudo:
        p.error("--pseudo is required for --stage train")
    return args


if __name__ == "__main__":
    a = parse_args()
    build_pseudo(a) if a.stage == "pseudo" else train_pseudo(a)
