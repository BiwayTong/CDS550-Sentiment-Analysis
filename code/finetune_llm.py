#!/usr/bin/env python3
"""Fine-tune a decoder LLM (default Qwen3-8B-Base) with LoRA as an IMDB sentiment classifier.

CDS550 individual project, stage 4 (GPU). Same data rules as stages 2 and 3: only the
25,000 labelled reviews in train.csv are used for training; test labels are never used.

Why a decoder LLM
-----------------
The stage-2/3 encoders (DeBERTa-v3, RoBERTa, ELECTRA) were pre-trained on roughly the same
~160 GB of text and their margins correlate at 0.98-0.99, so another encoder adds almost no
information. A base LLM pre-trained on tens of trillions of tokens knows far more about films,
actors, irony and mixed reviews, and is expected to make *different* mistakes.

Method
------
* Input:  "Movie review:\\n" + review + "\\n\\nQuestion: Overall, is this review of the movie
  positive or negative?\\nAnswer:"  (the review is head+tail truncated so that the whole
  sequence fits in --max-len tokens; the prompt is never cut).
* Representation: the final hidden state of the last prompt token. Attention is causal, so
  that token has read the whole review and the question.
* Head: a 2-way linear layer initialised with the LM-head rows of " negative" and " positive".
  At update 0 the classifier is therefore the model's own zero-shot answer, which gives a
  meaningful update-0 check (AUC clearly above 0.5) before any training is spent.
* Training: LoRA (r=16, alpha=32, dropout 0.05) on all attention and MLP projections plus the
  head; AdamW, cosine schedule with 5 % warm-up, 1 epoch, 16 reviews per update, bf16 autocast,
  gradient checkpointing, length-grouped micro-batches under a token budget.
* The *final* weights are used (no checkpoint selection on the validation fold), so the
  validation score is an honest out-of-fold estimate.
* After training the adapter is saved, the model is freed and re-loaded from disk, and all
  predictions come from the re-loaded copy: the files on disk are exactly what produced the
  submission. A reload check compares both copies on a few hundred reviews.

Only *base* (pre-trained, not instruction-tuned) checkpoints are accepted, and nothing that was
fine-tuned for sentiment, embedding or re-ranking, because such models may have been trained on
IMDB labels (and half of our test reviews come from the original IMDB training split).
The one deliberate exception is stage 7 (preset ``qwen3.8-27b``): Qwen3.8-27B has no released base
checkpoint, so its post-trained checkpoint is allowed by name (POST_TRAINED_ALLOWED) for that preset
only. Its post-training data are not published and may include sentiment or IMDB-style tasks, so its
results carry a contamination caveat in the report; every other post-trained checkpoint is still refused.

Outputs in --output use the same layout as finetune_transformer.py, so run_cv.py, assemble_cv.py
and ensemble_cv.py work unchanged: holdout_predictions.csv, test_predictions.csv, submission.csv,
run_info.json, train_log.txt, plus adapter/ (LoRA weights, head.pt, llm_meta.json).

Run from the ``submission`` folder (Windows: ``py -3.12 code\\finetune_llm.py ...``)::

    # 2-3 minute end-to-end check with a tiny model (downloads 1.2 GB once)
    py -3.12 code/finetune_llm.py --data-dir data --folds 5 --fold 0 --preset qwen3-0.6b --output runs/_smoke_llm --smoke
    # the same check with the real model: memory, speed and an estimate of the full-fold time
    py -3.12 code/finetune_llm.py --data-dir data --folds 5 --fold 0 --preset qwen3-8b --output runs/_smoke_llm8b --smoke
    # one fold (the stage-4 pilot); it lands where run_cv.py expects fold 0
    py -3.12 code/finetune_llm.py --data-dir data --folds 5 --fold 0 --preset qwen3-8b --output runs/cv-qwen3/qwen3-f0 --eval-every 0.5
    # all remaining folds, resumable
    py -3.12 code/run_cv.py --trainer llm --preset qwen3-8b --data-dir data --folds 5 --output runs/cv-qwen3 --name qwen3
    # re-create the predictions of a finished fold from its saved adapter (reproducibility check)
    py -3.12 code/finetune_llm.py --data-dir data --folds 5 --fold 0 --output runs/cv-qwen3/qwen3-f0 --predict-only
    # stage 5 on a rented 96 GB GPU (Linux): the same recipe with Qwen3-14B-Base
    python code/run_cv.py --trainer llm --preset qwen3-14b --data-dir data --folds 5 --output runs/cv-qwen3-14b --name qwen3-14b --eval-every 0.5
    # stage 6 on the same GPU: Mistral-Small-24B-Base-2501 (another company's pre-training data)
    python code/run_cv.py --trainer llm --preset mistral-24b --data-dir data --folds 5 --output runs/cv-mistral-24b --name mistral-24b --eval-every 0.5
    # stage 7 (needs transformers >= 5.2 and flash-linear-attention): Qwen3.8-27B, post-trained (see above)
    python code/run_cv.py --trainer llm --preset qwen3.8-27b --data-dir data --folds 5 --output runs/cv-qwen38-27b --name qwen38-27b --eval-every 0.5
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util
import json
import math
import platform
import random
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finetune_transformer as ft  # noqa: E402  (numpy/pandas only at import time)

LABELS = ("negative", "positive")
PROMPT_VERSION = "v1"
PREFIX = "Movie review:\n"
SUFFIX = "\n\nQuestion: Overall, is this review of the movie positive or negative?\nAnswer:"
VERBALIZER = (" negative", " positive")          # index 0 = negative, 1 = positive
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

_COMMON = dict(lr=1e-4, epochs=1.0, seqs_per_update=16, token_budget=4096, max_micro=16,
               lora_r=16, lora_alpha=32, lora_dropout=0.05)
PRESETS = {
    # name: base checkpoint + training settings. All are Apache-2.0, not gated on Hugging Face.
    # qwen3-14b (stage 5, rented 96 GB GPU): the same recipe as qwen3-8b. Its bf16 weights alone take
    # ~30 GB, so it needs a GPU with about 48 GB or more; it starts without gradient checkpointing
    # (faster), and the memory probe switches checkpointing on by itself if the GPU is too small.
    "qwen3-14b": dict(_COMMON, model="Qwen/Qwen3-14B-Base", grad_ckpt=False),
    # mistral-24b (stage 6, rented 96 GB GPU): the same recipe on a base model from another company,
    # i.e. other pre-training data and another tokenizer (Tekken, 131k vocabulary). bf16 weights take
    # ~47 GB and the MLP is 32,768 wide, so it starts with gradient checkpointing on.
    # fix_mistral_regex: the Hugging Face tokenizer.json of the Mistral Small 3 family ships a wrong
    # pre-tokenizer regex (transformers >= 4.57.2 can patch it on load); check_tokenizer.py compares the
    # result with Mistral's own tokenizer (mistral_common) before any GPU time is spent.
    "mistral-24b": dict(_COMMON, model="mistralai/Mistral-Small-24B-Base-2501", grad_ckpt=True,
                        tokenizer_kwargs={"fix_mistral_regex": True}),
    # qwen3.8-27b (stage 7, rented GPU with >= 80 GB): the same recipe on Qwen3.8-27B. Three things differ
    # from the other presets, each switched on only by the keys below:
    #  * post_trained: the checkpoint is post-trained (no -Base exists); allowed via POST_TRAINED_ALLOWED;
    #  * loader "vlm": the checkpoint is a vision-language model (Qwen3_5ForConditionalGeneration, needs
    #    transformers >= 5.2); it is loaded whole and only its text backbone is used (no images are passed);
    #  * lora_targets "text-linear": 48 of its 64 layers are Gated DeltaNet (linear attention) whose
    #    projections have other names than q/k/v/o_proj, so LoRA goes on every linear layer of the text
    #    backbone (attention of both kinds + MLP), and never on the vision tower, the MTP head or lm_head.
    # bf16 weights ~55 GB, so gradient checkpointing is on from the start.
    "qwen3.8-27b": dict(_COMMON, model="Qwen/Qwen3.8-27B", grad_ckpt=True, post_trained=True,
                        loader="vlm", lora_targets="text-linear"),
    "qwen3-8b": dict(_COMMON, model="Qwen/Qwen3-8B-Base", grad_ckpt=True),
    "qwen3-4b": dict(_COMMON, model="Qwen/Qwen3-4B-Base", grad_ckpt=True),
    "qwen3-1.7b": dict(_COMMON, model="Qwen/Qwen3-1.7B-Base", grad_ckpt=True),
    "qwen3-0.6b": dict(_COMMON, model="Qwen/Qwen3-0.6B-Base", grad_ckpt=False),  # smoke tests only
}

# Checkpoints that may have seen sentiment labels (instruction/chat tuning, sentiment classifiers,
# embedding and re-ranking models trained on MTEB-style data that includes IMDB) are refused.
FORBIDDEN_RE = re.compile(
    r"imdb|sst-?2|sentiment|emotion|polarity|yelp|amazon|instruct|chat|classif|"
    r"sft|dpo|orpo|kto|rlhf|reward|embed|rerank|(?:^|[-_/])(?:it|gte|bge|e5)(?:$|[-_/])",
    re.I)


# Post-trained checkpoints accepted *only* when the preset says post_trained=True (folder name, lower case).
POST_TRAINED_ALLOWED = ("qwen3.8-27b",)
# Module-path components that are never given LoRA under lora_targets="text-linear".
NON_TEXT_PARTS = ("visual", "vision", "vision_tower", "vision_model", "merger", "mtp", "lm_head")


# --------------------------------------------------------------------------- helpers (no torch)
def check_model_name(name: str, allow_post_trained: bool = False) -> None:
    """Raise ValueError unless ``name`` looks like a general *base* checkpoint.

    Only the last two path components are inspected (``org/name`` for a Hugging Face id, or
    ``parent/folder`` for a local download), so an unrelated folder higher up cannot trigger it.
    ``allow_post_trained`` (set by a preset with post_trained=True) lifts the Qwen3 "-Base" rule for the
    folder names in POST_TRAINED_ALLOWED only; the instruction/sentiment/embedding patterns still apply.
    """
    parts = [p for p in re.split(r"[\\/]+", str(name).strip()) if p]
    if not parts:
        raise ValueError("empty model name")
    tail = "/".join(parts[-2:])
    leaf = parts[-1].lower()
    m = FORBIDDEN_RE.search(tail)
    if m:
        raise ValueError(f"refusing model '{name}': '{m.group(0)}' suggests an instruction-tuned, "
                         "sentiment, embedding or re-ranking checkpoint, which may have been trained on "
                         "IMDB labels. Use a base checkpoint such as Qwen/Qwen3-8B-Base.")
    if allow_post_trained and leaf in POST_TRAINED_ALLOWED:
        return
    if "qwen3" in leaf and not leaf.endswith("-base"):
        raise ValueError(f"refusing model '{name}': for Qwen3 the post-trained (chat) checkpoints carry no "
                         "'Instruct' suffix; the pre-trained ones end in '-Base' (e.g. Qwen/Qwen3-8B-Base).")


def select_lora_targets(linear_names) -> list:
    """Full names of the linear layers that get LoRA under lora_targets="text-linear".

    Every linear layer of the text backbone is kept; layers of the vision tower, the multi-token-prediction
    head and the output head are dropped (matched on whole path components, see NON_TEXT_PARTS).
    """
    keep = []
    for name in linear_names:
        parts = str(name).split(".")
        if any(p in NON_TEXT_PARTS for p in parts):
            continue
        keep.append(str(name))
    if not keep:
        raise ValueError("no linear layer left for LoRA after removing the vision / MTP / output layers")
    return keep


def target_summary(names) -> dict:
    """{last name component: count}, for the log (e.g. {'q_proj': 16, 'in_proj_qkv': 48, ...})."""
    out: dict = {}
    for n in names:
        k = str(n).rsplit(".", 1)[-1]
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def hidden_of(config) -> int:
    """Hidden size of a text model's config, or of the text part of a vision-language config."""
    h = getattr(config, "hidden_size", None)
    if h is None and getattr(config, "text_config", None) is not None:
        h = getattr(config.text_config, "hidden_size", None)
    if h is None:
        raise ValueError("cannot find hidden_size in the model config")
    return int(h)


def find_text_backbone(m):
    """The decoder stack (embeddings + layers + final norm) inside a vision-language checkpoint."""
    for get in (lambda x: x.model.language_model, lambda x: x.language_model, lambda x: x.get_decoder()):
        try:
            tb = get(m)
        except (AttributeError, TypeError, NotImplementedError):
            continue
        if tb is not None and hasattr(tb, "layers"):
            return tb
    raise RuntimeError(f"cannot find the text backbone inside {type(m).__name__}")


def auto_preset(vram_gb: float) -> str:
    if vram_gb >= 22:
        return "qwen3-8b"
    if vram_gb >= 14:
        return "qwen3-4b"
    return "qwen3-1.7b"


def build_input(review_ids, prefix_ids, suffix_ids, bos_id, max_len: int, head: int) -> np.ndarray:
    """[BOS] + prefix + head/tail-truncated review + suffix, never longer than ``max_len``."""
    fixed = len(prefix_ids) + len(suffix_ids) + (1 if bos_id is not None else 0)
    max_body = int(max_len) - fixed
    if max_body < 16:
        raise ValueError(f"--max-len {max_len} leaves only {max_body} tokens for the review")
    body = ft.head_tail(review_ids, max_body, min(int(head), max_body))
    seq = ([int(bos_id)] if bos_id is not None else []) + [int(t) for t in prefix_ids] + \
        [int(t) for t in body] + [int(t) for t in suffix_ids]
    return np.asarray(seq, dtype=np.int64)


def _pack_micro(ex, lengths, token_budget: int, max_micro: int) -> list:
    """Greedy packing of ``ex`` (sorted by decreasing length) into micro-batches.

    A micro-batch is padded to its first (= longest) member, so the constraint is
    ``count * longest <= token_budget`` and ``count <= max_micro``. A single sequence longer
    than the budget still forms a micro-batch of one.
    """
    micro, cur = [], []
    for j in ex:
        j = int(j)
        if cur and ((len(cur) + 1) * int(lengths[cur[0]]) > token_budget or len(cur) >= max_micro):
            micro.append(np.asarray(cur, dtype=np.int64))
            cur = []
        cur.append(j)
    if cur:
        micro.append(np.asarray(cur, dtype=np.int64))
    return micro


def plan_updates(lengths, idx, seqs_per_update: int, token_budget: int, max_micro: int,
                 rng: np.random.Generator, group: int = 64) -> list:
    """One epoch of optimiser updates; each update is a list of micro-batches (row-index arrays).

    Rows are shuffled, cut into chunks of ``seqs_per_update * group``, sorted by length inside a
    chunk and cut into updates of ``seqs_per_update`` consecutive rows (similar lengths -> little
    padding). The updates are then shuffled. Every row of ``idx`` appears exactly once.
    """
    lengths = np.asarray(lengths)
    idx = np.asarray(idx, dtype=np.int64)
    if len(idx) == 0:
        return []
    idx = idx[rng.permutation(len(idx))]
    chunk = max(1, int(seqs_per_update) * int(group))
    updates = []
    for s in range(0, len(idx), chunk):
        c = idx[s:s + chunk]
        c = c[np.argsort(-lengths[c], kind="stable")]
        for u in range(0, len(c), seqs_per_update):
            updates.append(_pack_micro(c[u:u + seqs_per_update], lengths, token_budget, max_micro))
    order = rng.permutation(len(updates))
    return [updates[i] for i in order]


def budget_batches(lengths, idx, token_budget: int, max_bs: int) -> list:
    """Inference batches: longest first, packed under the same token rule."""
    lengths = np.asarray(lengths)
    idx = np.asarray(idx, dtype=np.int64)
    idx = idx[np.argsort(-lengths[idx], kind="stable")]
    return _pack_micro(idx, lengths, token_budget, max_bs)


def auc_score(y, s) -> float:
    """Rank-based ROC AUC (ties get half credit). NaN if one class is missing."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s, dtype=np.float64)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").values
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def cosine_lr(u: int, warm: int, total: int) -> float:
    """LR multiplier for update ``u`` (0-based): linear warm-up, then cosine decay to 0."""
    if u < warm:
        return (u + 1) / max(1, warm)
    prog = min(1.0, (u - warm) / max(1, total - warm))
    return 0.5 * (1.0 + math.cos(math.pi * prog))


class MemoryTight(RuntimeError):
    """The memory probe fitted only by using (almost) all of the GPU memory."""


# --------------------------------------------------------------------------- arguments
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="folder with train.csv, test.csv, sample_submission.csv")
    p.add_argument("--output", required=True, help="run folder (created if missing)")
    p.add_argument("--preset", default="auto", choices=["auto", *PRESETS],
                   help="model + training settings (auto: qwen3-8b with >= 22 GB of GPU memory)")
    p.add_argument("--model", help="override the checkpoint of the preset (Hugging Face id or local folder)")
    p.add_argument("--lr", type=float, help="peak learning rate of the LoRA weights")
    p.add_argument("--epochs", type=float)
    p.add_argument("--seqs-per-update", type=int, help="reviews per optimiser update (effective batch)")
    p.add_argument("--token-budget", type=int, help="max padded tokens per training micro-batch")
    p.add_argument("--max-micro", type=int, help="max reviews per training micro-batch")
    p.add_argument("--lora-r", type=int)
    p.add_argument("--lora-alpha", type=float)
    p.add_argument("--lora-dropout", type=float)
    ck = p.add_mutually_exclusive_group()
    ck.add_argument("--grad-ckpt", dest="grad_ckpt", action="store_true", help="force gradient checkpointing")
    ck.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false", help="disable gradient checkpointing")
    p.set_defaults(grad_ckpt=None)
    p.add_argument("--head-lr-mult", type=float, default=2.0, help="learning-rate multiplier of the 2-way head")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=float, default=0.05, help="fraction of updates with linear warm-up")
    p.add_argument("--max-len", type=int, default=1024, help="max tokens per sequence, prompt included")
    p.add_argument("--head", type=int, default=256, help="review tokens kept from the start of long reviews")
    p.add_argument("--holdout-frac", type=float, default=0.10, help="hold-out share when --folds is 0")
    p.add_argument("--folds", type=int, default=0, help="K-fold mode: number of stratified folds (0 = single hold-out)")
    p.add_argument("--fold", type=int, default=0, help="K-fold mode: which fold to run (0-based)")
    p.add_argument("--seed", type=int, default=42, help="training seed (batch order, LoRA init, dropout)")
    p.add_argument("--split-seed", type=int, default=42, help="seed of the fold partition (keep 42)")
    p.add_argument("--eval-every", type=float, default=0.0,
                   help="also score the whole validation fold every N epochs during training (0 = only at the end)")
    p.add_argument("--check-rows", type=int, default=256, help="validation reviews used by the update-0 and reload checks")
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end run in OUTPUT/smoke with a time estimate")
    p.add_argument("--predict-only", action="store_true", help="re-load OUTPUT/adapter and predict again")
    p.add_argument("--overwrite", action="store_true", help="retrain even if OUTPUT already holds a finished run")
    p.add_argument("--fresh-start", action="store_true",
                   help="accepted for run_cv.py compatibility; training always starts from the named base checkpoint")
    p.add_argument("--verify-fresh", action="store_true",
                   help="accepted for run_cv.py compatibility; the update-0 check always runs")
    p.add_argument("--allow-cpu", action="store_true", help="allow running without a GPU (impractically slow)")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- main
def main(argv=None) -> None:
    args = parse_args(argv)
    try:
        import torch
        import torch.nn.functional as F
        import transformers
        from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:  # pragma: no cover - GPU machine only
        sys.exit(f"torch / transformers are missing ({e}); run this on the GPU machine.")
    try:
        import peft
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError:  # pragma: no cover
        sys.exit('peft is not installed. Run:  py -3.12 -m pip install "peft==0.18.1" "transformers==4.57.6"')
    transformers.logging.set_verbosity_error()
    TF_MAJOR = int(transformers.__version__.split(".")[0])
    t_start = time.time()

    # ---------------- output folder state
    out = Path(args.output) / ("smoke" if args.smoke else "")
    if args.smoke and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    adapter_dir = out / "adapter"
    marker = adapter_dir / "TRAINING_COMPLETE.json"
    info_path = out / "run_info.json"
    log = ft.Logger(out / "train_log.txt")
    log("=" * 70)
    log("command: " + " ".join(sys.argv))

    resume = False
    if args.predict_only:
        if not marker.exists():
            sys.exit(f"{marker} not found: this fold has not finished training (same --output as the training run?)")
    else:
        if info_path.exists() and not args.overwrite:
            sys.exit(f"{out} already holds a finished run (run_info.json). Use --predict-only to predict again, "
                     "or --overwrite to retrain from scratch.")
        if args.overwrite:
            if adapter_dir.exists():
                shutil.rmtree(adapter_dir)
            if info_path.exists():
                info_path.unlink()
            log("--overwrite: previous adapter and run_info.json removed")
        if marker.exists():
            resume = True
            log("training had already finished earlier (adapter/TRAINING_COMPLETE.json found): "
                "skipping training and continuing with the predictions")
        elif adapter_dir.exists():
            shutil.rmtree(adapter_dir)
            log("removed an incomplete adapter folder left by an interrupted run; training from scratch")

    # ---------------- device
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        free_b, total_b = torch.cuda.mem_get_info()
        log(f"GPU {gpu_name}: {total_b / 1024 ** 3:.1f} GB, {free_b / 1024 ** 3:.1f} GB free before loading "
            "(close games / browsers / other GPU programs if this is much lower than the total)")
        if not torch.cuda.is_bf16_supported():
            sys.exit("this script needs a GPU with bfloat16 support (RTX 30-series or newer)")
        bf16 = True
    else:
        if not args.allow_cpu:
            sys.exit("No CUDA GPU found (torch.cuda.is_available() is False). Use the GPU machine.")
        device, gpu_name, vram_gb, bf16 = "cpu", "none", float("nan"), False
    dtype = torch.bfloat16 if bf16 else torch.float32

    # ---------------- configuration (from the preset, or from the saved run when re-predicting)
    meta_saved = None
    if args.predict_only or resume:
        meta_saved = json.loads((adapter_dir / "llm_meta.json").read_text(encoding="utf-8"))
        cfg = dict(meta_saved["config"])
        preset = meta_saved.get("preset", "?")
        if args.model and args.model != cfg["model"]:
            sys.exit(f"--model {args.model} differs from the model this adapter was trained on ({cfg['model']})")
        log("configuration taken from the saved adapter (command-line training options are ignored)")
    else:
        preset = args.preset if args.preset != "auto" else auto_preset(vram_gb if device == "cuda" else 0.0)
        cfg = dict(PRESETS[preset])
        for key in ("model", "lr", "epochs", "seqs_per_update", "token_budget", "max_micro",
                    "lora_r", "lora_alpha", "lora_dropout"):
            if getattr(args, key) is not None:
                cfg[key] = getattr(args, key)
        if args.grad_ckpt is not None:
            cfg["grad_ckpt"] = bool(args.grad_ckpt)
        cfg.update(max_len=int(args.max_len), head=int(args.head), head_lr_mult=float(args.head_lr_mult),
                   label_smoothing=float(args.label_smoothing), weight_decay=float(args.weight_decay),
                   warmup=float(args.warmup), prompt_version=PROMPT_VERSION)
        cfg["token_budget"] = max(int(cfg["token_budget"]), int(cfg["max_len"]))
    try:
        check_model_name(cfg["model"], allow_post_trained=bool(cfg.get("post_trained", False)))
    except ValueError as e:
        sys.exit(str(e))
    epochs_full = float(cfg["epochs"])
    eval_every = float(args.eval_every)
    if args.smoke:
        cfg["epochs"] = 1.0
        eval_every = 0.5  # exercise the in-training evaluation path too
    log(f"device={device} ({gpu_name}) torch={torch.__version__} transformers={transformers.__version__} "
        f"peft={peft.__version__} preset={preset} dtype={dtype}")
    log("config: " + json.dumps(cfg))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---------------- data and split
    data = Path(args.data_dir)
    train = pd.read_csv(data / "train.csv")
    test = pd.read_csv(data / "test.csv")
    y_all = (train["sentiment"].values == LABELS[1]).astype(np.int64)
    if args.folds:
        train_idx, hold_idx = ft.kfold_indices(y_all, args.folds, args.fold, args.split_seed)
        split_desc = {"mode": "kfold", "folds": int(args.folds), "fold": int(args.fold),
                      "split_seed": int(args.split_seed)}
    else:
        hold_mask = ft.stratified_holdout(y_all, args.holdout_frac, args.split_seed)
        train_idx, hold_idx = np.flatnonzero(~hold_mask), np.flatnonzero(hold_mask)
        split_desc = {"mode": "holdout", "holdout_frac": float(args.holdout_frac),
                      "split_seed": int(args.split_seed)}
    if len(hold_idx) == 0:
        sys.exit("the validation part is empty")
    if meta_saved is not None and meta_saved.get("split") != split_desc:
        sys.exit(f"split {split_desc} differs from the one used in training {meta_saved.get('split')}; "
                 "pass the same --folds/--fold/--split-seed")
    test_idx = np.arange(len(test))
    full_train_idx, full_hold_idx = train_idx, hold_idx
    if args.smoke:
        rng0 = np.random.default_rng(0)
        train_idx = np.sort(rng0.choice(train_idx, min(384, len(train_idx)), replace=False))
        hold_idx = np.sort(rng0.choice(hold_idx, min(192, len(hold_idx)), replace=False))
        test_idx = np.arange(min(96, len(test)))
    log(f"train={len(train_idx):,} validation={len(hold_idx):,} test={len(test_idx):,} split={split_desc}")

    # ---------------- tokenizer and inputs
    tok_kwargs = cfg.get("tokenizer_kwargs", {})
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], **tok_kwargs)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    bos_id = None
    if tokenizer.bos_token_id is not None:
        if tokenizer("hello")["input_ids"][:1] == [tokenizer.bos_token_id]:
            bos_id = int(tokenizer.bos_token_id)
    prefix_ids = tokenizer(PREFIX, add_special_tokens=False)["input_ids"]
    suffix_ids = tokenizer(SUFFIX, add_special_tokens=False)["input_ids"]
    verb = []
    for w in VERBALIZER:
        t = tokenizer(w, add_special_tokens=False)["input_ids"]
        if len(t) != 1:
            log(f"note: verbalizer {w!r} is {len(t)} tokens {t}; its first token is used")
        verb.append(int(t[0]))
    if verb[0] == verb[1]:
        sys.exit(f"the verbalizer tokens collide: {verb}")
    if meta_saved is not None and meta_saved.get("verbalizer_ids") not in (None, verb):
        sys.exit(f"tokenizer gives verbalizer ids {verb}, training used {meta_saved.get('verbalizer_ids')}")
    log(f"prompt: {len(prefix_ids)} prefix + {len(suffix_ids)} suffix tokens, bos={bos_id}, pad={pad_id}, "
        f"verbalizer ids {dict(zip(VERBALIZER, verb))}")

    def encode(texts) -> list:
        docs = []
        for s in range(0, len(texts), 1000):
            chunk = [ft.clean_review(t) for t in texts[s:s + 1000]]
            for ids in tokenizer(chunk, add_special_tokens=False)["input_ids"]:
                docs.append(build_input(ids, prefix_ids, suffix_ids, bos_id, cfg["max_len"], cfg["head"]))
        return docs

    t0 = time.time()
    train_docs = encode(train["review"].tolist())
    test_docs = encode(test["review"].tolist())
    train_len = np.array([len(d) for d in train_docs], dtype=np.int64)
    test_len = np.array([len(d) for d in test_docs], dtype=np.int64)
    log(f"tokenised 50,000 reviews in {time.time() - t0:.0f}s; mean length {train_len.mean():.0f} tokens "
        f"(prompt included); {int(np.sum(train_len >= cfg['max_len'])):,} train / "
        f"{int(np.sum(test_len >= cfg['max_len'])):,} test reviews truncated to {cfg['max_len']}")

    rng_c = np.random.default_rng(1)
    check_idx = np.sort(rng_c.choice(hold_idx, min(int(args.check_rows), len(hold_idx)), replace=False))

    # ---------------- model helpers
    have_accel = importlib.util.find_spec("accelerate") is not None

    def load_kwargs() -> dict:
        kw = dict(attn_implementation="sdpa")
        kw["dtype" if TF_MAJOR >= 5 else "torch_dtype"] = dtype  # transformers 5 renamed the argument
        if device == "cuda" and have_accel:
            kw["device_map"] = {"": 0}
        return kw

    def placed(m):
        return m if (device == "cuda" and have_accel) else m.to(device)

    def amp():
        if bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    is_vlm = cfg.get("loader", "causal") == "vlm"
    if is_vlm:
        fast = {m: importlib.util.find_spec(m) is not None for m in ("fla", "causal_conv1d")}
        log(f"linear-attention kernels: flash-linear-attention {'found' if fast['fla'] else 'MISSING'}, "
            f"causal-conv1d {'found' if fast['causal_conv1d'] else 'missing (torch fallback for the conv)'}")
        if not fast["fla"]:
            log("WARNING: without flash-linear-attention the Gated DeltaNet layers run on a slow fp32 torch "
                "loop. Install it (requirements-cloud.txt) before the full run; the smoke ESTIMATE shows the cost.")

    def load_full_vlm():
        from transformers import AutoModelForImageTextToText
        try:
            return placed(AutoModelForImageTextToText.from_pretrained(cfg["model"], **load_kwargs()))
        except ValueError:  # not registered as image-text-to-text in this transformers version
            return placed(AutoModelForCausalLM.from_pretrained(cfg["model"], **load_kwargs()))

    def load_backbone():
        """Backbone for re-loading the saved adapter: the same module tree the adapter was trained on."""
        if not is_vlm:
            return placed(AutoModel.from_pretrained(cfg["model"], **load_kwargs()))
        full = load_full_vlm()
        tb = find_text_backbone(full)
        del full
        gc.collect()
        return tb

    def enable_ckpt(m) -> None:
        try:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            m.gradient_checkpointing_enable()
        m.enable_input_require_grads()

    def forward_logits(net, hd, ids_np, mask_np):
        ids = torch.from_numpy(ids_np).to(device)
        mask = torch.from_numpy(mask_np).to(device)
        with amp():
            hs = net(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        last = mask.sum(dim=1) - 1                      # right padding: last real token
        pooled = hs[torch.arange(hs.shape[0], device=hs.device), last].float()
        return hd(pooled)                               # fp32 head, outside autocast

    def margins_safe(net, hd, docs, b) -> np.ndarray:
        oom = False
        try:
            ids, mask = ft.collate([docs[j] for j in b], pad_id)
            lg = forward_logits(net, hd, ids, mask)
            return (lg[:, 1] - lg[:, 0]).float().cpu().numpy()
        except torch.cuda.OutOfMemoryError:
            if len(b) == 1:
                raise
            oom = True
        # Retry *outside* the except block: while it is active, the traceback still holds the
        # failed attempt's activations, so the halves would run out of memory as well.
        if oom:
            gc.collect()
            torch.cuda.empty_cache()
            h = len(b) // 2
            return np.concatenate([margins_safe(net, hd, docs, b[:h]), margins_safe(net, hd, docs, b[h:])])
        raise RuntimeError("unreachable")

    def predict(net, hd, docs, lengths, idx):
        """Margins (logit positive - logit negative) for rows ``idx`` of ``docs``, in ``idx`` order.

        The model is put in eval mode (no LoRA dropout) and its previous mode is restored even if
        an exception escapes, so a failed mid-training evaluation cannot leave training in eval mode.
        """
        was_training = net.training
        net.eval()
        idx = np.asarray(idx, dtype=np.int64)
        margin = np.zeros(len(idx), dtype=np.float64)
        pos = {int(j): k for k, j in enumerate(idx)}
        n_tok = 0
        try:
            with torch.no_grad():
                for b in budget_batches(lengths, idx, 4 * int(cfg["token_budget"]), 64):
                    for j, v in zip(b, margins_safe(net, hd, docs, b)):
                        margin[pos[int(j)]] = float(v)
                    n_tok += int(lengths[b].sum())
        finally:
            net.train(was_training)
        return margin, n_tok

    def gpu_gb(fn) -> float:
        return fn() / 1024 ** 3 if device == "cuda" else 0.0

    history: list = []
    timings: dict = {}
    fingerprint: dict = {}
    if meta_saved is not None:  # resumed or --predict-only: keep the record of the training run
        history.extend(meta_saved.get("history", []))
        timings.update(meta_saved.get("timings", {}))
        fingerprint.update(meta_saved.get("fingerprint", {}))

    # ---------------- training (everything that lives on the GPU is local to this function)
    def train_phase() -> dict:
        t_load = time.time()
        if is_vlm:
            causal = load_full_vlm()
            log(f"loaded class {type(causal).__name__}; using its text backbone {type(find_text_backbone(causal)).__name__}")
        else:
            causal = placed(AutoModelForCausalLM.from_pretrained(cfg["model"], **load_kwargs()))
        rows = causal.get_output_embeddings().weight.detach()[verb].float().clone()
        backbone = find_text_backbone(causal) if is_vlm else causal.base_model
        del causal
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        backbone.config.use_cache = False
        hidden = hidden_of(backbone.config)
        if cfg["grad_ckpt"]:
            enable_ckpt(backbone)
        if cfg.get("lora_targets", "default") == "text-linear":
            lin = [n for n, mod in backbone.named_modules() if isinstance(mod, torch.nn.Linear)]
            chosen = select_lora_targets(lin)
            targets = sorted({n.rsplit(".", 1)[-1] for n in chosen})
            log(f"LoRA targets (every linear layer of the text backbone): {target_summary(chosen)}")
        else:
            targets = list(TARGET_MODULES)
        alpha = float(cfg["lora_alpha"])
        lcfg = LoraConfig(r=int(cfg["lora_r"]), lora_alpha=int(alpha) if alpha.is_integer() else alpha,
                          lora_dropout=float(cfg["lora_dropout"]), bias="none",
                          target_modules=targets)
        model = get_peft_model(backbone, lcfg)
        head = torch.nn.Linear(hidden, 2, bias=True).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            head.weight.copy_(rows.to(device))
            head.bias.zero_()
        del rows
        model.train()
        n_lora = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        log(f"loaded {cfg['model']} in {time.time() - t_load:.0f}s; hidden {hidden}; LoRA r={cfg['lora_r']} "
            f"trains {n_lora / 1e6:.1f}M of {n_all / 1e9:.2f}B weights (+ {hidden * 2 + 2:,} head); "
            f"GPU memory {gpu_gb(torch.cuda.memory_allocated if device == 'cuda' else None):.1f} GB")

        # -- memory probe: the largest micro-batch the run can produce
        if device == "cuda":
            while True:
                L = int(cfg["max_len"])
                B = max(1, min(int(cfg["max_micro"]), int(cfg["token_budget"]) // L))
                torch.cuda.reset_peak_memory_stats()
                problem = None
                try:
                    ids = np.full((B, L), int(prefix_ids[0]) if prefix_ids else int(pad_id), dtype=np.int64)
                    lg = forward_logits(model, head, ids, np.ones_like(ids))
                    loss = F.cross_entropy(lg, torch.zeros(B, dtype=torch.long, device=device))
                    loss.backward()
                    del lg, loss
                    model.zero_grad(set_to_none=True)
                    head.zero_grad(set_to_none=True)
                    peak_res = gpu_gb(torch.cuda.max_memory_reserved)
                    if peak_res > 0.94 * vram_gb:
                        raise MemoryTight(f"probe used {peak_res:.1f} of {vram_gb:.1f} GB")
                    break
                except (torch.cuda.OutOfMemoryError, MemoryTight) as e:
                    problem = type(e).__name__
                # clean up outside the except block (the traceback pins the failed attempt's tensors)
                lg = loss = None
                model.zero_grad(set_to_none=True)
                head.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                if not cfg["grad_ckpt"]:
                    cfg["grad_ckpt"] = True
                    enable_ckpt(model.get_base_model())
                    log(f"memory probe: {problem} -> enabling gradient checkpointing")
                elif int(cfg["token_budget"]) > int(cfg["max_len"]):
                    cfg["token_budget"] = max(int(cfg["max_len"]), int(cfg["token_budget"]) // 2)
                    log(f"memory probe: {problem} -> token budget {cfg['token_budget']}")
                else:
                    sys.exit("Out of GPU memory even with one review per micro-batch. Close other GPU "
                             "programs, or use --preset qwen3-4b, or --max-len 512.")
            log(f"memory probe ok: {B} x {L} tokens per micro-batch, grad_ckpt={cfg['grad_ckpt']}, peak "
                f"{gpu_gb(torch.cuda.max_memory_allocated):.1f} GB allocated / {peak_res:.1f} GB reserved "
                f"of {vram_gb:.1f} GB")
            torch.cuda.reset_peak_memory_stats()

        # -- update-0 check: the untrained classifier is the zero-shot answer of the base model
        m0, _ = predict(model, head, train_docs, train_len, check_idx)
        auc0 = auc_score(y_all[check_idx], m0)
        acc0 = float(np.mean((m0 > 0) == (y_all[check_idx] == 1)))
        log(f"UPDATE-0 CHECK (zero-shot, nothing trained yet) on {len(check_idx)} validation reviews: "
            f"AUC {auc0:.3f}, accuracy {acc0:.2%}  <- AUC should be clearly above 0.5")
        if not auc0 > 0.6:
            log("WARNING: zero-shot AUC is not above 0.6. The pooling or the prompt may be wrong. If the "
                "validation accuracy at the end is not far above this, stop and send the log.")
        history.append({"update": 0, "epoch": 0.0, "check_rows": int(len(check_idx)),
                        "check_auc": auc0, "check_accuracy": acc0, "phase": "zero-shot"})
        fingerprint.update(zero_shot_auc=auc0, zero_shot_accuracy=acc0, check_rows=int(len(check_idx)))

        # -- optimiser and schedule
        lora_params = [p for p in model.parameters() if p.requires_grad]
        trainable = lora_params + [head.weight, head.bias]
        groups = [
            {"params": lora_params, "weight_decay": float(cfg["weight_decay"]), "lr": float(cfg["lr"])},
            {"params": [head.weight], "weight_decay": float(cfg["weight_decay"]),
             "lr": float(cfg["lr"]) * float(cfg["head_lr_mult"])},
            {"params": [head.bias], "weight_decay": 0.0, "lr": float(cfg["lr"]) * float(cfg["head_lr_mult"])},
        ]
        opt = torch.optim.AdamW(groups, lr=float(cfg["lr"]), betas=(0.9, 0.999), eps=1e-8)
        U = int(cfg["seqs_per_update"])
        upd_per_epoch = len(plan_updates(train_len, train_idx, U, int(cfg["token_budget"]),
                                         int(cfg["max_micro"]), np.random.default_rng(0)))
        total_upd = max(1, int(round(upd_per_epoch * float(cfg["epochs"]))))
        n_epochs = int(math.ceil(float(cfg["epochs"])))
        warm = max(1, int(float(cfg["warmup"]) * total_upd))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda u: cosine_lr(u, warm, total_upd))
        eval_every_upd = int(round(upd_per_epoch * eval_every)) if eval_every > 0 else 0
        log(f"{upd_per_epoch} updates/epoch x {U} reviews, {total_upd} updates total, warm-up {warm}, "
            f"token budget {cfg['token_budget']}, mid-training evaluation every "
            f"{eval_every_upd if eval_every_upd else 'never'} updates")

        def run_update(micro_list, n_upd):
            """Forward + backward of one optimiser update. Returns (loss sum, rows, tokens), or None if
            a loss was not finite. The loss is summed per review and divided by the reviews of the whole
            update, so splitting a micro-batch into smaller ones does not change the gradient."""
            loss_sum, rows_done, toks = 0.0, 0, 0
            for m in micro_list:
                ids, mask = ft.collate([train_docs[j] for j in m], pad_id)
                lg = forward_logits(model, head, ids, mask)
                yb = torch.from_numpy(y_all[m]).to(device)
                loss = F.cross_entropy(lg, yb, reduction="sum", label_smoothing=float(cfg["label_smoothing"]))
                if not torch.isfinite(loss):
                    return None
                (loss / n_upd).backward()
                loss_sum += float(loss.detach())
                rows_done += len(m)
                toks += int(train_len[m].sum())
            return loss_sum, rows_done, toks

        shuffle_rng = np.random.default_rng(args.seed)
        upd, skipped, n_oom, done = 0, 0, 0, False
        run_loss, run_n, tok_seen = 0.0, 0, 0
        t_train = time.time()
        t_mark, tok_mark = None, 0
        eval_s = 0.0  # seconds spent in mid-training evaluation; excluded from speed and ETA
        for epoch in range(n_epochs):
            for micro_list in plan_updates(train_len, train_idx, U, int(cfg["token_budget"]),
                                           int(cfg["max_micro"]), shuffle_rng):
                n_upd = sum(len(m) for m in micro_list)
                attempt, res = micro_list, None
                for _try in range(4):
                    oom = False
                    try:
                        res = run_update(attempt, n_upd)
                    except torch.cuda.OutOfMemoryError:
                        oom = True
                    if not oom:
                        break
                    # out of memory (another program took GPU memory, or fragmentation): throw away this
                    # update's partial gradients and redo it with every micro-batch cut in half
                    opt.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    n_oom += 1
                    if all(len(m) == 1 for m in attempt):
                        sys.exit("out of GPU memory during training even with one review per micro-batch. Close "
                                 "other GPU programs and run the same command again (this fold restarts).")
                    attempt = [h for m in attempt for h in ((m[:len(m) // 2], m[len(m) // 2:]) if len(m) > 1 else (m,))]
                    log(f"warning: out of GPU memory at update {upd + 1}; retrying it with {len(attempt)} "
                        f"smaller micro-batches ({n_oom} retries so far)")
                else:
                    sys.exit("out of GPU memory four times in one update. Close other GPU programs and run the "
                             "same command again (this fold restarts).")
                bad = res is None
                if not bad:
                    run_loss += res[0]
                    run_n += res[1]
                    tok_seen += res[2]
                if bad:
                    opt.zero_grad(set_to_none=True)
                    skipped += 1
                    log(f"warning: non-finite loss at update {upd + 1}; update skipped ({skipped} so far)")
                    if skipped > max(3, int(0.02 * total_upd)):
                        sys.exit("too many non-finite losses: lower --lr (e.g. 5e-5) or change --seed")
                else:
                    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                    opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                upd += 1
                if upd == 3:
                    t_mark, tok_mark = time.time(), tok_seen
                if upd % 25 == 0 or upd == total_upd:
                    el = time.time() - t_train
                    rate = (tok_seen - tok_mark) / max(1e-9, time.time() - t_mark - eval_s) if t_mark else 0.0
                    log(f"epoch {epoch + 1} update {upd}/{total_upd} loss {run_loss / max(1, run_n):.4f} "
                        f"lr {sched.get_last_lr()[0]:.2e} {rate:,.0f} tok/s elapsed {el / 60:.1f} min "
                        f"ETA {(el - eval_s) / upd * (total_upd - upd) / 60:.1f} min "
                        f"peak {gpu_gb(torch.cuda.max_memory_reserved if device == 'cuda' else None):.1f} GB")
                    run_loss, run_n = 0.0, 0
                if eval_every_upd and upd % eval_every_upd == 0 and upd < total_upd:
                    t_ev = time.time()
                    try:
                        mh, _ = predict(model, head, train_docs, train_len, hold_idx)
                        acc = float(np.mean((mh > 0) == (y_all[hold_idx] == 1)))
                        auc = auc_score(y_all[hold_idx], mh)
                        history.append({"update": upd, "epoch": round(upd / upd_per_epoch, 3),
                                        "holdout_accuracy": acc, "holdout_auc": auc})
                        log(f"  VALIDATION accuracy {acc:.4%} (AUC {auc:.4f}) at update {upd} "
                            f"(epoch {upd / upd_per_epoch:.2f})")
                    except Exception as e:  # never lose a training run to an evaluation bug
                        log(f"  WARNING: mid-training evaluation failed ({type(e).__name__}: {e}); "
                            "training continues")
                        log(traceback.format_exc())
                    if t_mark is not None:
                        eval_s += time.time() - t_ev
                if upd >= total_upd:
                    done = True
                    break
            if done:
                break
        train_s = time.time() - t_train
        if t_mark:
            rate = (tok_seen - tok_mark) / max(1e-9, time.time() - t_mark - eval_s)
        else:
            rate = tok_seen / max(1e-9, train_s - eval_s)
        timings["train_minutes"] = round(train_s / 60, 2)
        timings["mid_training_eval_minutes"] = round(eval_s / 60, 2)
        fingerprint.update(skipped_updates=int(skipped), oom_retries=int(n_oom))
        log(f"training finished in {train_s / 60:.1f} min ({rate:,.0f} tok/s without the "
            f"{eval_s / 60:.1f} min of mid-training evaluation, {skipped} skipped updates, "
            f"{n_oom} out-of-memory retries)")

        # -- save, then keep the in-memory answers on the check rows for the reload check
        adapter_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        torch.save({"weight": head.weight.detach().cpu(), "bias": head.bias.detach().cpu()}, adapter_dir / "head.pt")
        meta = {"config": cfg, "preset": preset, "split": split_desc, "verbalizer": list(VERBALIZER),
                "verbalizer_ids": verb, "prefix": PREFIX, "suffix": SUFFIX, "bos_id": bos_id, "pad_id": int(pad_id),
                "hidden_size": hidden, "seed": int(args.seed), "updates": int(upd),
                "train_command": " ".join(sys.argv),
                "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                             "peft": peft.__version__},
                # kept here too, so that a resumed or --predict-only run can still report them
                "history": list(history), "fingerprint": dict(fingerprint), "timings": dict(timings)}
        (adapter_dir / "llm_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        m_mem, _ = predict(model, head, train_docs, train_len, check_idx)
        marker.write_text(json.dumps({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "updates": int(upd)}),
                          encoding="utf-8")
        log(f"saved LoRA adapter + head -> {adapter_dir}")
        return {"m_mem": m_mem, "m0": m0, "train_rate": rate}

    train_res = None
    if not (args.predict_only or resume):
        train_res = train_phase()
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            left = gpu_gb(torch.cuda.memory_allocated)
            log(f"training model freed; {left:.1f} GB still allocated before re-loading")
            if left > 2.0:
                log("WARNING: the training model was not fully released; re-loading may run out of memory. "
                    "If it does, run the same command again: it resumes at the prediction step.")

    # ---------------- reload from disk and predict
    t_load = time.time()
    meta_now = json.loads((adapter_dir / "llm_meta.json").read_text(encoding="utf-8"))
    backbone = load_backbone()
    backbone.config.use_cache = False
    model = PeftModel.from_pretrained(backbone, str(adapter_dir), is_trainable=False)
    model.eval()
    hidden = hidden_of(backbone.config)
    if hidden != int(meta_now["hidden_size"]):
        sys.exit(f"hidden size {hidden} differs from the trained adapter ({meta_now['hidden_size']})")
    head = torch.nn.Linear(hidden, 2, bias=True).to(device=device, dtype=torch.float32)
    head.load_state_dict(torch.load(adapter_dir / "head.pt", map_location=device, weights_only=True))
    log(f"re-loaded base model + adapter from disk in {time.time() - t_load:.0f}s")

    reload_diff = None
    if train_res is not None:
        m_disk, _ = predict(model, head, train_docs, train_len, check_idx)
        reload_diff = float(np.max(np.abs(m_disk - train_res["m_mem"])))
        moved = float(np.mean(np.abs(train_res["m_mem"] - train_res["m0"])))
        log(f"reload check on {len(check_idx)} reviews: max |margin(reloaded) - margin(in memory)| = "
            f"{reload_diff:.4f}; training moved the margins by {moved:.3f} on average")
        if reload_diff > 0.1 and reload_diff > 0.2 * moved:
            sys.exit("RELOAD CHECK FAILED: the adapter on disk does not reproduce the trained model. "
                     "Do not use these predictions; send the log.")
        fingerprint.update(reload_max_abs_diff=reload_diff, mean_abs_margin_change=moved)

    t0 = time.time()
    m_hold, tok_h = predict(model, head, train_docs, train_len, hold_idx)
    m_test, tok_t = predict(model, head, test_docs, test_len, test_idx)
    pred_s = time.time() - t0
    timings["predict_minutes"] = round(pred_s / 60, 2)
    hold_acc = float(np.mean((m_hold > 0) == (y_all[hold_idx] == 1)))
    hold_auc = auc_score(y_all[hold_idx], m_hold)
    pd.DataFrame({"id": train["id"].values[hold_idx], "label": y_all[hold_idx],
                  "p_positive": ft.sigmoid(m_hold), "margin": m_hold}).to_csv(out / "holdout_predictions.csv", index=False)
    test_ids = test["id"].values[test_idx]
    pd.DataFrame({"id": test_ids, "p_positive": ft.sigmoid(m_test), "margin": m_test}).to_csv(
        out / "test_predictions.csv", index=False)
    log(f"VALIDATION accuracy {hold_acc:.4%} (AUC {hold_auc:.4f}) on {len(hold_idx):,} reviews; "
        f"test positive share {np.mean(m_test > 0):.4f}; prediction took {pred_s / 60:.1f} min")
    history.append({"update": int(meta_now.get("updates", 0)), "epoch": float(cfg["epochs"]),
                    "holdout_accuracy": hold_acc, "holdout_auc": hold_auc, "phase": "final (re-loaded)"})

    if not args.smoke and hold_acc < 0.6:
        if args.predict_only:
            sys.exit("validation accuracy < 60% when re-predicting a finished fold: the adapter, the base model "
                     "or the data differ from training. Do not use these predictions; send the log.")
        if marker.exists():
            marker.unlink()  # the next run of the same command then retrains this fold from scratch
        sys.exit("Training diverged (validation accuracy < 60%). run_info.json was not written and the adapter "
                 "was marked incomplete, so running the same command again retrains this fold. Add a lower "
                 "--lr (e.g. 5e-5) or another --seed.")

    sub_info = None
    if not args.smoke:
        sub_info = ft.write_submission(test_ids, m_test, data / "sample_submission.csv", out / "submission.csv")
        log(f"wrote {out / 'submission.csv'} sha256={sub_info['sha256']}")
        if args.folds:
            log("note: this is a single-fold prediction; the file to submit is the fold average written by "
                "assemble_cv.py (or the ensemble written by ensemble_cv.py / compare_runs.py)")
    else:
        if train_res is not None and train_res["train_rate"] > 0:
            tr_tok = float(train_len[full_train_idx].sum()) * epochs_full
            pr_tok = float(train_len[full_hold_idx].sum() + test_len.sum())
            pr_rate = (tok_h + tok_t) / max(1e-9, pred_s)
            est_tr = tr_tok / train_res["train_rate"] / 60
            est_pr = pr_tok / pr_rate / 60
            log(f"ESTIMATE for one full fold ({len(full_train_idx):,} train / {len(full_hold_idx):,} validation / "
                f"{len(test):,} test reviews): training ~{est_tr:.0f} min + prediction ~{est_pr:.0f} min "
                f"+ loading ~3 min = ~{est_tr + est_pr + 3:.0f} min")
            fingerprint.update(estimate_fold_minutes=round(est_tr + est_pr + 3, 1))
        log("SMOKE TEST OK - train -> evaluate -> save -> reload -> predict -> write all ran end to end.")

    info = json.loads(info_path.read_text(encoding="utf-8")) if (args.predict_only and info_path.exists()) else {}
    if meta_saved is not None:
        info.setdefault("train_command", meta_saved.get("train_command"))
        if resume:
            fingerprint["reload_check"] = "not run: training had finished in an earlier, interrupted process"
    info.update({
        "mode": "predict-only" if args.predict_only else ("smoke" if args.smoke else ("resumed" if resume else "train")),
        "command": " ".join(sys.argv), "preset": preset, "config": cfg, "trainer": "finetune_llm.py",
        "split": split_desc, "folds": int(args.folds), "fold": int(args.fold) if args.folds else None,
        "seed": int(meta_saved.get("seed", args.seed)) if meta_saved is not None else int(args.seed),
        "split_seed": int(args.split_seed),
        "n_train": int(len(train_idx)), "n_holdout": int(len(hold_idx)),
        "holdout_accuracy": hold_acc, "holdout_auc": hold_auc, "test_positive_share": float(np.mean(m_test > 0)),
        "prompt": {"version": PROMPT_VERSION, "prefix": PREFIX, "suffix": SUFFIX,
                   "verbalizer": list(VERBALIZER), "verbalizer_ids": verb},
        "submission": sub_info,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "transformers": transformers.__version__,
                        "peft": peft.__version__, "device": device, "gpu": gpu_name,
                        "vram_gb": None if math.isnan(vram_gb) else round(vram_gb, 1), "dtype": str(dtype)},
        "data_sha256": {f: ft.sha256_file(data / f) for f in ("train.csv", "test.csv")},
        "total_minutes": round((time.time() - t_start) / 60, 2),
        "history": history,
    })
    info.setdefault("fingerprint", {}).update(fingerprint)
    info.setdefault("timings", {}).update(timings)
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    log(f"done in {info['total_minutes']} min -> {out}")


if __name__ == "__main__":
    main()
