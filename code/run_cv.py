#!/usr/bin/env python3
"""Run every fold of a stratified K-fold split, one process at a time.

The single 10 % hold-out used in stage 2 has 2,500 reviews, which is too few to tell a
real improvement from noise (its standard error is 0.35 points of accuracy, so a genuine
+0.3 point change is invisible). This script runs all K folds of a trainer and leaves the
per-fold predictions in ``<output>/<name>-f<j>/`` for ``code/assemble_cv.py``, which turns
them into

* one out-of-fold prediction per training review (all 25,000), and
* one averaged prediction per test review, from K models — free bagging on top.

Fold 0 validates on exactly the 2,500 reviews of the stage-2 hold-out (10 folds) or on a
superset of them (5 folds), so its accuracy can be compared with the numbers in the report.

Two trainers are available:

* ``--trainer transformer`` (default): ``finetune_transformer.py`` (DeBERTa-v3 / RoBERTa / ELECTRA)
* ``--trainer llm``: ``finetune_llm.py`` (decoder LLM such as Qwen3-8B-Base with LoRA)

    # all 5 folds of DeBERTa-v3-large, resumable, then assemble
    python code/run_cv.py --data-dir data --folds 5 --output runs/cv-deberta --name deberta
    python code/assemble_cv.py --data-dir data --runs runs/cv-deberta --output ensemble-deberta
    # all 5 folds of Qwen3-8B-Base + LoRA (fold 0 is skipped if the pilot already produced it)
    python code/run_cv.py --trainer llm --preset qwen3-8b --data-dir data --folds 5 --output runs/cv-qwen3 --name qwen3

Each fold is a separate call of the trainer, so memory is released between folds and an
interrupted run can be continued by running the same command again (``--force`` re-runs
folds that are already complete, ``--only 2 3`` runs a subset). Options such as --max-len,
--head and --eval-every are passed on only when given, so each trainer keeps its own defaults.
Finished folds trained on a different model than the one requested are refused, so two model
families can never be mixed silently inside one folder.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINERS = {"transformer": HERE / "finetune_transformer.py", "llm": HERE / "finetune_llm.py"}
TRAINER = TRAINERS["transformer"]  # kept for backward compatibility
# what a finished fold folder must contain to count as done
REQUIRED = ("holdout_predictions.csv", "test_predictions.csv", "run_info.json")


def fold_dir(root: Path, name: str, fold: int) -> Path:
    return root / f"{name}-f{fold}"


def is_done(d: Path) -> bool:
    if not all((d / f).exists() for f in REQUIRED):
        return False
    try:  # a fold that died mid-write leaves an unreadable json
        json.loads((d / "run_info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return True


def fold_seconds(d: Path) -> float:
    try:
        info = json.loads((d / "run_info.json").read_text(encoding="utf-8"))
        return float(info.get("total_minutes", 0.0)) * 60.0
    except (OSError, ValueError):
        return 0.0


def fold_model(d: Path):
    """Model name recorded in a finished fold's run_info.json (None if unknown)."""
    try:
        info = json.loads((d / "run_info.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return (info.get("config") or {}).get("model")


def requested_model(args):
    """The checkpoint this command will train, or None when it depends on the GPU (preset auto)."""
    if args.model:
        return args.model
    if args.preset == "auto":
        return None
    sys.path.insert(0, str(HERE))
    if args.trainer == "llm":
        import finetune_llm as trainer_mod  # noqa: E402  (no torch at import time)
    else:
        import finetune_transformer as trainer_mod  # noqa: E402  (no torch at import time)
    preset = trainer_mod.PRESETS.get(args.preset)
    if preset is None:
        sys.exit(f"--preset {args.preset} is not a preset of the {args.trainer} trainer "
                 f"({', '.join(trainer_mod.PRESETS)})")
    return preset["model"]


def build_cmd(args, root: Path, fold: int) -> list:
    """Command line of one fold. Options left unset are not passed, so the trainer's defaults apply."""
    cmd = [args.python, str(TRAINERS[args.trainer]),
           "--data-dir", args.data_dir, "--output", str(fold_dir(root, args.name, fold)),
           "--folds", str(args.folds), "--fold", str(fold), "--preset", args.preset,
           "--seed", str(args.seed), "--split-seed", str(args.split_seed)]
    for flag, value in (("--max-len", args.max_len), ("--head", args.head), ("--eval-every", args.eval_every),
                        ("--model", args.model), ("--epochs", args.epochs), ("--lr", args.lr)):
        if value is not None:
            cmd += [flag, str(value)]
    if args.fresh_start:
        cmd += ["--fresh-start"]
    if args.verify_fresh:
        cmd += ["--verify-fresh"]
    if args.force and args.trainer == "llm" and is_done(fold_dir(root, args.name, fold)):
        cmd += ["--overwrite"]  # finetune_llm.py refuses to overwrite a finished fold otherwise
    return cmd


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data", help="folder with train.csv, test.csv, sample_submission.csv")
    p.add_argument("--folds", type=int, default=5, help="number of stratified folds (5 or 10 recommended)")
    p.add_argument("--output", required=True, help="root folder; folds go to <output>/<name>-f<j>/")
    p.add_argument("--name", default="cv", help="fold folder prefix, e.g. 'deberta' -> deberta-f0 ...")
    p.add_argument("--trainer", choices=sorted(TRAINERS), default="transformer",
                   help="transformer = finetune_transformer.py (default), llm = finetune_llm.py")
    p.add_argument("--preset", default="auto", help="preset passed to the trainer (auto = by VRAM)")
    p.add_argument("--model", help="override the model name of the preset")
    p.add_argument("--seed", type=int, default=42, help="training seed, shared by all folds")
    p.add_argument("--split-seed", type=int, default=42, help="fold-partition seed (keep 42)")
    p.add_argument("--max-len", type=int, help="passed on only when given (transformer default 512, llm 1024)")
    p.add_argument("--head", type=int, help="passed on only when given (transformer default 128, llm 256)")
    p.add_argument("--eval-every", type=float,
                   help="passed on only when given (transformer default 0.5 epochs, llm default 0 = end only)")
    p.add_argument("--epochs", type=float, help="override the preset's epoch count")
    p.add_argument("--lr", type=float)
    p.add_argument("--fresh-start", action="store_true",
                   help="pass --fresh-start to every fold; refuse local/fine-tuned initial checkpoints")
    p.add_argument("--verify-fresh", action="store_true",
                   help="evaluate each fold's newly initialised classifier at update 0")
    p.add_argument("--only", type=int, nargs="+", help="run only these fold numbers")
    p.add_argument("--force", action="store_true", help="re-run folds whose output already exists")
    p.add_argument("--time-budget-min", type=float, default=0,
                   help="stop starting new folds after this many minutes (0 = no limit); "
                        "re-run the same command later to continue")
    p.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    p.add_argument("--python", default=sys.executable, help="interpreter used for the folds")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    folds = args.only if args.only else list(range(args.folds))
    for f in folds:
        if not 0 <= f < args.folds:
            sys.exit(f"--only {f} is outside 0..{args.folds - 1}")

    # never mix two model families inside one folder
    want = requested_model(args)
    if want is not None and not args.force:
        other = {f: fold_model(fold_dir(root, args.name, f)) for f in range(args.folds)
                 if is_done(fold_dir(root, args.name, f))}
        other = {f: m for f, m in other.items() if m is not None and m != want}
        if other:
            sys.exit(f"finished folds {sorted(other)} were trained on {sorted(set(other.values()))}, but this "
                     f"command would train {want}. Use the same --preset/--model as before, another --output, "
                     "or --force to retrain those folds.")

    todo = [f for f in folds if args.force or not is_done(fold_dir(root, args.name, f))]
    done_before = [f for f in folds if f not in todo]
    print(f"trainer {args.trainer}: folds {folds} of {args.folds}: {len(done_before)} already complete "
          f"{done_before}, {len(todo)} to run {todo}", flush=True)
    if not todo:
        print("nothing to do; run code/assemble_cv.py to combine the folds.", flush=True)
        return

    est = sum(fold_seconds(fold_dir(root, args.name, f)) for f in done_before)
    est_fold_min = (est / len(done_before) / 60) if (done_before and est) else 0.0
    if est_fold_min:
        print(f"a finished fold took about {est_fold_min:.1f} min here, "
              f"so about {est_fold_min * len(todo):.0f} min remain (before new folds are timed).", flush=True)

    t0 = time.time()
    ran, failed, last_min = [], [], est_fold_min
    for f in todo:
        elapsed = (time.time() - t0) / 60
        if args.time_budget_min and ran and last_min and elapsed + last_min > args.time_budget_min:
            print(f"\ntime budget {args.time_budget_min:.0f} min reached after fold(s) {ran}; "
                  f"stopping before fold {f} (a fold needs about {last_min:.0f} min). "
                  f"Re-run this command to continue.", flush=True)
            break
        cmd = build_cmd(args, root, f)
        print("\n" + "=" * 78, flush=True)
        print(f"fold {f}/{args.folds}  ({len(ran) + 1} of {len(todo)} to run, "
              f"{elapsed:.1f} min elapsed)", flush=True)
        print("  " + " ".join(cmd), flush=True)
        print("=" * 78, flush=True)
        if args.dry_run:
            ran.append(f)
            continue
        tf = time.time()
        rc = subprocess.call(cmd)
        if rc != 0 or not is_done(fold_dir(root, args.name, f)):
            print(f"fold {f} FAILED (exit {rc}); stopping so the log stays readable. "
                  f"Fix the error and re-run this command; finished folds are skipped.", flush=True)
            failed.append(f)
            break
        mins = (time.time() - tf) / 60
        last_min = max(last_min, mins) if last_min else mins
        ran.append(f)
        print(f"fold {f} done in {mins:.1f} min", flush=True)

    print("\n" + "-" * 78)
    if args.dry_run:
        print(f"dry run: would run folds {ran}; nothing was executed. Already complete: {done_before}")
        return
    print(f"completed now: {ran}   already there: {done_before}   failed: {failed}")
    remaining = [f for f in folds if not is_done(fold_dir(root, args.name, f))]
    print(f"still missing: {remaining if remaining else 'none - ready to assemble'}")
    if not remaining and not args.dry_run:
        print(f"\nnext: python code/assemble_cv.py --data-dir {args.data_dir} "
              f"--runs {args.output} --output ensemble-{args.name}")


if __name__ == "__main__":
    main()
