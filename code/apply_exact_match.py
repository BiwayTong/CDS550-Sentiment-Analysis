#!/usr/bin/env python3
"""Apply only conflict-free exact train/test text matches to a submission.

This is an independent, deterministic post-processing step.  It never reads test labels:
the replacement label comes only from an identical review text in train.csv.  A text with
conflicting training labels is rejected instead of being guessed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", required=True)
    p.add_argument("--test", required=True)
    p.add_argument("--submission", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)

    train_path, test_path = Path(args.train), Path(args.test)
    sub_path, out_path = Path(args.submission), Path(args.output)
    train, test, sub = read_csv(train_path), read_csv(test_path), read_csv(sub_path)
    if not train or not test or not sub:
        raise SystemExit("train, test and submission must all contain data rows")
    for rows, name, cols in ((train, "train", {"review", "sentiment"}),
                             (test, "test", {"id", "review"}),
                             (sub, "submission", {"id", "sentiment"})):
        missing = cols - set(rows[0])
        if missing:
            raise SystemExit(f"{name} is missing columns {sorted(missing)}")

    labels: dict[str, str] = {}
    for row in train:
        text = row["review"]
        label = row["sentiment"]
        previous = labels.setdefault(text, label)
        if previous != label:
            raise SystemExit("conflicting train labels for an identical review text")

    test_by_id = {row["id"]: row["review"] for row in test}
    if len(test_by_id) != len(test):
        raise SystemExit("test.csv contains duplicate ids")
    sub_ids = [row["id"] for row in sub]
    if len(set(sub_ids)) != len(sub_ids):
        raise SystemExit("submission contains duplicate ids")
    if set(sub_ids) != set(test_by_id):
        raise SystemExit("submission ids do not match test.csv exactly")

    changed = []
    for row in sub:
        if row["id"] not in test_by_id:
            raise SystemExit(f"submission id {row['id']} is not present in test.csv")
        target = labels.get(test_by_id[row["id"]])
        if target is not None and target != row["sentiment"]:
            changed.append({"id": row["id"], "from": row["sentiment"], "to": target})
            row["sentiment"] = target

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "sentiment"])
        writer.writeheader()
        writer.writerows({"id": r["id"], "sentiment": r["sentiment"]} for r in sub)
    audit = {
        "train": str(train_path), "test": str(test_path), "input_submission": str(sub_path),
        "output_submission": str(out_path), "train_sha256": sha256(train_path),
        "test_sha256": sha256(test_path), "input_sha256": sha256(sub_path),
        "output_sha256": sha256(out_path), "n_train_unique_texts": len(labels),
        "n_test_rows_with_exact_train_match": sum(test_by_id[r["id"]] in labels for r in test),
        "n_changed": len(changed), "changed": changed,
    }
    out_path.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
