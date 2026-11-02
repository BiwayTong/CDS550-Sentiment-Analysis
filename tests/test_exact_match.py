"""Tests for the auditable exact train/test text-match post-processing step."""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import apply_exact_match as em  # noqa: E402


def write_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


class ExactMatchTests(unittest.TestCase):
    def test_changes_only_conflict_free_exact_matches_and_writes_audit(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            write_csv(d / "train.csv", ["id", "review", "sentiment"], [
                {"id": "1", "review": "same", "sentiment": "positive"},
                {"id": "2", "review": "other", "sentiment": "negative"},
            ])
            write_csv(d / "test.csv", ["id", "review"], [
                {"id": "10", "review": "same"}, {"id": "11", "review": "new"},
            ])
            write_csv(d / "sub.csv", ["id", "sentiment"], [
                {"id": "10", "sentiment": "negative"}, {"id": "11", "sentiment": "positive"},
            ])
            out = d / "out.csv"
            em.main(["--train", str(d / "train.csv"), "--test", str(d / "test.csv"),
                     "--submission", str(d / "sub.csv"), "--output", str(out)])
            rows = em.read_csv(out)
            self.assertEqual(rows, [{"id": "10", "sentiment": "positive"},
                                    {"id": "11", "sentiment": "positive"}])
            audit = json.loads(out.with_suffix(".audit.json").read_text())
            self.assertEqual(audit["n_changed"], 1)
            self.assertEqual(audit["changed"][0]["id"], "10")

    def test_rejects_conflicting_training_labels(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            write_csv(d / "train.csv", ["id", "review", "sentiment"], [
                {"id": "1", "review": "same", "sentiment": "positive"},
                {"id": "2", "review": "same", "sentiment": "negative"},
            ])
            write_csv(d / "test.csv", ["id", "review"], [{"id": "10", "review": "same"}])
            write_csv(d / "sub.csv", ["id", "sentiment"], [{"id": "10", "sentiment": "positive"}])
            with self.assertRaises(SystemExit):
                em.main(["--train", str(d / "train.csv"), "--test", str(d / "test.csv"),
                         "--submission", str(d / "sub.csv"), "--output", str(d / "out.csv")])

    def test_rejects_submission_id_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            write_csv(d / "train.csv", ["id", "review", "sentiment"],
                      [{"id": "1", "review": "same", "sentiment": "positive"}])
            write_csv(d / "test.csv", ["id", "review"], [{"id": "10", "review": "new"}])
            write_csv(d / "sub.csv", ["id", "sentiment"], [{"id": "99", "sentiment": "positive"}])
            with self.assertRaises(SystemExit):
                em.main(["--train", str(d / "train.csv"), "--test", str(d / "test.csv"),
                         "--submission", str(d / "sub.csv"), "--output", str(d / "out.csv")])


if __name__ == "__main__":
    unittest.main(verbosity=2)
