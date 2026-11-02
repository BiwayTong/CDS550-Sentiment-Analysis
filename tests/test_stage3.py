"""Tests for the stage-3 cross-validation helpers. NumPy/pandas only, no torch, no GPU."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import assemble_cv as ac  # noqa: E402
import ensemble_cv as ec  # noqa: E402
import finetune_transformer as ft  # noqa: E402
import run_cv as rc  # noqa: E402

LABELS = ("negative", "positive")


def toy_labels(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, n)


class KFoldPartition(unittest.TestCase):
    def test_folds_partition_every_row_exactly_once(self):
        y = toy_labels()
        seen = np.zeros(len(y), dtype=int)
        for f in range(5):
            tr, va = ft.kfold_indices(y, 5, f, 42)
            self.assertEqual(len(tr) + len(va), len(y))
            self.assertEqual(len(np.intersect1d(tr, va)), 0)
            seen[va] += 1
        self.assertTrue((seen == 1).all())

    def test_folds_are_balanced_and_stratified(self):
        y = toy_labels(1000)
        for folds in (5, 10):
            sizes = []
            for f in range(folds):
                tr, va = ft.kfold_indices(y, folds, f, 42)
                sizes.append(len(va))
                # each class contributes its share, within one review
                for c in (0, 1):
                    self.assertLessEqual(abs(np.sum(y[va] == c) / np.sum(y == c) - 1 / folds), 0.005)
            self.assertLessEqual(max(sizes) - min(sizes), 2)

    def test_fold_zero_matches_the_classic_holdout(self):
        """This is what makes the CV scores comparable with the stage-2 hold-out numbers."""
        y = toy_labels(25000, seed=3)
        for folds in (5, 10):
            for seed in (42, 7):
                tr, va = ft.kfold_indices(y, folds, 0, seed)
                hold = ft.stratified_holdout(y, 1.0 / folds, seed)
                np.testing.assert_array_equal(np.sort(va), np.flatnonzero(hold))

    def test_deterministic_and_seed_sensitive(self):
        y = toy_labels()
        np.testing.assert_array_equal(ft.kfold_indices(y, 5, 2, 42)[1], ft.kfold_indices(y, 5, 2, 42)[1])
        self.assertFalse(np.array_equal(ft.kfold_indices(y, 5, 2, 42)[1], ft.kfold_indices(y, 5, 2, 7)[1]))

    def test_rejects_bad_fold_numbers(self):
        y = toy_labels(100)
        for folds, fold in ((1, 0), (5, 5), (5, -1)):
            with self.assertRaises(ValueError):
                ft.kfold_indices(y, folds, fold, 42)


class RunCvHelpers(unittest.TestCase):
    def test_fold_dir_naming(self):
        self.assertEqual(rc.fold_dir(Path("runs/cv"), "deberta", 3), Path("runs/cv/deberta-f3"))

    def test_is_done_requires_all_outputs_and_valid_json(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            self.assertFalse(rc.is_done(d))
            for f in rc.REQUIRED:
                (d / f).write_text("id\n1\n", encoding="utf-8")
            (d / "run_info.json").write_text("{ broken", encoding="utf-8")
            self.assertFalse(rc.is_done(d), "a half-written run_info.json must not count as done")
            (d / "run_info.json").write_text(json.dumps({"total_minutes": 12.5}), encoding="utf-8")
            self.assertTrue(rc.is_done(d))
            self.assertAlmostEqual(rc.fold_seconds(d), 750.0)

    def test_fold_seconds_defaults_to_zero(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(rc.fold_seconds(Path(td)), 0.0)


class Statistics(unittest.TestCase):
    def test_wilson_interval_brackets_the_estimate_and_shrinks_with_n(self):
        lo, hi = ac.wilson(970, 1000)
        self.assertLess(lo, 0.97)
        self.assertGreater(hi, 0.97)
        lo2, hi2 = ac.wilson(9700, 10000)
        self.assertLess(hi2 - lo2, hi - lo)
        self.assertEqual(ac.wilson(0, 0), (0.0, 0.0))

    def test_wilson_matches_the_normal_approximation_for_typical_accuracy(self):
        # at 97 % with 25,000 rows the two should agree to about a hundredth of a point
        lo, hi = ac.wilson(24250, 25000)
        half = 1.96 * np.sqrt(0.97 * 0.03 / 25000)
        self.assertAlmostEqual((lo + hi) / 2, 0.97, places=3)
        self.assertAlmostEqual((hi - lo) / 2, half, places=3)

    def test_mcnemar_ignores_agreements(self):
        a = np.array([True, True, False, False, True])
        b = np.array([True, True, True, False, True])
        r = ac.mcnemar_exact(a, b)  # b fixes one case, breaks none
        self.assertEqual((r["a_only"], r["b_only"], r["n_discordant"]), (0, 1, 1))
        self.assertAlmostEqual(r["p_value"], 1.0)  # one discordant pair cannot be significant

    def test_mcnemar_labels_match_the_assembler_message(self):
        # In the assembler call, a is the new fold model and b is the old stage-2 model.
        # The returned names must therefore be: a_only = fold fixed / stage-2 broken,
        # b_only = stage-2 fixed / fold broken.
        a = np.array([True, False, True, False], dtype=bool)
        b = np.array([False, False, True, True], dtype=bool)
        r = ac.mcnemar_exact(a, b)
        self.assertEqual((r["a_only"], r["b_only"]), (1, 1))

    def test_assembler_uses_new_model_as_first_mcnemar_argument(self):
        src = Path(ac.__file__).read_text(encoding="utf-8")
        self.assertIn("mc = mcnemar_exact(a, b)", src)
        self.assertNotIn("mc = mcnemar_exact(b, a)", src)

    def test_mcnemar_identical_classifiers(self):
        a = np.array([True, False, True])
        r = ac.mcnemar_exact(a, a.copy())
        self.assertEqual(r["n_discordant"], 0)
        self.assertEqual(r["p_value"], 1.0)

    def test_mcnemar_symmetric_and_detects_a_large_consistent_gap(self):
        # 10 reviews where a is wrong and b is right, 90 where both agree
        a = np.array([1] * 10 + [1] * 90, dtype=bool)
        a[:10] = False
        b = np.ones(100, dtype=bool)
        r = ac.mcnemar_exact(a, b)
        self.assertEqual((r["a_only"], r["b_only"]), (0, 10))
        self.assertAlmostEqual(r["p_value"], 2.0 * sum(math.comb(10, i) for i in range(0, 1)) / 2 ** 10)
        self.assertLess(r["p_value"], 0.05)
        r2 = ac.mcnemar_exact(b, a)  # the test is two-sided, so the p-value is symmetric
        self.assertAlmostEqual(r["p_value"], r2["p_value"])


class AssembleCv(unittest.TestCase):
    """End-to-end check of the assembler on synthetic folds in a temporary folder."""

    def _write_case(self, td, folds=5, acc=0.97, seed=0):
        data = Path(td) / "data"
        data.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        n, nt = 2000, 400
        y = toy_labels(n, seed)
        train = pd.DataFrame({"id": np.arange(1, n + 1),
                              "review": ["word " * 50] * n,
                              "sentiment": np.where(y == 1, "positive", "negative")})
        test = pd.DataFrame({"id": np.arange(10001, 10001 + nt), "review": ["word " * 50] * nt})
        train.to_csv(data / "train.csv", index=False)
        test.to_csv(data / "test.csv", index=False)
        pd.DataFrame({"id": test["id"], "sentiment": "positive"}).to_csv(
            data / "sample_submission.csv", index=False)

        root = Path(td) / "runs" / "cv"
        for f in range(folds):
            tr_i, va_i = ft.kfold_indices(y, folds, f, 42)
            good = rng.random(len(va_i)) < acc
            pred = np.where(good, y[va_i], 1 - y[va_i])
            margin = np.where(pred == 1, 1.0, -1.0) * rng.gamma(5, 1.0, len(va_i))
            d = root / f"cv-f{f}"
            d.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"id": train["id"].values[va_i], "label": y[va_i],
                          "p_positive": ft.sigmoid(margin), "margin": margin}).to_csv(
                d / "holdout_predictions.csv", index=False)
            tm = rng.normal(0, 4, nt) + np.where(pred.mean() > 0.5, 4, -4)
            pd.DataFrame({"id": test["id"].values, "p_positive": ft.sigmoid(tm),
                          "margin": tm}).to_csv(d / "test_predictions.csv", index=False)
            (d / "run_info.json").write_text(json.dumps(
                {"config": {"model": "microsoft/deberta-v3-large", "epochs": 2}, "total_minutes": 15.0}),
                encoding="utf-8")
        return data, root

    def _run(self, td, folds=5, extra=()):
        data, root = self._write_case(td, folds=folds)
        out = Path(td) / "out"
        ac.main(["--data-dir", str(data), "--runs", str(root), "--output", str(out), *extra])
        return data, out

    def test_writes_a_complete_submission_and_oof_vector(self):
        with tempfile.TemporaryDirectory() as td:
            data, out = self._run(td)
            sub = pd.read_csv(out / "submission.csv")
            sample = pd.read_csv(data / "sample_submission.csv")
            self.assertEqual(list(sub.columns), ["id", "sentiment"])
            self.assertEqual(len(sub), len(sample))
            self.assertEqual(set(sub["id"]), set(sample["id"]))
            self.assertLessEqual(set(sub["sentiment"]), set(LABELS))
            oof = pd.read_csv(out / "oof_predictions.csv")
            self.assertEqual(len(oof), 2000)
            self.assertEqual(oof["id"].nunique(), 2000)
            summary = json.loads((out / "cv_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["n_folds"], 5)
            self.assertEqual(summary["n_oof"], 2000)
            self.assertAlmostEqual(summary["oof_accuracy"], summary["oof_correct"] / 2000)

    def test_refuses_incomplete_folds(self):
        with tempfile.TemporaryDirectory() as td:
            data, root = self._write_case(td, folds=5)
            (root / "cv-f4").rename(Path(td) / "cv-f4-moved")
            with self.assertRaises(SystemExit):
                ac.main(["--data-dir", str(data), "--runs", str(root),
                         "--output", str(Path(td) / "out")])

    def test_refuses_overlapping_folds(self):
        with tempfile.TemporaryDirectory() as td:
            data, root = self._write_case(td, folds=5)
            # make fold 1 a copy of fold 0: the partition check should catch it
            for f in ("holdout_predictions.csv", "test_predictions.csv", "run_info.json"):
                (root / f"cv-f1" / f).write_bytes((root / "cv-f0" / f).read_bytes())
            with self.assertRaises(SystemExit):
                ac.main(["--data-dir", str(data), "--runs", str(root), "--output", str(Path(td) / "out")])

    def test_discovers_the_fold_count_from_the_folder(self):
        with tempfile.TemporaryDirectory() as td:
            data, root = self._write_case(td, folds=4)
            out = Path(td) / "out"
            ac.main(["--data-dir", str(data), "--runs", str(root), "--output", str(out)])
            self.assertEqual(json.loads((out / "cv_summary.json").read_text())["n_folds"], 4)

    def test_no_submission_flag_still_writes_oof(self):
        with tempfile.TemporaryDirectory() as td:
            data, out = self._run(td, extra=("--no-submission",))
            self.assertFalse((out / "submission.csv").exists())
            self.assertTrue((out / "oof_predictions.csv").exists())


class DiscoverPrefix(unittest.TestCase):
    """run_cv.py --name deberta writes deberta-f0…, so the prefix must not be assumed to be 'cv'."""

    def _dirs(self, td, names):
        root = Path(td)
        for n in names:
            (root / n).mkdir()
        return root

    def test_discovers_the_prefix_and_fold_numbers(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._dirs(td, ["deberta-f0", "deberta-f2", "deberta-f1"])
            self.assertEqual(ac.discover_prefix(root), ("deberta", [0, 1, 2]))

    def test_explicit_name_wins(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._dirs(td, ["deberta-f0", "deberta-f1"])
            self.assertEqual(ac.discover_prefix(root, "deberta"), ("deberta", [0, 1]))

    def test_rejects_a_name_that_is_not_there(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._dirs(td, ["deberta-f0"])
            with self.assertRaises(SystemExit):
                ac.discover_prefix(root, "cv")

    def test_rejects_two_families_in_one_folder(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._dirs(td, ["deberta-f0", "electra-f0"])
            with self.assertRaises(SystemExit):
                ac.discover_prefix(root)

    def test_rejects_an_empty_or_missing_folder(self):
        with tempfile.TemporaryDirectory() as td:
            root = self._dirs(td, ["not-a-fold-folder", "deberta-fX"])
            with self.assertRaises(SystemExit):
                ac.discover_prefix(root)
            with self.assertRaises(SystemExit):
                ac.discover_prefix(Path(td) / "does-not-exist")

    def test_end_to_end_without_name_matches_name(self):
        """Assembling a deberta-f* folder must work without --name, as documented."""
        with tempfile.TemporaryDirectory() as td:
            data, root = AssembleCv()._write_case(td, folds=4)
            for f in range(4):  # rename cv-fN -> deberta-fN, as run_cv.py --name deberta does
                (root / f"cv-f{f}").rename(root / f"deberta-f{f}")
            out = Path(td) / "out"
            ac.main(["--data-dir", str(data), "--runs", str(root), "--output", str(out)])
            self.assertEqual(json.loads((out / "cv_summary.json").read_text())["n_folds"], 4)


class EnsembleCvLoading(unittest.TestCase):
    def _case(self, td, bad_oof=None, report_labels=True):
        data = Path(td) / "data"
        data.mkdir(parents=True, exist_ok=True)
        n, nt = 500, 100
        rng = np.random.default_rng(0)
        y = toy_labels(n)
        train = pd.DataFrame({"id": np.arange(1, n + 1), "review": ["w"] * n,
                              "sentiment": np.where(y == 1, "positive", "negative")})
        test = pd.DataFrame({"id": np.arange(1001, 1001 + nt), "review": ["w"] * nt})
        train.to_csv(data / "train.csv", index=False)
        test.to_csv(data / "test.csv", index=False)
        pd.DataFrame({"id": test["id"], "sentiment": "positive"}).to_csv(
            data / "sample_submission.csv", index=False)

        fam = Path(td) / "fam"
        fam.mkdir(parents=True, exist_ok=True)
        ids = train["id"].values
        labels = y if report_labels else 1 - y
        pd.DataFrame({"id": ids, "label": labels, "margin": rng.normal(0, 3, n)}).to_csv(
            fam / "oof_predictions.csv", index=False)
        pd.DataFrame({"id": test["id"].values, "margin": rng.normal(0, 3, nt)}).to_csv(
            fam / "test_predictions.csv", index=False)
        return data, fam, train, test

    def test_loads_a_family_and_aligns_to_train_order(self):
        with tempfile.TemporaryDirectory() as td:
            data, fam, train, test = self._case(td)
            name, oof, tst = ec.load_family(fam, "a", train, test)
            self.assertEqual(name, "a")
            self.assertEqual(len(oof), len(train))
            self.assertEqual(len(tst), len(test))

    def test_rejects_label_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            data, fam, train, test = self._case(td, report_labels=False)
            with self.assertRaises(SystemExit):
                ec.load_family(fam, "a", train, test)

    def test_rejects_missing_oof(self):
        with tempfile.TemporaryDirectory() as td:
            data, fam, train, test = self._case(td)
            (fam / "oof_predictions.csv").unlink()
            with self.assertRaises(SystemExit):
                ec.load_family(fam, "a", train, test)

    def test_rejects_incomplete_oof(self):
        with tempfile.TemporaryDirectory() as td:
            data, fam, train, test = self._case(td)
            o = pd.read_csv(fam / "oof_predictions.csv").iloc[:-1]
            o.to_csv(fam / "oof_predictions.csv", index=False)
            with self.assertRaises(SystemExit):
                ec.load_family(fam, "a", train, test)

    def test_requires_name_equals_folder(self):
        with self.assertRaises(SystemExit):
            ec.main(["--oof", "no-equals-sign", "--output", "x"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
