"""Tests for the stage-2 helpers (transformer data handling and ensembling). NumPy/pandas only."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import analyze_ensemble as ae  # noqa: E402
import analyze_stage2 as an  # noqa: E402
import predict_windows as pw  # noqa: E402
import pseudo_label as pl  # noqa: E402
import ensemble as ens  # noqa: E402
import finetune_transformer as ft  # noqa: E402


class FreshStartConfig(unittest.TestCase):
    def test_fresh_start_flags_are_exposed(self):
        ns = ft.parse_args(["--data-dir", "data", "--output", "runs/new",
                            "--fresh-start", "--verify-fresh"])
        self.assertTrue(ns.fresh_start)
        self.assertTrue(ns.verify_fresh)

class TransformerHelpers(unittest.TestCase):
    def test_clean_review(self):
        self.assertEqual(ft.clean_review("Good.<br /><br />A &amp; B  <i>x</i>"), "Good. A & B x")
        self.assertEqual(ft.clean_review("I &lt;3 it"), "I <3 it")  # entity unescaped after tag removal

    def test_head_tail(self):
        self.assertEqual(ft.head_tail(range(10), 6, 2), [0, 1, 6, 7, 8, 9])
        self.assertEqual(ft.head_tail(range(4), 6, 2), [0, 1, 2, 3])
        self.assertEqual(ft.head_tail(range(10), 6, 9), [0, 1, 2, 3, 4, 5])

    def test_holdout_is_stratified_and_deterministic(self):
        y = np.r_[np.zeros(500, int), np.ones(500, int)]
        a, b = ft.stratified_holdout(y, 0.1, 42), ft.stratified_holdout(y, 0.1, 42)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(int(a[y == 0].sum()), 50)
        self.assertEqual(int(a[y == 1].sum()), 50)

    def test_batches_cover_every_index_once(self):
        lengths = np.random.default_rng(0).integers(5, 500, 333)
        batches = ft.length_grouped_batches(lengths, np.arange(333), 8, np.random.default_rng(1))
        flat = np.concatenate(batches)
        self.assertEqual(sorted(flat.tolist()), list(range(333)))
        self.assertTrue(all(len(b) <= 8 for b in batches))

    def test_collate(self):
        ids, mask = ft.collate([np.array([5, 6, 7]), np.array([8])], pad_id=0)
        self.assertEqual(ids.tolist(), [[5, 6, 7], [8, 0, 0]])
        self.assertEqual(mask.tolist(), [[1, 1, 1], [1, 0, 0]])

    def test_refuses_sentiment_checkpoints(self):
        for name in ("lvwerra/distilbert-imdb", "textattack/roberta-base-SST-2",
                     "siebert/sentiment-roberta-large-english"):
            self.assertIsNotNone(ft.FORBIDDEN_MODEL_RE.search(name))
        for name in ("microsoft/deberta-v3-large", "roberta-large"):
            self.assertIsNone(ft.FORBIDDEN_MODEL_RE.search(name))

    def test_write_submission_order_and_format(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            pd.DataFrame({"id": [3, 1, 2], "sentiment": "positive"}).to_csv(d / "s.csv", index=False)
            info = ft.write_submission(np.array([1, 2, 3]), np.array([1.0, -1.0, 2.0]), d / "s.csv", d / "o.csv")
            sub = pd.read_csv(d / "o.csv")
            self.assertEqual(sub["id"].tolist(), [3, 1, 2])
            self.assertEqual(sub["sentiment"].tolist(), ["positive", "positive", "negative"])
            self.assertEqual(info["positive"], 2)


class EnsembleHelpers(unittest.TestCase):
    def test_logreg_separable_direction(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, 2))
        y = (X[:, 0] - 0.5 * X[:, 1] > 0).astype(float)
        w = ens.fit_logreg(X, y)
        self.assertGreater(w[1], 0)
        self.assertLess(w[2], 0)
        self.assertGreater(np.mean((ens.predict_margin(w, X) > 0) == y), 0.97)

    def test_cv_accuracy_prefers_informative_model(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 1000)
        good = (2 * y - 1) * 2 + rng.normal(size=1000)
        noise = rng.normal(size=1000)
        acc = ens.cv_accuracy(np.column_stack([good, noise]), y, "stack", repeats=2)
        self.assertGreater(acc, 0.95)

    def test_folds_stratified(self):
        y = np.r_[np.zeros(100, int), np.ones(50, int)]
        f = ens.stratified_folds(y, 10, 0)
        for k in range(10):
            self.assertEqual(int(np.sum((f == k) & (y == 0))), 10)
            self.assertEqual(int(np.sum((f == k) & (y == 1))), 5)


class AnalysisHelpers(unittest.TestCase):
    def test_metrics_and_wilson(self):
        y = np.array([1, 1, 0, 0])
        m = an.metrics(np.array([True, False, True, False]), y)
        self.assertEqual(m["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertAlmostEqual(m["accuracy"], 0.5)
        lo, hi = an.wilson(2418, 2500)
        self.assertLess(lo, 0.9672)
        self.assertGreater(hi, 0.9672)
        self.assertAlmostEqual(lo, 0.9595, places=3)


class LongInputHelpers(unittest.TestCase):
    def test_window_segments_cover_first_and_last_token(self):
        ids = np.arange(1000)
        segs = pw.window_segments(ids, 510, 255)
        self.assertTrue(np.array_equal(segs[0], ids[:510]))
        self.assertTrue(np.array_equal(segs[-1], ids[-510:]))     # last window ends at the end
        self.assertTrue(all(len(s) == 510 for s in segs))
        self.assertEqual(segs[0][0], 0)

    def test_window_segments_short_review_is_one_segment(self):
        self.assertEqual(len(pw.window_segments(np.arange(100), 510, 255)), 1)

    def test_window_segments_respects_max_windows(self):
        segs = pw.window_segments(np.arange(9000), 510, 255, max_windows=4)
        self.assertEqual(len(segs), 4)
        self.assertTrue(np.array_equal(segs[-1], np.arange(9000)[-510:]))

    def test_review_segments_strategies(self):
        ids = np.arange(2000)
        self.assertEqual(len(pw.review_segments(ids, "head_tail", 510, 128, 510, 255)), 1)
        self.assertEqual(len(pw.review_segments(ids, "tta", 510, 128, 510, 255)), 3)
        n_win = len(pw.review_segments(ids, "windows", 510, 128, 510, 255))
        self.assertEqual(len(pw.review_segments(ids, "windows_tta", 510, 128, 510, 255)), n_win + 1)

    def test_aggregate_averages_per_review(self):
        owner = np.array([0, 0, 1, 2, 2, 2])
        m = pw.aggregate(owner, np.array([1.0, 3.0, -2.0, 1.0, 2.0, 6.0]), 3)
        self.assertTrue(np.allclose(m, [2.0, -2.0, 3.0]))

    def test_add_special_wraps_cls_sep(self):
        out = pw.add_special(np.array([7, 8]), cls_id=1, sep_id=2)
        self.assertEqual(out.tolist(), [1, 7, 8, 2])


class PseudoLabelHelpers(unittest.TestCase):
    def test_read_column_orders_by_id(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t); (t / "run").mkdir()
            pd.DataFrame({"id": [3, 1, 2], "p_positive": [0.9, 0.1, 0.2]}).to_csv(
                t / "run" / "test_predictions.csv", index=False)
            got = pl.read_column(t / "run" / "test_predictions.csv", np.array([1, 2, 3]), "p_positive")
            self.assertTrue(np.allclose(got, [0.1, 0.2, 0.9]))

    def test_conf_rule_requires_agreement(self):
        """P is (reviews, models). Review 0: both models say positive. Review 1: the models split
        0.85 / 0.10, so its mean falls below 0.5 and only review 0 may be pseudo-labelled."""
        P = np.array([[0.9, 0.9], [0.85, 0.10]])
        keep, label, _, agree, _ = pl.conf_rule(P, 1.0, 0.5)
        self.assertTrue(np.array_equal(agree, [True, False]))
        self.assertTrue(np.array_equal(keep, [True, False]))
        self.assertEqual(label.tolist(), [1, 0])

    def test_conf_rule_keeps_both_classes_separately(self):
        """A single global probability cut would drop the positive side, because the models are
        more confident on negatives. The rule must keep half of each predicted class."""
        neg = np.linspace(0.01, 0.49, 50)     # 50 reviews the models call negative
        pos = np.linspace(0.51, 0.99, 50)     # 50 reviews the models call positive
        P = np.column_stack([np.concatenate([neg, pos])] * 2)
        keep, label, _, _, thr = pl.conf_rule(P, 0.5, 0.5)
        self.assertEqual(keep.sum(), 50)
        self.assertEqual(int((label[keep] == 0).sum()), 25)
        self.assertEqual(int((label[keep] == 1).sum()), 25)
        self.assertLess(thr[0], 0.5)          # the negative cut is a lower bound
        self.assertGreater(thr[1], 0.5)
        self.assertLess(thr[0], thr[1])

    def test_conf_rule_min_prob_can_reject_everything(self):
        P = np.full((10, 2), 0.6)
        keep, _, _, _, _ = pl.conf_rule(P, 0.5, 0.99)
        self.assertEqual(keep.sum(), 0)

    def test_build_pseudo_survives_a_class_being_dropped(self):
        """If every review is predicted negative, the positive threshold stays undefined.
        Reporting must not crash on it (this used to raise a formatting TypeError)."""
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            n = 60
            ids = np.arange(1, n + 1)
            y = np.tile([0, 1], n // 2)
            (t / "data").mkdir()
            pd.DataFrame({"id": ids, "review": ["r"] * n,
                          "sentiment": ["positive" if v else "negative" for v in y]}).to_csv(
                t / "data" / "train.csv", index=False)
            pd.DataFrame({"id": np.arange(n + 1, n + 21), "review": ["r"] * 20}).to_csv(
                t / "data" / "test.csv", index=False)
            # one model, all-negative predictions on the test set
            (t / "r0").mkdir()
            pd.DataFrame({"id": np.arange(n + 1, n + 21), "p_positive": np.full(20, 0.01),
                          "margin": np.full(20, -5.0)}).to_csv(t / "r0" / "test_predictions.csv", index=False)
            # hold-out predictions, so the calibration table can run too
            import finetune_transformer as ft2
            mask = ft2.stratified_holdout(y, 0.10, 42)
            pd.DataFrame({"id": ids[mask], "label": y[mask],
                          "p_positive": np.where(y[mask] == 1, 0.02, 0.02),
                          "margin": np.full(int(mask.sum()), -5.0)}).to_csv(
                t / "r0" / "holdout_predictions.csv", index=False)
            argv = ["--data-dir", str(t / "data"), "--stage", "pseudo", "--runs", str(t / "r0"),
                    "--output", str(t / "p.csv")]
            pl.build_pseudo(pl.parse_args(argv))       # must not raise
            out = pd.read_csv(t / "p.csv")
            self.assertTrue((out["label"] == 0).all())
            info = json.loads((t / "p.json").read_text())
            self.assertIsNone(info["threshold_positive"])
            self.assertEqual(info["n_kept"], len(out))

    def test_conf_rule_keeps_only_the_confident_share(self):
        P = np.column_stack([np.linspace(0.90, 0.999, 100), np.linspace(0.90, 0.999, 100)])
        keep, _, _, _, _ = pl.conf_rule(P, 0.2, 0.5)
        self.assertEqual(keep.sum(), 20)


class MultiRunEnsembleHelpers(unittest.TestCase):
    def test_mcnemar_exact(self):
        self.assertAlmostEqual(ae.mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(ae.mcnemar_exact(5, 5), 1.0)
        self.assertAlmostEqual(ae.mcnemar_exact(0, 5), 2 / 32)  # two-sided, all 5 discordant pairs one way
        self.assertAlmostEqual(ae.mcnemar_exact(17, 12), 0.4583, places=3)

    def test_mean_combiner_standardises_each_model(self):
        X = np.column_stack([np.array([-1.0, 1.0, 3.0]), 100 * np.array([1.0, -1.0, 0.0])])
        m = ae.mean_combiner(X, X)
        z = (X - X.mean(0)) / X.std(0)
        self.assertTrue(np.allclose(m, z.mean(1)))
        # a model with a 100x larger scale does not dominate
        self.assertTrue(np.allclose(ae.mean_combiner(X * [1, 0.01], X * [1, 0.01]), m))


if __name__ == "__main__":
    unittest.main()
