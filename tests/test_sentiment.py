"""Numerical regression tests for the handwritten sentiment pipeline.

Run from the submission directory with ``python -m unittest discover -s tests -v``.
The core tests need NumPy only. Install ``requirements-dev.txt`` to enable the
optional SciPy/scikit-learn reference comparisons.

Compatibility note: this project's default sublinear TF is ``log1p(count)``.
scikit-learn's ``sublinear_tf=True`` instead uses ``1 + log(count)``. The
reference test therefore preweights counts with log1p and explicitly disables
scikit-learn's sublinear TF; linear TF is compared directly with equal settings.
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest
import weakref

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

# Resolve the submitted package without installing it or depending on the cwd.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from sentiment.evaluation import (accuracy, classification_metrics,
                                  confusion_matrix, stratified_kfold)
from sentiment.features import (NgramCountVectorizer, TfidfTransformer,
                                clean_text, ngrams, tokenize)
from sentiment.models import (LogisticRegression, MultinomialNB,
                              NBLogisticRegression)
from sentiment.sparse import CSRMatrix

try:
    from scipy.sparse import csr_matrix as scipy_csr
    from sklearn.feature_extraction.text import CountVectorizer as SkCountVectorizer
    from sklearn.feature_extraction.text import TfidfTransformer as SkTfidfTransformer
    from sklearn.linear_model import LogisticRegression as SkLogisticRegression
    from sklearn.metrics import (accuracy_score, confusion_matrix as sk_confusion_matrix,
                                 f1_score, precision_score, recall_score)
    from sklearn.naive_bayes import MultinomialNB as SkMultinomialNB
except ImportError:
    HAS_SKLEARN = False
else:
    HAS_SKLEARN = True


def as_csr(dense: np.ndarray) -> CSRMatrix:
    """Build canonical CSR from a dense fixture, retaining empty rows/columns."""
    dense = np.asarray(dense, dtype=np.float32)
    row, col = np.nonzero(dense)
    indptr = np.concatenate(([0], np.cumsum(np.bincount(row, minlength=dense.shape[0]))))
    return CSRMatrix(dense[row, col], col, indptr, dense.shape)


def as_dense(matrix: CSRMatrix) -> np.ndarray:
    dense = np.zeros(matrix.shape, dtype=np.float32)
    np.add.at(dense, (matrix.row_ids(), matrix.indices), matrix.data)
    return dense


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(np.asarray(values) / 2.0))


class SparseMatrixTests(unittest.TestCase):
    def setUp(self):
        self.dense = np.array([[0, 2, 0, -1], [0, 0, 0, 0],
                               [3, 0, 4, 0], [0, 0, 0, 0]], dtype=np.float32)
        self.matrix = as_csr(self.dense)

    def test_products_and_masked_column_sums_match_dense_algebra(self):
        weights = np.array([0.3, -2.0, 0.7, 1.2])
        residuals = np.array([0.4, 0.2, -0.8, 2.0])
        mask = np.array([True, False, True, False])
        assert_allclose(self.matrix.dot(weights), self.dense @ weights, atol=1e-12)
        assert_allclose(self.matrix.rdot(residuals), self.dense.T @ residuals, atol=1e-12)
        assert_allclose(self.matrix.column_sums(), self.dense.sum(axis=0))
        assert_allclose(self.matrix.column_sums(mask), self.dense[mask].sum(axis=0))
        self.assertEqual(self.matrix.nnz, np.count_nonzero(self.dense))

    def test_fit_local_float64_products_match_current_matrix_values(self):
        weights = np.array([0.3, -2.0, 0.7, 1.2])
        residuals = np.array([0.4, 0.2, -0.8, 2.0])
        data64 = self.matrix.data.astype(np.float64)
        assert_allclose(self.matrix.dot(weights, data64=data64), self.dense @ weights, atol=1e-12)
        assert_allclose(self.matrix.rdot(residuals, data64=data64), self.dense.T @ residuals,
                        atol=1e-12)
        # A matrix must not retain a stale conversion across independent fits/products.
        self.matrix.data[0] = 5
        current = as_dense(self.matrix)
        assert_allclose(self.matrix.dot(weights), current @ weights, atol=1e-12)
        assert_allclose(self.matrix.rdot(residuals), current.T @ residuals, atol=1e-12)
        fresh_data64 = self.matrix.data.astype(np.float64)
        assert_allclose(self.matrix.dot(weights, data64=fresh_data64), current @ weights,
                        atol=1e-12)
        assert_allclose(self.matrix.rdot(residuals, data64=fresh_data64), current.T @ residuals,
                        atol=1e-12)

    def test_scaling_normalisation_and_value_transforms(self):
        rows = np.array([2.0, -1.0, 0.5, 0.0], dtype=np.float32)
        columns = np.array([-1.0, 0.5, 2.0, 3.0], dtype=np.float32)
        assert_allclose(as_dense(self.matrix.scale_rows(rows)), self.dense * rows[:, None])
        assert_allclose(as_dense(self.matrix.scale_columns(columns)), self.dense * columns)
        norms = np.linalg.norm(self.dense, axis=1, keepdims=True)
        expected = self.dense / np.where(norms == 0, 1.0, norms)
        assert_allclose(as_dense(self.matrix.l2_normalize_rows()), expected, atol=1e-7)
        assert_array_equal(as_dense(self.matrix.binarize()), (self.dense != 0).astype(np.float32))
        assert_array_equal(as_dense(self.matrix.map_values(lambda x: x * 2)), self.dense * 2)
        assert_array_equal(as_dense(self.matrix), self.dense)

    def test_empty_matrix_products_keep_declared_dimensions(self):
        for shape in ((0, 4), (3, 4), (3, 0)):
            with self.subTest(shape=shape):
                matrix = as_csr(np.zeros(shape))
                assert_array_equal(matrix.dot(np.ones(shape[1])), np.zeros(shape[0]))
                assert_array_equal(matrix.rdot(np.ones(shape[0])), np.zeros(shape[1]))
                self.assertEqual(matrix.l2_normalize_rows().shape, shape)


class FeatureTests(unittest.TestCase):
    def test_cleaning_tokenisation_and_ngrams(self):
        text = "<b>GREAT</b><BR />Don't &amp; can't!  10/10? https://example.com/a  www.foo.com"
        self.assertEqual(clean_text(text), "great don't & can't! 10/10?")
        tokens = ["great", "don't", "can't", "!", "10", "10", "?"]
        self.assertEqual(tokenize(text), tokens)
        self.assertEqual(ngrams(["good", "!", "movie"], (1, 3)),
                         ["good", "!", "movie", "good !", "! movie", "good ! movie"])
        self.assertEqual(ngrams([], (1, 3)), [])

    def test_document_frequency_pruning_and_deterministic_order(self):
        texts = ["GOOD good! movie", "Good movie?", "bad movie!", "bad plot?"]
        vectorizer = NgramCountVectorizer(ngram_range=(1, 1), min_df=2, max_df=0.75)
        counts = vectorizer.fit_transform(iter(texts))
        self.assertEqual(vectorizer.feature_names_, ["movie", "!", "?", "bad", "good"])
        assert_array_equal(vectorizer.document_frequency_, [3, 2, 2, 2, 2])
        self.assertEqual(vectorizer.n_documents_, 4)
        assert_array_equal(as_dense(counts), [[1, 1, 0, 0, 2], [1, 0, 1, 0, 1],
                                             [1, 1, 0, 1, 0], [0, 0, 1, 1, 0]])
        assert_array_equal(as_dense(vectorizer.transform(["unknown", "good good?"])),
                           [[0, 0, 0, 0, 0], [0, 0, 1, 0, 2]])
        limited = NgramCountVectorizer(ngram_range=(1, 1), min_df=2,
                                      max_df=0.75, max_features=3).fit(texts)
        self.assertEqual(limited.feature_names_, ["movie", "!", "?"])
        self.assertEqual(vectorizer.transform([]).shape, (0, 5))

    def test_single_pass_fit_transform_preserves_separate_fit_transform_results(self):
        texts = ["good good! movie", "<b>good</b> movie?", "", "bad plot!", "bad movie?"]
        configurations = [dict(ngram_range=(1, 3), min_df=1, max_df=1.0),
                          dict(ngram_range=(1, 2), min_df=2, max_df=0.8, max_features=5),
                          dict(ngram_range=(2, 3), min_df=9, max_df=1.0)]
        for kwargs in configurations:
            with self.subTest(kwargs=kwargs):
                separate = NgramCountVectorizer(**kwargs).fit(texts)
                single_pass = NgramCountVectorizer(**kwargs)
                result = single_pass.fit_transform(iter(texts))
                self.assertEqual(single_pass.vocabulary_, separate.vocabulary_)
                self.assertEqual(single_pass.feature_names_, separate.feature_names_)
                assert_array_equal(single_pass.document_frequency_, separate.document_frequency_)
                assert_array_equal(as_dense(result), as_dense(separate.transform(texts)))
                self.assertEqual(result.indptr[-1], result.nnz)
                # Canonical column order is needed by downstream sparse consumers.
                for row in range(result.shape[0]):
                    columns = result.indices[result.indptr[row]:result.indptr[row + 1]]
                    self.assertTrue(np.all(np.diff(columns) > 0))
        empty = NgramCountVectorizer(min_df=1, max_df=1.0).fit_transform(iter([]))
        self.assertEqual(empty.shape, (0, 0))

    def test_tfidf_matches_dense_definition_and_keeps_empty_rows(self):
        dense = np.array([[1, 3, 0, 0], [0, 2, 4, 0], [0, 0, 0, 0]], dtype=np.float32)
        counts = as_csr(dense)
        expected_idf = np.log((1 + len(dense)) / (1 + np.count_nonzero(dense, axis=0))) + 1
        for sublinear in (False, True):
            for norm in (False, True):
                with self.subTest(sublinear=sublinear, norm=norm):
                    transformer = TfidfTransformer(sublinear_tf=sublinear, norm=norm)
                    result = transformer.fit_transform(counts)
                    expected = (np.log1p(dense) if sublinear else dense) * expected_idf
                    if norm:
                        lengths = np.linalg.norm(expected, axis=1, keepdims=True)
                        expected /= np.where(lengths == 0, 1.0, lengths)
                    assert_allclose(transformer.idf_, expected_idf, rtol=1e-7)
                    assert_allclose(as_dense(result), expected, rtol=2e-7, atol=1e-7)
                    assert_array_equal(as_dense(counts), dense)
                    held_out = as_csr(np.array([[0, 0, 0, 1]], dtype=np.float32))
                    expected_new = np.log1p(1.0) if sublinear else 1.0
                    expected_new = 1.0 if norm else expected_new * expected_idf[-1]
                    assert_allclose(as_dense(transformer.transform(held_out)),
                                    [[0, 0, 0, expected_new]], rtol=2e-7)


class ModelTests(unittest.TestCase):
    def test_logistic_fit_snapshot_is_reused_released_and_fresh_on_refit(self):
        class TrackingMatrix(CSRMatrix):
            def dot(self, weights, *, data64=None):
                if data64 is not None:
                    self.snapshots.append((id(data64), weakref.ref(data64)))
                return super().dot(weights, data64=data64)

            def rdot(self, residuals, *, data64=None):
                if data64 is not None:
                    self.snapshots.append((id(data64), weakref.ref(data64)))
                return super().rdot(residuals, data64=data64)

        dense = np.array([[-2, 0], [-1, 1], [1, 0], [2, 1]], dtype=np.float32)
        labels = np.array([0, 0, 1, 1])
        source = as_csr(dense)
        matrix = TrackingMatrix(source.data, source.indices, source.indptr, source.shape)
        matrix.snapshots = []
        model = LogisticRegression(C=0.7).fit(matrix, labels)
        self.assertGreater(len(matrix.snapshots), 2)
        self.assertEqual(len({identity for identity, _ in matrix.snapshots}), 1)
        self.assertIsNone(matrix.snapshots[0][1]())

        # Refitting the same object must observe mutations, just like a fresh CSR.
        matrix.data[0] = -4
        mutated = as_dense(matrix)
        model.fit(matrix, labels)
        fresh = LogisticRegression(C=0.7).fit(as_csr(mutated), labels)
        assert_array_equal(model.coef_, fresh.coef_)
        self.assertEqual(model.intercept_, fresh.intercept_)
        self.assertEqual(model.loss_trace_, fresh.loss_trace_)
        self.assertTrue(all(ref() is None for _, ref in matrix.snapshots))

    def test_naive_bayes_matches_manual_likelihoods(self):
        dense = np.array([[3, 1, 0], [1, 1, 0], [0, 1, 4], [0, 0, 2], [0, 1, 1]],
                         dtype=np.float32)
        labels = np.array([0, 0, 1, 1, 1])
        model = MultinomialNB(alpha=0.5).fit(as_csr(dense), labels)
        counts = np.array([[4.5, 2.5, 0.5], [0.5, 2.5, 7.5]])
        log_prob = np.log(counts / counts.sum(axis=1, keepdims=True))
        priors = np.log([2 / 5, 3 / 5])
        test = np.array([[0, 0, 0], [2, 0, 0], [0, 0, 2], [1, 2, 1]], dtype=np.float32)
        scores = test @ (log_prob[1] - log_prob[0]) + priors[1] - priors[0]
        assert_allclose(model.feature_log_prob_, log_prob, atol=1e-12)
        assert_allclose(model.class_log_prior_, priors, atol=1e-12)
        assert_allclose(model.decision_function(as_csr(test)), scores, atol=1e-12)
        assert_allclose(model.predict_proba(as_csr(test)), sigmoid(scores), atol=1e-12)
        assert_array_equal(model.predict(as_csr(test)), scores > 0)

    def test_logistic_regression_matches_independent_one_dimensional_optimum(self):
        dense = np.array([[-2], [-1], [1], [2]], dtype=np.float32)
        labels = np.array([0, 0, 1, 1])
        C = 0.7
        # Solve the derivative's unique zero by bisection, independently of L-BFGS.
        lo, hi = 0.0, 10.0
        for _ in range(80):
            weight = (lo + hi) / 2
            derivative = np.mean((sigmoid(dense[:, 0] * weight) - labels) * dense[:, 0])
            derivative += weight / (C * len(labels))
            if derivative > 0:
                hi = weight
            else:
                lo = weight
        expected = (lo + hi) / 2
        for fit_intercept in (False, True):
            with self.subTest(fit_intercept=fit_intercept):
                model = LogisticRegression(C=C, max_iter=500,
                                           fit_intercept=fit_intercept).fit(as_csr(dense), labels)
                assert_allclose(model.coef_, [expected], atol=1e-4)
                self.assertAlmostEqual(model.intercept_, 0.0, places=8)
                assert_array_equal(model.predict(as_csr(dense)), labels)
                assert_allclose(model.predict_proba(as_csr(dense)),
                                sigmoid(dense[:, 0] * expected), atol=3e-5)
                self.assertLess(model.loss_trace_[-1], model.loss_trace_[0])
                self.assertTrue(np.all(np.diff(model.loss_trace_) <= 1e-12))

    def test_nb_lr_ratio_uses_presence_and_probabilities_match_scores(self):
        dense = np.array([[3, 0, 1], [2, 0, 0], [0, 4, 1], [0, 1, 1]], dtype=np.float32)
        labels = np.array([0, 0, 1, 1])
        model = NBLogisticRegression(alpha=0.5, beta=0.75, max_iter=500).fit(as_csr(dense), labels)
        positive = np.array([0.5, 2.5, 2.5])
        negative = np.array([2.5, 0.5, 1.5])
        expected_ratio = np.log(positive / positive.sum()) - np.log(negative / negative.sum())
        assert_allclose(model.r_, expected_ratio, rtol=1e-7, atol=1e-7)
        test = np.array([[2, 0, 1], [1, 0, 1], [0, 0, 0], [0, 7, 1]], dtype=np.float32)
        score = ((test != 0).astype(np.float32) * model.r_) @ model.coef_ + model.intercept_
        assert_allclose(model.decision_function(as_csr(test)), score, atol=1e-12)
        assert_allclose(model.predict_proba(as_csr(test)), sigmoid(score), atol=1e-12)
        self.assertEqual(model.decision_function(as_csr(test))[0],
                         model.decision_function(as_csr(test))[1])


class EvaluationTests(unittest.TestCase):
    def test_metrics_match_known_confusion_matrix(self):
        truth = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        predictions = np.array([0, 0, 0, 1, 0, 0, 1, 1])
        assert_array_equal(confusion_matrix(truth, predictions), [[3, 1], [2, 2]])
        self.assertEqual(accuracy(truth, predictions), 5 / 8)
        metrics = classification_metrics(truth, predictions)
        for name, expected in {"accuracy": 5 / 8, "precision_positive": 2 / 3,
                               "recall_positive": 1 / 2, "f1_positive": 4 / 7,
                               "specificity_negative": 3 / 4}.items():
            self.assertAlmostEqual(metrics[name], expected)
        self.assertEqual(metrics["confusion_matrix"], [[3, 1], [2, 2]])

    def test_confusion_matrix_accepts_numeric_binary_types(self):
        for dtype in (np.bool_, np.int8, np.uint32, np.float32, np.float64):
            with self.subTest(dtype=dtype):
                assert_array_equal(confusion_matrix(np.array([0, 0, 1, 1], dtype=dtype),
                                                    np.array([0, 1, 0, 1], dtype=dtype)),
                                   [[1, 1], [1, 1]])
        assert_array_equal(confusion_matrix(np.array([], dtype=int), np.array([], dtype=int)),
                           np.zeros((2, 2), dtype=np.int64))

    def test_confusion_matrix_rejects_mismatched_nonbinary_or_nonvector_labels(self):
        invalid = [([0, 1], [1]), ([0], [0, 1]), ([[0, 1]], [[0, 1]]),
                   (0, 1), ([0, 2], [0, 1]), ([-1, 1], [0, 1]),
                   ([0, 0.5], [0, 1]), ([0, np.nan], [0, 1]),
                   ([0, np.inf], [0, 1]), (["0", "1"], [0, 1]),
                   (np.array([0, 1], dtype=object), [0, 1]),
                   ([0, 1], [0, 0.5])]
        for truth, predictions in invalid:
            with self.subTest(truth=truth, predictions=predictions):
                with self.assertRaises(ValueError):
                    confusion_matrix(truth, predictions)
                with self.assertRaises(ValueError):
                    classification_metrics(truth, predictions)

    def test_metrics_zero_denominators_return_finite_zero(self):
        for truth, predictions in (([0, 1, 1], [0, 0, 0]), ([0, 0], [0, 0])):
            metrics = classification_metrics(np.array(truth), np.array(predictions))
            for name in ("precision_positive", "recall_positive", "f1_positive"):
                self.assertEqual(metrics[name], 0.0)

    def test_stratified_folds_partition_data_without_leakage(self):
        labels = np.array([0] * 13 + [1] * 11)
        folds = stratified_kfold(labels, n_splits=5, seed=42)
        repeated = stratified_kfold(labels, n_splits=5, seed=42)
        all_validation = []
        class_sizes = []
        for (train, validation), (train_again, validation_again) in zip(folds, repeated):
            assert_array_equal(train, train_again)
            assert_array_equal(validation, validation_again)
            self.assertEqual(len(np.intersect1d(train, validation)), 0)
            assert_array_equal(np.sort(np.concatenate((train, validation))), np.arange(len(labels)))
            all_validation.extend(validation)
            class_sizes.append(np.bincount(labels[validation], minlength=2))
        assert_array_equal(np.sort(all_validation), np.arange(len(labels)))
        self.assertTrue(np.all(np.ptp(np.array(class_sizes), axis=0) <= 1))


@unittest.skipUnless(HAS_SKLEARN, "optional reference tests require scipy and scikit-learn")
class SklearnReferenceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(23)
        dense = rng.poisson(2.0, size=(90, 9)).astype(np.float32)
        dense[rng.random_sample(dense.shape) < 0.65] = 0
        dense[-1] = 0
        self.dense = dense
        self.matrix = as_csr(dense)
        self.reference_matrix = scipy_csr(dense)
        probabilities = sigmoid(dense @ np.array([-0.9, 0.7, 0.2, -0.1, 0.8,
                                                   -0.5, 0.3, -0.7, 0.1]) + 0.15)
        self.labels = rng.binomial(1, probabilities)

    def test_count_vectorizer_matches_sklearn_with_project_token_pattern(self):
        texts = ["Good good <b>movie</b>!", "Good movie?", "bad movie!",
                 "bad plot?", "Don't call www.example.com a movie."]
        own = NgramCountVectorizer(ngram_range=(1, 2), min_df=2, max_df=1.0)
        own_counts = own.fit_transform(texts)
        reference = SkCountVectorizer(preprocessor=clean_text, lowercase=False,
                                     token_pattern=r"[a-z0-9]+(?:'[a-z]+)?|[!?]",
                                     ngram_range=(1, 2), min_df=2, max_df=1.0)
        ref_counts = reference.fit_transform(texts)
        self.assertEqual(set(own.feature_names_), set(reference.get_feature_names_out()))
        order = [reference.vocabulary_[term] for term in own.feature_names_]
        assert_array_equal(as_dense(own_counts), ref_counts[:, order].toarray())
        assert_array_equal(own.document_frequency_,
                           np.asarray((ref_counts[:, order] != 0).sum(axis=0)).ravel())
        held_out = ["good movie! good movie!", "unseen words"]
        assert_array_equal(as_dense(own.transform(held_out)),
                           reference.transform(held_out)[:, order].toarray())

    def test_linear_tf_matches_sklearn_with_equal_parameters(self):
        for norm in (False, True):
            with self.subTest(norm=norm):
                own = TfidfTransformer(sublinear_tf=False, norm=norm)
                reference = SkTfidfTransformer(sublinear_tf=False, smooth_idf=True,
                                              use_idf=True, norm="l2" if norm else None)
                expected = reference.fit_transform(self.reference_matrix)
                result = own.fit_transform(self.matrix)
                assert_allclose(own.idf_, reference.idf_, rtol=2e-7)
                assert_allclose(as_dense(result), expected.toarray(), rtol=3e-7, atol=1e-7)

    def test_default_log1p_tf_matches_explicit_sklearn_compatibility_mode(self):
        # Never compare against sklearn sublinear_tf=True: that is a different TF formula.
        weighted = self.reference_matrix.copy()
        weighted.data = np.log1p(weighted.data)
        reference = SkTfidfTransformer(sublinear_tf=False, smooth_idf=True,
                                      use_idf=True, norm="l2")
        expected = reference.fit_transform(weighted)
        own = TfidfTransformer()
        assert_allclose(as_dense(own.fit_transform(self.matrix)), expected.toarray(),
                        rtol=3e-7, atol=1e-7)
        assert_allclose(own.idf_, reference.idf_, rtol=2e-7)

    def test_naive_bayes_matches_sklearn_probabilities(self):
        own = MultinomialNB(alpha=0.7).fit(self.matrix, self.labels)
        reference = SkMultinomialNB(alpha=0.7, fit_prior=True).fit(
            self.reference_matrix, self.labels)
        assert_allclose(own.feature_log_prob_, reference.feature_log_prob_, atol=1e-12)
        assert_allclose(own.class_log_prior_, reference.class_log_prior_, atol=1e-12)
        assert_allclose(own.predict_proba(self.matrix),
                        reference.predict_proba(self.reference_matrix)[:, 1], atol=1e-12)
        assert_array_equal(own.predict(self.matrix), reference.predict(self.reference_matrix))

    def test_logistic_regression_matches_sklearn_coefficients_and_predictions(self):
        for fit_intercept in (False, True):
            with self.subTest(fit_intercept=fit_intercept):
                own = LogisticRegression(C=0.8, max_iter=1000,
                                         fit_intercept=fit_intercept).fit(self.matrix, self.labels)
                reference = SkLogisticRegression(C=0.8, solver="lbfgs", max_iter=1000,
                                                 fit_intercept=fit_intercept, tol=1e-10).fit(
                    self.reference_matrix, self.labels)
                assert_allclose(own.coef_, reference.coef_[0], atol=1e-3, rtol=0)
                assert_allclose(own.intercept_, np.ravel(reference.intercept_)[0],
                                atol=1e-3, rtol=0)
                assert_allclose(own.predict_proba(self.matrix),
                                reference.predict_proba(self.reference_matrix)[:, 1], atol=3e-4)
                assert_array_equal(own.predict(self.matrix), reference.predict(self.reference_matrix))

    def test_nb_lr_matches_scipy_and_sklearn_composition(self):
        C, alpha, beta = 0.9, 0.5, 0.65
        binary = self.reference_matrix.copy()
        binary.data.fill(1)
        p = np.asarray(binary[self.labels == 1].sum(axis=0)).ravel().astype(np.float64) + alpha
        q = np.asarray(binary[self.labels == 0].sum(axis=0)).ravel().astype(np.float64) + alpha
        ratio = (np.log(p / p.sum()) - np.log(q / q.sum())).astype(np.float32)
        scaled = binary.multiply(ratio).tocsr()
        reference = SkLogisticRegression(C=C, solver="lbfgs", max_iter=1000,
                                         tol=1e-10).fit(scaled, self.labels)
        weights = (1 - beta) * np.abs(reference.coef_[0]).mean() + beta * reference.coef_[0]
        scores = np.asarray(scaled @ weights).ravel() + reference.intercept_[0]
        own = NBLogisticRegression(C=C, alpha=alpha, beta=beta, max_iter=1000).fit(
            self.matrix, self.labels)
        assert_allclose(own.r_, ratio, atol=1e-7)
        assert_allclose(own.lr_.coef_, reference.coef_[0], atol=1e-3, rtol=0)
        assert_allclose(own.coef_, weights, atol=1e-3, rtol=0)
        assert_allclose(own.predict_proba(self.matrix), sigmoid(scores), atol=3e-4)
        assert_array_equal(own.predict(self.matrix), scores > 0)

    def test_metrics_match_sklearn(self):
        predictions = (self.dense[:, 0] + self.dense[:, 1] > 2).astype(int)
        own = classification_metrics(self.labels, predictions)
        self.assertEqual(own["accuracy"], accuracy_score(self.labels, predictions))
        self.assertEqual(own["precision_positive"],
                         precision_score(self.labels, predictions, zero_division=0))
        self.assertEqual(own["recall_positive"],
                         recall_score(self.labels, predictions, zero_division=0))
        self.assertAlmostEqual(own["f1_positive"],
                               f1_score(self.labels, predictions, zero_division=0))
        assert_array_equal(own["confusion_matrix"],
                           sk_confusion_matrix(self.labels, predictions, labels=[0, 1]))


if __name__ == "__main__":
    unittest.main()
