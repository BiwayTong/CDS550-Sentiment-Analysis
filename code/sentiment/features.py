"""Text preprocessing and n-gram feature extraction (NumPy only).

Pipeline:  raw review -> clean_text -> tokenize -> word n-grams -> sparse counts
           -> (optional) TF-IDF weighting.
"""
from __future__ import annotations

import html
import re
from collections import Counter

import numpy as np

from .sparse import CSRMatrix

_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_SPACE_RE = re.compile(r"\s+")
# Words (keeping internal apostrophes such as don't / it's) plus the two
# punctuation marks that carry sentiment on their own.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[!?]")


def clean_text(text: str) -> str:
    """Lower-case and strip the HTML artefacts that the IMDB reviews contain."""
    text = html.unescape(str(text))
    text = _BR_RE.sub(" ", text)
    text = _TAG_RE.sub(" ", text)
    text = _URL_RE.sub(" ", text)
    text = text.lower()
    return _SPACE_RE.sub(" ", text).strip()


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(clean_text(text))


def ngrams(tokens: list[str], ngram_range: tuple[int, int]) -> list[str]:
    lo, hi = ngram_range
    out: list[str] = []
    n_tokens = len(tokens)
    for n in range(lo, hi + 1):
        if n == 1:
            out.extend(tokens)
            continue
        joined = [" ".join(tokens[i:i + n]) for i in range(n_tokens - n + 1)]
        out.extend(joined)
    return out


class NgramCountVectorizer:
    """Bag of word n-grams with document-frequency pruning.

    Parameters mirror the familiar scikit-learn ``CountVectorizer`` so the code
    reads naturally, but everything is implemented here with Python dicts and
    NumPy arrays only.
    """

    def __init__(self, ngram_range: tuple[int, int] = (1, 2), min_df: int = 2, max_df: float = 0.95,
                 max_features: int | None = None):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features
        self.vocabulary_: dict[str, int] = {}
        self.feature_names_: list[str] = []
        self.document_frequency_: np.ndarray | None = None
        self.n_documents_: int = 0

    # ---------------------------------------------------------------- helpers
    def _doc_terms(self, text: str) -> list[str]:
        return ngrams(tokenize(text), self.ngram_range)

    def _set_vocabulary(self, document_frequencies, n_docs: int) -> np.ndarray:
        """Prune/order terms and return their positions in the input sequence."""
        self.n_documents_ = n_docs
        max_df_abs = self.max_df * n_docs if isinstance(self.max_df, float) else self.max_df
        kept = [(term, df, i) for i, (term, df) in enumerate(document_frequencies)
                if self.min_df <= df <= max_df_abs]
        # Deterministic ordering: most frequent first, ties broken alphabetically.
        kept.sort(key=lambda item: (-item[1], item[0]))
        if self.max_features is not None:
            kept = kept[: self.max_features]
        self.feature_names_ = [term for term, _, _ in kept]
        self.vocabulary_ = {term: i for i, term in enumerate(self.feature_names_)}
        self.document_frequency_ = np.array([df for _, df, _ in kept], dtype=np.int64)
        return np.fromiter((i for _, _, i in kept), dtype=np.int32, count=len(kept))

    # -------------------------------------------------------------------- fit
    def fit(self, texts) -> "NgramCountVectorizer":
        df_counter: Counter[str] = Counter()
        n_docs = 0
        for text in texts:
            n_docs += 1
            df_counter.update(set(self._doc_terms(text)))
        self._set_vocabulary(df_counter.items(), n_docs)
        return self

    # -------------------------------------------------------------- transform
    def transform(self, texts) -> CSRMatrix:
        vocab = self.vocabulary_
        data_parts: list[np.ndarray] = []
        index_parts: list[np.ndarray] = []
        indptr = [0]
        total = 0
        for text in texts:
            counts = Counter(vocab[t] for t in self._doc_terms(text) if t in vocab)
            if counts:
                cols = np.fromiter(counts.keys(), dtype=np.int32, count=len(counts))
                vals = np.fromiter(counts.values(), dtype=np.float32, count=len(counts))
                order = np.argsort(cols, kind="stable")
                index_parts.append(cols[order])
                data_parts.append(vals[order])
                total += len(counts)
            indptr.append(total)
        data = np.concatenate(data_parts) if data_parts else np.zeros(0, np.float32)
        indices = np.concatenate(index_parts) if index_parts else np.zeros(0, np.int32)
        return CSRMatrix(data, indices, np.asarray(indptr, dtype=np.int64), (len(indptr) - 1, len(vocab)))

    def fit_transform(self, texts) -> CSRMatrix:
        """Extract each document once, then prune and remap sparse counts.

        Only compact integer column IDs and float32 counts are retained per
        document. Raw reviews and their n-gram lists are not cached, and the
        temporary candidate vocabulary is discarded after fitting.
        """
        candidates: dict[str, int] = {}
        document_frequencies: list[int] = []
        data_parts: list[np.ndarray] = []
        index_parts: list[np.ndarray] = []
        for text in texts:
            counts = Counter(self._doc_terms(text))
            cols = np.empty(len(counts), dtype=np.int32)
            for i, term in enumerate(counts):
                col = candidates.get(term)
                if col is None:
                    col = len(candidates)
                    candidates[term] = col
                    document_frequencies.append(0)
                document_frequencies[col] += 1
                cols[i] = col
            index_parts.append(cols)
            data_parts.append(np.fromiter(counts.values(), dtype=np.float32, count=len(counts)))

        n_docs = len(index_parts)
        # Keep only the compact insertion-ordered term list while finalising
        # the vocabulary.  Dropping the dictionary first avoids retaining two
        # large hash tables (candidate and fitted vocabulary) at once.
        candidate_terms = list(candidates)
        del candidates
        kept_columns = self._set_vocabulary(zip(candidate_terms, document_frequencies), n_docs)
        remap = np.full(len(candidate_terms), -1, dtype=np.int32)
        remap[kept_columns] = np.arange(len(kept_columns), dtype=np.int32)
        # Release unpruned n-gram strings before assembling the final matrix.
        del candidate_terms
        document_frequencies.clear()

        indptr = np.zeros(n_docs + 1, dtype=np.int64)
        for row in range(n_docs):
            cols = remap[index_parts[row]]
            keep = cols >= 0
            cols = cols[keep]
            vals = data_parts[row][keep]
            order = np.argsort(cols, kind="stable")
            index_parts[row] = cols[order]
            data_parts[row] = vals[order]
            indptr[row + 1] = indptr[row] + len(cols)
        data = np.concatenate(data_parts) if data_parts else np.zeros(0, np.float32)
        indices = np.concatenate(index_parts) if index_parts else np.zeros(0, np.int32)
        return CSRMatrix(data, indices, indptr, (n_docs, len(self.vocabulary_)))


class TfidfTransformer:
    """Sub-linear TF-IDF with smoothed IDF and L2 row normalisation.

    This project's sublinear TF uses ``log1p(tf)``. It differs from
    scikit-learn's ``1 + log(tf)`` and is retained to reproduce the reported
    experiments.
    """

    def __init__(self, sublinear_tf: bool = True, norm: bool = True):
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.idf_: np.ndarray | None = None

    def fit(self, counts: CSRMatrix) -> "TfidfTransformer":
        n_docs = counts.shape[0]
        df = np.bincount(counts.indices, minlength=counts.shape[1]).astype(np.float64)
        self.idf_ = (np.log((1.0 + n_docs) / (1.0 + df)) + 1.0).astype(np.float32)
        return self

    def transform(self, counts: CSRMatrix) -> CSRMatrix:
        x = counts.map_values(lambda v: np.log1p(v)) if self.sublinear_tf else counts
        x = x.scale_columns(self.idf_)
        return x.l2_normalize_rows() if self.norm else x

    def fit_transform(self, counts: CSRMatrix) -> CSRMatrix:
        return self.fit(counts).transform(counts)
