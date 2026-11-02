"""A minimal compressed-sparse-row (CSR) matrix built only on NumPy.

The document-term matrices used in this project are extremely sparse
(about 0.1% non-zero entries), so a dense array would need tens of
gigabytes.  This module implements just the handful of operations the
feature extractor and the models need:

* ``X @ w``          (matrix-vector product, used for scoring)
* ``X.T @ r``        (transposed product, used for gradients / feature sums)
* row L2 scaling and column scaling (used by TF-IDF and NB feature weighting)
* binarisation and value mapping (used by NB-LR and sublinear TF-IDF)

Everything is vectorised with ``numpy.bincount`` / ``numpy.repeat`` so the
hot loops run in C even though SciPy and scikit-learn are not required.
"""
from __future__ import annotations

import numpy as np


class CSRMatrix:
    """Compressed sparse row matrix with float32 data."""

    def __init__(self, data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, shape: tuple[int, int]):
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.indices = np.ascontiguousarray(indices, dtype=np.int32)
        self.indptr = np.ascontiguousarray(indptr, dtype=np.int64)
        self.shape = (int(shape[0]), int(shape[1]))
        if self.indptr.shape[0] != self.shape[0] + 1:
            raise ValueError("indptr length must equal n_rows + 1")
        if self.data.shape[0] != self.indices.shape[0]:
            raise ValueError("data and indices must have the same length")
        self._row_ids: np.ndarray | None = None

    # ------------------------------------------------------------------ basics
    @property
    def nnz(self) -> int:
        return int(self.data.shape[0])

    def row_lengths(self) -> np.ndarray:
        return np.diff(self.indptr)

    def row_ids(self) -> np.ndarray:
        """Row index of every stored entry (cached, needed for bincount tricks)."""
        if self._row_ids is None:
            self._row_ids = np.repeat(np.arange(self.shape[0], dtype=np.int64), self.row_lengths())
        return self._row_ids

    # ------------------------------------------------------------ linear algebra
    def dot(self, w: np.ndarray, *, data64: np.ndarray | None = None) -> np.ndarray:
        """Return ``X @ w`` as a float64 vector of length n_rows.

        ``data64`` may supply a temporary float64 copy of the current stored
        values, with the same shape, for repeated products during one fit.
        It is never cached on the matrix.
        """
        w = np.asarray(w, dtype=np.float64)
        data = self.data.astype(np.float64) if data64 is None else data64
        prod = data * w[self.indices]
        return np.bincount(self.row_ids(), weights=prod, minlength=self.shape[0])

    def rdot(self, r: np.ndarray, *, data64: np.ndarray | None = None) -> np.ndarray:
        """Return ``X.T @ r`` as a float64 vector of length n_cols.

        ``data64`` has the same temporary-copy contract as in :meth:`dot`.
        """
        r = np.asarray(r, dtype=np.float64)
        data = self.data.astype(np.float64) if data64 is None else data64
        prod = data * r[self.row_ids()]
        return np.bincount(self.indices, weights=prod, minlength=self.shape[1])

    def column_sums(self, mask: np.ndarray | None = None) -> np.ndarray:
        """Sum of each column, optionally restricted to the rows where ``mask`` is True."""
        if mask is None:
            return np.bincount(self.indices, weights=self.data.astype(np.float64), minlength=self.shape[1])
        return self.rdot(mask.astype(np.float64))

    # ------------------------------------------------------------------ scaling
    def scale_rows(self, factors: np.ndarray) -> "CSRMatrix":
        factors = np.asarray(factors, dtype=np.float32)
        return CSRMatrix(self.data * factors[self.row_ids()], self.indices, self.indptr, self.shape)

    def scale_columns(self, factors: np.ndarray) -> "CSRMatrix":
        factors = np.asarray(factors, dtype=np.float32)
        return CSRMatrix(self.data * factors[self.indices], self.indices, self.indptr, self.shape)

    def l2_normalize_rows(self) -> "CSRMatrix":
        sq = np.bincount(self.row_ids(), weights=self.data.astype(np.float64) ** 2, minlength=self.shape[0])
        norms = np.sqrt(sq)
        norms[norms == 0] = 1.0
        return self.scale_rows(1.0 / norms)

    def binarize(self) -> "CSRMatrix":
        return CSRMatrix(np.ones_like(self.data), self.indices, self.indptr, self.shape)

    def map_values(self, fn) -> "CSRMatrix":
        return CSRMatrix(fn(self.data), self.indices, self.indptr, self.shape)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"CSRMatrix(shape={self.shape}, nnz={self.nnz})"
