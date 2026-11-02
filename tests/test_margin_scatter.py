"""Regression tests for the margin-scatter helper in pseudo_label.

The bug these guard against: the hold-out is stored as a *scatter* of row numbers through
train.csv (e.g. [3, 17, 42, ...]) while the documents that get predicted are a *compact*
list indexed 0..n-1. Writing a prediction back through a row-number dictionary raised
``KeyError: 0`` at the first in-training evaluation, i.e. only after an epoch had already
been paid for. NumPy only: no torch needed.
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import pseudo_label as pl  # noqa: E402


class ScatterMargin(unittest.TestCase):
    def test_writes_each_prediction_at_its_position(self):
        margin = np.zeros(5)
        pl.scatter_margin(margin, np.array([0, 3, 4]), np.array([1.5, -2.0, 0.5]))
        np.testing.assert_array_equal(margin, [1.5, 0.0, 0.0, -2.0, 0.5])

    def test_tolerates_shuffled_batch_order(self):
        margin = np.zeros(4)
        pl.scatter_margin(margin, np.array([2, 0, 3, 1]), np.array([20.0, 0.0, 30.0, 10.0]))
        np.testing.assert_array_equal(margin, [0.0, 10.0, 20.0, 30.0])

    def test_does_not_need_keyzero_to_exist_in_a_row_index(self):
        """A batch index is a position, not a train.csv row number."""
        margin = np.zeros(3)
        row_numbers = {999: 0, 1000: 1, 1001: 2}   # positions 0,1,2 are absent as keys
        pl.scatter_margin(margin, np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(margin, [1.0, 2.0, 3.0])
        self.assertNotIn(0, row_numbers)

    def test_covers_every_document_exactly_once_across_batches(self):
        rng = np.random.default_rng(0)
        n = 2500
        lengths = rng.integers(10, 512, n)
        margin = np.full(n, np.nan)
        order = np.arange(n)
        for b in [order[i:i + 32] for i in range(0, n, 32)]:
            pl.scatter_margin(margin, b, np.arange(len(b), dtype=float))
        self.assertFalse(np.isnan(margin).any())

    def test_source_does_not_scatter_through_a_row_index(self):
        """Guards the actual bug: `margin[pos[int(j)]] = v` inside pseudo_label."""
        src = Path(pl.__file__).read_text(encoding="utf-8")
        self.assertNotIn("pos[int(j)]", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
