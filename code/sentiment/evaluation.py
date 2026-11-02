"""Evaluation helpers: metrics and stratified cross-validation folds."""
from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Rows = true class (0 = negative, 1 = positive), columns = predicted class.

    Inputs must be same-length 1D arrays of numeric 0/1 labels. Booleans and
    exact floating-point 0.0/1.0 are accepted; fractional labels are rejected.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must be same-length 1D arrays")
    for labels in (y_true, y_pred):
        if labels.dtype.kind not in "biuf" or not np.all((labels == 0) | (labels == 1)):
            raise ValueError("labels must be numeric 0 or 1")
    encoded = 2 * y_true.astype(np.int64, copy=False) + y_pred.astype(np.int64, copy=False)
    return np.bincount(encoded, minlength=4).reshape(2, 2)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "precision_positive": float(precision),
        "recall_positive": float(recall),
        "f1_positive": float(f1),
        "specificity_negative": float(tn / max(tn + fp, 1)),
        "confusion_matrix": cm.tolist(),
    }


def stratified_kfold(y: np.ndarray, n_splits: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``[(train_idx, valid_idx), ...]`` with the class ratio preserved in every fold."""
    rng = np.random.RandomState(seed)
    y = np.asarray(y)
    fold_of = np.empty(len(y), dtype=np.int64)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        fold_of[idx] = np.arange(len(idx)) % n_splits
    folds = []
    for k in range(n_splits):
        valid = np.where(fold_of == k)[0]
        train = np.where(fold_of != k)[0]
        folds.append((train, valid))
    return folds
