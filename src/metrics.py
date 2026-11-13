"""Evaluation metric for BirdCLEF 2026.

The competition scores submissions with "a version of macro-averaged ROC-AUC that
skips classes that have no true positive labels" (see
``References/kaggle - overview.md`` and https://www.kaggle.com/code/metric/birdclef-roc-auc).

This module implements that rule for local validation. It deliberately has no
dependency on PyTorch so it can be unit-tested in a lightweight environment;
tensors are accepted through duck typing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def _to_numpy(values: Any) -> np.ndarray:
    """Convert numpy arrays, lists or PyTorch tensors to a float numpy array."""
    if hasattr(values, "detach"):  # torch.Tensor without importing torch
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def scored_classes(solution: Any) -> np.ndarray:
    """Boolean mask of the classes that can be scored.

    A class is scored when it has at least one positive label (the competition
    rule) and at least one negative label. The second condition only matters for
    small local folds: ROC-AUC is undefined when every row is positive, and the
    hidden test set is never in that situation.
    """
    y = _to_numpy(solution)
    positives = y.sum(axis=0)
    return (positives > 0) & (positives < y.shape[0])


def macro_roc_auc(solution: Any, submission: Any) -> float:
    """Macro ROC-AUC over the classes that have true positives.

    Args:
        solution:   Binary ground truth, shape (n_rows, n_classes).
        submission: Predicted scores or probabilities, same shape.

    Returns:
        Unweighted mean of the per-class ROC-AUC over the scored classes.

    Raises:
        ValueError: if shapes differ, labels are not binary, or no class can be scored.
    """
    y = _to_numpy(solution)
    p = _to_numpy(submission)
    if y.ndim != 2 or y.shape != p.shape:
        raise ValueError(f"solution and submission must be 2-D with equal shapes, got {y.shape} and {p.shape}")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("solution must contain only 0/1 labels (do not pass MixUp soft targets)")
    if not np.isfinite(p).all():
        raise ValueError("submission contains NaN or infinite values")

    mask = scored_classes(y)
    if not mask.any():
        raise ValueError("no class has both positive and negative labels; the metric is undefined")
    return float(roc_auc_score(y[:, mask], p[:, mask], average="macro"))


if __name__ == "__main__":
    targets = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0]])
    preds = np.array([[0.9, 0.1, 0.8], [0.2, 0.8, 0.3], [0.1, 0.7, 0.9], [0.6, 0.4, 0.2]])
    # Class 2 has no positives and is skipped.
    print(f"macro ROC-AUC (2 scored classes): {macro_roc_auc(targets, preds):.4f}")
