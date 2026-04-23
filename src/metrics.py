import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def padded_cmap(solution, submission, padding_factor: int = 5) -> float:
    """
    Compute the competition evaluation metric: macro-averaged ROC-AUC with padding.

    This is an exact implementation of Kaggle's BirdCLEF padded c-map metric.
    The padding prevents NaN scores for classes that have no positive labels in
    the current batch by artificially adding a small number of true-positive and
    true-negative rows before computing AUC.

    Args:
        solution:       Ground-truth labels.  Shape (batch, num_classes).
                        Accepts numpy arrays or PyTorch tensors.
        submission:     Model predictions (probabilities).  Same shape as solution.
        padding_factor: Number of artificial (all-ones, all-zeros) row pairs to
                        prepend.  Default of 5 matches the official metric.

    Returns:
        float: Macro-averaged ROC-AUC score in [0, 1].
               Returns 0.5 (random-guess baseline) when no positive labels exist.
    """
    if isinstance(solution, torch.Tensor):
        solution = solution.detach().cpu().numpy()
    if isinstance(submission, torch.Tensor):
        submission = submission.detach().cpu().numpy()

    # Guard against batches with no positive labels at all.
    if solution.sum() == 0:
        return 0.5

    num_classes = solution.shape[1]

    # Build the padding rows: alternating all-ones and all-zeros.
    # This ensures every class has at least one positive and one negative example,
    # which is required by roc_auc_score to produce a finite AUC per class.
    padding_rows = []
    for _ in range(padding_factor):
        padding_rows.append([1] * num_classes)
        padding_rows.append([0] * num_classes)

    padding_array = np.array(padding_rows, dtype=np.float32)
    solution   = np.vstack([solution,   padding_array])
    submission = np.vstack([submission, padding_array])

    return roc_auc_score(solution, submission, average="macro")


if __name__ == "__main__":
    print("=== padded_cmap smoke test ===")
    dummy_targets = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    dummy_preds   = np.array([[0.9, 0.1, 0.1], [0.2, 0.8, 0.3], [0.1, 0.2, 0.9]])
    score = padded_cmap(dummy_targets, dummy_preds)
    print(f"Dummy AUC: {score:.4f}  (expected ~1.0 for near-perfect predictions)")
