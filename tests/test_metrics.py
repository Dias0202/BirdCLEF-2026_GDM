import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from metrics import macro_roc_auc, scored_classes


def test_class_without_positives_is_skipped_not_rewarded():
    y = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0]])
    # Class 0 is perfectly ranked, class 1 is ranked backwards; class 2 has no positives.
    p = np.array([[0.9, 0.1, 0.2], [0.1, 0.05, 0.3], [0.8, 0.3, 0.1], [0.2, 0.02, 0.4]])

    assert scored_classes(y).tolist() == [True, True, False]
    assert macro_roc_auc(y, p) == pytest.approx(0.5)  # mean(1.0, 0.0); class 2 contributes nothing


def test_matches_per_class_mean_over_scored_classes():
    rng = np.random.default_rng(0)
    y = (rng.random((200, 6)) < 0.2).astype(int)
    y[:, 4] = 0  # no positives
    p = rng.random((200, 6))
    expected = np.mean([roc_auc_score(y[:, k], p[:, k]) for k in (0, 1, 2, 3, 5)])
    assert macro_roc_auc(y, p) == pytest.approx(expected)


def test_class_with_only_positives_is_skipped():
    y = np.array([[1, 1], [0, 1], [1, 1]])
    p = np.array([[0.9, 0.1], [0.1, 0.5], [0.8, 0.9]])
    assert scored_classes(y).tolist() == [True, False]
    assert macro_roc_auc(y, p) == pytest.approx(1.0)


def test_rejects_soft_labels_shape_mismatch_and_unscorable_input():
    with pytest.raises(ValueError, match="0/1"):
        macro_roc_auc(np.array([[0.3, 0.7], [0.7, 0.3]]), np.array([[0.1, 0.9], [0.9, 0.1]]))
    with pytest.raises(ValueError, match="shape"):
        macro_roc_auc(np.zeros((3, 2)), np.zeros((3, 3)))
    with pytest.raises(ValueError, match="undefined"):
        macro_roc_auc(np.zeros((3, 2)), np.full((3, 2), 0.5))
