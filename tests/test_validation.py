import numpy as np
import pandas as pd
import pytest

from validation import make_folds, recording_groups, train_val_split


def _metadata(n_authors: int = 30, per_author: int = 4, n_unknown: int = 10) -> pd.DataFrame:
    rows = []
    for a in range(n_authors):
        for r in range(per_author):
            rows.append({"author": f"Recordist {a}", "filename": f"sp{a % 5}/XC{a:03d}{r}.ogg", "primary_label": f"sp{a % 5}"})
    for u in range(n_unknown):
        rows.append({"author": "Unknown" if u % 2 else None, "filename": f"sp{u % 5}/iNat{u:03d}.ogg", "primary_label": f"sp{u % 5}"})
    return pd.DataFrame(rows)


def test_no_recordist_or_file_on_both_sides_of_any_fold():
    df = _metadata()
    folds = make_folds(df, n_splits=5, seed=0)
    assert set(folds) == set(range(5))
    for k in range(5):
        train, val = df[folds != k], df[folds == k]
        known = lambda d: set(d["author"].dropna()) - {"Unknown"}  # noqa: E731
        assert not known(train) & known(val)
        assert not set(train["filename"]) & set(val["filename"])


def test_unknown_authors_are_not_merged_into_one_group():
    df = _metadata()
    groups = recording_groups(df)
    unknown = df["author"].isna() | (df["author"] == "Unknown")
    assert groups[unknown].nunique() == unknown.sum()
    assert groups[~unknown].nunique() == 30


def test_train_val_split_partitions_rows_and_is_deterministic():
    df = _metadata()
    train, val = train_val_split(df, fold=1, n_splits=5, seed=42)
    assert len(train) + len(val) == len(df)
    assert not set(train["filename"]) & set(val["filename"])
    train2, val2 = train_val_split(df, fold=1, n_splits=5, seed=42)
    pd.testing.assert_frame_equal(val, val2)


def test_rejects_bad_arguments():
    df = _metadata()
    with pytest.raises(ValueError):
        train_val_split(df, fold=5, n_splits=5)
    with pytest.raises(KeyError):
        make_folds(df.drop(columns="author"))
    with pytest.raises(ValueError):
        make_folds(df.head(3), n_splits=5)
    assert np.all(make_folds(df) >= 0)
