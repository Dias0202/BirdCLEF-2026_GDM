"""Grouped cross-validation folds for train_audio.

Each row of ``train.csv`` is one recording file, so a random split by row already
keeps every file on one side. That is not enough: recordists upload several
recordings from the same session, site and individual animals, and those
near-duplicates on both sides of a split inflate validation scores.

Folds are therefore grouped by recordist (``author``). Recordings whose author is
missing or ``Unknown`` cannot be linked to a session, so each of them forms its
own group (grouping them together would create one giant artificial group).
Folds are stratified by ``primary_label`` so rare species are spread across folds
as evenly as the grouping allows.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

UNKNOWN_AUTHORS = frozenset({"", "unknown"})


def recording_groups(df: pd.DataFrame, author_col: str = "author", file_col: str = "filename") -> pd.Series:
    """Group key per row: the recordist, or the file itself when the recordist is unknown."""
    missing = [c for c in (author_col, file_col) if c not in df.columns]
    if missing:
        raise KeyError(f"train metadata is missing columns: {missing}")
    if df[file_col].isna().any():
        raise ValueError(f"{file_col} has missing values")

    author = df[author_col].fillna("").astype(str).str.strip()
    unknown = author.str.lower().isin(UNKNOWN_AUTHORS)
    return ("author:" + author).where(~unknown, "file:" + df[file_col].astype(str))


def make_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    label_col: str = "primary_label",
) -> np.ndarray:
    """Fold id (0..n_splits-1) for every row, grouped by recordist and stratified by label."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    groups = recording_groups(df)
    if groups.nunique() < n_splits:
        raise ValueError(f"only {groups.nunique()} groups for {n_splits} folds")

    folds = np.full(len(df), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with warnings.catch_warnings():
        # Species with fewer recordings than folds trigger a warning; expected here.
        warnings.filterwarnings("ignore", message="The least populated class", category=UserWarning)
        for fold, (_, val_idx) in enumerate(splitter.split(df, df[label_col], groups)):
            folds[val_idx] = fold
    return folds


def train_val_split(
    df: pd.DataFrame,
    fold: int = 0,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training and validation frames for one fold."""
    if not 0 <= fold < n_splits:
        raise ValueError(f"fold must be in [0, {n_splits - 1}]")
    folds = make_folds(df, n_splits=n_splits, seed=seed)
    is_val = folds == fold
    return df.loc[~is_val].reset_index(drop=True), df.loc[is_val].reset_index(drop=True)
