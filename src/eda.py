import pandas as pd

from config import TRAIN_CSV_PATH, TAXONOMY_CSV_PATH


def run_eda():
    """
    Load and validate the training metadata CSV files.

    Reads train.csv and taxonomy.csv, prints a brief summary, and checks
    that the number of unique species labels in the training set matches
    the taxonomy catalogue.

    Returns:
        tuple: (train_df, taxonomy_df) on success, or (None, None) if a file
               is not found (e.g., the competition data has not been downloaded yet).
    """
    print("=== Exploratory Data Analysis ===")

    try:
        train_df    = pd.read_csv(TRAIN_CSV_PATH)
        taxonomy_df = pd.read_csv(TAXONOMY_CSV_PATH)

        print(f"train.csv    loaded — shape: {train_df.shape}")
        print(f"taxonomy.csv loaded — shape: {taxonomy_df.shape}")

        print("\n[train.csv — first two rows]")
        print(train_df.head(2).to_string())

        print("\n[taxonomy.csv — first two rows]")
        print(taxonomy_df.head(2).to_string())

        num_train_classes = train_df["primary_label"].nunique()
        print(f"\nUnique species (primary_label) in train.csv: {num_train_classes}")

        if num_train_classes == len(taxonomy_df):
            print("Class count matches taxonomy.csv.")
        else:
            print(
                f"WARNING: {num_train_classes} classes in train.csv vs "
                f"{len(taxonomy_df)} rows in taxonomy.csv — counts differ."
            )

        return train_df, taxonomy_df

    except FileNotFoundError as exc:
        print(f"ERROR: File not found — {exc}")
        print("Make sure the competition data has been downloaded (run main.py).")
        return None, None


if __name__ == "__main__":
    run_eda()
