"""
dataset_profiler.py
-------------------
Gives the user a quick, beginner-friendly summary of ANY dataset before
training: size, column types, missing values, duplicates and target
statistics. It also flags columns that LOOK like identifiers.

Identifier columns are only REPORTED (and later confirmed with the user
in main.py) — they are never removed silently.
"""

import pandas as pd

# Identifier heuristics (kept conservative and documented in the README):
#   - a text column is suspicious when more than 35% of its values are
#     unique AND it has at least 20 distinct values (names, IDs, codes)
#   - a numeric column is suspicious when EVERY value is unique and there
#     are more than 10 rows (a counter, e.g. customer_id)
TEXT_UNIQUE_RATIO = 0.35
TEXT_MIN_UNIQUE = 20
NUMERIC_MIN_ROWS = 10


def detect_column_types(df: pd.DataFrame, exclude: tuple = ()):
    """Detect numerical and categorical feature columns from the data types.

    Anything pandas sees as a number (int/float) is numerical; everything
    else (text, categories) is categorical. The target column is excluded
    because it is what we predict, not a feature.
    """
    numeric = [
        col for col in df.columns
        if col not in exclude and pd.api.types.is_numeric_dtype(df[col])
    ]
    categorical = [col for col in df.columns if col not in exclude and col not in numeric]
    return numeric, categorical


def detect_identifier_candidates(df: pd.DataFrame, target: str):
    """Return a list of (column, n_unique, reason) tuples that LOOK like IDs.

    We only report these — removing them is the user's decision, because a
    column that looks like an ID in one dataset can be a real feature in
    another (e.g. a product code that encodes a category).
    """
    candidates = []
    for col in df.columns:
        if col == target:
            continue
        series = df[col]
        n_unique = int(series.nunique(dropna=True))
        if n_unique == 0:
            continue
        if pd.api.types.is_numeric_dtype(series):
            if n_unique == int(series.notna().sum()) and len(series) > NUMERIC_MIN_ROWS:
                candidates.append((col, n_unique, "every value is unique"))
        else:
            ratio = n_unique / max(int(series.notna().sum()), 1)
            if ratio > TEXT_UNIQUE_RATIO and n_unique >= TEXT_MIN_UNIQUE:
                candidates.append((col, n_unique, "mostly unique text values"))
    return candidates


def profile_dataset(df: pd.DataFrame, target: str) -> dict:
    """Collect all profile numbers in one dict (used for printing + logic)."""
    numeric, categorical = detect_column_types(df, exclude=(target,))
    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric": numeric,
        "categorical": categorical,
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_stats": df[target].describe(),
        "identifier_candidates": detect_identifier_candidates(df, target),
    }


def print_profile(df: pd.DataFrame, target: str, dataset_name: str) -> dict:
    """Print the DATASET PROFILE section and return the profile dict."""
    profile = profile_dataset(df, target)
    stats = profile["target_stats"]

    print("\n" + "=" * 55)
    print("DATASET PROFILE")
    print("=" * 55)
    print()
    print(f"Dataset           : {dataset_name}")
    print(f"Rows              : {profile['rows']}")
    print(f"Columns           : {profile['columns']}")
    print(f"Target            : {target}")
    print(f"Problem Type      : Regression")
    print()
    print(f"Numerical Columns : {len(profile['numeric'])}")
    print(f"Categorical       : {len(profile['categorical'])}")
    print()
    print(f"Missing Values    : {profile['missing_values']}")
    print(f"Duplicate Rows    : {profile['duplicate_rows']}")

    print("\nTarget Statistics")
    print("-----------------")
    print(f"Min               : {stats['min']:,.2f}")
    print(f"Max               : {stats['max']:,.2f}")
    print(f"Mean              : {stats['mean']:,.2f}")
    print(f"Median            : {stats['50%']:,.2f}")

    print("\nColumn Overview")
    print("---------------")
    print(f"{'column':<20}{'dtype':<12}{'missing':>8}{'unique':>8}")
    for col in df.columns:
        print(
            f"{col:<20}{str(df[col].dtype):<12}"
            f"{int(df[col].isna().sum()):>8}{int(df[col].nunique()):>8}"
        )

    if profile["identifier_candidates"]:
        print("\nPossible identifier columns (review before training):")
        for col, n_unique, _ in profile["identifier_candidates"]:
            print(f"  - {col} ({n_unique} unique values)")

    print("=" * 55)
    return profile
