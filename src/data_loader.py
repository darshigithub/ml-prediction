"""
data_loader.py
--------------
Loads ANY CSV dataset and validates the user's target column.

The framework is generic: it never assumes column names like "rating" or
"price". The caller tells us which file to load and which column is the
target; this module makes sure both exist and that the target is numeric
(this project does regression only).

Expected user mistakes raise a DatasetError, which main.py turns into a
short, friendly ERROR message instead of a long Python traceback.
"""

import os

import pandas as pd


class DatasetError(Exception):
    """Raised for expected, user-facing dataset problems."""


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame, with friendly errors."""
    if not os.path.isfile(path):
        raise DatasetError(f"Dataset file not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # unreadable / corrupt CSV
        raise DatasetError(f"Could not read CSV file '{path}': {exc}") from exc

    if df.empty:
        raise DatasetError(f"Dataset file '{path}' contains no rows.")

    print(f"[Load] Loaded {len(df)} rows and {len(df.columns)} columns from {path}")
    return df


def validate_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Make sure the target column exists and is usable for regression.

    Regression needs NUMBERS to predict. If the target column contains
    text that is not numeric, we stop with a clear error instead of
    guessing (classification may come in a future phase).
    """
    if target not in df.columns:
        available = "\n".join(f"  - {col}" for col in df.columns)
        raise DatasetError(
            f"Target column '{target}' was not found.\nAvailable columns:\n{available}"
        )

    if pd.api.types.is_numeric_dtype(df[target]):
        return df

    # Numbers stored as text (e.g. "4.3") can be rescued; real text cannot.
    converted = pd.to_numeric(df[target], errors="coerce")
    new_missing = int(converted.isna().sum() - df[target].isna().sum())
    if new_missing > 0:
        raise DatasetError(
            f"Target column '{target}' must contain numeric values for regression.\n"
            f"{new_missing} value(s) are not numeric.\n"
            "Classification support will be added in a future phase."
        )

    df[target] = converted
    print(f"[Load] Converted target column '{target}' from text to numbers")
    return df
