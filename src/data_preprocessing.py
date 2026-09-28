"""
data_preprocessing.py
---------------------
Handles everything related to the raw dataset:
  1. Load the CSV with pandas
  2. Basic data cleaning
  3. Split into features (X) and target (y)
  4. Encode categorical columns with One-Hot Encoding

This module is intentionally small and heavily commented so beginners
can follow every ML step.
"""

import os

import pandas as pd

# Column that we want to predict (the "target")
TARGET_COLUMN = "rating"

# Columns that are pure identifiers and carry no predictive signal,
# so we drop them before training.
DROP_COLUMNS = ["restaurant_name"]

# Columns we expect the CSV to contain
REQUIRED_COLUMNS = ["cuisine", "city", "votes", "average_cost",
                    "delivery_time", "online_order"]

# Path helpers so the script works no matter where you run it from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "zomato_restaurants.csv")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Step 1: Load the restaurant CSV into a pandas DataFrame."""
    df = pd.read_csv(path)
    print(f"[Load] Loaded {len(df)} rows and {len(df.columns)} columns from {path}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Basic data cleaning.

    Real datasets almost always have problems. Here we handle the most
    common ones with simple, readable steps:
      - remove exact duplicate rows
      - convert numeric columns to numbers (bad values -> NaN)
      - drop rows with missing values in important columns
      - clip ratings to the realistic 1.0 - 5.0 range
    """
    before = len(df)

    # 2a. Drop exact duplicate rows (same restaurant entry twice)
    df = df.drop_duplicates()

    # 2b. Make sure numeric columns really are numeric.
    #     errors="coerce" turns bad strings (e.g. "NEW", "-") into NaN.
    for col in ["rating", "votes", "average_cost", "delivery_time"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 2c. Standardise the online_order column to "Yes"/"No"
    df["online_order"] = df["online_order"].astype(str).str.strip().str.title()
    df["online_order"] = df["online_order"].map({"Yes": "Yes", "No": "No"})
    df = df.dropna(subset=["online_order"])

    # 2d. Remove rows with any missing value in the columns we need
    needed = REQUIRED_COLUMNS + [TARGET_COLUMN]
    df = df.dropna(subset=needed)

    # 2e. Sanity ranges: ratings live on a 1-5 scale, counts are positive
    df = df[(df[TARGET_COLUMN] >= 1) & (df[TARGET_COLUMN] <= 5)]
    df = df[df["votes"] >= 0]
    df = df[df["average_cost"] > 0]
    df = df[df["delivery_time"] > 0]

    # 2f. Strip extra spaces from text columns
    for col in ["cuisine", "city", "online_order"]:
        df[col] = df[col].astype(str).str.strip()

    print(f"[Clean] Removed {before - len(df)} bad/duplicate rows -> {len(df)} rows left")
    return df


def split_features_target(df: pd.DataFrame):
    """Step 3: Separate the features (X) from the target (y)."""
    X = df.drop(columns=[TARGET_COLUMN] + DROP_COLUMNS, errors="ignore")
    y = df[TARGET_COLUMN]
    print(f"[Split] Features: {list(X.columns)} | Target: {TARGET_COLUMN}")
    return X, y


def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """Step 4: Encode categorical columns using One-Hot Encoding.

    ML models only understand numbers, so text columns such as
    `cuisine` and `city` must be converted. One-Hot Encoding creates
    a new 0/1 column for each category (e.g. cuisine_Biryani).

    dtype=float keeps everything numeric and avoids bool columns,
    which keeps Linear Regression happy.
    """
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    X_encoded = pd.get_dummies(X, columns=categorical, drop_first=True, dtype=float)
    print(f"[Encode] One-hot encoded columns: {categorical} "
          f"({X.shape[1]} -> {X_encoded.shape[1]} features)")
    return X_encoded
