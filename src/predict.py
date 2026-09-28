"""
predict.py
----------
Predicts the target for ONE new row, asking the user for feature values
dynamically based on the dataset's columns.

The model is a full sklearn Pipeline (preprocessing + model), so the raw
values typed by the user go through EXACTLY the same preprocessing that
was fitted on the training data.

Interactive input is VALIDATED before prediction:
  - categorical values must match a category seen in training
    (whitespace is trimmed, matching is case-insensitive, typos are
    REJECTED — there is deliberately NO fuzzy/auto-correct matching)
  - numeric values must parse as numbers
This protects beginners from typos like "Baangalore" silently producing
a misleading prediction. It is CLI-only: the underlying pipeline keeps
OneHotEncoder(handle_unknown="ignore"), so programmatic use with truly
new categories remains safe and crash-free.
"""

import pandas as pd

from data_preprocessing import get_known_categories
from dataset_profiler import detect_column_types

# How many category values to show per categorical column before
# collapsing the rest into "(+N more)" so the prompt stays readable.
MAX_SHOWN_CATEGORIES = 12


def show_categories(col: str, categories: list) -> None:
    """Print a categorical column's prompt with its known categories."""
    print(f"{col}")
    if categories:
        shown = ", ".join(sorted(categories)[:MAX_SHOWN_CATEGORIES])
        more = "" if len(categories) <= MAX_SHOWN_CATEGORIES else (
            f" (+{len(categories) - MAX_SHOWN_CATEGORIES} more)"
        )
        print(f"Available values: {shown}{more}")


def validate_categorical_input(raw: str, col: str, known: list):
    """Normalize and validate one categorical value against known ones.

    Rules (intentionally minimal — no fuzzy matching):
      1. trim surrounding whitespace ("  Bangalore " -> "Bangalore")
      2. case-insensitive exact match ("YES" -> "Yes", "bangalore" ->
         "Bangalore"), so capitalization never creates a new category
      3. anything else is rejected (returns None)

    Returns the canonical training-category spelling, or None if invalid.
    """
    lookup = {value.casefold(): value for value in known}
    return lookup.get(raw.casefold())


def parse_numeric_input(raw: str, col: str):
    """Convert one numeric input, printing a friendly error on failure.

    Accepts ints and floats, including 0 and negatives — the framework
    adds no business rules of its own. Returns the number, or None if
    the text is not a valid number (the caller asks again).
    """
    try:
        return float(raw) if "." in raw else int(raw)
    except ValueError:
        print(f"Invalid numeric value '{raw}' for '{col}'. "
              "Please enter a number (e.g. 1200 or 3.5).")
        return None


def ask_for_prediction_input(df: pd.DataFrame, target: str,
                             known_categories: dict = None):
    """Ask the user for a value of every feature column, one by one.

    - Numerical feature   -> keep asking until a valid number is entered.
    - Categorical feature -> validated against the categories the fitted
      OneHotEncoder knows; invalid values are asked again.

    known_categories: {column: [values]} from the fitted pipeline
    (get_known_categories). Columns missing from it fall back to the
    dataset's own unique values, so validation degrades gracefully.

    Returns the collected {column: value} dict, or None if the user
    cancelled (empty input) or stdin closed (EOF).
    """
    numeric_cols, categorical_cols = detect_column_types(df, exclude=(target,))
    known_categories = known_categories or {}

    print("\n" + "=" * 55)
    print("NEW PREDICTION")
    print("=" * 55)
    print("\nEnter values for prediction (empty line = cancel)\n")

    values = {}
    for col in numeric_cols + categorical_cols:
        kind = "number" if col in numeric_cols else "text"

        if col in categorical_cols:
            # Prefer categories from the FITTED encoder; fall back to the
            # cleaned dataset if the pipeline could not be introspected.
            known = known_categories.get(
                col, [str(v) for v in df[col].dropna().unique()]
            )
            show_categories(col, known)

        while True:
            try:
                raw = input(f"{col} ({kind}) > ").strip()
            except EOFError:
                return None  # stdin closed — treat like cancel
            if raw == "":
                return None  # user cancelled

            if col in numeric_cols:
                number = parse_numeric_input(raw, col)
                if number is not None:
                    values[col] = number
                    break
            else:
                normalized = validate_categorical_input(raw, col, known)
                if normalized is not None:
                    if normalized != raw:
                        print(f"  Interpreted '{raw}' as '{normalized}'.")
                    values[col] = normalized
                    break
                print(f"Invalid value '{raw}' for '{col}'.")
                if known:
                    print("Allowed values:")
                    for value in sorted(known):
                        print(f"  {value}")
                print("Please try again.")

    return values


def predict_new_row(pipeline, new_values: dict) -> float:
    """Run one raw feature row through the full pipeline and predict.

    The pipeline applies the SAME preprocessing fitted during training,
    so no manual encoding/reindexing is needed anymore.
    """
    new_df = pd.DataFrame([new_values])
    prediction = pipeline.predict(new_df)[0]
    return float(prediction)


def print_prediction(target: str, new_values: dict, prediction: float,
                     model_name: str) -> None:
    """Print the prediction in a simple, readable text format."""
    print("\n" + "-" * 40)
    print("Prediction Result")
    print("-" * 40)
    for col, val in new_values.items():
        print(f"{col:<14}: {val}")
    print()
    print(f"Predicted {target}: {prediction:,.4f}")
    print(f"Model Used: {model_name}")
    print("-" * 40)
