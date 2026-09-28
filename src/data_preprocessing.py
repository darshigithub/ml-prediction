"""
data_preprocessing.py
---------------------
Builds the GENERIC preprocessing pipeline used for every dataset:

    Numerical features   -> SimpleImputer(median) -> StandardScaler
    Categorical features -> SimpleImputer(most frequent) -> OneHotEncoder

Both branches are combined with a ColumnTransformer and wrapped in a
sklearn Pipeline together with the model.

WHY a Pipeline instead of encoding everything up-front?
  1. NO DATA LEAKAGE: the imputer, scaler and encoder are FITTED on the
     training split only, then reused to transform the test split. The
     test set therefore never influences the preprocessing.
  2. UNKNOWN CATEGORIES: OneHotEncoder(handle_unknown="ignore") means a
     category the model never saw during training (e.g. a new city) does
     not crash the app at prediction time — it just contributes nothing.
  3. SAME CODE PATH for training, testing and final prediction, so the
     new input row is always preprocessed exactly like the training data.

This module contains NO dataset-specific names — the columns are detected
from the data itself (see dataset_profiler.detect_column_types).

It also owns the small helpers that INTROSPECT a fitted pipeline:
get_transformed_feature_names (chart labels) and get_known_categories
(interactive prediction validation).
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """Create the ColumnTransformer for a given list of feature columns.

    - Numerical:  fill missing values with the MEDIAN (robust to outliers),
                  then scale so Linear Regression treats all features fairly.
    - Categorical: fill missing values with the most frequent value, then
                  One-Hot encode. handle_unknown="ignore" keeps new/unseen
                  categories from crashing predictions.
    """
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])


def build_full_pipeline(preprocessor: ColumnTransformer, model) -> Pipeline:
    """Attach a model to the preprocessor in one sklearn Pipeline.

    Fitting this pipeline fits the preprocessor AND the model in one call,
    which makes it impossible to accidentally leak test information.
    """
    return Pipeline(steps=[
        ("preprocessing", preprocessor),
        ("model", model),
    ])


def get_known_categories(pipeline: Pipeline) -> dict:
    """Return the categories the FITTED OneHotEncoder knows, per column.

    Result shape: {column_name: [category, category, ...]}

    WHY read this from the pipeline instead of the raw DataFrame: these
    are the EXACT values the encoder was fitted on, so interactive
    prediction can validate user input against them (see predict.py).
    The ColumnTransformer stores one categories_ array per categorical
    column, in the same order as the column list we passed in.
    """
    try:
        preprocessor = pipeline.named_steps["preprocessing"]
        for name, _, columns in preprocessor.transformers_:
            if name != "cat":
                continue
            encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]
            return {
                col: [str(value) for value in categories]
                for col, categories in zip(columns, encoder.categories_)
            }
    except Exception:
        # If the pipeline layout ever changes, validation degrades safely:
        # predict.py falls back to the dataset's own unique values.
        pass
    return {}


def get_transformed_feature_names(fitted_pipeline: Pipeline) -> list:
    """Return the feature names AFTER preprocessing (one-hot expands columns).

    Used by the visualization module so charts label the expanded features
    correctly (e.g. city_Bangalore) no matter which dataset is used.
    Falls back to generic names if sklearn cannot provide them.
    """
    try:
        names = fitted_pipeline.named_steps["preprocessing"].get_feature_names_out()
        cleaned = [str(n).split("__", 1)[-1] for n in names]  # strip "num__"/"cat__"
        return cleaned
    except Exception:
        # Extremely defensive fallback; should not happen with sklearn >= 1.0
        model = fitted_pipeline.named_steps["model"]
        return [f"feature_{i}" for i in range(getattr(model, "n_features_in_", 0))]


def clean_dataframe(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Light, GENERIC cleaning that is safe for any dataset.

    Kept deliberately small (no dataset-specific rules):
      - drop exact duplicate rows
      - strip spaces from text columns (prevents "Bangalore " != "Bangalore")
      - remove rows where the TARGET is missing (cannot train without it)

    Missing FEATURE values are NOT dropped here — the pipeline imputers
    handle them, which keeps more rows for training.
    """
    before = len(df)
    df = df.drop_duplicates()

    # Strip spaces from every non-numeric column (prevents
    # "Bangalore " != "Bangalore"). We check is_numeric_dtype rather than
    # dtype == object because pandas 3.x uses the new "str" dtype for text.
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()

    df = df.dropna(subset=[target])

    removed = before - len(df)
    print(f"[Clean] Removed {removed} duplicate/invalid rows -> {len(df)} rows left")
    return df
