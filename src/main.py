"""
main.py
-------
Entry point of the GENERIC tabular regression framework.

You provide:
    --data    path to any CSV dataset
    --target  the numeric column to predict

The framework does the rest automatically:
    profile -> clean -> split -> preprocess (no leakage)
    -> 5-fold CV on TRAINING data -> select best by Mean CV R²
    -> train all models -> final evaluation on untouched TEST data
    -> visualize -> retrain best on ALL data -> predict one new row

Examples:
    python src/main.py --data data/datasets/zomato_restaurants.csv --target rating
    python src/main.py --data data/datasets/house_prices.csv --target price
    python src/main.py --data data/datasets/house_prices.csv --target price --no-predict
"""

import argparse
import os
import sys

from cross_validation import print_cv_summary, select_best_model
from data_loader import DatasetError, load_csv, validate_target
from data_preprocessing import (
    build_preprocessor,
    clean_dataframe,
    get_known_categories,
    get_transformed_feature_names,
)
from dataset_profiler import (
    detect_column_types,
    detect_identifier_candidates,
    print_profile,
)
from evaluate_models import print_best_model_summary, print_comparison_table
from predict import ask_for_prediction_input, predict_new_row, print_prediction
from train_models import (
    cross_validate_all_models,
    retrain_best_model,
    split_train_test,
    train_and_compare,
)
from visualization import generate_training_insights


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments (--data and --target are required)."""
    parser = argparse.ArgumentParser(
        description="Generic tabular regression framework: train and compare "
                    "4 regression models on any CSV dataset.",
    )
    parser.add_argument("--data", required=True,
                        help="Path to the CSV dataset file.")
    parser.add_argument("--target", required=True,
                        help="Name of the numeric column to predict.")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of rows held back for testing (default: 0.20).")
    parser.add_argument("--no-predict", action="store_true",
                        help="Skip the interactive prediction for a new row.")
    args = parser.parse_args()

    if not 0.0 < args.test_size < 1.0:
        parser.error("--test-size must be between 0 and 1 (e.g. 0.2)")
    return args


def confirm_identifier_removal(candidates) -> list:
    """Ask the user about each column that LOOKS like an identifier.

    We never drop columns silently: a column that looks like an ID in one
    dataset can be a real feature in another. The user decides.
    Returns the list of columns to drop (possibly empty).
    """
    to_drop = []
    for col, n_unique, _ in candidates:
        while True:
            try:
                answer = input(
                    f"Drop possible identifier column '{col}' "
                    f"({n_unique} unique values)? (y/n) > "
                ).strip().lower()
            except EOFError:
                answer = "n"
            if answer in ("y", "yes"):
                to_drop.append(col)
                break
            if answer in ("n", "no", ""):
                break
            print("  Please answer y or n.")
    return to_drop


def main() -> int:
    # Windows consoles often default to a non-UTF-8 encoding (cp1252),
    # which cannot print symbols like ₹. Force UTF-8 output first.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    args = parse_args()

    print("=" * 55)
    print("GENERIC ML REGRESSION FRAMEWORK")
    print("=" * 55)

    # --- [Load] -------------------------------------------------------------
    df = load_csv(args.data)
    df = validate_target(df, args.target)

    # --- [Profile] ----------------------------------------------------------
    dataset_name = os.path.basename(args.data)
    print_profile(df, args.target, dataset_name)

    # --- [Clean] ------------------------------------------------------------
    df = clean_dataframe(df, args.target)

    # Identifier columns are only dropped with explicit user confirmation.
    candidates = detect_identifier_candidates(df, args.target)
    if candidates:
        print("\nPossible identifier columns detected.")
        to_drop = confirm_identifier_removal(candidates)
        if to_drop:
            df = df.drop(columns=to_drop)
            print(f"[Clean] Dropped identifier column(s): {', '.join(to_drop)}")

    # --- Features / target split ---------------------------------------------
    y = df[args.target]
    X = df.drop(columns=[args.target])

    # --- [Preprocess] info (actual fitting happens INSIDE each pipeline) ----
    numeric_features, categorical_features = detect_column_types(
        df, exclude=(args.target,)
    )
    print("\n[Preprocess]")
    print(f"Numerical features  : {len(numeric_features)} -> impute median + scale")
    print(f"Categorical features: {len(categorical_features)} -> impute mode + one-hot")

    # --- [Split] RAW data; preprocessing is fitted on TRAIN only (no leakage)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=args.test_size)

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # --- [CV] 5-fold cross-validation, TRAINING split only -------------------
    # The test set is NOT passed here — it stays untouched for final
    # evaluation, and preprocessing is fitted fresh inside every fold.
    cv_results = cross_validate_all_models(preprocessor, X_train, y_train)
    cv_best_name = select_best_model(cv_results)
    print_cv_summary(cv_results, cv_best_name, len(X_train), len(X_test))

    # --- [Train] all four models on the training split ------------------------
    # Selection inside uses the highest Mean CV R² (primary signal);
    # test-set metrics are computed for the FINAL evaluation only.
    best_name, results_df, pipelines = train_and_compare(
        preprocessor, X_train, X_test, y_train, y_test, cv_results=cv_results
    )

    n_encoded = len(get_transformed_feature_names(pipelines["Linear Regression"]))
    print(f"[Preprocess] Encoded features: {n_encoded} (after one-hot expansion)")

    # --- [Final Test Evaluation] on the untouched holdout ----------------------
    print("\n" + "=" * 55)
    print(" FINAL TEST SET EVALUATION")
    print("=" * 55)
    print("These rows were seen by NO model during CV or training.")
    print_comparison_table(results_df)
    print_best_model_summary(best_name, results_df, cv_results=cv_results)

    # --- [Visualizations] ------------------------------------------------------
    generate_training_insights(pipelines, results_df, X_test, y_test, args.target,
                               cv_results=cv_results)

    # --- [Retrain] the winner on ALL rows for the final model -----------------
    final_pipeline = retrain_best_model(best_name, preprocessor, X, y)

    # --- [Predict] one new row, asking for values dynamically ------------------
    if args.no_predict:
        print("\n[Done] Interactive prediction skipped (--no-predict).")
        return 0

    # Known categories come from the FITTED pipeline (OneHotEncoder
    # categories_), so validation matches what the model was trained on.
    new_values = ask_for_prediction_input(
        df, args.target, known_categories=get_known_categories(final_pipeline)
    )
    if new_values is None:
        print("\nPrediction cancelled.")
        return 0

    prediction = predict_new_row(final_pipeline, new_values)
    print_prediction(args.target, new_values, prediction, best_name)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except DatasetError as exc:
        # Expected, user-facing problems: short message, no stack trace.
        print(f"\nERROR: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(130)
