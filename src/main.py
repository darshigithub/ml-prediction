"""
main.py
-------
Runs the complete beginner-friendly ML workflow from the command line:

    1. Load the dummy Zomato dataset
    2. Clean it
    3. Split into features (X) and target (y)
    4. Encode categorical columns
    5. Train / test split
    6. Train 4 regression models
    7. Evaluate with MAE, RMSE, R-squared
    8. Print a comparison table
    9. Select the best model
   10. Predict the rating for a new restaurant

Run everything with:
    python src/main.py
"""

import sys

from data_preprocessing import clean_data, encode_features, load_data, split_features_target
from predict import predict_new_restaurant, print_prediction
from train_models import retrain_best_model, split_train_test, train_and_compare
from visualization import generate_training_insights


def main():
    # Windows consoles often default to a non-UTF-8 encoding (cp1252),
    # which cannot print the ₹ symbol. Force UTF-8 output first.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 55)
    print(" Zomato Restaurant Rating Prediction (ML Project)")
    print("=" * 55)

    # --- Steps 1-2: load and clean -----------------------------------------
    df = load_data()
    df = clean_data(df)

    # --- Steps 3-4: features/target split + encoding -----------------------
    X, y = split_features_target(df)
    X_encoded = encode_features(X)

    # --- Step 5: train/test split -------------------------------------------
    X_train, X_test, y_train, y_test = split_train_test(X_encoded, y)

    # --- Steps 6-9: train, evaluate, compare, select ------------------------
    # NOTE: train_and_compare now also returns the fitted models so the
    # visualization module can draw charts from the same models.
    best_name, results_df, models = train_and_compare(X_train, X_test, y_train, y_test)

    # --- Training insights (ADDITIVE): charts under outputs/ ----------------
    generate_training_insights(models, results_df, X_test, y_test)

    # --- Step 9b: retrain the winner on all data ----------------------------
    final_model = retrain_best_model(best_name, X_encoded, y)

    # --- Step 10: predict for a brand-new restaurant ------------------------
    # Change these values to try different restaurants!
    new_restaurant = {
        "cuisine": "Biryani",
        "city": "Bangalore",
        "votes": 5000,
        "average_cost": 500,
        "delivery_time": 35,
        "online_order": "Yes",
    }

    predicted_rating = predict_new_restaurant(
        final_model, list(X_encoded.columns), new_restaurant
    )
    print_prediction(new_restaurant, predicted_rating, best_name)


if __name__ == "__main__":
    main()
