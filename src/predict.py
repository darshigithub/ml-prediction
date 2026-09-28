"""
predict.py
----------
Uses the trained model to predict the rating of a brand-new restaurant.

The new restaurant goes through the SAME encoding as the training data,
which is why we keep the training feature columns and reindex the
one-hot encoded input to match them.
"""

import pandas as pd


def predict_new_restaurant(model, training_columns, new_restaurant: dict) -> float:
    """Predict the rating for one new restaurant.

    Parameters
    ----------
    model            : a trained scikit-learn / XGBoost model
    training_columns : the exact column list the model was trained on
    new_restaurant   : dict with keys:
                       cuisine, city, votes, average_cost,
                       delivery_time, online_order

    Returns
    -------
    The predicted rating as a float (clipped to the 1.0 - 5.0 scale).
    """
    # 1. Put the single restaurant into a one-row DataFrame.
    #    Column order does not matter here, we fix it in step 3.
    new_df = pd.DataFrame([new_restaurant])

    # 2. Apply the SAME one-hot encoding used during training.
    #    Categories that never appeared in training would simply get
    #    their own new column, so we only keep what the model knows.
    new_encoded = pd.get_dummies(new_df)

    # 3. Reindex to the training columns:
    #      - columns the model knows     -> kept (0 if absent here)
    #      - columns the model never saw -> dropped
    new_encoded = new_encoded.reindex(columns=training_columns, fill_value=0)

    # 4. Predict! (pass a DataFrame so sklearn knows the feature names)
    prediction = float(model.predict(new_encoded)[0])

    # Ratings live on a 1-5 scale, so clip any wild output.
    return max(1.0, min(5.0, prediction))


def print_prediction(new_restaurant: dict, predicted_rating: float, model_name: str):
    """Step 10: Print the prediction in a simple, readable text format."""
    rupees = f"₹{new_restaurant['average_cost']}"
    print("\n" + "-" * 40)
    print("Restaurant Rating Prediction")
    print("-" * 40)
    print(f"Cuisine       : {new_restaurant['cuisine']}")
    print(f"City          : {new_restaurant['city']}")
    print(f"Votes         : {new_restaurant['votes']}")
    print(f"Average Cost  : {rupees}")
    print(f"Delivery Time : {new_restaurant['delivery_time']} minutes")
    print(f"Online Order  : {new_restaurant['online_order']}")
    print()
    print(f"Predicted Rating: {predicted_rating:.2f}")
    print(f"Model Used: {model_name}")
    print("-" * 40)
