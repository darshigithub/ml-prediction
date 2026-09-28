"""
train_models.py
---------------
Trains four regression models and compares them:

  1. Linear Regression
  2. Decision Tree Regressor
  3. Random Forest Regressor
  4. XGBoost Regressor

Each model is evaluated with MAE, RMSE and R-squared, and the best
model (highest R-squared) is selected automatically.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

RANDOM_STATE = 42   # keeps the train/test split reproducible
TEST_SIZE = 0.2     # 20% of rows are kept aside for testing


def split_train_test(X, y):
    """Step 5: Split the data into a training set and a testing set.

    We train on 80% of the rows and evaluate on the 20% the model
    has never seen. random_state makes the split reproducible.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"[Split] Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def build_models():
    """Step 6: Define the four regression models.

    random_state / n_estimators notes:
      - random_state makes tree-based models reproducible.
      - n_jobs=-1 uses all CPU cores where supported.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,       # number of trees in the forest
            random_state=42,
            n_jobs=-1,              # use all CPU cores
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,       # number of trees
            learning_rate=0.1,      # step size shrinkage
            max_depth=5,            # depth of each tree
            eval_metric="rmse",     # ADDED for the training curve chart only:
                                    # lets XGBoost record RMSE during boosting.
                                    # Does NOT change how the model learns.
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate_model(name, model, X_test, y_test):
    """Step 7: Compute MAE, RMSE and R-squared for one trained model.

    - MAE  (Mean Absolute Error):  average miss in rating points. Lower = better.
    - RMSE (Root Mean Squared Error): like MAE but punishes big misses more. Lower = better.
    - R2   (R-squared):  how much of the rating variation the model explains.
      Higher = better, 1.0 is perfect.
    """
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    # RMSE = square root of the mean squared error.
    # (newer scikit-learn versions removed the `squared=` argument,
    #  so taking np.sqrt ourselves works on every version.)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


def train_and_compare(X_train, X_test, y_train, y_test):
    """Step 6-8: Train every model, evaluate it, and print a comparison table.

    Returns the fitted models as well, so the visualization module can
    draw insight charts from the SAME models that produced the metrics.
    Training/selection logic is unchanged.
    """
    results = []
    models = {}

    for name, model in build_models().items():
        print(f"[Train] Training {name} ...")
        if name == "XGBoost":
            # ADDED eval_set: asks XGBoost to record validation RMSE each
            # round so the training curve can be plotted. The learned model
            # is the same as before (same data, same hyperparameters).
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)
        models[name] = model
        results.append(evaluate_model(name, model, X_test, y_test))

    # Step 8: comparison table, sorted by R2 (best first)
    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)
    print("\n=== Model Comparison (test set) ===")
    print(results_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Step 9: select the best model (highest R-squared)
    best_name = results_df.iloc[0]["Model"]
    print(f"\n[Select] Best model based on R2: {best_name}")
    return best_name, results_df, models


def retrain_best_model(best_name, X, y):
    """Step 9 (optional but useful): retrain the winning model on ALL data.

    After comparing models we know which one to trust. Retraining on the
    full dataset gives the final model a little extra information before
    we use it for new predictions.
    """
    model = build_models()[best_name]
    model.fit(X, y)
    return model
