"""
train_models.py
---------------
Trains four regression models on ANY tabular dataset and compares them:

  1. Linear Regression
  2. Decision Tree Regressor
  3. Random Forest Regressor
  4. XGBoost Regressor

Each model is wrapped in a Pipeline together with the preprocessing, so
every model sees the data through the SAME fitted preprocessing.

Model SELECTION is driven by 5-fold cross-validation on the TRAINING
split (see cross_validation.py): the model with the highest MEAN CV R²
wins. The holdout test set is used for final evaluation only, and the
selected model is later retrained on all data for the final prediction.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from cross_validation import run_cross_validation, select_best_model
from data_preprocessing import build_full_pipeline

RANDOM_STATE = 42   # keeps every split and model reproducible
TEST_SIZE = 0.2     # 20% of rows are kept aside for testing


def split_train_test(X, y, test_size: float = TEST_SIZE):
    """Split the data into a training set and a testing set.

    We train on most rows and evaluate on the fraction the model has
    never seen. random_state makes the split reproducible.
    NOTE: the split happens BEFORE preprocessing is fitted, which prevents
    data leakage (see data_preprocessing.py).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    print(f"[Split] Train rows: {len(X_train)} | Test rows: {len(X_test)}")
    return X_train, X_test, y_train, y_test


def build_models() -> dict:
    """Define the four regression models (nothing dataset-specific here).

    Notes for beginners:
      - random_state makes tree-based models reproducible.
      - n_jobs=-1 uses all CPU cores where supported.
      - XGBoost gets eval_metric="rmse" so it RECORDS validation RMSE
        each boosting round for the training-curve chart. This does not
        change how the model learns.
    """
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            eval_metric="rmse",  # for the training-curve chart only
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def build_pipeline(name: str, preprocessor):
    """Wrap one model + the shared preprocessor into a single Pipeline."""
    return build_full_pipeline(preprocessor, build_models()[name])


def cross_validate_all_models(preprocessor, X_train, y_train) -> dict:
    """Run 5-fold CV for all four models on the TRAINING split only.

    Thin bridge to cross_validation.run_cross_validation. Each model gets
    a FRESH pipeline per fold, so preprocessing is fitted inside every
    fold (no leakage). The test set is never touched here.
    """
    return run_cross_validation(
        model_names=list(build_models().keys()),
        pipeline_builder=lambda name: build_pipeline(name, preprocessor),
        X_train=X_train,
        y_train=y_train,
    )


def evaluate_model(name, pipeline, X_test, y_test) -> dict:
    """Compute MAE, RMSE and R-squared for one trained pipeline.

    - MAE  (Mean Absolute Error): average miss in target units. Lower = better.
    - RMSE (Root Mean Squared Error): punishes big misses more. Lower = better.
    - R2   (R-squared): how much of the target variation the model explains.
      Higher = better, 1.0 is perfect.
    """
    predictions = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    # RMSE = square root of the mean squared error. We compute it with
    # np.sqrt because newer scikit-learn removed the `squared=` argument.
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    return {"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2}


def train_and_compare(preprocessor, X_train, X_test, y_train, y_test,
                      cv_results: dict = None):
    """Train every model, evaluate it on the test set, and select a winner.

    The SAME preprocessor instance is cloned into each model pipeline by
    sklearn (each fit starts fresh), so preprocessing is fitted on the
    training data only — never on the test data.

    cv_results : optional output of cross_validation.run_cross_validation.
                 When given, the winner is the model with the highest MEAN
                 CV R² (the primary, more reliable signal). Without it we
                 fall back to the single test-set R².

    Returns (best_name, results_df, pipelines); results_df is the TEST-set
    comparison table, used for the final evaluation section.
    """
    results = []
    pipelines = {}

    for name in build_models():
        print(f"[Train] Training {name} ...")
        pipeline = build_pipeline(name, preprocessor)
        if name == "XGBoost":
            # XGBoost's eval_set must ALREADY be preprocessed — a pipeline
            # fit would hand it the raw test rows. So we fit the
            # preprocessing on the TRAIN split only, transform the TEST
            # split with it (transforming never leaks information), and
            # fit the model directly. The pipeline stays consistent: its
            # internal preprocessor and model are the same fitted objects,
            # so pipeline.predict() behaves exactly like this fit.
            pre = pipeline.named_steps["preprocessing"]
            X_train_t = pre.fit_transform(X_train, y_train)
            X_test_t = pre.transform(X_test)
            pipeline.named_steps["model"].fit(
                X_train_t, y_train,
                eval_set=[(X_test_t, y_test)],
                verbose=False,
            )
        else:
            pipeline.fit(X_train, y_train)
        pipelines[name] = pipeline
        results.append(evaluate_model(name, pipeline, X_test, y_test))

    # Comparison table, sorted by R2 (best first). The formatted display
    # of this table lives in evaluate_models.py (called from main.py).
    results_df = pd.DataFrame(results).sort_values("R2", ascending=False)

    # Model selection: highest MEAN CV R² (primary signal) when CV results
    # are available; otherwise the single test-set R².
    if cv_results:
        best_name = select_best_model(cv_results)
        print(f"\n[Select] Best model based on Mean CV R2: {best_name}")
    else:
        best_name = results_df.iloc[0]["Model"]
        print(f"\n[Select] Best model based on test R2: {best_name}")
    return best_name, results_df, pipelines


def retrain_best_model(best_name: str, preprocessor, X, y):
    """Retrain the winning model on ALL data (train + test).

    After comparing models we know which one to trust. Retraining on the
    full dataset gives the final model more information before we use it
    for new predictions.
    """
    print(f"[Retrain] Retraining {best_name} on the full dataset ...")
    pipeline = build_pipeline(best_name, preprocessor)
    pipeline.fit(X, y)
    return pipeline
