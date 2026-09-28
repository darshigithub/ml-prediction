"""
cross_validation.py
-------------------
5-fold cross-validation for model comparison and selection.

WHY cross-validation? A single train/test split can be lucky or unlucky:
performance varies with which rows happen to land in the test set.
Running 5-fold CV on the TRAINING data gives a more reliable picture
(mean) and shows how stable each model is (standard deviation).

THE GOLDEN RULE — no test-set leakage:
  Cross-validation uses ONLY the training split. The holdout test set
  stays completely untouched until the FINAL evaluation after selection.

NO PREPROCESSING LEAKAGE inside CV either:
  Each fold gets a FRESH clone of the full preprocessing+model pipeline
  (sklearn does this automatically when fitting/predicting per fold), so
  the imputer/scaler/encoder are fitted on that fold's training portion
  only, then transform the fold's validation portion.

XGBoost note: for CV we use a plain pipeline.fit (no eval_set) — the
training-curve chart comes from the separate post-selection training run
in train_models.py, which handles the eval_set special case.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, cross_validate

CV_FOLDS = 5  # number of cross-validation folds


def run_cross_validation(model_names: list, pipeline_builder,
                         X_train, y_train, n_folds: int = CV_FOLDS) -> dict:
    """Run k-fold cross-validation for every model on the TRAINING data.

    pipeline_builder : function(model_name) -> fresh sklearn Pipeline
                       (preprocessing + model). A NEW pipeline per fold is
                       essential: it guarantees preprocessing is fitted
                       inside each fold, never on the full training set.
    X_train, y_train : the TRAINING split only — never the test data.

    Returns structured results (no printing), one entry per model:
        {
          "Linear Regression": {
              "mean_r2": ..., "std_r2": ...,
              "mean_mae": ..., "std_mae": ...,
              "mean_rmse": ..., "std_rmse": ...,
              "fold_r2": [...],       # kept for optional debugging
          }, ...
        }
    """
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    scoring = {
        "r2": make_scorer(r2_score),
        "neg_mae": make_scorer(mean_absolute_error, greater_is_better=False),
        "neg_mse": make_scorer(mean_squared_error, greater_is_better=False),
    }

    cv_results = {}
    for name in model_names:
        print(f"[CV] Cross-validating {name} ({n_folds} folds) ...")
        pipeline = pipeline_builder(name)  # fresh pipeline for this model
        scores = cross_validate(
            pipeline, X_train, y_train,
            cv=kfold, scoring=scoring, n_jobs=1,
        )

        # Flip the negative losses back to positive MAE / RMSE.
        mae_folds = -scores["test_neg_mae"]
        rmse_folds = np.sqrt(-scores["test_neg_mse"])
        r2_folds = scores["test_r2"]

        cv_results[name] = {
            "mean_r2": float(np.mean(r2_folds)),
            "std_r2": float(np.std(r2_folds)),
            "mean_mae": float(np.mean(mae_folds)),
            "std_mae": float(np.std(mae_folds)),
            "mean_rmse": float(np.mean(rmse_folds)),
            "std_rmse": float(np.std(rmse_folds)),
            "fold_r2": [float(v) for v in r2_folds],
        }
    return cv_results


def select_best_model(cv_results: dict) -> str:
    """Pick the model with the highest MEAN CV R-squared.

    This is the PRIMARY selection signal now: it averages over 5 folds,
    so it is much harder to fool by an unlucky split than one test score.
    """
    return max(cv_results, key=lambda name: cv_results[name]["mean_r2"])


def cv_results_to_frame(cv_results: dict) -> pd.DataFrame:
    """Turn the structured CV results into a DataFrame (mean columns first)."""
    rows = []
    for name, r in cv_results.items():
        rows.append({
            "Model": name,
            "Mean R2": r["mean_r2"], "Std R2": r["std_r2"],
            "Mean MAE": r["mean_mae"], "Std MAE": r["std_mae"],
            "Mean RMSE": r["mean_rmse"], "Std RMSE": r["std_rmse"],
        })
    return pd.DataFrame(rows).sort_values("Mean R2", ascending=False)


def print_cv_summary(cv_results: dict, best_name: str, n_train: int,
                     n_test: int, n_folds: int = CV_FOLDS) -> None:
    """Print the beginner-friendly CV section shown in the CLI.

    Mean R² is the selection metric; MAE/RMSE means give the full
    picture. Column widths adapt to the data (a price target has huge
    MAE/RMSE, a rating target fits in a few characters).
    """
    print("\n" + "=" * 55)
    print(" 5-FOLD CROSS-VALIDATION")
    print("=" * 55)
    print(f"\nTraining data: {n_train} rows (used for CV)")
    print(f"Test data: {n_test} rows (untouched until final evaluation)")
    print(f"CV folds: {n_folds}")

    rows = [(name, cv_results[name]) for name in
            sorted(cv_results, key=lambda n: cv_results[n]["mean_r2"],
                   reverse=True)]
    r2_cells = [f"{r['mean_r2']:.4f}" for _, r in rows]
    std_cells = [f"{r['std_r2']:.4f}" for _, r in rows]
    mae_cells = [f"{r['mean_mae']:,.4f}" for _, r in rows]
    rmse_cells = [f"{r['mean_rmse']:,.4f}" for _, r in rows]

    r2_w = max(len("Mean R²"), *(len(s) for s in r2_cells)) + 1
    std_w = max(len("Std R²"), *(len(s) for s in std_cells)) + 1
    mae_w = max(len("Mean MAE"), *(len(s) for s in mae_cells)) + 1
    rmse_w = max(len("Mean RMSE"), *(len(s) for s in rmse_cells)) + 1
    table_w = 23 + r2_w + std_w + mae_w + rmse_w

    print("\n" + "-" * table_w)
    print(f" {'Model':<22}{'Mean R²':>{r2_w}}{'Std R²':>{std_w}}"
          f"{'Mean MAE':>{mae_w}}{'Mean RMSE':>{rmse_w}}")
    print("-" * table_w)
    for (name, _), r2_s, std_s, mae_s, rmse_s in zip(
            rows, r2_cells, std_cells, mae_cells, rmse_cells):
        print(f" {name:<22}{r2_s:>{r2_w}}{std_s:>{std_w}}"
              f"{mae_s:>{mae_w}}{rmse_s:>{rmse_w}}")
    print("-" * table_w)
    print(f"\nSelected Model: {best_name}")
    print("Selection Metric: Mean CV R²")
    print("=" * 55)
