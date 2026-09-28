"""
visualization.py
----------------
Training-insight charts for ANY dataset. Separated from the ML training
code — nothing here changes how models are trained, evaluated, selected,
or used for predictions.

Every value shown or plotted comes from the models and metrics of the
CURRENT run. Titles and labels use the ACTUAL target column name, so the
same code produces "Actual vs Predicted price" for a house dataset and
"Actual vs Predicted rating" for the Zomato dataset.

Generated files (in outputs/):
  model_comparison.png                  R² per model (bar chart)
  actual_vs_predicted.png               actual vs predicted scatter, 4 models
  linear_regression_coefficients.png    Linear Regression coefficients
  random_forest_feature_importance.png  Random Forest feature importances
  decision_tree.png                     Decision Tree structure (top levels)
  xgboost_training_curve.png            XGBoost RMSE per boosting round
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless backend: save files, no GUI window needed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import plot_tree

from data_preprocessing import get_transformed_feature_names

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

MODEL_KEYS = {
    "linear": "Linear Regression",
    "tree": "Decision Tree",
    "forest": "Random Forest",
    "xgb": "XGBoost",
}


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Saved outputs/{filename}")
    return f"outputs/{filename}"


def _model(pipeline):
    """Unwrap the estimator from its pipeline ('model' step)."""
    return pipeline.named_steps["model"]


# ---------------------------------------------------------------------------
# 0. Cross-validation chart (Mean CV R² with Std as error bars) — OPTIONAL
# ---------------------------------------------------------------------------
def plot_cv_model_comparison(cv_results):
    """Bar chart of Mean CV R² per model; error bar = Std CV R².

    This is the SELECTION view (5-fold CV on training data). The test-set
    chart below (model_comparison.png) stays untouched as the final
    evaluation view.
    """
    names = sorted(cv_results, key=lambda n: cv_results[n]["mean_r2"])
    means = [cv_results[n]["mean_r2"] for n in names]
    stds = [cv_results[n]["std_r2"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(names, means, xerr=stds, color="#4C72B0",
                   capsize=4, error_kw={"alpha": 0.6})
    ax.bar_label(bars, labels=[f"{m:.4f}" for m in means], padding=3)
    ax.set_xlabel("Mean CV R² (5-fold) — error bar = Std CV R²")
    ax.set_title("Model Comparison — 5-Fold Cross-Validation (training data)")
    # R² can be negative for very weak models; keep the axis honest.
    ax.set_xlim(min(0.0, min(means) - max(stds)), 1.05)
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, "cv_model_comparison.png")


# ---------------------------------------------------------------------------
# 1. Model comparison chart (uses the ACTUAL R² from the current run)
# ---------------------------------------------------------------------------
def plot_model_comparison(results_df):
    """Bar chart of R² per model. `results_df` is the real comparison table."""
    df = results_df.sort_values("R2", ascending=True)  # lowest -> highest for barh
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(df["Model"], df["R2"], color="#4C72B0")
    ax.bar_label(bars, labels=[f"{v:.4f}" for v in df["R2"]], padding=3)
    ax.set_xlabel("R² (higher is better)")
    ax.set_title("Model Comparison — R² on the test set")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, "model_comparison.png")


# ---------------------------------------------------------------------------
# 2. Actual vs Predicted scatter (one subplot per model, real predictions)
# ---------------------------------------------------------------------------
def plot_actual_vs_predicted(pipelines, X_test, y_test, target):
    """Scatter plots of actual vs predicted target for all four models."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    lo = float(y_test.min()) - 0.1
    hi = float(y_test.max()) + 0.1

    for ax, name in zip(axes.flat, MODEL_KEYS.values()):
        preds = pipelines[name].predict(X_test)  # real predictions, test rows
        ax.scatter(y_test, preds, alpha=0.6, s=25, color="#4C72B0")
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)  # perfect-prediction line
        ax.set_title(name)
        ax.set_xlabel(f"Actual {target}")
        ax.set_ylabel(f"Predicted {target}")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Actual vs Predicted {target} (test set)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, "actual_vs_predicted.png")


# ---------------------------------------------------------------------------
# 3. Linear Regression coefficients (print + bar chart)
# ---------------------------------------------------------------------------
def show_linear_coefficients(pipeline, feature_names, target):
    """Print learned coefficients and save a bar chart, sorted by impact.

    Feature names come from the FITTED preprocessing pipeline, so
    one-hot columns are labelled correctly (e.g. city_Bangalore).
    """
    model = _model(pipeline)
    coefs = model.coef_
    order = np.argsort(np.abs(coefs))[::-1]  # strongest impact first
    rows = [(feature_names[i], float(coefs[i])) for i in order]

    print(f"\nLinear Regression — learned coefficients ({len(rows)} features)")
    print("Feature                  Coefficient")
    for fname, val in rows:
        print(f"{fname:<25}{val:+.5f}")

    labels = [r[0] for r in rows][::-1]
    values = [r[1] for r in rows][::-1]
    colors = ["#C44E52" if v < 0 else "#55A868" for v in values]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient value")
    ax.set_title(
        f"Feature Coefficients\n(green pushes {target} up, red pulls it down)"
    )
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, "linear_regression_coefficients.png")


# ---------------------------------------------------------------------------
# 4. Random Forest feature importance (print + bar chart, sorted desc)
# ---------------------------------------------------------------------------
def show_random_forest_importance(pipeline, feature_names):
    """Print feature_importances_ (sorted) and save a bar chart.

    IMPORTANT: importance only tells us which features the tree USED most.
    It does NOT say whether a feature pushes the target up or down — for
    direction, look at the Linear Regression coefficients instead.
    """
    importances = _model(pipeline).feature_importances_
    order = np.argsort(importances)[::-1]  # highest importance first
    rows = [(feature_names[i], float(importances[i])) for i in order]

    print("\nRandom Forest — feature importance (sorted)")
    print("Feature                  Importance")
    for fname, val in rows:
        print(f"{fname:<25}{val:.5f}")

    labels = [r[0] for r in rows][::-1]
    values = [r[1] for r in rows][::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(labels, values, color="#4C72B0")
    ax.set_xlabel("Importance (fraction of total)")
    ax.set_title("Feature Importance (Random Forest)")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, "random_forest_feature_importance.png")


# ---------------------------------------------------------------------------
# 5. Decision Tree visualization (display-only depth limit; model unchanged)
# ---------------------------------------------------------------------------
def plot_decision_tree(pipeline, feature_names):
    """Plot the trained Decision Tree with sklearn's plot_tree.

    The tree itself is NOT changed — we only DRAW the top 3 levels so the
    image stays readable (a full tree would be a wall of boxes).
    """
    fig, ax = plt.subplots(figsize=(20, 11))
    plot_tree(
        _model(pipeline),
        feature_names=list(feature_names),
        filled=True,
        rounded=True,
        max_depth=3,          # display-only limit, training params untouched
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Decision Tree Regressor — top 3 levels (full tree is deeper)",
                 fontsize=14)
    return _save(fig, "decision_tree.png")


# ---------------------------------------------------------------------------
# 6. XGBoost training curve (from evals_result_ recorded during training)
# ---------------------------------------------------------------------------
def plot_xgboost_training_curve(pipeline):
    """Plot validation RMSE per boosting round.

    Uses the eval_set recorded in train_and_compare(); if unavailable
    (older runs / API change), we skip gracefully instead of failing.
    """
    try:
        results = _model(pipeline).evals_result()
        rmse = results["validation_0"]["rmse"]
    except (AttributeError, KeyError):
        print("[Viz] XGBoost eval history not available — skipping training curve")
        return None

    rounds = np.arange(1, len(rmse) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rounds, rmse, color="#4C72B0")
    ax.set_xlabel("Boosting round")
    ax.set_ylabel("RMSE (test set, lower is better)")
    ax.set_title("XGBoost — RMSE per boosting round")
    ax.grid(alpha=0.3)
    return _save(fig, "xgboost_training_curve.png")


# ---------------------------------------------------------------------------
# Entry point called by main.py AFTER model training
# ---------------------------------------------------------------------------
def generate_training_insights(pipelines, results_df, X_test, y_test, target,
                               cv_results: dict = None):
    """Generate all charts for the current run.

    Parameters
    ----------
    pipelines  : dict of fitted sklearn Pipelines from train_and_compare()
    results_df : the actual comparison table (MAE/RMSE/R²) from this run
    X_test     : RAW test features (the pipelines preprocess internally)
    y_test     : true target values for the test rows
    target     : target column name, used for chart titles
    cv_results : optional CV summary; adds cv_model_comparison.png when given
    """
    _ensure_output_dir()
    feature_names = get_transformed_feature_names(pipelines["Linear Regression"])

    print("\n" + "=" * 55)
    print("TRAINING INSIGHTS")
    print("=" * 55)

    # 0. CV comparison chart (the selection view) when CV was run
    if cv_results:
        plot_cv_model_comparison(cv_results)

    # 1. Model comparison chart (actual R² values)
    plot_model_comparison(results_df)

    # 2. Actual vs predicted scatter for every model
    plot_actual_vs_predicted(pipelines, X_test, y_test, target)

    # 3. Linear Regression coefficients
    coef_chart = show_linear_coefficients(
        pipelines[MODEL_KEYS["linear"]], feature_names, target
    )
    print(f"* Coefficient chart: {coef_chart}")

    # 4. Decision Tree visualization
    tree_chart = plot_decision_tree(pipelines[MODEL_KEYS["tree"]], feature_names)
    print(f"\nDecision Tree\n* Tree visualization: {tree_chart}")

    # 5. Random Forest feature importance
    imp_chart = show_random_forest_importance(
        pipelines[MODEL_KEYS["forest"]], feature_names
    )
    print(f"* Feature importance chart: {imp_chart}")

    # 6. XGBoost training curve
    xgb_chart = plot_xgboost_training_curve(pipelines[MODEL_KEYS["xgb"]])
    print(f"\nXGBoost\n* Training curve: {xgb_chart or 'skipped'}")

    # Final summary block (paths only, so it is easy to find the charts)
    print("\nModel comparison chart:")
    print("outputs/model_comparison.png")
    print("\nActual vs predicted chart:")
    print("outputs/actual_vs_predicted.png")
    print("=" * 55)
