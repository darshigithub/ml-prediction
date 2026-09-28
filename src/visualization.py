"""
visualization.py
----------------
ADDITIVE module: training insights + charts. Kept completely separate from
the existing ML training code in train_models.py — nothing here changes how
models are trained, evaluated, selected, or used for predictions.

Every value shown or plotted comes from the models and metrics of the
CURRENT run. Nothing is invented.

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
def plot_actual_vs_predicted(models, X_test, y_test):
    """Scatter plots of actual vs predicted ratings for all four models."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    lo = float(min(y_test.min(), y_test.min())) - 0.1
    hi = float(max(y_test.max(), y_test.max())) + 0.1

    for ax, name in zip(axes.flat, MODEL_KEYS.values()):
        preds = models[name].predict(X_test)  # real predictions, test rows only
        ax.scatter(y_test, preds, alpha=0.6, s=25, color="#4C72B0")
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)  # perfect-prediction line
        ax.set_title(name)
        ax.set_xlabel("Actual rating")
        ax.set_ylabel("Predicted rating")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(alpha=0.3)

    fig.suptitle("Actual vs Predicted ratings (test set)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    return _save(fig, "actual_vs_predicted.png")


# ---------------------------------------------------------------------------
# 3. Linear Regression coefficients (print + bar chart)
# ---------------------------------------------------------------------------
def show_linear_coefficients(model, feature_names):
    """Print learned coefficients and save a bar chart, sorted by impact."""
    coefs = model.coef_
    order = np.argsort(np.abs(coefs))[::-1]  # strongest impact first
    rows = [(feature_names[i], float(coefs[i])) for i in order]

    print("\nLinear Regression — learned coefficients")
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
    ax.set_title("Linear Regression — feature coefficients\n(green pushes rating up, red pulls it down)")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, "linear_regression_coefficients.png")


# ---------------------------------------------------------------------------
# 4. Random Forest feature importance (print + bar chart, sorted desc)
# ---------------------------------------------------------------------------
def show_random_forest_importance(model, feature_names):
    """Print feature_importances_ (sorted) and save a bar chart."""
    importances = model.feature_importances_
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
    ax.set_title("Random Forest — feature importance")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, "random_forest_feature_importance.png")


# ---------------------------------------------------------------------------
# 5. Decision Tree visualization (display-only depth limit; model unchanged)
# ---------------------------------------------------------------------------
def plot_decision_tree(model, feature_names):
    """Plot the trained Decision Tree with sklearn's plot_tree.

    The tree itself is NOT changed — we only DRAW the top 3 levels so the
    image stays readable (a full tree would be a wall of boxes).
    """
    fig, ax = plt.subplots(figsize=(20, 11))
    plot_tree(
        model,
        feature_names=list(feature_names),
        filled=True,
        rounded=True,
        max_depth=3,          # display-only limit, training params untouched
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Decision Tree Regressor — top 3 levels (full tree is deeper)", fontsize=14)
    return _save(fig, "decision_tree.png")


# ---------------------------------------------------------------------------
# 6. XGBoost training curve (from evals_result_ recorded during training)
# ---------------------------------------------------------------------------
def plot_xgboost_training_curve(model):
    """Plot validation RMSE per boosting round.

    Uses the eval_set recorded in train_and_compare(); if unavailable
    (older runs / API change), we skip gracefully instead of failing.
    """
    try:
        results = model.evals_result()
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
def generate_training_insights(models, results_df, X_test, y_test):
    """Print the TRAINING INSIGHTS section and generate all charts.

    Parameters
    ----------
    models     : dict of fitted models from train_and_compare()
    results_df : the actual comparison table (MAE/RMSE/R²) from this run
    X_test     : encoded test features (column order = training order)
    y_test     : true ratings for the test rows
    """
    _ensure_output_dir()
    feature_names = list(X_test.columns)

    print("\n" + "=" * 55)
    print("TRAINING INSIGHTS")
    print("=" * 55)

    # 1. Model comparison chart (actual R² values)
    plot_model_comparison(results_df)

    # 2. Actual vs predicted scatter for every model
    plot_actual_vs_predicted(models, X_test, y_test)

    # 3. Linear Regression coefficients
    coef_chart = show_linear_coefficients(models[MODEL_KEYS["linear"]], feature_names)
    print(f"* Coefficient chart: {coef_chart}")

    # 4. Decision Tree visualization
    tree_chart = plot_decision_tree(models[MODEL_KEYS["tree"]], feature_names)
    print(f"\nDecision Tree\n* Tree visualization: {tree_chart}")

    # 5. Random Forest feature importance
    imp_chart = show_random_forest_importance(models[MODEL_KEYS["forest"]], feature_names)
    print(f"* Feature importance chart: {imp_chart}")

    # 6. XGBoost training curve
    xgb_chart = plot_xgboost_training_curve(models[MODEL_KEYS["xgb"]])
    print(f"\nXGBoost\n* Training curve: {xgb_chart or 'skipped'}")

    # Final summary block (paths only, so it is easy to find the charts)
    print("\nModel comparison chart:")
    print("outputs/model_comparison.png")
    print("\nActual vs predicted chart:")
    print("outputs/actual_vs_predicted.png")
    print("=" * 55)
