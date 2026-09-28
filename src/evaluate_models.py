"""
evaluate_models.py
------------------
Small helpers that turn trained pipelines into beginner-readable output:
the model comparison table and the final "report card" for the winner.

All metric computation itself lives in train_models.evaluate_model so the
numbers are produced in exactly one place.
"""


def print_comparison_table(results_df) -> None:
    """Print the classic comparison table in a fixed, readable layout.

    Column widths adapt to the data: a target like `price` has values in
    the millions, while a rating target fits in a few characters.

    MAE  -> lower is better
    RMSE -> lower is better
    R²   -> higher is better
    """
    mae_strings = [f"{v:,.4f}" for v in results_df["MAE"]]
    rmse_strings = [f"{v:,.4f}" for v in results_df["RMSE"]]
    r2_strings = [f"{v:.4f}" for v in results_df["R2"]]

    mae_w = max(len("MAE"), *(len(s) for s in mae_strings)) + 2
    rmse_w = max(len("RMSE"), *(len(s) for s in rmse_strings)) + 2
    r2_w = max(len("R²"), *(len(s) for s in r2_strings)) + 2

    print("\n=== Model Comparison ===")
    header = f"{'Model':<22}{'MAE':>{mae_w}}{'RMSE':>{rmse_w}}{'R²':>{r2_w}}"
    print(header)
    print("-" * len(header))
    for (_, row), mae_s, rmse_s, r2_s in zip(
        results_df.iterrows(), mae_strings, rmse_strings, r2_strings
    ):
        print(f"{row['Model']:<22}{mae_s:>{mae_w}}{rmse_s:>{rmse_w}}{r2_s:>{r2_w}}")
    print("-" * len(header))
    print("Best = highest R² (most variation explained)")


def print_best_model_summary(best_name: str, results_df, cv_results=None) -> None:
    """Print a short summary of the winning model's metrics.

    The winner row is looked up BY NAME (selection is driven by Mean CV R²,
    so the winner is not necessarily the first row of the test table).
    When CV results are available, the CV numbers are shown too so the
    difference between selection signal (CV) and final evaluation (test)
    stays visible for beginners.
    """
    row = results_df[results_df["Model"] == best_name].iloc[0]
    print("\n" + "=" * 55)
    print(f"BEST MODEL: {best_name}")
    print("=" * 55)
    print(f"MAE  : {row['MAE']:,.4f}   (average miss, lower is better)")
    print(f"RMSE : {row['RMSE']:,.4f}   (punishes big misses, lower is better)")
    print(f"R²   : {row['R2']:.4f}   (explained variation, higher is better)")
    if cv_results and best_name in cv_results:
        cv = cv_results[best_name]
        print(f"CV   : Mean R² {cv['mean_r2']:.4f} ± {cv['std_r2']:.4f}"
              "   (5-fold on training data, used for selection)")
    print("=" * 55)
