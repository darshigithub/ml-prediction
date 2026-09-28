# Project Context — ml-zomato-prediction

> Reference doc for maintaining continuity across sessions. Read this first
> before making changes, and update it whenever the project evolves.

## 1. What this project is

A **beginner-friendly Machine Learning regression project** that predicts a
restaurant's rating from a dummy Zomato-like dataset. Everything runs from
the **command line only**.

**Explicitly OUT of scope (do not add):**
Power BI, web apps, FastAPI, LLMs, chatbots, GenUI, databases, Docker.

## 2. Current structure (verified working)

```
D:\AIML\ml-prediction\
├── data\zomato_restaurants.csv    # 450 dummy records, 8 columns, rating = target
├── scripts\generate_data.py       # optional: regenerates the CSV (seed=42)
├── src\
│   ├── data_preprocessing.py      # Steps 1-4: load, clean, X/y split, one-hot encode
│   ├── train_models.py            # Steps 5-9: split, 4 models, MAE/RMSE/R², select best
│   ├── visualization.py           # ADDITIVE: insight charts under outputs/
│   ├── predict.py                 # Step 10: predict new restaurant + formatted output
│   └── main.py                    # orchestrates the full workflow
├── outputs\                       # generated charts (6 PNGs), safe to delete
├── requirements.txt               # pandas, numpy, scikit-learn, xgboost, matplotlib
├── README.md
└── PROJECT_CONTEXT.md             # this file
```

## 3. Environment facts

- OS: Windows, shell is **Git Bash** (use POSIX syntax: `mv`, `rm`, forward slashes).
- Python **3.13.7** — system install at `C:\Python313` (has NO packages).
- Project venv: **`.venv\`** at project root — all deps installed here
  (pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, xgboost 3.4.1,
  matplotlib 3.11.2).
- **Run everything with the venv python:** `.venv/Scripts/python ...`
  (plain `python` hits the bare system install and fails on `import pandas`).
- `requirements.txt` is the source of truth for deps.

## 4. How to run

```bash
.venv/Scripts/python src/main.py              # full workflow
.venv/Scripts/python scripts/generate_data.py # regenerate dataset
```

End-user command (documented in README): `python src/main.py`
after `pip install -r requirements.txt` in an activated venv.

## 5. Design decisions (keep these consistent)

- **Target:** `rating`. **Dropped from features:** `restaurant_name` (identifier).
- **Encoding:** `pd.get_dummies(..., drop_first=True, dtype=float)` — NOT
  ColumnTransformer/sklearn pipelines. `dtype=float` avoids bool columns
  that upset Linear Regression.
- **Prediction alignment:** new restaurants are one-hot encoded then
  `reindex(columns=training_columns, fill_value=0)` so unknown categories
  become 0. Any new feature engineering must keep this contract.
- **Best model selection:** highest R² on the test split; the winner is then
  **retrained on ALL rows** before final prediction.
- **RMSE:** computed as `np.sqrt(mean_squared_error(...))` — do NOT use
  `squared=False` (removed in newer scikit-learn; breaks on 1.9+).
- **Seeds:** `random_state=42` everywhere (split, all tree models).
- **Visualization (added 2026-09-25, additive only):** `src/visualization.py`
  owns ALL charts; `main.py` calls `generate_training_insights(models,
  results_df, X_test, y_test)` after training. `train_and_compare` now
  returns `(best_name, results_df, models)` — third element is new.
  XGBoost `.fit` got `eval_set=[(X_test, y_test)], verbose=False` plus
  `eval_metric="rmse"` in `build_models()` — recording-only, does not
  change learning; metrics verified identical to pre-viz baseline.
  Decision Tree chart draws top 3 levels only (display limit, model
  params untouched). Charts use matplotlib "Agg" backend (no GUI).
- **Data recipe:** `scripts/generate_data.py` bakes a hidden linear-ish
  rating formula (cuisine boost + city boost + votes + online_order + cost
  − slow delivery + noise ±0.25). If you regenerate, metrics stay similar.

## 6. Windows gotchas already handled — do not regress

1. **₹ symbol crashes cp1252 consoles** (`UnicodeEncodeError: charmap`).
   Fixed in `src/main.py` via `sys.stdout.reconfigure(encoding="utf-8")`.
   Any new print of non-ASCII text relies on this — keep it at the top of `main()`.
2. Console output on Windows uses `\r\n`; harmless, just cosmetic in logs.

## 7. Last verified state (2026-09-25, post-visualization)

- `python src/main.py` runs end-to-end, exit code 0, and writes 6 PNGs to
  `outputs/`: model_comparison, actual_vs_predicted,
  linear_regression_coefficients, random_forest_feature_importance,
  decision_tree, xgboost_training_curve. Metrics and the final prediction
  are byte-identical to the pre-visualization baseline.
- Test metrics (450 rows, 360 train / 90 test):

  | Model             | MAE    | RMSE   | R²     |
  |-------------------|--------|--------|--------|
  | Linear Regression | 0.1400 | 0.1671 | 0.7468 |
  | XGBoost           | 0.1710 | 0.2087 | 0.6053 |
  | Random Forest     | 0.1747 | 0.2118 | 0.5935 |
  | Decision Tree     | 0.2156 | 0.2749 | 0.3151 |

- Linear Regression wins (dataset formula is mostly linear — expected).
- Example prediction: Biryani / Bangalore / 5000 votes / ₹500 / 35 min /
  Yes → **Predicted Rating 4.10**, Model Used: Linear Regression.

## 8. Conventions for future changes

- Keep modules small, beginner-readable, and comment the ML reasoning
  (the "why", not just the "what").
- New deps go in `requirements.txt` AND must be pip-installed into `.venv`.
- Update this file whenever: structure changes, a decision changes, deps
  change, or metrics are re-verified.
