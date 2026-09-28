# Generic ML Regression Framework 📊

A simple, beginner-friendly **Machine Learning** framework that predicts a
**numeric target** from any CSV dataset. It started as a Zomato restaurant
rating predictor and is now a reusable, dataset-agnostic tabular regression
tool — same code, different data.

Everything runs from the **command line** — no web app, no database, no
dashboards. Just clean, commented Python.

## Project Purpose

Give it a CSV file and the name of the numeric column you want to predict.
The framework automatically:

- profiles the dataset (size, column types, missing values, duplicates)
- detects numerical vs categorical feature columns
- preprocesses them (imputation + scaling + one-hot encoding, no leakage)
- runs **5-fold cross-validation on the training data** (Mean/Std R², MAE, RMSE)
- trains 4 regression models and compares them on the test set (MAE / RMSE / R²)
- selects the best model by **highest Mean CV R²** and retrains it on all data
- draws training-insight charts
- asks you for one new row's values and predicts the target

## Supported Algorithms

| # | Model | Notes |
|---|-------|-------|
| 1 | Linear Regression | fast baseline, interpretable coefficients |
| 2 | Decision Tree Regressor | visualizeable rules |
| 3 | Random Forest Regressor | 300 trees, robust |
| 4 | XGBoost Regressor | gradient boosting, 300 rounds |

`random_state=42` everywhere for reproducibility.

## Dataset Requirements

- A CSV file with **at least one numeric column to use as the target**
- Any mix of numerical and categorical feature columns
- The target is supplied by YOU via `--target` (never guessed)
- Regression only: the target must contain numbers

## Installation

```bash
# 1. (optional) create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash
# .venv\Scripts\activate           # Windows CMD/PowerShell
# source .venv/bin/activate        # Linux/macOS

# 2. install the required libraries
pip install -r requirements.txt
```

Dependencies: pandas, numpy, scikit-learn, xgboost, matplotlib.

## CLI Usage

```bash
python src/main.py --data <path-to-csv> --target <column-to-predict>
```

| Argument | Required | Meaning |
|----------|----------|---------|
| `--data` | yes | path to the CSV dataset |
| `--target` | yes | numeric column to predict |
| `--test-size` | no | test fraction, default `0.20` |
| `--no-predict` | no | skip the interactive prediction at the end |

## Example 1 — Zomato Restaurant Ratings

`data/datasets/zomato_restaurants.csv` (450 synthetic records, kept as the reference
example dataset):

| Column           | Meaning                                | Example      |
|------------------|----------------------------------------|--------------|
| `restaurant_name`| fake restaurant name (identifier)      | Grill Garden |
| `cuisine`        | food type (categorical)                | Biryani      |
| `city`           | Indian city (categorical)              | Bangalore    |
| `rating`         | **target** – rating from 1.0 to 5.0    | 4.3          |
| `votes`          | number of user votes (numeric)         | 5000         |
| `average_cost`   | average cost for two, ₹ (numeric)      | 500          |
| `delivery_time`  | delivery time in minutes (numeric)     | 35           |
| `online_order`   | accepts online orders? (categorical)   | Yes          |

```bash
.venv/Scripts/python src/main.py --data data/datasets/zomato_restaurants.csv --target rating
```

## Example 2 — House Prices

`data/datasets/house_prices.csv` (300 synthetic records) — a completely
different problem for the SAME framework:

```bash
.venv/Scripts/python src/main.py --data data/datasets/house_prices.csv --target price
```

The framework automatically detects:

```
Numerical  : area, bedrooms, bathrooms, age
Categorical: city, property_type
Target     : price
```

To regenerate that CSV: `python scripts/generate_house_data.py`

## Output Metrics

### Cross-validation (training data only)

Every model is evaluated with 5-fold CV on the **training split**. For
each model the framework reports the Mean and Std of R², MAE and RMSE
across the 5 folds. sklearn returns MAE/RMSE as negative losses during
CV; they are converted back to positive values before display.

### Final test evaluation (holdout)

The untouched test set then gives the classic table:

- **MAE** — Mean Absolute Error (lower is better)
- **RMSE** — Root Mean Squared Error, computed with
  `np.sqrt(mean_squared_error(...))` (lower is better)
- **R²** — proportion of target variation explained (higher is better)

### How they are used

- **Selection**: the model with the **highest Mean CV R²** wins. Averaging
  over 5 folds makes the choice far harder to fool by one lucky split.
- **Final evaluation**: the test set is touched by NO model during CV or
  training — it provides the unbiased check after selection.
- **Final model**: the winner is **retrained on all rows** (train + test)
  before the interactive prediction.

## Model Selection: Cross-Validation vs Test Set

```
Full dataset
    ├── 80% Training ── 5-fold CV ──► Mean CV R² ──► SELECT model
    │
    └── 20% Test ── (untouched until the end) ──► FINAL evaluation
                        │
                        ▼
              Retrain selected model on ALL data → prediction
```

CV and test numbers can legitimately disagree (they measure different
things). If they do, the CV ranking decides — that is the point of using
it. The CLI shows both tables so you can compare.

## Visualizations

After training, six charts are saved to `outputs/` (labels use your actual
target name, e.g. "Actual vs Predicted price"):

| File | What it shows |
|------|----------------|
| `cv_model_comparison.png` | Mean CV R² per model, error bar = Std CV R² (selection view) |
| `model_comparison.png` | R² of all four models on the test set (bar chart) |
| `actual_vs_predicted.png` | actual vs predicted scatter for each model |
| `linear_regression_coefficients.png` | learned coefficients (generic "Feature Coefficients") |
| `random_forest_feature_importance.png` | Random Forest feature importances (importance ≠ direction) |
| `decision_tree.png` | Decision Tree (top 3 levels drawn for readability; model untouched) |
| `xgboost_training_curve.png` | XGBoost validation RMSE per boosting round |

The `outputs/` folder can be deleted at any time; it is regenerated on the
next run.

## Project Structure

```
ml-prediction/
│
├── data/
│   └── datasets/
│       ├── zomato_restaurants.csv # reference example dataset (450 records)
│       └── house_prices.csv       # second example dataset (300 records)
│
├── scripts/
│   ├── generate_data.py           # regenerates the Zomato CSV (seed=42)
│   └── generate_house_data.py     # regenerates the house CSV (seed=42)
│
├── src/
│   ├── data_loader.py             # generic CSV load + target validation
│   ├── dataset_profiler.py        # profile + feature-type detection
│   ├── data_preprocessing.py      # ColumnTransformer pipeline builder
│   ├── train_models.py            # 4 models inside sklearn Pipelines
│   ├── evaluate_models.py         # comparison table + best-model summary
│   ├── visualization.py           # generic insight charts (outputs/*.png)
│   ├── predict.py                 # dynamic, interactive new-row prediction
│   └── main.py                    # CLI: ties the whole workflow together
│
├── outputs/                       # generated charts (6 PNGs), safe to delete
├── requirements.txt
├── README.md
└── PROJECT_CONTEXT.md
```

## How Preprocessing Works (and why)

```
Raw dataset
    ↓  train/test split on RAW data (default 80/20, random_state=42)
    ↓  5-fold CV on TRAINING rows: each fold fits a FRESH pipeline
    ↓  (preprocessing + model) on that fold's training portion only
    ↓  final training run: fit preprocessing on TRAIN ONLY
    ↓  transform TRAIN and TEST with it
Numerical  → SimpleImputer(median) → StandardScaler
Categorical→ SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown="ignore")
    ↓  ColumnTransformer, wrapped with each model in a sklearn Pipeline
```

- **No data leakage**: preprocessing statistics are learned from training
  rows only — inside CV, each fold fits its own fresh pipeline, so
  validation rows never influence the imputer/scaler/encoder either.
- **Unknown categories are safe**: the pipeline keeps
  `OneHotEncoder(handle_unknown="ignore")`, so a category never seen in
  training does not crash programmatic predictions.
- **One code path**: training, testing and final prediction all use the
  same fitted pipeline, so inputs are always preprocessed identically.

## Interactive Prediction Input

After training, the framework asks for every feature value to predict one
new row. Categorical input is validated against the categories the fitted
pipeline knows (read directly from the OneHotEncoder's `categories_` —
nothing is hardcoded):

- surrounding whitespace is trimmed (`"  Bangalore"` → `Bangalore`)
- matching is case-insensitive (`"yes"`, `"YES"`, `"YeS"` → `Yes`)
- unknown values are **rejected** with the list of allowed values and
  asked again — a typo like `"Baangalore"` never silently produces a
  prediction
- there is deliberately **no fuzzy/auto-correct matching**

Numeric input accepts ints and floats (including 0 and negatives) and
asks again with a clear error on invalid text. An empty line or EOF
(Ctrl+D) cancels the prediction safely.

## Limitations

- Regression only — classification is out of scope (for now)
- No hyperparameter tuning — sensible fixed defaults
- Identifier columns are only *reported*; you decide whether to drop them
- Very high-cardinality text columns explode into many one-hot columns
- Light cleaning only: duplicates dropped, target-missing rows removed,
  missing *feature* values imputed (never silently dropped)

## Future Roadmap

- Classification support (categorical targets)
- Hyperparameter optimization
- Model persistence (save/load the final pipeline)
- Larger dataset handling and alternative encoders (e.g. target encoding)
