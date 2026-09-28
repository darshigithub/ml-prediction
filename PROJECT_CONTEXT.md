# Project Context — ml-prediction (Generic Regression Framework)

> Reference doc for maintaining continuity across sessions. Read this first
> before making changes, and update it whenever the project evolves.

## 1. What this project is

A **beginner-friendly GENERIC tabular regression framework**, run entirely
from the command line. The user provides:

- a CSV dataset (`--data`)
- a numeric target column name (`--target`)

The framework automatically: profiles the dataset, detects feature types,
preprocesses (no leakage), runs **5-fold CV on the TRAINING split**,
selects the best model by **highest Mean CV R²**, trains all 4 models,
evaluates them on the untouched TEST set (MAE/RMSE/R²), retrains the
winner on all data, draws generic charts, and predicts one new row with
dynamically asked input.

**Explicitly OUT of scope (do not add):**
Power BI, web apps, FastAPI, LLMs, chatbots, GenUI, databases, Docker,
classification, AutoML, hyperparameter optimization, deep learning.

## 2. Current structure (verified working)

```
D:\AIML\ml-prediction\
├── data\
│   └── datasets\
│       ├── zomato_restaurants.csv   # 450 records, rating target (reference dataset)
│       └── house_prices.csv         # 300 records, price target (2nd example)
├── scripts\
│   ├── generate_data.py           # regenerates Zomato CSV (seed=42)
│   └── generate_house_data.py     # regenerates house CSV (seed=42)
├── src\
│   ├── data_loader.py             # generic CSV load + DatasetError + target validation
│   ├── dataset_profiler.py        # profile dict, feature-type + ID detection
│   ├── data_preprocessing.py      # ColumnTransformer/Pipeline builders + light clean
│   ├── cross_validation.py        # 5-fold CV (train split only) + Mean CV R² selection
│   ├── train_models.py            # split + 4 model pipelines + test metrics + retrain
│   ├── evaluate_models.py         # formatted comparison table + best-model summary
│   ├── visualization.py           # generic charts under outputs/ (Agg backend)
│   ├── predict.py                 # dynamic prediction input + print
│   └── main.py                    # CLI orchestration
├── outputs\                       # 7 generated PNGs (incl. cv_model_comparison), safe to delete
├── requirements.txt
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
.venv/Scripts/python src/main.py --data data/datasets/zomato_restaurants.csv --target rating
.venv/Scripts/python src/main.py --data data/datasets/house_prices.csv --target price
.venv/Scripts/python src/main.py --data data/datasets/house_prices.csv --target price --no-predict
.venv/Scripts/python src/main.py --data data/datasets/house_prices.csv --target price --test-size 0.25
```

User errors (missing file, bad/missing target, non-numeric target) raise
`DatasetError` in `data_loader.py`; `main.py` catches it and prints a short
`ERROR: ...` message with exit code 1 (no stack trace).

## 5. Design decisions (keep these consistent)

- **Target:** always user-supplied via `--target`; NEVER auto-guessed.
  Must be numeric (regression only).
- **Feature typing:** automatic from dtypes (`dataset_profiler.detect_column_types`);
  numeric -> numerical, everything else -> categorical. No hardcoded names.
- **Preprocessing:** sklearn `ColumnTransformer` + `Pipeline`:
  numeric -> `SimpleImputer(median)` + `StandardScaler`;
  categorical -> `SimpleImputer(most_frequent)` + `OneHotEncoder(handle_unknown="ignore")`.
  `handle_unknown="ignore"` is REQUIRED so unseen categories at prediction
  time do not crash.
- **NO DATA LEAKAGE:** split RAW data first; preprocessing is fitted on
  TRAIN only (inside each pipeline). Never fit a preprocessor on the full
  dataset before splitting.
- **XGBoost + eval_set gotcha:** a Pipeline `.fit` cannot take
  `model__eval_set` (it would receive RAW rows). In `train_and_compare`
  the preprocessor is fitted on TRAIN, TEST is `transform`ed, and the
  XGBoost model is fitted directly on transformed data. The pipeline keeps
  the same fitted objects, so `pipeline.predict` stays consistent.
- **Identifier columns:** only DETECTED and reported (heuristics in
  `dataset_profiler`); dropped only after an explicit y/n prompt per column
  (`main.confirm_identifier_removal`). Never removed silently.
- **Model selection (changed 2026-09-28):** highest **Mean CV R²** from
  5-fold CV on the TRAINING split (`cross_validation.select_best_model`) —
  NOT the single test-set R². The test set is only for final evaluation.
- **5-fold CV (added 2026-09-28):** `cross_validation.py` uses
  `KFold(n_splits=5, shuffle=True, random_state=42)` + sklearn
  `cross_validate` with the full preprocessing+model pipeline, so
  preprocessing is fitted FRESH inside every fold (no leakage). Runs on
  `X_train` only — the test set is never passed. scorers: `r2`,
  `neg_mae`, `neg_mse` (flipped back to positive MAE/RMSE for display).
  Results: structured dict `{model: {mean_r2, std_r2, mean_mae, std_mae,
  mean_rmse, std_rmse, fold_r2}}` + `print_cv_summary` table +
  `cv_model_comparison.png` (Mean R² bars, Std error bars).
- **XGBoost in CV:** plain `pipeline.fit` inside `cross_validate` — NO
  eval_set in folds (the training-curve chart comes from the separate
  post-selection training run in `train_and_compare`, which handles the
  eval_set preprocessing special case).
- **RMSE:** `np.sqrt(mean_squared_error(...))` — do NOT use `squared=False`
  (removed in newer scikit-learn).
- **Seeds:** `random_state=42` everywhere (split, all tree models).
- **Visualization:** labels use the ACTUAL target name ("Actual vs
  Predicted price"). Expanded one-hot feature names come from
  `get_transformed_feature_names(pipeline)` (strips `num__`/`cat__`).
  Decision Tree chart draws top 3 levels only (display-only limit).
- **Generic prediction:** `predict.ask_for_prediction_input` asks for every
  feature dynamically (validated numeric input; text with known-category
  hints; EOF/empty line = cancel). The raw row goes through the full
  pipeline — no manual encoding anywhere.
- **Interactive categorical validation (added 2026-09-28):** input is
  validated against categories read from the FITTED OneHotEncoder
  (`data_preprocessing.get_known_categories`, uses `categories_`).
  Rules: trim whitespace, case-insensitive exact match (`.casefold()`),
  normalize to the training spelling, REJECT anything else and re-ask.
  Deliberately NO fuzzy/auto-correct matching. CLI-only guard: the
  pipeline keeps `OneHotEncoder(handle_unknown="ignore")` so programmatic
  use with unseen categories stays crash-free. Unknown-category
  predictions remain possible only outside the interactive loop.
- **Interactive prompts:** wrap `input()` in try/EOFError so piped/CI runs
  exit cleanly instead of raising.
- **Data recipes:** `scripts/generate_data.py` (Zomato, hidden rating
  formula) and `scripts/generate_house_data.py` (hidden price formula).
  Regenerating keeps metrics similar.

## 6. Windows gotchas already handled — do not regress

1. **₹ symbol crashes cp1252 consoles** (`UnicodeEncodeError: charmap`).
   Fixed in `src/main.py` via `sys.stdout.reconfigure(encoding="utf-8")`.
   Keep it at the top of `main()`.
2. Console output on Windows uses `\r\n`; harmless, just cosmetic in logs.
3. **pandas 3.x `str` dtype:** text columns are the new `str` dtype, NOT
   `object`. Never test for text with `dtype == object` — use
   `not pd.api.types.is_numeric_dtype(...)`. Also: pandas' new
   dtype-mismatch reindexing FILLS WITH NaN instead of raising
   (silent column wipe) — never rely on `.reindex` for column alignment;
   the pipeline architecture made it unnecessary.

## 7. Last verified state (2026-09-28, post-CV)

- Zomato run completes end-to-end with 5-fold CV, exit 0, 7 PNGs.
  Zomato CV results (360 train rows, KFold 5, shuffle, rs=42) — SELECTION
  table (Mean R² decides):

  | Model             | Mean R² | Std R² | Mean MAE | Mean RMSE |
  |-------------------|---------|--------|----------|-----------|
  | Linear Regression | 0.6693  | 0.1085 | 0.1348   | 0.1751    |
  | XGBoost           | 0.6555  | 0.0697 | 0.1461   | 0.1791    |
  | Random Forest     | 0.6245  | 0.0617 | 0.1531   | 0.1876    |
  | Decision Tree     | 0.1707  | 0.2627 | 0.2167   | 0.2757    |

  Selected: Linear Regression (Mean CV R²). FINAL TEST metrics (90 rows,
  untouched) — UNCHANGED from before CV was added:

  | Model             | MAE    | RMSE   | R²     |
  |-------------------|--------|--------|--------|
  | Linear Regression | 0.1400 | 0.1671 | 0.7468 |
  | Random Forest     | 0.1603 | 0.1942 | 0.6580 |
  | XGBoost           | 0.1607 | 0.1978 | 0.6454 |
  | Decision Tree     | 0.2022 | 0.2539 | 0.4159 |

  Prediction unchanged: Biryani/Bangalore/5000/500/35/Yes -> 4.0966.
- House prices (240 train / 60 test): CV selects XGBoost (Mean R² 0.9422
  ± 0.0275); final test XGBoost R² 0.9767. With `--test-size 0.25`
  (225/75) CV selects Linear Regression (Mean R² 0.9282 vs XGBoost
  0.9257) — proof selection follows CV, not the test table.
- Verified: `--no-predict` exits cleanly; unknown target / missing file /
  non-numeric target errors unchanged; categorical validation (case /
  whitespace / typo rejection) intact; `requirements.txt` unchanged
  (no new deps — sklearn provides cross_validate/KFold).
- NOTE (2026-09-28): several tool outputs returned corrupted/mismatched
  content during this session; all results above were re-verified with
  clean single-command runs before being recorded here.
- Error tests verified: unknown target, missing file, non-numeric target
  (all clean `ERROR:` + exit 1). Interactive validation verified: `yes` /
  `YES` / `YeS` -> `Yes`, `bangalore` / `apartment` -> canonical spelling,
  `  Bangalore` trimmed, `Baangalore` and `Vila` rejected with allowed
  values listed and re-asked; empty input / EOF still cancels cleanly.
  House-price run (240/60 split) — XGBoost best, R² 0.9767; auto-detection
  found area/bedrooms/bathrooms/age numerical, city/property_type
  categorical. NOTE: dataset CSVs now live in `data/datasets/`.

## 8. Conventions for future changes

- Keep modules small, beginner-readable, and comment the ML reasoning
  (the "why", not just the "what").
- New deps go in `requirements.txt` AND must be pip-installed into `.venv`.
- Update this file whenever: structure changes, a decision changes, deps
  change, or metrics are re-verified.
