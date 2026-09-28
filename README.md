# Zomato Restaurant Rating Prediction 🍽️📊

A simple, beginner-friendly **Machine Learning** project that predicts a
restaurant's rating from historical restaurant data (a dummy Zomato-like
dataset).

Everything runs from the **command line** — no web app, no database, no
dashboards. Just clean, commented Python.

## Project Structure

```
ml-zomato-prediction/
│
├── data/
│   └── zomato_restaurants.csv     # dummy dataset (450 records)
│
├── outputs/                       # generated training-insight charts (PNG)
│
├── scripts/
│   └── generate_data.py           # (optional) regenerate the dummy CSV
│
├── src/
│   ├── data_preprocessing.py      # load, clean, encode
│   ├── train_models.py            # train 4 models + evaluation
│   ├── visualization.py           # training-insight charts (outputs/*.png)
│   ├── predict.py                 # predict for a new restaurant
│   └── main.py                    # runs the whole workflow
│
├── requirements.txt
└── README.md
```

## Dataset

`data/zomato_restaurants.csv` contains ~450 synthetic (dummy) records with:

| Column           | Meaning                                | Example      |
|------------------|----------------------------------------|--------------|
| `restaurant_name`| fake restaurant name                   | Grill Garden |
| `cuisine`        | food type                              | Biryani      |
| `city`           | Indian city                            | Bangalore    |
| `rating`         | **target** – rating from 1.0 to 5.0    | 4.3          |
| `votes`          | number of user votes                   | 5000         |
| `average_cost`   | average cost for two (₹)               | 500          |
| `delivery_time`  | delivery time in minutes               | 35           |
| `online_order`   | accepts online orders?                 | Yes          |

## ML Workflow

1. **Load** the CSV with pandas
2. **Clean** the data (duplicates, bad values, missing rows, sanity ranges)
3. **Separate** features (X) and target (y = `rating`)
4. **Encode** categorical columns with One-Hot Encoding
5. **Split** into 80% train / 20% test
6. **Train** four regression models:
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - XGBoost Regressor
7. **Evaluate** every model with **MAE**, **RMSE** and **R²**
8. **Compare** all models in a printed table
9. **Select** the best model (highest R²) and retrain it on all data
10. **Predict** the rating for a brand-new restaurant

## Setup & Run

```bash
# 1. (optional) create and activate a virtual environment
python -m venv .venv
source .venv/Scripts/activate      # Windows Git Bash
# .venv\Scripts\activate           # Windows CMD/PowerShell
# source .venv/bin/activate        # Linux/macOS

# 2. install the required libraries
pip install -r requirements.txt

# 3. run the project
python src/main.py
```

That's it — one command runs the full workflow.

## Training Insights (Visualizations)

After training, the script prints a **TRAINING INSIGHTS** section and saves
six charts to `outputs/`:

| File | What it shows |
|------|----------------|
| `model_comparison.png` | R² of all four models (bar chart) |
| `actual_vs_predicted.png` | actual vs predicted scatter for each model |
| `linear_regression_coefficients.png` | learned Linear Regression coefficients |
| `random_forest_feature_importance.png` | Random Forest feature importances (sorted) |
| `decision_tree.png` | the trained Decision Tree (top 3 levels drawn) |
| `xgboost_training_curve.png` | XGBoost validation RMSE per boosting round |

All charts use the **actual** values from the current run — nothing is
invented. The `outputs/` folder can be deleted at any time; it is
regenerated on the next run.

## Example Output

```
=== Model Comparison (test set) ===
            Model    MAE   RMSE     R2
Linear Regression 0.1400 0.1671 0.7468
          XGBoost 0.1710 0.2087 0.6053
    Random Forest 0.1747 0.2118 0.5935
    Decision Tree 0.2156 0.2749 0.3151

[Select] Best model based on R2: Linear Regression

=======================================================
TRAINING INSIGHTS
=======================================================
[Viz] Saved outputs/model_comparison.png
[Viz] Saved outputs/actual_vs_predicted.png

Linear Regression — learned coefficients
Feature                  Coefficient
cuisine_Burger           -0.48252
cuisine_Pizza            -0.43266
...
votes                    +0.00002

Random Forest — feature importance (sorted)
Feature                  Importance
online_order_Yes         0.25989
votes                    0.19963
...

Model comparison chart:
outputs/model_comparison.png

Actual vs predicted chart:
outputs/actual_vs_predicted.png
=======================================================

----------------------------------------
Restaurant Rating Prediction
----------------------------------------
Cuisine       : Biryani
City          : Bangalore
Votes         : 5000
Average Cost  : ₹500
Delivery Time : 35 minutes
Online Order  : Yes

Predicted Rating: 4.10
Model Used: Linear Regression
----------------------------------------
```

## Try Other Restaurants

Open `src/main.py` and edit the `new_restaurant` dictionary:

```python
new_restaurant = {
    "cuisine": "Pizza",
    "city": "Mumbai",
    "votes": 800,
    "average_cost": 300,
    "delivery_time": 45,
    "online_order": "No",
}
```

## Notes

- The dataset is **dummy** (generated with a hidden rating formula), so the
  metrics look optimistic — that's expected and fine for learning.
- To regenerate the CSV with a fresh random sample:
  `python scripts/generate_data.py`
