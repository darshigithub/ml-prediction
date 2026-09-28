"""
Script to generate a small synthetic HOUSE PRICE dataset.

This dataset exists to prove that the framework is generic: the same
code that predicts Zomato ratings must also work on a completely
different tabular problem (predicting `price` from area, rooms, age,
city and property type).

Run ONLY if you want to regenerate data/datasets/house_prices.csv:

    python scripts/generate_house_data.py
"""

import os
import random

import pandas as pd

# Make results reproducible: same "random" numbers every run
random.seed(42)

N_HOUSES = 300  # number of sample records to generate

# --- Realistic pools of values -------------------------------------------
CITIES = ["Bangalore", "Mumbai", "Delhi", "Mysore", "Chennai"]
PROPERTY_TYPES = ["Apartment", "House", "Villa"]

# Price per sq.ft baseline per city (bigger market = costlier)
CITY_PRICE_FACTOR = {
    "Mumbai": 1.60, "Bangalore": 1.15, "Delhi": 1.10,
    "Chennai": 1.00, "Mysore": 0.70,
}
# Villas cost more per sq.ft than plain houses/apartments
TYPE_PRICE_FACTOR = {"Villa": 1.30, "House": 1.05, "Apartment": 1.00}

rows = []
for _ in range(N_HOUSES):
    city = random.choice(CITIES)
    property_type = random.choice(PROPERTY_TYPES)

    area = random.randint(600, 2600)                # square feet
    bedrooms = max(1, min(5, area // 550 + random.choice([0, 0, 1])))
    bathrooms = max(1, bedrooms - random.choice([0, 1]))
    age = random.randint(0, 25)                     # years since built

    # --- Hidden formula so the models have real signal to learn ---------
    base = area * 7000                              # ₹ per sq.ft baseline
    base *= CITY_PRICE_FACTOR[city]
    base *= TYPE_PRICE_FACTOR[property_type]
    base += bedrooms * 250_000                      # extra per bedroom
    base += bathrooms * 120_000                     # extra per bathroom
    base *= max(0.75, 1.0 - age * 0.012)            # older = slightly cheaper
    base *= random.uniform(0.96, 1.04)              # market noise
    price = int(round(base / 10_000) * 10_000)      # round to ₹10,000

    rows.append({
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age": age,
        "city": city,
        "property_type": property_type,
        "price": price,
    })

df = pd.DataFrame(rows)

# Save to <root>/data/datasets/house_prices.csv
out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "datasets")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "house_prices.csv")
df.to_csv(out_path, index=False)

print(f"Generated {len(df)} records -> {os.path.abspath(out_path)}")
print(df.head())
