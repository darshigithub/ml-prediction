"""
Script to generate a dummy (synthetic) Zomato-like restaurant dataset.

Run this ONLY if you want to regenerate `data/datasets/zomato_restaurants.csv`
with a new random sample. The project already ships with a generated CSV,
so normally you do NOT need to run this script.

    python scripts/generate_data.py
"""

import os
import random

import pandas as pd

# Make results reproducible: same "random" numbers every run
random.seed(42)

N_RESTAURANTS = 450  # number of sample records to generate

# --- Realistic pools of values -------------------------------------------
CUISINES = ["Biryani", "North Indian", "South Indian", "Chinese", "Pizza",
            "Burger", "Desserts", "Seafood", "Mughlai", "Cafe"]
CITIES = ["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Chennai",
          "Pune", "Kolkata", "Jaipur"]
FIRST_NAMES = ["Spice", "Tandoor", "Curry", "Royal", "Urban", "Golden",
               "Paradise", "Grill", "Masala", "Biryani", "Kebab", "Dosa",
               "Wok", "Cheese", "Sweet", "Coastal", "Punjabi", "Hyderabadi"]
SECOND_NAMES = ["House", "Kitchen", "Junction", "Garden", "Palace", "Hub",
                "Cafe", "Express", "Point", "Court", "Bay", "Street"]
ONLINE_ORDER = ["Yes", "No"]

# Base popularity per cuisine (used to make ratings look realistic)
CUISINE_RATING_BOOST = {
    "Biryani": 0.35, "Mughlai": 0.30, "Seafood": 0.25, "North Indian": 0.15,
    "South Indian": 0.15, "Cafe": 0.10, "Desserts": 0.05,
    "Chinese": 0.0, "Pizza": -0.05, "Burger": -0.10,
}

# Some cities are simply bigger food markets -> slightly higher ratings
CITY_RATING_BOOST = {
    "Bangalore": 0.15, "Hyderabad": 0.15, "Mumbai": 0.10, "Delhi": 0.05,
    "Pune": 0.05, "Chennai": 0.0, "Jaipur": -0.05, "Kolkata": -0.10,
}

rows = []
for _ in range(N_RESTAURANTS):
    cuisine = random.choice(CUISINES)
    city = random.choice(CITIES)
    online_order = random.choice(ONLINE_ORDER)

    name = f"{random.choice(FIRST_NAMES)} {random.choice(SECOND_NAMES)}"

    # Votes: most restaurants have few votes, a few are very popular
    votes = max(10, int(random.lognormvariate(6.2, 1.3)))

    # Average cost for two (INR)
    average_cost = random.choice([100, 150, 200, 250, 300, 350, 400,
                                  450, 500, 600, 700, 800, 1000])

    # Delivery time in minutes (online-order places are usually faster)
    base_time = 20 if online_order == "Yes" else 30
    delivery_time = random.randint(base_time, base_time + 30)

    # --- Build a "hidden formula" for the rating so models have signal ----
    rating = 3.0
    rating += CUISINE_RATING_BOOST[cuisine]                    # cuisine effect
    rating += CITY_RATING_BOOST[city]                          # city effect
    rating += min(votes, 10000) / 10000 * 0.6                  # popularity
    rating += 0.3 if online_order == "Yes" else 0.0            # convenience
    rating += min(average_cost, 800) / 800 * 0.3               # premium places
    rating -= max(delivery_time - 35, 0) / 50 * 0.5            # slow delivery
    # Add some random noise, clip to a realistic 5-point scale
    rating += random.uniform(-0.25, 0.25)
    rating = round(min(5.0, max(2.0, rating)), 1)

    rows.append({
        "restaurant_name": name,
        "cuisine": cuisine,
        "city": city,
        "rating": rating,
        "votes": votes,
        "average_cost": average_cost,
        "delivery_time": delivery_time,
        "online_order": online_order,
    })

df = pd.DataFrame(rows)

# Save next to the project root: <root>/data/datasets/zomato_restaurants.csv
# out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "datasets")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "zomato_restaurants.csv")
df.to_csv(out_path, index=False)

print(f"Generated {len(df)} records -> {os.path.abspath(out_path)}")
print(df.head())
