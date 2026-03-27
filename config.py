"""
config.py — Single source of truth for all pipeline settings.
Edit this file to change paths, cities, truck specs, or model params.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))

# ── Input CSV ──────────────────────────────────────────────────────────────
# Three ways to provide the input CSV (checked in this order):
#   1. --input flag at runtime:   python pipeline.py --input my_data.csv
#   2. Drop-zone folder:          place any single .csv in the inputs/ folder
#   3. Fallback default below:    edit INPUT_CSV_DEFAULT to a fixed path
#
# You should rarely need to edit INPUT_CSV_DEFAULT — use option 1 or 2 instead.
INPUT_CSV_DEFAULT = os.path.join(BASE_DIR, "sales_input.csv")
INPUTS_DIR        = os.path.join(BASE_DIR, "inputs")   # drop-zone folder

OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")

# Intermediate checkpoints (set to None to skip saving)
CHECKPOINT_XGBOOST   = os.path.join(OUTPUT_DIR, "checkpoint_1_xgboost.xlsx")
CHECKPOINT_INVENTORY = os.path.join(OUTPUT_DIR, "checkpoint_2_inventory.xlsx")
FINAL_OUTPUT         = os.path.join(OUTPUT_DIR, "Route.xlsx")

# ── Cities & Coordinates ───────────────────────────────────────────────────
CITIES = ['Ahemdabad', 'Bangalore', 'Delhi', 'Mumbai']

CITY_COORDS = {
    'Delhi':     (28.7041, 77.1025),
    'Mumbai':    (19.0760, 72.8777),
    'Bangalore': (12.9716, 77.5946),
    'Ahemdabad': (23.0225, 72.5714),
}

# ── XGBoost Hyperparameters ────────────────────────────────────────────────
XGBOOST_PARAMS = {
    "objective":        "reg:squarederror",
    "eval_metric":      "rmse",
    "eta":              0.08,
    "max_depth":        3,
    "min_child_weight": 8,
    "subsample":        0.7,
    "colsample_bytree": 0.6,
    "lambda":           5.0,
    "alpha":            2.0,
    "seed":             42,
}

XGBOOST_ROUNDS          = 500
XGBOOST_EARLY_STOPPING  = 20
TRAIN_RATIO             = 0.70
VAL_RATIO               = 0.15

# ── Truck / Route Config ───────────────────────────────────────────────────
TRUCK_CONFIG = {
    "capacity":      5000,   # pieces per truck
    "cost_per_km":   50,     # ₹ per km
    "fixed_cost":    5000,   # ₹ per trip
    "loading_time":  1.5,    # hours per stop
    "speed":         60,     # km/h
    "max_hours":     10,     # max daily driving hours
}
