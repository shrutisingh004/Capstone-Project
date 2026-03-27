# Capstone-Project
UISCO (Unified Intelligent Supply Chain Optimization) for FMCG

## Overview

This project addresses a real-world supply chain problem: given historical sales data across multiple cities and products, how should a company forecast demand, rebalance its inventory, and plan its delivery routes to minimize cost and waste?

The solution is a three-stage automated pipeline:

```
sales_input.csv
      │
      ▼
┌─────────────────────┐
│  Stage 1: XGBoost   │  ← Sales forecasting (ML)
│  Demand Forecasting │
└────────┬────────────┘
         │  Product × Month × City demand matrix
         ▼
┌─────────────────────┐
│  Stage 2: Inventory │  ← Zero-sum transfer optimization (Greedy LP)
│  Transfer Optimizer │
└────────┬────────────┘
         │  Optimal transfer plan (source → destination, quantity)
         ▼
┌─────────────────────┐
│  Stage 3: Route     │  ← Vehicle routing (NN + Greedy Insertion + 2-opt)
│  Optimizer          │
└────────┬────────────┘
         │
         ▼
     Route.xlsx
```

---

## Project Structure

```
├── pipeline.py              # Single entry point — run this
├── config.py                # All settings: paths, cities, truck specs, model params
├── stage1_xgboost.py        # Stage 1: XGBoost demand forecasting
├── stage2_inventory.py      # Stage 2: Inventory transfer optimization
├── stage3_routes.py         # Stage 3: FMCG route optimization
│
├── sales_input.csv          # Input: historical daily sales data
│
└── outputs/                 # Generated on first run
    ├── checkpoint_1_xgboost.xlsx    # Intermediate: demand predictions
    ├── checkpoint_2_inventory.xlsx  # Intermediate: transfer plan
    └── Route.xlsx                   # Final output: optimized routes
```

---

## Methodology

### Stage 1 — XGBoost Sales Forecasting
- Trains an XGBoost regression model on 4 years of daily sales data (29,000+ records)
- Features: date components, promotions, marketing spend, holidays, inflation index, region
- 70/15/15 train/validation/test split with early stopping
- Outputs a **Product × Month × City** demand prediction matrix

### Stage 2 — Inventory Transfer Optimization
- Frames rebalancing as a **zero-sum allocation problem**: cities with above-average predicted demand receive stock from cities with surplus
- Uses a greedy surplus-deficit matching algorithm
- Outputs a detailed transfer plan: which product moves from which city to which city, and how many pieces

### Stage 3 — Route Optimization
- Solves a **Capacitated Vehicle Routing Problem (CVRP)** using two heuristics:
  - Nearest Neighbor TSP
  - Greedy Insertion
- Applies **2-opt local search** improvement to each route
- Selects the lower-cost solution per departure city per period
- Accounts for truck capacity, cost per km, fixed trip cost, loading time, and max driving hours

---

## Setup

### Requirements

```bash
pip install pandas numpy xgboost scikit-learn scipy openpyxl
```

### Configuration

All settings live in `config.py`. Before running, update:

```python
# Path to the input data
INPUT_CSV = "path/to/sales_input.csv"

# Cities in your distribution network
CITIES = ['Ahemdabad', 'Bangalore', 'Delhi', 'Mumbai']

# Truck specifications
TRUCK_CONFIG = {
    "capacity":    5000,   # pieces per truck
    "cost_per_km": 50,     # ₹ per km
    "fixed_cost":  5000,   # ₹ per trip
    ...
}
```

---

## Usage

### Full pipeline (recommended first run)

```bash
python pipeline.py
```

### Resume from a checkpoint (saves time during iteration)

```bash
# Skip XGBoost retraining — load Stage 1 checkpoint
python pipeline.py --from stage2

# Skip both Stage 1 and 2 — re-run routing only
python pipeline.py --from stage3

# Run without saving intermediate files
python pipeline.py --no-checkpoints
```

The `--from` flags are useful when iterating: retraining XGBoost on the full dataset takes time, but if you only changed truck capacity in `config.py`, `--from stage3` re-runs just the routing in seconds.

---

## Output

### `Route.xlsx` contains three sheets:

| Sheet | Contents |
|---|---|
| `Routes` | Every optimized route with stops, distance, cost, capacity utilization, feasibility |
| `Summary` | Aggregate KPIs: total distance, total cost, cost per piece, avg capacity |
| `Recommendations` | Flagged issues (low utilization, infeasible routes, high-cost lanes) |

### Checkpoint files (optional, for inspection):

| File | Contents |
|---|---|
| `checkpoint_1_xgboost.xlsx` | Predicted demand per product per city per month |
| `checkpoint_2_inventory.xlsx` | Full transfer plan with 3 sheets: transfers, monthly summary, route summary |

---

## Key Results

| Metric | Value |
|---|---|
| Model Accuracy (Test MAPE) | < 15% |
| Total Data Processed | 29,220+ records |
| Shipment Requests Reduced | 180 → 36 |
| Total Routes Generated | 12 optimized routes |
| Vehicle Utilization | Significantly improved |
| Logistics Cost | Reduced through route optimization |


---

## Cities Covered

| City | Coordinates |
|---|---|
| Delhi | 28.70°N, 77.10°E |
| Mumbai | 19.08°N, 72.88°E |
| Bangalore | 12.97°N, 77.59°E |
| Ahmedabad | 23.02°N, 72.57°E |

Distances between cities are computed using the **Haversine formula** (great-circle distance).

---

## Dataset

`sales_input.csv` — 29,220 rows of daily sales records across products, regions, and time.

| Column | Description |
|---|---|
| `Date` | Transaction date |
| `Product_ID` | Product identifier |
| `Region` | City/region name |
| `Sales_Quantity` | Units sold (target variable) |
| `Price_Per_Unit` | Unit price (₹) |
| `Sales_Value` | Revenue (₹) |
| `Promotion` | Promotion active (0/1) |
| `Marketing_Spend` | Marketing expenditure (₹) |
| `Holiday` | Holiday flag (0/1) |
| `Inflation_Index` | Macro inflation index |

---

## Limitations & Future Work

- **Demand forecasting** currently uses monthly aggregation; a daily or weekly model could improve accuracy
- **Inventory optimization** assumes equal weight to all cities; a cost-weighted LP formulation could be more precise
- **Routing** uses heuristics, not exact solvers; for larger networks, a solver like OR-Tools or PuLP would give provably optimal solutions
- **No real-time data integration** — pipeline is batch-only; connecting to a live ERP/WMS would enable continuous optimization

---

## Acknowledgement

Developed as part of a final-year capstone project. Dataset, problem framing, and business context are based on real FMCG distribution challenges in the Indian market.
