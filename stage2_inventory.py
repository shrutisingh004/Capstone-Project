"""
stage2_inventory.py — Zero-sum inventory transfer optimization.

Input  : XGBoost prediction matrix (Product_ID, Month, city columns)
Output : transfers DataFrame with columns:
         Product_ID, Month, Source_City, Destination_City,
         Transfer_Quantity, Source_Surplus_Before, Dest_Deficit_Before,
         Source_Original_Demand, Dest_Original_Demand, Mean_Demand
"""

import pandas as pd
import numpy as np

import config


# ── Public entry point ─────────────────────────────────────────────────────

def run(matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute optimal inventory transfers from prediction matrix.

    Parameters
    ----------
    matrix : DataFrame with columns [Product_ID, Month, <city1>, <city2>, ...]
             Values are integer piece counts.

    Returns
    -------
    pd.DataFrame of transfer records.
    """
    print("\n" + "=" * 70)
    print("STAGE 2 — Inventory Transfer Optimization")
    print("=" * 70)

    cities = _detect_cities(matrix)
    print(f"   Cities detected : {cities}")
    print(f"   Products        : {matrix['Product_ID'].nunique()}")
    print(f"   Months          : {matrix['Month'].nunique()}")

    transfers = _compute_transfers(matrix, cities)

    if transfers.empty:
        print("⚠️  No transfers needed — all cities balanced.")
        return transfers

    _print_summary(transfers)
    print(f"\n✅ Stage 2 complete — {len(transfers)} transfer records")
    return transfers


# ── Internal helpers ───────────────────────────────────────────────────────

def _detect_cities(matrix: pd.DataFrame) -> list:
    """Return city columns present in matrix (intersection with config.CITIES)."""
    known = set(config.CITIES)
    detected = [c for c in matrix.columns if c in known]
    if not detected:
        # Fallback: any column that isn't Product_ID or Month
        detected = [c for c in matrix.columns if c not in ("Product_ID", "Month")]
    return detected


def _compute_transfers(matrix: pd.DataFrame, cities: list) -> pd.DataFrame:
    all_transfers = []

    for _, row in matrix.iterrows():
        product_id  = row["Product_ID"]
        month       = row["Month"]
        city_demands = {city: int(row[city]) for city in cities if city in row}

        transfers = _calculate_transfers(product_id, month, city_demands)
        all_transfers.extend(transfers)

    return pd.DataFrame(all_transfers) if all_transfers else pd.DataFrame()


def _calculate_transfers(product_id, month, city_demands: dict) -> list:
    """
    Greedy zero-sum allocation.
    Surplus cities send stock to deficit cities relative to mean demand.
    """
    mean_demand     = np.mean(list(city_demands.values()))
    surplus_deficit = {city: demand - mean_demand for city, demand in city_demands.items()}

    surplus_cities  = {c: v         for c, v in surplus_deficit.items() if v > 0}
    deficit_cities  = {c: abs(v)    for c, v in surplus_deficit.items() if v < 0}

    if not surplus_cities or not deficit_cities:
        return []   # All balanced, or all one-sided — no internal transfer possible

    surplus_rem = dict(sorted(surplus_cities.items(), key=lambda x: x[1], reverse=True))
    deficit_rem = dict(sorted(deficit_cities.items(), key=lambda x: x[1], reverse=True))

    transfers = []
    for src in list(surplus_rem):
        if surplus_rem[src] <= 0:
            continue
        for dst in list(deficit_rem):
            if deficit_rem[dst] <= 0:
                continue

            qty = min(surplus_rem[src], deficit_rem[dst])
            if qty > 0.5:
                transfers.append({
                    "Product_ID":            product_id,
                    "Month":                 month,
                    "Source_City":           src,
                    "Destination_City":      dst,
                    "Transfer_Quantity":     round(qty),
                    "Source_Surplus_Before": round(surplus_cities[src]),
                    "Dest_Deficit_Before":   round(deficit_cities[dst]),
                    "Source_Original_Demand": city_demands[src],
                    "Dest_Original_Demand":   city_demands[dst],
                    "Mean_Demand":           round(mean_demand, 2),
                })

            surplus_rem[src] -= qty
            deficit_rem[dst] -= qty

    return transfers


def _print_summary(df: pd.DataFrame):
    print(f"\n   Total pieces to transfer : {df['Transfer_Quantity'].sum():,}")
    print(f"   Avg transfer size        : {df['Transfer_Quantity'].mean():.0f} pieces")
    print(f"   Largest single transfer  : {df['Transfer_Quantity'].max():,} pieces")

    print("\n   Outbound by city:")
    for city, total in df.groupby("Source_City")["Transfer_Quantity"].sum().sort_values(ascending=False).items():
        print(f"      {city}: {total:,} pieces")

    print("\n   Inbound by city:")
    for city, total in df.groupby("Destination_City")["Transfer_Quantity"].sum().sort_values(ascending=False).items():
        print(f"      {city}: {total:,} pieces")


# ── Checkpoint helper ──────────────────────────────────────────────────────

def save_checkpoint(transfers: pd.DataFrame, path: str):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        transfers.to_excel(writer, sheet_name="Detailed_Transfers", index=False)

        # Monthly pivot
        monthly = transfers.groupby(["Product_ID", "Month"])["Transfer_Quantity"].sum().unstack(fill_value=0)
        monthly.to_excel(writer, sheet_name="Monthly_Summary")

        # Route summary
        route_summary = (
            transfers.groupby(["Source_City", "Destination_City"])
            .agg(Transfer_Quantity=("Transfer_Quantity", "sum"),
                 Number_of_Transfers=("Product_ID", "count"))
            .reset_index()
            .sort_values("Transfer_Quantity", ascending=False)
        )
        route_summary.to_excel(writer, sheet_name="Route_Summary", index=False)

    print(f"   💾 Checkpoint saved → {path}")
