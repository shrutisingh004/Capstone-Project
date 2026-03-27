"""
stage3_routes.py — FMCG route optimization (Nearest Neighbour + Greedy Insertion + 2-opt).

Input  : transfers DataFrame (output of stage2_inventory.py)
Output : routes DataFrame + summary report dict
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

import config


# ── Public entry point ─────────────────────────────────────────────────────

def run(transfers: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Optimise delivery routes from inventory transfer plan.

    Parameters
    ----------
    transfers : DataFrame from stage2_inventory.run()

    Returns
    -------
    routes_df : DataFrame of optimised routes
    report    : dict of KPI metrics + recommendations
    """
    print("\n" + "=" * 70)
    print("STAGE 3 — Route Optimisation")
    print("=" * 70)

    optimizer = FMCGRouteOptimizer(config.TRUCK_CONFIG)
    distances  = optimizer.calculate_distances(config.CITY_COORDS)

    consolidated = optimizer.consolidate_shipments(transfers, period="monthly")
    routes_df    = optimizer.optimize_routes(consolidated, distances)
    report       = optimizer.generate_report(routes_df)

    _print_report(report)
    print(f"\n✅ Stage 3 complete — {len(routes_df)} routes generated")
    return routes_df, report


# ── FMCGRouteOptimizer class ───────────────────────────────────────────────

class FMCGRouteOptimizer:
    def __init__(self, cfg: Dict):
        self.config = cfg

    # ── Distance calculation ───────────────────────────────────────────────

    def calculate_distances(self, city_coords: Dict) -> Dict:
        distances = {}
        cities = list(city_coords.keys())
        for c1 in cities:
            for c2 in cities:
                if c1 != c2:
                    distances[(c1, c2)] = self._haversine(city_coords[c1], city_coords[c2])
        return distances

    def _haversine(self, coord1: Tuple, coord2: Tuple) -> float:
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        R = 6371
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(a))

    # ── Consolidation ─────────────────────────────────────────────────────

    def consolidate_shipments(self, df: pd.DataFrame, period: str = "monthly") -> pd.DataFrame:
        df = df.copy()
        if period == "weekly":
            df["Period"] = ((df["Month"] - 1) // 1).astype(int) + 1
        elif period == "monthly":
            df["Period"] = df["Month"]
        else:
            df["Period"] = 1

        consolidated = (
            df.groupby(["Period", "Source_City", "Destination_City"])
            .agg(
                Transfer_Quantity=("Transfer_Quantity", "sum"),
                Product_ID=("Product_ID", lambda x: ", ".join(x.astype(str).unique())),
            )
            .reset_index()
        )

        print(f"   Consolidation: {len(df)} → {len(consolidated)} shipments "
              f"({len(df) / max(len(consolidated), 1):.1f}x reduction)")
        return consolidated

    # ── Route optimisation ─────────────────────────────────────────────────

    def optimize_routes(self, shipments: pd.DataFrame, distances: Dict) -> pd.DataFrame:
        all_routes = []

        for (period, source), group in shipments.groupby(["Period", "Source_City"]):
            destinations = [
                {"city": row["Destination_City"], "quantity": row["Transfer_Quantity"], "products": row["Product_ID"]}
                for _, row in group.iterrows()
            ]

            routes_nn      = self._nearest_neighbor_tsp(source, destinations, distances)
            routes_greedy  = self._greedy_insertion(source, destinations, distances)

            cost_nn      = self._calculate_total_cost(routes_nn,     source, distances)
            cost_greedy  = self._calculate_total_cost(routes_greedy, source, distances)

            best_routes  = routes_nn     if cost_nn <= cost_greedy else routes_greedy
            algorithm    = "Nearest Neighbor" if cost_nn <= cost_greedy else "Greedy Insertion"

            for route_num, route in enumerate(best_routes, 1):
                all_routes.append(self._format_route(route, source, period, route_num, algorithm, distances))

        return pd.DataFrame(all_routes)

    # ── TSP heuristics ─────────────────────────────────────────────────────

    def _nearest_neighbor_tsp(self, source: str, destinations: List[Dict], distances: Dict) -> List[List[Dict]]:
        routes    = []
        remaining = destinations.copy()

        while remaining:
            route, current, current_load = [], source, 0

            while remaining:
                nearest, min_dist = None, float("inf")
                for dest in remaining:
                    if current_load + dest["quantity"] <= self.config["capacity"]:
                        d = distances.get((current, dest["city"]), float("inf"))
                        if d < min_dist:
                            min_dist, nearest = d, dest
                if nearest is None:
                    break
                route.append(nearest)
                remaining.remove(nearest)
                current       = nearest["city"]
                current_load += nearest["quantity"]

            if route:
                routes.append(self._two_opt(route, source, distances))

        return routes

    def _greedy_insertion(self, source: str, destinations: List[Dict], distances: Dict) -> List[List[Dict]]:
        routes    = []
        remaining = destinations.copy()

        while remaining:
            route        = [max(remaining, key=lambda d: distances.get((source, d["city"]), 0))]
            remaining.remove(route[0])
            current_load = route[0]["quantity"]

            while remaining:
                best_dest, best_pos, best_cost = None, None, float("inf")
                for dest in remaining:
                    if current_load + dest["quantity"] > self.config["capacity"]:
                        continue
                    for pos in range(len(route) + 1):
                        cost = self._insertion_cost(route, dest, pos, source, distances)
                        if cost < best_cost:
                            best_cost, best_dest, best_pos = cost, dest, pos
                if best_dest is None:
                    break
                route.insert(best_pos, best_dest)
                remaining.remove(best_dest)
                current_load += best_dest["quantity"]

            routes.append(route)

        return routes

    def _insertion_cost(self, route, dest, position, source, distances):
        temp = route[:position] + [dest] + route[position:]
        return self._route_distance(temp, source, distances)

    def _two_opt(self, route: List[Dict], source: str, distances: Dict) -> List[Dict]:
        if len(route) < 3:
            return route
        best, improved, iterations = route.copy(), True, 0
        while improved and iterations < 100:
            improved, iterations = False, iterations + 1
            for i in range(len(best) - 1):
                for j in range(i + 2, len(best)):
                    candidate = best[: i + 1] + best[i + 1: j][::-1] + best[j:]
                    if self._route_distance(candidate, source, distances) < self._route_distance(best, source, distances):
                        best, improved = candidate, True
                        break
                if improved:
                    break
        return best

    # ── Distance / cost helpers ────────────────────────────────────────────

    def _route_distance(self, route: List[Dict], source: str, distances: Dict) -> float:
        if not route:
            return 0
        total  = distances.get((source, route[0]["city"]), 0)
        total += sum(distances.get((route[i]["city"], route[i + 1]["city"]), 0) for i in range(len(route) - 1))
        total += distances.get((route[-1]["city"], source), 0)
        return total

    def _calculate_total_cost(self, routes: List[List[Dict]], source: str, distances: Dict) -> float:
        return sum(
            self._route_distance(r, source, distances) * self.config["cost_per_km"] + self.config["fixed_cost"]
            for r in routes
        )

    def _format_route(self, route: List[Dict], source: str, period: int,
                      route_num: int, algorithm: str, distances: Dict) -> Dict:
        load      = sum(d["quantity"] for d in route)
        distance  = self._route_distance(route, source, distances)
        travel_h  = distance / self.config["speed"]
        total_h   = travel_h + len(route) * self.config["loading_time"]
        var_cost  = distance * self.config["cost_per_km"]
        total_cost = var_cost + self.config["fixed_cost"]

        return {
            "Period":           period,
            "Source":           source,
            "Route_Number":     route_num,
            "Algorithm":        algorithm,
            "Stops":            " → ".join([source] + [d["city"] for d in route] + [source]),
            "Destinations":     ", ".join(d["city"] for d in route),
            "Num_Stops":        len(route),
            "Total_Load":       load,
            "Distance_KM":      round(distance, 1),
            "Travel_Time_H":    round(travel_h, 1),
            "Total_Time_H":     round(total_h, 1),
            "Variable_Cost":    round(var_cost, 2),
            "Fixed_Cost":       self.config["fixed_cost"],
            "Total_Cost":       round(total_cost, 2),
            "Capacity_Util_%":  round(load / self.config["capacity"] * 100, 1),
            "Feasible":         total_h <= self.config["max_hours"],
            "Cost_Per_KM":      round(total_cost / distance, 2) if distance > 0 else 0,
            "Cost_Per_Piece":   round(total_cost / load, 2)     if load > 0     else 0,
        }

    # ── Reporting ──────────────────────────────────────────────────────────

    def generate_report(self, routes_df: pd.DataFrame) -> Dict:
        td = routes_df["Distance_KM"].sum()
        tc = routes_df["Total_Cost"].sum()
        tl = routes_df["Total_Load"].sum()

        report = {
            "total_routes":        len(routes_df),
            "total_distance_km":   td,
            "total_cost":          tc,
            "total_load":          tl,
            "avg_capacity_util":   routes_df["Capacity_Util_%"].mean(),
            "cost_per_km":         tc / td if td > 0 else 0,
            "cost_per_piece":      tc / tl if tl > 0 else 0,
            "feasible_routes":     (routes_df["Feasible"] == True).sum(),
            "infeasible_routes":   (routes_df["Feasible"] == False).sum(),
            "multi_stop_routes":   (routes_df["Num_Stops"] > 1).sum(),
            "recommendations":     [],
        }

        # Recommendations
        if report["avg_capacity_util"] < 60:
            under = (routes_df["Capacity_Util_%"] < 50).sum()
            report["recommendations"].append({
                "priority":       "HIGH",
                "category":       "Capacity",
                "issue":          f"Low avg capacity utilisation ({report['avg_capacity_util']:.1f}%)",
                "recommendation": f"Use smaller vehicles or consolidate further. {under} routes under 50%.",
                "impact":         f"Potential savings: ₹{under * self.config['fixed_cost'] * 0.3:,.0f}",
            })

        if report["infeasible_routes"] > 0:
            report["recommendations"].append({
                "priority":       "MEDIUM",
                "category":       "Operations",
                "issue":          f"{report['infeasible_routes']} routes exceed time limit",
                "recommendation": "Implement multi-day scheduling or relay drivers",
                "impact":         "Improved compliance & driver satisfaction",
            })

        q75 = routes_df["Cost_Per_Piece"].quantile(0.75)
        high_cost_n = (routes_df["Cost_Per_Piece"] > q75).sum()
        if high_cost_n > 0:
            report["recommendations"].append({
                "priority":       "MEDIUM",
                "category":       "Cost",
                "issue":          f"{high_cost_n} routes have high cost-per-piece",
                "recommendation": "Consider 3PL for these lanes",
                "impact":         "Potential 40–60% cost reduction on these routes",
            })

        return report


# ── Checkpoint helper ──────────────────────────────────────────────────────

def save_checkpoint(routes_df: pd.DataFrame, report: dict, path: str):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        routes_df.to_excel(writer, sheet_name="Routes", index=False)

        summary = pd.DataFrame({
            "Metric": [
                "Total Routes", "Total Distance (km)", "Total Cost (₹)",
                "Total Load (pieces)", "Avg Capacity Util (%)",
                "Cost per KM (₹)", "Cost per Piece (₹)",
                "Feasible Routes", "Multi-stop Routes",
            ],
            "Value": [
                report["total_routes"],
                f"{report['total_distance_km']:,.0f}",
                f"{report['total_cost']:,.2f}",
                f"{report['total_load']:,.0f}",
                f"{report['avg_capacity_util']:.1f}",
                f"{report['cost_per_km']:.2f}",
                f"{report['cost_per_piece']:.2f}",
                report["feasible_routes"],
                report["multi_stop_routes"],
            ],
        })
        summary.to_excel(writer, sheet_name="Summary", index=False)

        if report["recommendations"]:
            pd.DataFrame(report["recommendations"]).to_excel(
                writer, sheet_name="Recommendations", index=False
            )

    print(f"   💾 Final output saved → {path}")


def _print_report(report: dict):
    print(f"\n   Total routes         : {report['total_routes']}")
    print(f"   Total distance       : {report['total_distance_km']:,.0f} km")
    print(f"   Total cost           : ₹{report['total_cost']:,.2f}")
    print(f"   Cost per piece       : ₹{report['cost_per_piece']:.2f}")
    print(f"   Avg capacity util    : {report['avg_capacity_util']:.1f}%")
    print(f"   Feasible routes      : {report['feasible_routes']} / {report['total_routes']}")

    if report["recommendations"]:
        print("\n   💡 Recommendations:")
        for r in report["recommendations"]:
            print(f"      [{r['priority']}] {r['category']} — {r['issue']}")
            print(f"         → {r['recommendation']}")
