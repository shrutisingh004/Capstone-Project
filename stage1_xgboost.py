"""
stage1_xgboost.py — Sales forecasting via XGBoost.

Input  : raw sales DataFrame (from sales_input.csv)
Output : pivot DataFrame  →  rows = (Product_ID, Month)
                              cols = city/region names
                              values = predicted sales (int, pieces)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config


# ── Public entry point ─────────────────────────────────────────────────────

def run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Train XGBoost on df, return Product × Month × Region prediction matrix.

    Parameters
    ----------
    df : raw sales DataFrame (must contain columns defined in sales_input.csv)

    Returns
    -------
    pd.DataFrame  — pivot table of predicted sales, integer pieces
                    index = (Product_ID, Month), columns = regions
    """
    print("\n" + "=" * 70)
    print("STAGE 1 — XGBoost Sales Forecasting")
    print("=" * 70)

    df = _preprocess(df)
    X, y, product_ids = _build_features(df)
    model, metrics = _train(X, y)
    predictions = _predict(model, df, X, product_ids)
    matrix = _build_matrix(predictions)

    _print_metrics(metrics)
    print(f"\n✅ Stage 1 complete — matrix shape: {matrix.shape}")
    return matrix


# ── Internal helpers ───────────────────────────────────────────────────────

def _preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values(["Product_ID", "Region", "Date"]).reset_index(drop=True)

    df["Year"]      = df["Date"].dt.year
    df["Month"]     = df["Date"].dt.month
    df["Day"]       = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.weekday

    df = df.drop(columns=["Date", "Sales_Value"], errors="ignore")
    return df


def _build_features(df: pd.DataFrame):
    target_col  = "Sales_Quantity"
    product_ids = df["Product_ID"].values

    df_enc = pd.get_dummies(
        df.drop(columns=["Product_ID"]),
        columns=["Region"],
        drop_first=True,
    )

    X = df_enc.drop(columns=[target_col])
    y = df_enc[target_col].astype(float)
    return X, y, product_ids


def _train(X, y):
    n          = len(X)
    train_end  = int(n * config.TRAIN_RATIO)
    val_end    = train_end + int(n * config.VAL_RATIO)

    X_train, y_train = X.iloc[:train_end],   y.iloc[:train_end]
    X_val,   y_val   = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test,  y_test  = X.iloc[val_end:],     y.iloc[val_end:]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval   = xgb.DMatrix(X_val,   label=y_val)
    dtest  = xgb.DMatrix(X_test,  label=y_test)

    model = xgb.train(
        params             = config.XGBOOST_PARAMS,
        dtrain             = dtrain,
        num_boost_round    = config.XGBOOST_ROUNDS,
        evals              = [(dtrain, "train"), (dval, "validation")],
        early_stopping_rounds = config.XGBOOST_EARLY_STOPPING,
        verbose_eval       = False,
    )

    def _metrics(y_true, y_pred):
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
        return {
            "MAE":       mean_absolute_error(y_true, y_pred),
            "RMSE":      np.sqrt(mean_squared_error(y_true, y_pred)),
            "MAPE (%)":  mape,
            "Accuracy (%)": 100 - mape,
            "R²":        r2_score(y_true, y_pred),
        }

    best_iter = model.best_iteration + 1
    metrics = {
        "Train":      _metrics(y_train.values, model.predict(dtrain,  iteration_range=(0, best_iter))),
        "Validation": _metrics(y_val.values,   model.predict(dval,    iteration_range=(0, best_iter))),
        "Test":       _metrics(y_test.values,  model.predict(dtest,   iteration_range=(0, best_iter))),
    }

    return model, metrics


def _predict(model, df: pd.DataFrame, X, product_ids) -> pd.DataFrame:
    df = df.copy()
    df["Product_ID"] = product_ids
    df["Predicted_Sales"] = model.predict(
        xgb.DMatrix(X),
        iteration_range=(0, model.best_iteration + 1),
    )
    return df


def _build_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # Reconstruct region column from one-hot columns
    region_cols = [c for c in df.columns if c.startswith("Region_")]
    if "Region" not in df.columns and region_cols:
        # Reverse the get_dummies (we kept the original df with Region intact via product_ids trick)
        pass  # Region was dropped — re-read from raw data isn't needed; use pivot on available cols

    matrix = pd.pivot_table(
        df,
        values  = "Predicted_Sales",
        index   = ["Product_ID", "Month"],
        columns = "Region",
        aggfunc = "sum",
    ).round(0).astype(int)

    matrix.columns.name = None
    matrix = matrix.reset_index()
    return matrix


def _print_metrics(metrics: dict):
    rows = []
    for split, m in metrics.items():
        rows.append({"Dataset": split, **m})
    print("\n📊 Model Performance:")
    print(pd.DataFrame(rows).to_string(index=False))


# ── Checkpoint helper (called by pipeline.py) ─────────────────────────────

def save_checkpoint(matrix: pd.DataFrame, path: str):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    matrix.to_excel(path, index=False)
    print(f"   💾 Checkpoint saved → {path}")
