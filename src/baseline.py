"""
baseline.py
-----------
Trains Random Forest and a shallow Neural Network (MLP) baseline models,
then saves the better one to models/best_baseline.pkl.

Usage:
    python src/baseline.py --data data/processed/cleaned_vehicles.csv \
                           --model-out models/best_baseline.pkl \
                           --model-type rf          # or mlp / both (default: both)
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder

from utils import load_processed, prepare_features, save_model

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Trainer functions
# ---------------------------------------------------------------------------

def train_random_forest(X_train, X_test, y_train, y_test):
    print("\n── Training Random Forest ─────────────────────────────")
    cat_cols = X_train.select_dtypes(include=["category", "object"]).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=["category", "object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("rf", RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    print(f"   Random Forest test R² (log scale): {r2:.4f}")
    return model, r2


def train_mlp(X_train, X_test, y_train, y_test):
    print("\n── Training Shallow Neural Network ────────────────────")
    cat_cols = X_train.select_dtypes(include=["category", "object"]).columns.tolist()
    num_cols = X_train.select_dtypes(exclude=["category", "object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("mlp", MLPRegressor(hidden_layer_sizes=(64,), max_iter=200, random_state=42, early_stopping=True))
    ])

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    print(f"   MLP test R² (log scale): {r2:.4f}")
    return model, r2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_path: str, model_out: str, model_type: str) -> None:
    print(f"[load] Reading {data_path} ...")
    df = load_processed(data_path)

    # Extra cleaning applied in the original notebooks (price & model filter)
    df = df[(df["price"] > 2_000) & (df["price"] < 85_000)]
    df = df[df["model"] != "other"]
    print(f"[load] {len(df):,} rows after noise removal")

    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    if model_type in ("rf", "both"):
        rf_model, rf_r2 = train_random_forest(X_train, X_test, y_train, y_test)
        results["rf"] = (rf_model, rf_r2)

    if model_type in ("mlp", "both"):
        mlp_model, mlp_r2 = train_mlp(X_train, X_test, y_train, y_test)
        results["mlp"] = (mlp_model, mlp_r2)

    # Pick the best model
    best_name = max(results, key=lambda k: results[k][1])
    best_model, best_r2 = results[best_name]

    print(f"\n🏆 Best baseline model: {best_name.upper()}  (R² = {best_r2:.4f})")
    save_model(best_model, model_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train baseline car-price models.")
    parser.add_argument("--data",       default="data/processed/cleaned_vehicles.csv")
    parser.add_argument("--model-out",  default="models/best_baseline.pkl")
    parser.add_argument("--model-type", default="both",
                        choices=["rf", "mlp", "both"],
                        help="Which baseline model(s) to train (default: both, saves best)")
    args = parser.parse_args()
    main(args.data, args.model_out, args.model_type)
