"""
train.py
--------
Trains both XGBoost and LightGBM models, then saves the better one
(by test-set R²) to models/best_model.pkl.

Usage:
    python src/train.py --data data/processed/cleaned_vehicles.csv \
                        --model-out models/best_model.pkl \
                        --model-type lgbm          # or xgb / both (default: both)
"""

import argparse
import warnings

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from utils import load_processed, prepare_features, save_model

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Trainer functions
# ---------------------------------------------------------------------------

def train_lightgbm(X_train, X_test, y_train, y_test):
    print("\n── Training LightGBM ──────────────────────────────────")
    model = lgb.LGBMRegressor(
        n_estimators=10_000,
        learning_rate=0.01,
        num_leaves=124,
        importance_type="gain",
        random_state=42,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(500),
        ],
    )
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    print(f"   LightGBM test R² (log scale): {r2:.4f}")
    return model, r2


def train_xgboost(X_train, X_test, y_train, y_test):
    print("\n── Training XGBoost ───────────────────────────────────")
    # XGBoost needs numeric-only; encode categoricals
    X_tr = X_train.copy()
    X_te = X_test.copy()
    for col in X_tr.select_dtypes(include="category").columns:
        X_tr[col] = X_tr[col].cat.codes
        X_te[col] = X_te[col].cat.codes

    model = xgb.XGBRegressor(
        n_estimators=1_000,
        max_depth=6,
        learning_rate=0.05,
        early_stopping_rounds=30,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        verbosity=0,
    )
    model.fit(
        X_tr, y_train,
        eval_set=[(X_te, y_test)],
        verbose=False,
    )
    preds = model.predict(X_te)
    r2 = r2_score(y_test, preds)
    print(f"   XGBoost test R² (log scale): {r2:.4f}")
    return model, r2


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_path: str, model_out: str, model_type: str) -> None:
    print(f"[load] Reading {data_path} ...")
    df = load_processed(data_path)

    # Extra cleaning applied in the LightGBM notebook (price & model filter)
    df = df[(df["price"] > 2_000) & (df["price"] < 85_000)]
    df = df[df["model"] != "other"]
    print(f"[load] {len(df):,} rows after noise removal")

    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}

    if model_type in ("lgbm", "both"):
        lgbm_model, lgbm_r2 = train_lightgbm(X_train, X_test, y_train, y_test)
        results["lgbm"] = (lgbm_model, lgbm_r2)

    if model_type in ("xgb", "both"):
        xgb_model, xgb_r2 = train_xgboost(X_train, X_test, y_train, y_test)
        results["xgb"] = (xgb_model, xgb_r2)

    # Pick the best model
    best_name = max(results, key=lambda k: results[k][1])
    best_model, best_r2 = results[best_name]

    print(f"\n🏆 Best model: {best_name.upper()}  (R² = {best_r2:.4f})")
    save_model(best_model, model_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train car-price models.")
    parser.add_argument("--data",       default="data/processed/cleaned_vehicles.csv")
    parser.add_argument("--model-out",  default="models/best_model.pkl")
    parser.add_argument("--model-type", default="both",
                        choices=["lgbm", "xgb", "both"],
                        help="Which model(s) to train (default: both, saves best)")
    args = parser.parse_args()
    main(args.data, args.model_out, args.model_type)
