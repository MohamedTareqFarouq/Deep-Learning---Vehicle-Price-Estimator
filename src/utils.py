"""
utils.py
--------
Shared helper functions used across training, evaluation, and prediction.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Categorical columns used by both models
# ---------------------------------------------------------------------------

CAT_COLS = [
    "manufacturer", "model", "cylinders", "fuel",
    "title_status", "transmission", "drive", "type",
    "paint_color", "state",
]

FEATURE_COLS_DROP = ["price", "year"]   # columns to exclude when building X


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_processed(path: str = "data/processed/cleaned_vehicles.csv") -> pd.DataFrame:
    """Load the preprocessed CSV and cast categorical columns."""
    df = pd.read_csv(path)
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) dropping price/year; applies log1p to y."""
    X = df.drop(columns=[c for c in FEATURE_COLS_DROP if c in df.columns])
    y = np.log1p(df["price"])
    return X, y


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model, path: str = "models/best_model.pkl") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved → {path}")


def load_model(path: str = "models/best_model.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def predict_price(model, X: pd.DataFrame) -> np.ndarray:
    """Run model inference and reverse the log1p transform."""
    log_preds = model.predict(X)
    return np.expm1(log_preds)
