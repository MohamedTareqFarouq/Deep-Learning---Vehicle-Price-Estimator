"""
preprocess.py
-------------
Loads raw vehicles.csv, cleans and engineers features,
and saves the processed dataset to data/processed/cleaned_vehicles.csv.

Usage:
    python src/preprocess.py --input data/raw/vehicles.csv \
                              --output data/processed/cleaned_vehicles.csv
"""

import argparse
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COLUMNS_TO_DROP = [
    "id", "url", "region_url", "vin", "image_url",
    "region", "lat", "long", "description", "county", "size", "condition",
]

MANUFACTURER_FIXES = {
    "porche": "porsche",
    "rover": "land rover",
    "alfa-romeo": "alfa romeo",
    "aston-martin": "aston martin",
}

MANUFACTURER_THRESHOLD = 100
MODEL_THRESHOLD = 150

PRICE_MIN = 2_000
PRICE_MAX = 85_000
YEAR_MIN = 1995
ODOMETER_MAX = 350_000

GROUP_COLS = ["manufacturer", "type", "fuel", "drive"]


# ---------------------------------------------------------------------------
# Step helpers
# ---------------------------------------------------------------------------

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=COLUMNS_TO_DROP, errors="ignore")


def clean_cylinders(df: pd.DataFrame) -> pd.DataFrame:
    df["cylinders"] = (
        df["cylinders"]
        .str.extract(r"(\d+)")
        .astype("Int64")
    )
    return df


def clean_manufacturer(df: pd.DataFrame) -> pd.DataFrame:
    df["manufacturer"] = df["manufacturer"].replace(MANUFACTURER_FIXES)
    counts = df["manufacturer"].value_counts()
    rare = counts[counts < MANUFACTURER_THRESHOLD].index
    df["manufacturer"] = df["manufacturer"].replace(rare, "other")
    df = df.dropna(subset=["manufacturer"])
    return df


def _clean_model_name(name: str) -> str:
    if pd.isna(name):
        return "other"
    name = str(name).lower().strip()
    if re.match(r"^f\s?-?150", name):
        return "f-150"
    if re.match(r"^f\s?-?250", name):
        return "f-250"
    if re.match(r"^f\s?-?350", name):
        return "f-350"
    if "town" in name and "country" in name:
        return "town and country"
    return name.split()[0]


def clean_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["model"])
    df["model"] = df["model"].apply(_clean_model_name)
    counts = df["model"].value_counts()
    rare = counts[counts < MODEL_THRESHOLD].index
    df["model"] = df["model"].replace(rare, "other")
    return df


def filter_price(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["price"] >= PRICE_MIN) & (df["price"] <= PRICE_MAX)].copy()


def clean_year(df: pd.DataFrame) -> pd.DataFrame:
    median_year = df["year"].median()
    df["year"] = df["year"].fillna(median_year).astype(int)
    df = df[df["year"] >= YEAR_MIN]
    return df


def impute_cylinders(df: pd.DataFrame) -> pd.DataFrame:
    """Group-mode imputation for cylinders; electric → 0."""
    df.loc[df["fuel"] == "electric", "cylinders"] = 0

    valid_mask = df[GROUP_COLS].notna().all(axis=1)
    known = df[df["cylinders"].notna() & valid_mask]

    group_modes = (
        known.groupby(GROUP_COLS)["cylinders"]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    )

    def _fill(row):
        if not pd.isna(row["cylinders"]):
            return row["cylinders"]
        if row["fuel"] == "electric":
            return 0
        if row[GROUP_COLS].isna().any():
            return np.nan
        key = tuple(row[col] for col in GROUP_COLS)
        return group_modes.loc[key] if key in group_modes else np.nan

    df["cylinders"] = df.apply(_fill, axis=1)
    return df


def fill_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df["fuel"] = df["fuel"].fillna("other")
    df["paint_color"] = df["paint_color"].fillna("unknown")
    df["transmission"] = df["transmission"].fillna("unknown")
    df["title_status"] = df["title_status"].fillna("unknown")
    df["drive"] = df["drive"].fillna("unknown")
    df["cylinders"] = df["cylinders"].astype(object).fillna("unknown")
    return df


def filter_odometer(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["odometer"] <= ODOMETER_MAX]


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df["car_age"] = 2026 - df["year"]
    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def preprocess(input_path: str, output_path: str) -> pd.DataFrame:
    print(f"[1/10] Loading data from {input_path} ...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"       {df.shape[0]:,} rows × {df.shape[1]} columns")

    print("[2/10] Dropping irrelevant columns ...")
    df = drop_columns(df)

    print("[3/10] Cleaning cylinders ...")
    df = clean_cylinders(df)

    print("[4/10] Cleaning manufacturer ...")
    df = clean_manufacturer(df)

    print("[5/10] Cleaning model ...")
    df = clean_model(df)

    print("[6/10] Filtering price ...")
    df = filter_price(df)

    print("[7/10] Cleaning year ...")
    df = clean_year(df)

    print("[8/10] Imputing cylinders ...")
    df = impute_cylinders(df)

    print("[9/10] Filling remaining categoricals & filtering odometer ...")
    df = fill_categoricals(df)
    df = filter_odometer(df)

    print("[10/10] Feature engineering (car_age) ...")
    df = feature_engineering(df)
    df = df.copy()

    print(f"\n✅ Preprocessing complete. Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    df.to_csv(output_path, index=False)
    print(f"   Saved → {output_path}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw vehicles CSV.")
    parser.add_argument("--input",  default="data/raw/vehicles.csv",
                        help="Path to raw CSV (default: data/raw/vehicles.csv)")
    parser.add_argument("--output", default="data/processed/cleaned_vehicles.csv",
                        help="Where to save cleaned CSV")
    args = parser.parse_args()
    preprocess(args.input, args.output)
