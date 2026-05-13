"""
predict.py
----------
Make a price prediction for a single car from the command line.

Usage:
    python src/predict.py \
        --manufacturer ford \
        --model f-150 \
        --year 2018 \
        --odometer 65000 \
        --fuel gas \
        --transmission automatic \
        --drive 4wd \
        --type pickup \
        --cylinders 8 \
        --title_status clean \
        --paint_color white \
        --state tx
"""

import argparse
import warnings

import numpy as np
import pandas as pd

from utils import load_model, predict_price, CAT_COLS

warnings.filterwarnings("ignore")


def build_input(args: argparse.Namespace) -> pd.DataFrame:
    """Build a single-row DataFrame matching the training feature schema."""
    car_age = 2026 - int(args.year)

    row = {
        "manufacturer":  args.manufacturer,
        "model":         args.model,
        "odometer":      float(args.odometer),
        "fuel":          args.fuel,
        "title_status":  args.title_status,
        "transmission":  args.transmission,
        "drive":         args.drive,
        "type":          args.type,
        "cylinders":     str(args.cylinders),
        "paint_color":   args.paint_color,
        "state":         args.state,
        "car_age":       car_age,
    }

    df = pd.DataFrame([row])
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a car's price.")
    parser.add_argument("--model-path",   default="models/best_model.pkl")
    parser.add_argument("--manufacturer", required=True)
    parser.add_argument("--model",        required=True, dest="model")
    parser.add_argument("--year",         required=True, type=int)
    parser.add_argument("--odometer",     required=True, type=float)
    parser.add_argument("--fuel",         default="gas")
    parser.add_argument("--transmission", default="automatic")
    parser.add_argument("--drive",        default="4wd")
    parser.add_argument("--type",         default="sedan")
    parser.add_argument("--cylinders",    default="6")
    parser.add_argument("--title_status", default="clean")
    parser.add_argument("--paint_color",  default="unknown")
    parser.add_argument("--state",        default="ca")
    args = parser.parse_args()

    print(f"\n[load] Loading model from {args.model_path} ...")
    trained_model = load_model(args.model_path)

    X = build_input(args)

    price = predict_price(trained_model, X)[0]

    print("\n" + "─" * 40)
    print(f"  Estimated Price: ${price:,.0f}")
    print("─" * 40 + "\n")


if __name__ == "__main__":
    main()
