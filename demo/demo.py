"""
demo/demo.py
------------
Interactive command-line demo.  Loads the best saved model and lets the
user type in car details to get an instant price estimate.

Usage:
    python demo/demo.py
    python demo/demo.py --model models/best_model.pkl
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from repo root or from demo/ directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils import load_model, predict_price, CAT_COLS

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIVIDER = "─" * 50

def prompt(label: str, default: str) -> str:
    value = input(f"  {label} [{default}]: ").strip()
    return value if value else default


def collect_car_info() -> dict:
    print(f"\n{DIVIDER}")
    print("  Enter car details (press Enter to use default)")
    print(DIVIDER)
    manufacturer = prompt("Manufacturer (e.g. ford, toyota)", "ford")
    model        = prompt("Model        (e.g. f-150, camry)",  "f-150")
    year         = int(prompt("Year         (e.g. 2018)",       "2018"))
    odometer     = float(prompt("Odometer     (miles, e.g. 60000)", "60000"))
    fuel         = prompt("Fuel         (gas/diesel/electric/hybrid/other)", "gas")
    transmission = prompt("Transmission (automatic/manual/other)", "automatic")
    drive        = prompt("Drive        (4wd/fwd/rwd/unknown)", "4wd")
    vtype        = prompt("Type         (sedan/SUV/truck/pickup/van/…)", "sedan")
    cylinders    = prompt("Cylinders    (4/6/8/…)", "6")
    title_status = prompt("Title status (clean/rebuilt/salvage/…)", "clean")
    paint_color  = prompt("Paint color  (white/black/silver/…)", "unknown")
    state        = prompt("State        (2-letter, e.g. ca)", "ca")

    return dict(
        manufacturer=manufacturer,
        model=model,
        odometer=odometer,
        fuel=fuel,
        transmission=transmission,
        drive=drive,
        type=vtype,
        cylinders=str(cylinders),
        title_status=title_status,
        paint_color=paint_color,
        state=state,
        car_age=2026 - year,
    )


def build_dataframe(info: dict) -> pd.DataFrame:
    df = pd.DataFrame([info])
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(model_path: str) -> None:
    print("\n🚗  Car Price Predictor — Interactive Demo")
    print(f"    Model: {model_path}\n")

    try:
        model = load_model(model_path)
    except FileNotFoundError:
        print(f"❌  Model file not found: {model_path}")
        print("    Run `python src/train.py` first to generate a model.")
        sys.exit(1)

    while True:
        info = collect_car_info()
        X    = build_dataframe(info)

        price = predict_price(model, X)[0]

        print(f"\n{DIVIDER}")
        print(f"  📊 Estimated Price: ${price:,.0f}")
        print(DIVIDER)

        again = input("\n  Predict another car? (y/n) [y]: ").strip().lower()
        if again in ("n", "no"):
            print("\n  Goodbye! 👋\n")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive car-price demo.")
    parser.add_argument("--model", default="models/best_model.pkl",
                        help="Path to trained model pickle")
    args = parser.parse_args()
    main(args.model)
