"""
evaluate.py
-----------
Loads a saved model and a processed dataset, runs evaluation, and prints
a full metrics report plus optional plots.

Usage:
    python src/evaluate.py --data  data/processed/cleaned_vehicles.csv \
                            --model models/best_model.pkl \
                            --plots            # optional: show matplotlib charts
"""

import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from utils import load_processed, prepare_features, load_model, predict_price

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae  = mean_absolute_error(actual, predicted)
    mse  = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2   = r2_score(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    return dict(MAE=mae, MSE=mse, RMSE=rmse, R2=r2, MAPE=mape)


def print_report(metrics: dict, model_label: str = "Model") -> None:
    print("\n" + "=" * 45)
    print(f"  Evaluation Report — {model_label}")
    print("=" * 45)
    print(f"  MAE  : ${metrics['MAE']:>12,.2f}")
    print(f"  MSE  : {metrics['MSE']:>15,.2f}")
    print(f"  RMSE : ${metrics['RMSE']:>12,.2f}")
    print(f"  R²   : {metrics['R2']:>15.4f}")
    print(f"  MAPE : {metrics['MAPE']:>14.2f}%")
    print("=" * 45)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_results(actual: np.ndarray, predicted: np.ndarray,
                 metrics: dict) -> None:
    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig)

    # 1 – Predicted vs Actual
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(actual, predicted, alpha=0.3, s=10, color="steelblue")
    lim = [min(actual.min(), predicted.min()), max(actual.max(), predicted.max())]
    ax1.plot(lim, lim, "r--", label="Perfect prediction")
    ax1.set_xlabel("Actual Price ($)")
    ax1.set_ylabel("Predicted Price ($)")
    ax1.set_title(f"Predicted vs Actual  (R²={metrics['R2']:.3f})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2 – Residuals
    residuals = actual - predicted
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(predicted, residuals, alpha=0.3, s=10, color="coral")
    ax2.axhline(0, color="red", linestyle="--")
    ax2.set_xlabel("Predicted Price ($)")
    ax2.set_ylabel("Residual ($)")
    ax2.set_title("Residual Plot")
    ax2.grid(True, alpha=0.3)

    # 3 – Error distribution
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.hist(residuals, bins=60, color="mediumseagreen", edgecolor="white")
    ax3.set_xlabel("Prediction Error ($)")
    ax3.set_ylabel("Count")
    ax3.set_title("Distribution of Prediction Errors")
    ax3.grid(True, alpha=0.3)

    # 4 – Metrics summary text
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    summary = (
        f"MAE   : ${metrics['MAE']:,.0f}\n"
        f"RMSE  : ${metrics['RMSE']:,.0f}\n"
        f"R²    : {metrics['R2']:.4f}\n"
        f"MAPE  : {metrics['MAPE']:.2f}%"
    )
    ax4.text(0.5, 0.5, summary, transform=ax4.transAxes,
             fontsize=18, va="center", ha="center",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="lightyellow",
                       edgecolor="steelblue", linewidth=2))
    ax4.set_title("Metrics Summary", fontsize=14)

    plt.suptitle("Car Price Prediction — Evaluation", fontsize=16, y=1.01)
    plt.tight_layout()
    plt.show()


def worst_predictions(X_test: pd.DataFrame,
                      actual: np.ndarray,
                      predicted: np.ndarray,
                      n: int = 10) -> pd.DataFrame:
    results = X_test.copy()
    results["actual_price"]    = actual
    results["predicted_price"] = predicted
    results["abs_error"]       = np.abs(actual - predicted)
    results["pct_error"]       = results["abs_error"] / actual * 100
    return results.sort_values("abs_error", ascending=False).head(n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_path: str, model_path: str, show_plots: bool) -> None:
    print(f"[load] Reading data from {data_path} ...")
    df = load_processed(data_path)
    df = df[(df["price"] > 2_000) & (df["price"] < 85_000)]
    df = df[df["model"] != "other"]

    X, y_log = prepare_features(df)
    _, X_test, _, y_log_test = train_test_split(X, y_log, test_size=0.2, random_state=42)

    print(f"[load] Loading model from {model_path} ...")
    model = load_model(model_path)

    print("[eval] Running predictions ...")
    actual    = np.expm1(y_log_test.values)
    predicted = predict_price(model, X_test)

    metrics = compute_metrics(actual, predicted)
    print_report(metrics, model_label=type(model).__name__)

    print("\n── Top 10 Worst Predictions (by absolute error) ──")
    worst = worst_predictions(X_test, actual, predicted)
    print(worst[["actual_price", "predicted_price", "abs_error", "pct_error"]].to_string(index=False))

    if show_plots:
        plot_results(actual, predicted, metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a saved car-price model.")
    parser.add_argument("--data",   default="data/processed/cleaned_vehicles.csv")
    parser.add_argument("--model",  default="models/best_model.pkl")
    parser.add_argument("--plots",  action="store_true",
                        help="Show matplotlib evaluation plots")
    args = parser.parse_args()
    main(args.data, args.model, args.plots)
