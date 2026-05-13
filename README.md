# 🚗 Car Price Prediction

A machine-learning pipeline that predicts used-car prices from Craigslist listings.  
It compares **XGBoost** and **LightGBM** models (with log-transform on the target) and saves the best one for inference.

---

## Repository Structure

```
car-price-prediction/
│
├── src/
│   ├── preprocess.py   # Raw → cleaned CSV
│   ├── train.py        # Train XGBoost / LightGBM, save best model
│   ├── evaluate.py     # Metrics report + optional plots
│   ├── predict.py      # CLI single-car prediction
│   └── utils.py        # Shared helpers (load, save, feature prep)
│
├── models/
│   └── best_model.pkl  # Saved after training (git-ignored)
│
├── demo/
│   └── demo.py         # Interactive terminal demo
│
├── data/
│   ├── raw/            # Place vehicles.csv here (git-ignored)
│   └── processed/      # cleaned_vehicles.csv generated here
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Quick Start

### 1 – Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

### 2 – Get the data

Download the Kaggle dataset **"vehicles-csv"** by `mdnizamsapiee` and place `vehicles.csv` in `data/raw/`:

```bash
# Automatic download (requires kaggle credentials)
python - <<'EOF'
import kagglehub, shutil, os
path = kagglehub.dataset_download("mdnizamsapiee/vehicles-csv")
for f in os.listdir(path):
    if f.endswith(".csv"):
        shutil.copy(os.path.join(path, f), "data/raw/vehicles.csv")
        print("Copied →", f)
EOF
```

### 3 – Preprocess

```bash
python src/preprocess.py
# → data/processed/cleaned_vehicles.csv
```

### 4 – Train

```bash
python src/train.py                       # trains both, saves best
python src/train.py --model-type lgbm    # LightGBM only
python src/train.py --model-type xgb     # XGBoost only
```

### 5 – Evaluate

```bash
python src/evaluate.py
python src/evaluate.py --plots           # also show charts
```

### 6 – Predict a single car

```bash
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
    --state tx
```

### 7 – Interactive demo

```bash
python demo/demo.py
```

---

## Models

| Model | Key hyperparameters | Target |
|-------|---------------------|--------|
| LightGBM | `n_estimators=10 000`, `lr=0.01`, `num_leaves=124`, early stopping | `log1p(price)` |
| XGBoost | `n_estimators=1 000`, `max_depth=6`, `lr=0.05`, early stopping | `log1p(price)` |

Predictions are back-transformed with `np.expm1`.

---

## Preprocessing Summary

| Step | Action |
|------|--------|
| Drop columns | id, url, vin, image_url, description, county, size, condition, … |
| Cylinders | Extract digits, impute via group mode (`manufacturer × type × fuel × drive`) |
| Manufacturer | Fix typos, group rare brands (<100 listings) → `other` |
| Model | Normalize F-series trucks, keep first token, group rare (<150) → `other` |
| Price | Keep $2 000 – $85 000 |
| Year | Impute with median, keep ≥ 1995, engineer `car_age = 2026 − year` |
| Odometer | Drop > 350 000 miles |
| Categoricals | Fill NaN with `unknown` / `other` |

---

## Requirements

- Python 3.10+
- See `requirements.txt` for package versions
