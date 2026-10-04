# DA Price Forecaster — Train / Validation / Test Report

_Generated: 2026-10-03_

## Methodology

Chronological train / validation / test — no shuffling, no future leakage.
Walk-forward CV (4 folds) on the development set selects the model.
The final 12-month held-out test set (never seen during selection) gives the unbiased accuracy below.
The serving forecaster (da_price_forecaster.py) retrains on all history for live daily prediction.

Skill score = 1 − MAE_model / MAE_naive; positive = beats naive 24h persistence.

---

## Data

- Development : 50,135 rows  (2020-01-15 → 2025-10-03)
- Test (held out): 8,761 rows  (2025-10-03 → 2026-10-03)
- Selected model : **RandomForest**

---

## Walk-forward CV — Validation MAE (model selection)

| Model | MAE (EUR/MWh) |
|-------|-----------|
| LightGBM           | 32.72 |
| XGBoost            | 33.30 |
| RandomForest       | 32.27 |

---

## Held-out TEST — Unbiased Accuracy

| Metric | Value |
|--------|-------|
| MAE        | 18.61 EUR/MWh |
| RMSE       | 25.42 EUR/MWh |
| Bias (ME)  | +2.57 EUR/MWh |
| Skill vs naive | +2.6% |

---

## Feature Importance (RandomForest — top 10)

| Feature | Importance |
|---------|-----------|
| lag_24h | 1 |
| roll_mean_168h | 0 |
| lag_48h | 0 |
| roll_mean_24h | 0 |
| lag_168h | 0 |
| lag_336h | 0 |
| roll_std_24h | 0 |
| dow | 0 |
| price_diff_24h | 0 |
| dow_sin | 0 |
