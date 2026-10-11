# Inflow Forecaster — Train / Validation / Test Report

_Generated: 2026-10-03_

## Methodology

Chronological train / validation / test — no shuffling, no future leakage.
Walk-forward CV (4 folds) on development set selects the model.
Final 12-month held-out test set gives unbiased accuracy below.
Source: 11-year hourly research dataset (2015-2025), aggregated to daily means.
Serving forecaster (reservoir_inflow_forecaster.py) retrains on full history daily.

Skill score = 1 − MAE_model / MAE_naive; positive = beats naive 1-day persistence.

---

## Data

- Development : 3,803 days  (2015-01-31 → 2025-06-29)
- Test (held out): 366 days  (2025-06-30 → 2026-06-30)
- Selected model : **RandomForest**

---

## Walk-forward CV — Validation MAE (model selection)

| Model | MAE (m3/h) |
|-------|-----------|
| LightGBM           | 119356 |
| XGBoost            | 112924 |
| RandomForest       | 110684 |

---

## Held-out TEST — Unbiased Accuracy

| Metric | Value |
|--------|-------|
| MAE        | 135897 m3/h |
| RMSE       | 192279 m3/h |
| Bias (ME)  | -36312 m3/h |
| Skill vs naive | -28.4% |

---

## Feature Importance (RandomForest — top 10)

| Feature | Importance |
|---------|-----------|
| roll_mean_7d | 0 |
| lag_1d | 0 |
| roll_mean_30d | 0 |
| roll_std_7d | 0 |
| doy_cos | 0 |
| lag_7d | 0 |
| doy_sin | 0 |
| lag_30d | 0 |
| inflow_diff_1d | 0 |
| month_cos | 0 |
