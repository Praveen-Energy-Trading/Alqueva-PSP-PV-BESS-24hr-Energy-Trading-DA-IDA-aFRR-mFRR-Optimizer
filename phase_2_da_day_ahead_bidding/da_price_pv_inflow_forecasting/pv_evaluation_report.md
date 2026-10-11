# PV Forecaster — Train / Validation / Test Report

_Generated: 2026-10-03_

## Methodology

Chronological train / validation / test — no shuffling, no future leakage.
Walk-forward CV (4 folds) on development set selects the model.
Final 12-month held-out test set (never seen during selection) gives unbiased accuracy below.
Serving forecaster (pv_power_forecaster.py) retrains on all history daily.

Skill score = 1 − MAE_model / MAE_naive; positive = beats naive 24h persistence.

---
### PV — Global Horizontal Irradiance

- Development  : 91,847 rows (2015-01-08 → 2025-06-30)
- Test (held out): 8,761 rows (2025-06-30 → 2026-06-30)
- Selected model : **RandomForest**

**Walk-forward CV (validation MAE — model selection):**

| Model | MAE (W/m2) |
|-------|-----------|
| LightGBM           | 12.28 |
| XGBoost            | 12.00 |
| RandomForest       | 11.84 |

**Held-out TEST (unbiased):**

| Metric | Value |
|--------|-------|
| MAE        | 18.94 W/m2 |
| RMSE       | 52.44 W/m2 |
| Bias (ME)  | +0.51 W/m2 |
| Skill vs naive | -7.0% |

**Feature Importance (RandomForest — top 10):**

| Feature | Importance |
|---------|-----------|
| lag_24h_GHI | 1 |
| lag_48h_GHI | 0 |
| lag_168h_GHI | 0 |
| clearsky_ghi | 0 |
| doy | 0 |
| roll_mean_24h_GHI | 0 |
| lag_kt_24h | 0 |
| roll_std_24h_GHI | 0 |
| month_cos | 0 |
| month_sin | 0 |

---

### PV — Ambient Temperature

- Development  : 91,847 rows (2015-01-08 → 2025-06-30)
- Test (held out): 8,761 rows (2025-06-30 → 2026-06-30)
- Selected model : **LightGBM**

**Walk-forward CV (validation MAE — model selection):**

| Model | MAE (degC) |
|-------|-----------|
| LightGBM           | 0.25 |
| XGBoost            | 0.28 |
| RandomForest       | 0.41 |

**Held-out TEST (unbiased):**

| Metric | Value |
|--------|-------|
| MAE        | 1.37 degC |
| RMSE       | 2.00 degC |
| Bias (ME)  | -0.21 degC |
| Skill vs naive | -51.1% |

**Feature Importance (LightGBM — top 10):**

| Feature | Importance |
|---------|-----------|
| roll_std_24h_T_amb | 5235 |
| lag_24h_T_amb | 4926 |
| roll_mean_24h_T_amb | 4640 |
| doy | 4198 |
| lag_48h_T_amb | 3499 |
| lag_168h_T_amb | 3173 |
| hour_sin | 1368 |
| month_sin | 1195 |
| clearsky_ghi | 1152 |
| month_cos | 1036 |
