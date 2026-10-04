# IDA Price Forecaster — Evaluation Report

Generated: 2026-10-04

## Data
- Source: `ida3_training_data_2024_2025.xlsx` (OMIE/ENTSO-E SIDC intraday results)
- Range : 2024-06-13 to 2026-10-02
- Gate   : IDA3 (H12-H24, closes D 10:00 CET; H1-H11 frozen after IDA1/IDA2)
- Model : gate-specific spread model (LightGBM/XGBoost/RandomForest, auto-selected by walk-forward CV)
- Target: spread = price_IDA - price_DA [EUR/MWh]

## Walk-forward CV (2024-06-20 to 2025-10-01, 4 folds, real rows only)
| Model | MAE EUR/MWh (spread) |
|---|---|
| Naive | inf |
| LightGBM | 8.1712 |
| XGBoost | 7.3035 |
| RandomForest | 7.1450 **SELECTED** |

## Hold-out Test (2025-10-02 to 2026-10-02, real rows only)
Predicted hour by hour with the model's own previous-hour spread, exactly as the live forecaster does.

| Metric | Value |
|---|---|
| Naive MAE (spread=0) | 12.5369 EUR/MWh |
| RandomForest MAE | 13.1635 EUR/MWh |
| Skill score | -5.0% |

*Negative skill: the model does not beat the naive (spread = 0) baseline on this window.*

## Per-Hour-Bucket Test Breakdown
| Bucket | Naive MAE | RandomForest MAE | Skill |
|---|---|---|---|
| Midday (H12-H16) | 11.32 | 12.25 | -8.3% |
| Evening (H17-H24) | 13.15 | 13.62 | -3.6% |

## IDA3 Gate (Production)
- Tradable hours: H12-H24 (H1-H11 frozen after IDA1/IDA2; gate closes D 10:00 CET)
- Dedicated model trained on IDA3 SIDC clearing prices only
- Final intraday gate; no further re-optimisation after this