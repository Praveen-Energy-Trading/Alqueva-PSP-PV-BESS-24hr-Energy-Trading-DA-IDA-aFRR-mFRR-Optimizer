# IDA Price Forecaster — Evaluation Report

Generated: 2026-10-04

## Data
- Source: `ida1_training_data_2024_2025.xlsx` (OMIE/ENTSO-E SIDC intraday results)
- Range : 2024-06-13 to 2026-10-02
- Gate   : IDA1 (H1-H24, closes D-1 15:00 CET)
- Model : gate-specific spread model (LightGBM/XGBoost/RandomForest, auto-selected by walk-forward CV)
- Target: spread = price_IDA - price_DA [EUR/MWh]

## Walk-forward CV (2024-06-20 to 2025-10-01, 4 folds, real rows only)
| Model | MAE EUR/MWh (spread) |
|---|---|
| Naive | inf |
| LightGBM | 4.0089 |
| XGBoost | 3.5127 |
| RandomForest | 3.5064 **SELECTED** |

## Hold-out Test (2025-10-02 to 2026-10-02, real rows only)
Predicted hour by hour with the model's own previous-hour spread, exactly as the live forecaster does.

| Metric | Value |
|---|---|
| Naive MAE (spread=0) | 6.4291 EUR/MWh |
| RandomForest MAE | 6.4975 EUR/MWh |
| Skill score | -1.1% |

*Negative skill: the model does not beat the naive (spread = 0) baseline on this window.*

## Per-Hour-Bucket Test Breakdown
| Bucket | Naive MAE | RandomForest MAE | Skill |
|---|---|---|---|
| Off-peak (H1-H6, H23-H24) | 5.39 | 5.45 | -1.1% |
| Peak (H7-H22) | 6.89 | 6.96 | -1.0% |

## IDA1 Gate (Production)
- Tradable hours: H1-H24 (all hours; gate closes D-1 15:00 CET)
- Dedicated model trained on IDA1 SIDC clearing prices only
- Baseline for IDA2 re-optimisation