# IDA Price Forecaster — Evaluation Report

Generated: 2026-10-04

## Data
- Source: `ida2_training_data_2024_2025.xlsx` (OMIE/ENTSO-E SIDC intraday results)
- Range : 2024-06-13 to 2026-10-02
- Gate   : IDA2 (H3-H24, closes D-1 22:00 CET; H1-H2 frozen after IDA1)
- Model : gate-specific spread model (LightGBM/XGBoost/RandomForest, auto-selected by walk-forward CV)
- Target: spread = price_IDA - price_DA [EUR/MWh]

## Walk-forward CV (2024-06-20 to 2025-10-01, 4 folds, real rows only)
| Model | MAE EUR/MWh (spread) |
|---|---|
| Naive | inf |
| LightGBM | 6.6667 |
| XGBoost | 5.8130 **SELECTED** |
| RandomForest | 5.8701 |

## Hold-out Test (2025-10-02 to 2026-10-02, real rows only)
Predicted hour by hour with the model's own previous-hour spread, exactly as the live forecaster does.

| Metric | Value |
|---|---|
| Naive MAE (spread=0) | 9.9037 EUR/MWh |
| XGBoost MAE | 10.1099 EUR/MWh |
| Skill score | -2.1% |

*Negative skill: the model does not beat the naive (spread = 0) baseline on this window.*

## Per-Hour-Bucket Test Breakdown
| Bucket | Naive MAE | XGBoost MAE | Skill |
|---|---|---|---|
| Off-peak (H3-H6, H23-H24) | 9.15 | 9.32 | -1.8% |
| Peak (H7-H22) | 10.21 | 10.44 | -2.3% |

## IDA2 Gate (Production)
- Tradable hours: H3-H24 (H1-H2 frozen after IDA1; gate closes D-1 22:00 CET)
- Dedicated model trained on IDA2 SIDC clearing prices only
- Baseline for IDA3 re-optimisation