# XBID Price Forecaster — Evaluation Report

Generated: 2026-10-04

## Data
- Source: `xbid_training_data_2024_2025.xlsx` (OMIE continuous-intraday MedioPT price)
- Range : 2024-06-13 to 2026-10-02
- Gate  : XBID continuous (H1-H24; closes 1h before each delivery period)
- Note  : XBID price = OMIE's volume-weighted Portuguese continuous-market mean (MedioPT).
- Model : gate-specific spread model (LightGBM/XGBoost/RandomForest, auto-selected by walk-forward CV)
- Target: spread = price_XBID - price_DA [EUR/MWh]

## Walk-forward CV (2024-06-20 to 2025-10-01, 4 folds, real rows only)
| Model | MAE EUR/MWh (spread) |
|---|---|
| Naive | inf |
| LightGBM | 15.5266 |
| XGBoost | 13.7646 **SELECTED** |
| RandomForest | 14.0250 |

## Hold-out Test (2025-10-02 to 2026-10-02, real rows only)
Predicted hour by hour with the model's own previous-hour spread, exactly as the live forecaster does.

| Metric | Value |
|---|---|
| Naive MAE (spread=0) | 18.4794 EUR/MWh |
| XGBoost MAE | 18.6728 EUR/MWh |
| Skill score | -1.0% |

*Negative skill: the model does not beat the naive (spread = 0) baseline on this window.*

## Per-Hour-Bucket Test Breakdown
| Bucket | Naive MAE | XGBoost MAE | Skill |
|---|---|---|---|
| Off-peak (H1-H6, H23-H24) | 22.50 | 22.54 | -0.2% |
| Peak (H7-H22) | 16.72 | 16.98 | -1.6% |

## XBID Gate (Production)
- Tradable hours: H1-H24 (gate closes 1h before each delivery period)
- Two check windows: W1 (D-1 18:30 CET), W2 (D 09:30 CET)
- Per-order cap: xbid_max_volume_per_order_mw (config)
- Order placed only if avg gain > xbid_min_spread_eur_mwh (no-churn)
- Spread std wider than IDA3 (~14 vs ~11 EUR/MWh): closer to delivery