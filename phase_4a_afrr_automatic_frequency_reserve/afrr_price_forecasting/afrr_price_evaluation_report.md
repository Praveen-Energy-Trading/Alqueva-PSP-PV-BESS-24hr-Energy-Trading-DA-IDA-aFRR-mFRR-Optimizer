# aFRR Cap-Price Forecaster — Evaluation Report

Generated: 2026-10-03

## Data
- Source: `afrr_training_data_2019_2025.xlsx` (REN aFRR band clearing prices, mercado.ren.pt)
- Range : 2019-01-01 to 2026-10-03
- Gate  : aFRR capacity market (H1-H24, daily auction, gate closes D-1 before DA)
- Models: two separate models — cap_up (upward reserve) and cap_dn (downward reserve)
- Target: cap_up_EUR_MW, cap_dn_EUR_MW (availability payment, not energy)

## cap_up Model
Held-out test 2025-10-03 to 2026-10-03 (real rows only); naive = same hour on the previous day.

| Model | CV MAE EUR/MW |
|---|---|
| Naive | inf |
| LightGBM | 24.4873 |
| XGBoost | 20.9509 **SELECTED** |
| RandomForest | 22.6619 |

| Metric | Value |
|---|---|
| Naive MAE | 4.2504 EUR/MW |
| XGBoost MAE | 15.3936 EUR/MW |
| Skill score | -262.2% |

## cap_dn Model
Held-out test 2025-10-03 to 2026-10-03 (real rows only); naive = same hour on the previous day.

| Model | CV MAE EUR/MW |
|---|---|
| Naive | inf |
| LightGBM | 24.4873 |
| XGBoost | 20.9509 **SELECTED** |
| RandomForest | 22.6619 |

| Metric | Value |
|---|---|
| Naive MAE | 4.2175 EUR/MW |
| XGBoost MAE | 16.1812 EUR/MW |
| Skill score | -283.7% |

## aFRR Gate (Production)
- Daily capacity auction: offer submitted D-1 before DA gate (gate closes ~D-1 08:00 CET)
- cap_up > 0: plant commits headroom above committed energy to provide upward reserve
- cap_dn > 0: plant commits headroom below committed energy to provide downward reserve
- No MW sold twice (PR-11): reserve headroom bounded by energy position
- FAT: 5 minutes (PICASSO harmonised, since 4 Dec 2024)
- Cap ceiling: 250 EUR/MW (REN regulatory cap)