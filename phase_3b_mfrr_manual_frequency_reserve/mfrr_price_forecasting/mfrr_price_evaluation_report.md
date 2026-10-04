# mFRR Cap-Price Forecaster — Evaluation Report

Generated: 2026-10-03

## Data
- Source: `mfrr_training_data_2024_2025.xlsx` (REN mFRR AP_PRECO, mercado.ren.pt — documented capacity-price proxy)
- Range : 2024-11-27 (REN MARI accession) to 2026-10-02
- Gate  : mFRR capacity market (H1-H24, daily auction, gate closes D-1 before DA)
- Models: two separate models — cap_up (upward reserve) and cap_dn (downward reserve)
- Note  : history starts 2024-11-27 (REN joins MARI); 3 CV folds; 6-month hold-out test

## cap_up Model
Held-out test 2026-04-02 to 2026-10-02 (real rows only); naive = same hour on the previous day.

| Model | CV MAE EUR/MW |
|---|---|
| Naive | inf |
| LightGBM | 35.1807 |
| XGBoost | 32.3058 **SELECTED** |
| RandomForest | 32.3398 |

| Metric | Value |
|---|---|
| Naive MAE | 47.6291 EUR/MW |
| XGBoost MAE | 47.5558 EUR/MW |
| Skill score | +0.2% |

## cap_dn Model
Held-out test 2026-04-02 to 2026-10-02 (real rows only); naive = same hour on the previous day.

| Model | CV MAE EUR/MW |
|---|---|
| Naive | inf |
| LightGBM | 35.1807 |
| XGBoost | 32.3058 **SELECTED** |
| RandomForest | 32.3398 |

| Metric | Value |
|---|---|
| Naive MAE | 47.6291 EUR/MW |
| XGBoost MAE | 47.5558 EUR/MW |
| Skill score | +0.2% |

## mFRR Gate (Production)
- Daily capacity auction: offer submitted D-1 before DA gate (gate closes ~D-1 08:00 CET)
- FAT: 12.5 minutes (MARI harmonised, REN joined MARI 27 Nov 2024)
- Platform: MARI (Manually Activated Reserves Initiative)
- mFRR sized from headroom REMAINING after aFRR commitment (PR-11 stack)
- No MW sold twice: mFRR + aFRR headroom bounded by energy position
- Cap ceiling: 250 EUR/MW (REN regulatory cap)
- Independent forecast (not derived from aFRR): MARI and PICASSO are separate markets