# Phase 0 — Forecast catalogue and as-of audit

Status: draft for review, 2026-10-11. Part of the programme in `docs/ML_Forecasting_Plan.pdf`.
Machine-readable version: `config/forecast_catalogue.yaml` (validated by `tests/test_forecast_catalogue.py`).

## 1. What this phase fixes

For every forecast the pipeline uses: which decision consumes it, the **as-of time** (the moment the
decision is taken), and therefore which information may be used. Information published after the as-of
time is forbidden as an input. The rules are written once, in the catalogue, and later phases (evaluation
framework, leakage tests) read them.

All times are CET. D is the delivery day.

## 2. The day, as a timeline of what is known

| Time | Event | Becomes known |
|---|---|---|
| D-2 ~12:45 | Day-ahead result for D-1 published | DA prices of D-1 |
| D-1 11:00 | Pipeline trigger for the DA bid (gate closes 12:00) | prices up to D-1, weather forecasts issued before 11:00 |
| D-1 ~12:45 | **Day-ahead result for D published** | DA price of every hour of D |
| D-1 afternoon | aFRR band market, then mFRR band market (exact hours set by a REN notice, not public) | band results (not public in time) |
| D-1 14:10 / 15:00 | IDA1 trigger / gate close | |
| D-1 ~15:30 | IDA1 result for D published | IDA1 prices of D |
| D-1 21:10 / 22:00 | IDA2 trigger / gate close | |
| D-1 ~22:30 | IDA2 result published | IDA2 prices of D |
| D 09:10 / 10:00 | IDA3 trigger / gate close | |
| D ~10:30 | IDA3 result published | IDA3 prices of D |
| D-1 18:30, 22:30, D 03:00, 06:00, 09:30, 12:00 | XBID windows W1-W6 | continuous trades up to each window |

## 3. Decision map: which forecast feeds which decision

| Forecast | Decision | Where it is used | As-of gate |
|---|---|---|---|
| Day-ahead price | Day-ahead bid; energy-price reference for reserves | `run_da`, intraday gates, backtest | DA (D-1 11:00) |
| PV irradiance and temperature | PV availability in every optimisation | `run_da`, `ida_reoptimiser` | DA (D-1 11:00) |
| Reservoir inflow | Water balance | `run_da`, `ida_reoptimiser` | DA (D-1 11:00) |
| IDA1 / IDA2 / IDA3 price | Intraday re-optimisation | `ida_reoptimiser`, backtest | each gate |
| XBID price | Continuous-market checks | `xbid_optimiser`, backtest | each window |
| aFRR capacity price | aFRR offer and expected revenue | `run_afrr`, backtest | after DA result |
| mFRR price series | mFRR offer; **no revenue consumer today** (capacity is paid at the aFRR price in the future-band scenario) | `run_mfrr` | after aFRR band |

Not forecast today: imbalance price and activation volumes. Settlement uses real or fallback prices and a
simulated activation model.

## 4. Audit findings (evidence in the code)

Severity: high = changes decisions or the honesty of reported skill; medium = limits skill or testing;
low/info = housekeeping or opportunity.

| # | Severity | Finding | Evidence |
|---|---|---|---|
| F1 | **high** | **aFRR/mFRR forecasters: serving features differ from training.** At prediction time the DA price is set to the price of the same hour 7 days earlier, the 24-hour rolling mean to that same value, and the rolling std to the constant 0, while training used the real values. | `afrr_price_forecaster.py` `_build_pred_rows` (lines 193-221); `mfrr_price_forecaster.py` lines 203-223 |
| F2 | **high** | **Intraday gates feed a *forecast* of the DA price,** although the actual DA result is published (D-1 ~12:45) before every intraday gate. | `ida_reoptimiser.py` line 88 (`forecast_da_prices`) |
| F1b | medium | Intraday forecasters: training uses an **expanding within-day** DA mean and std (hour 1 sees one value), prediction uses the **full-day** mean and std for every hour. | `ida1_price_forecaster.py` lines 195-198 (training) vs 229-230 (serving); same code in IDA2, IDA3, XBID |
| F3 | medium | IDA2, IDA3 and XBID reuse the IDA1 feature set. The IDA1 result (known at IDA2), the IDA1 and IDA2 results (known at IDA3) and early XBID trades are unused. | `_feature_cols` in the four modules |
| F4 | medium | `spread_lag_h1` is the previous hour of the **same auction**, not observable at the as-of time. Training uses actual values, serving uses predicted ones (recursive). The evaluation was already made autoregressive, but the model is still trained with teacher forcing. | `ida1_price_forecaster.py` line 204 and the autoregressive loop (lines 112-125) |
| F5 | medium | Reserve forecasters use only a 7-day lag. The band price of D-1, known at the as-of time, is unused (consistent with the poor aFRR skill measured earlier). | `afrr_price_forecaster.py` `_feature_cols` |
| F6 | medium | Model comparison uses a **Wilcoxon test on at most 4 fold MAEs**. There is no Diebold-Mariano test on daily errors. (The plan document said "no significance tests"; the precise statement is "only a weak one".) | `ml_train_val_test_common.py` line 92 |
| F7 | medium | Training files contain **SYNTHETIC rows**: IDA1 7.8%, IDA2 2.1%, IDA3 1.7%, XBID 0.7%, mFRR 2.2%, aFRR 0.1%; the PV weather file runs to 2027-03 with 6,168 synthetic rows. Phase 3 must check they never enter training. | counts of the `source` column, see section 5 |
| F8 | medium | PV weather rows are labelled `PLANT_SENSOR` for 2015 onward. Provenance unverified. | `pv_training_data_from_2015.xlsx` |
| F11 | medium | Inflow `lag_1d` treats yesterday's daily value as known at 11:00; the publication delay of the research dataset is unknown. | `reservoir_inflow_forecaster.py` lines 253-255 |
| F9 | low | Stale docstrings (IDA2 "H3-H24", data ranges "2020-2025") after the data rebuild. | module headers |
| F10 | info | The day-ahead forecaster has no fundamentals (load, wind, solar, gas, carbon) and only 12 months of 15-minute history. | `_feature_cols`, `_EXCEL_ISP_PATH` |
| F12 | info | The mFRR price forecast has no revenue consumer today; it could become an activation-price forecaster. | `run_mfrr.py` |

What this means: the skill numbers measured earlier were honest for the evaluation setup, but F1 and F1b
mean the **served** forecasts do not behave like the evaluated ones, so the live skill of the intraday and
reserve forecasts is probably worse than the evaluation suggests. F2 is the opposite direction: the
intraday gates ignore information they are entitled to.

## 5. Data snapshot (preview of Phase 1)

| Dataset | Rows | Span | Real / synthetic |
|---|---|---|---|
| DA hourly | 59,400 | 2020-01-01 to 2026-10-10 | all OMIE_LIVE |
| DA 15-minute | 36,000 | 2025-10-01 to 2026-10-10 | all live |
| PV weather | 106,944 | 2015-01-01 to 2027-03-14 | 100,776 PLANT_SENSOR, 6,168 SYNTHETIC |
| Inflow | 4,373 | 2015-01-01 to 2026-12-21 | 4,199 research data, 174 climatology fill |
| IDA1 / IDA2 / IDA3 | 20,376 each | 2024-06-13 to 2026-10-09 | 7.8% / 2.1% / 1.7% synthetic |
| XBID | 20,376 | 2024-06-13 to 2026-10-09 | 0.7% synthetic |
| aFRR | 68,136 | 2019-01-01 to 2026-10-09 | 0.1% synthetic |
| mFRR | 16,368 | 2024-11-27 to 2026-10-09 | 2.2% synthetic |

## 6. Rule proposed for the intraday and reserve gates (decision for you)

From the timeline in section 2: **every gate after the day-ahead result must use the actual DA result,
not a forecast of it** (F2), and each later gate must also use the earlier gates' results (F3). This is a
design rule now; the code change belongs to Phase 7 (features) and Phase 14 (serving).

## 7. Open points

1. **Order of targets.** Drafted as: day-ahead price (priority 1), then IDA/XBID and aFRR (2), then PV, inflow, mFRR (3). Confirm or change.
2. **Reserve gate hours.** The aFRR and mFRR band-market hours are set by a REN notice that is not public; the as-of rule uses "after the DA result, before IDA1". Acceptable?
3. **Rule for F2.** Use the actual DA result at all later gates (recommended).
4. **Imbalance and activation forecasts.** Out of scope for now (settlement uses real or fallback prices). Add later?

## 8. Exit criterion

Met when the points above are confirmed: a signed-off catalogue with an as-of time and a consumer for
every target. The file and its validator are in place.
