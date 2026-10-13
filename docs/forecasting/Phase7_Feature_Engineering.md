# Phase 7 — Feature engineering and feature store

Status: built and tested, 2026-10-11. Part of the programme in `docs/ML_Forecasting_Plan.pdf`.
Code: `common_layer/feature_engineering/` (calendar, catalogue, builder, store, selection). Definitions:
`config/feature_catalogue.yaml`, `config/feature_sets.yaml`, `config/feature_selection.yaml`; generated result:
`config/feature_selection_results.yaml`. Tools: `tools/run_feature_selection.py`, `tools/evaluate_forecaster.py --feature-set`.
Tests: `tests/test_feature_engineering.py` (20).

## 1. Result in one paragraph

Every feature a model may use is defined once, in a catalogue, and built by **one function** that serves training,
evaluation and live prediction, so a feature cannot differ between training and serving (findings F1 and F1b of the
Phase 0 audit). What a feature reads is derived from its definition, and whether it is **available** for a target is
computed against that target's information set and the publication times of Phase 5: no feature can use data its gate would
not have, and a feature that read too far would be caught by the point-in-time check. The catalogue holds 58 features
(25–55 available per target). A stability check across periods keeps only 2–6 features per
target; on the day-ahead price the full set is modestly better than the small one for LightGBM, mostly on large errors.

## 2. Calendar and market rules (`calendar.py`)

Hour (with sine and cosine), weekday, month, annual cycle, weekend, **national holidays of Portugal** (Easter-based
ones included; the Portuguese holidays suspended in 2013–2015 are handled), **bridge days** (a Monday before a Tuesday holiday,
a Friday after a Thursday holiday), days to the next and since the last Portuguese holiday, **daylight-saving day type**
(−1 for the 23-hour day, +1 for the 25-hour day), and the market rules in force that day (day-ahead product length, price
floor). Regional and local holidays are not included.

## 3. The catalogue and its availability

| Group | Features | What |
|---|---|---|
| Calendar and market rule | 15 | as above |
| Lags and rolling statistics | 22 | the modelled quantity 1, 2, 7, 14 days earlier; 7- and 28-day means, standard deviation, minimum, maximum by hour and over all hours; yesterday's mean, maximum, minimum; the same windows ending 2 or 3 days back for targets published later |
| Base price (spread targets) | 9 | the day-ahead price of the hour and of the day (mean, maximum, minimum, rank, midday-minus-evening shape), yesterday's, and the change |
| Day-ahead market results | 2 | the Portuguese matched purchase and sale volumes of day D (for the intraday gates) |
| Other markets | 10 | earlier auctions of the same day, spreads of earlier days, aFRR, imbalance and mFRR lags |

Availability follows from what each feature reads. Examples the check produces: the continuous-market target cannot use
yesterday's price (its summary is published the next morning), so it gets windows ending two days back; the mFRR target only
three days back; the day-ahead target cannot use any market result of day D but can use yesterday's; the intraday gates can
use the day-ahead volumes and flows of day D (published with the price), which are a demand and interconnection proxy. Every available feature reaches its target at least 30 minutes before the as-of time (the
day-ahead file is the tightest input, and a gate that depends on it has a late policy from Phase 5).

**Imbalance lags start at six days.** REN publishes settled imbalance prices about four days late (the API returned
HTTP 500 for 8–11 October on 11 October), so the latest day usable at the day-ahead gate is D-6.

## 4. One builder, and point-in-time correctness

`FeatureBuilder.build(view, D)` is the only code that computes features. `point_in_time_check` builds the features of a
day twice, from the as-of view and from the same rows with nothing masked, and requires identical values: a feature that
depended on data the target may not know would differ. It passes for every available feature of six targets on ordinary
days and the spring daylight-saving day; a feature deliberately made to read one day too far is caught. A feature that is not
available for a target is refused when the builder is created, not silently left empty.

## 5. Feature store (`store.py`)

SQLite, one table per target and feature set, one row per delivery hour, keyed by UTC time. The table records the feature set's
**version**: a 12-character hash of every member's id, kind, parameters and version. A changed definition is a new version, the
old rows are dropped and rebuilt rather than mixed. Reading from the store gives exactly what the builder gives (tested);
writing a day twice replaces it.

## 6. Feature selection with stability checks (`selection.py`, development data only)

Method: in each period LightGBM is trained on the first 70% of the days; the increase of the mean absolute error (EUR/MWh)
from shuffling a feature on the last 30% is its permutation importance. A feature is kept if its share of the total positive
importance is at least 1% in at least 75% of the periods. The day-ahead price and the aFRR price use five yearly periods
(2021–2025); the intraday and reserve markets, whose history starts in mid-2024, use consecutive blocks of about 100 days of
their own development history. No period reaches the locked test period (the tool refuses).

| Target | Candidates | Kept | Rank correlation between periods | Selected features |
|---|---|---|---|---|
| Day-ahead price | 45 | 5 | 0.41 | hour, weekday, same hour yesterday, same hour last week, yesterday's maximum |
| IDA1 | 53 | 9 | 0.32 | hour, same hour last week, yesterday's minimum, the day-ahead price and its deviation from the day mean, the day's maximum and minimum of the day-ahead price, its change since yesterday, the Portuguese sales volume of day D |
| IDA2 | 54 | 2 | 0.22 | day-ahead price, **IDA1 spread of the same day** |
| IDA3 | 55 | 2 | 0.27 | **IDA1 and IDA2 spreads of the same day** |
| Continuous market | 41 | 6 | 0.21 | hour, weekday, 2-day-old daily mean, day-ahead deviation and change, **IDA1 spread of the same day** |
| aFRR price | 43 | 6 | 0.29 | weekday, same hour yesterday, 28-day same-hour mean, the Portuguese purchase and sale volumes of day D, aFRR lag |
| mFRR price | 25 | 3 | 0.34 | hour (sine and cosine), same hour two weeks earlier |

Readings: the later auctions are explained by **the earlier auctions of the same day** (IDA2 by IDA1, IDA3 by IDA1 and IDA2,
the continuous market by IDA1), the information that finding F3 said the old forecasters left unused. The day-ahead price is
explained by its own recent history and the calendar; the market results and the imbalance and mFRR lags rarely matter. The rank
correlation between periods is low (0.19–0.44): which feature is most important changes with the market regime, so the selection
is a stable *core* rather than a ranking. Permutation importance splits credit between correlated features, so a feature can be left
out only because a close relative carries the signal. The hold-out error of the day-ahead model in each period (63, 33, 19, 26
and 16 EUR/MWh for 2021–2025) shows again how much harder a rising market is for a tree.

## 7. Does a bigger feature set help? (day-ahead price, development window, untuned)

2023-01-01 to 2025-10-11, 1,015 days, 24,359 hours; the `baseline_v1` set has 9 features for this target, `full_v1` 45 (the Spanish features were
removed from the catalogue on 2026-10-13, and this comparison was re-run without them).

| Model | Feature set | SSE | MAE | RMSE |
|---|---|---|---|---|
| LightGBM | baseline_v1 | 15.47 M | 18.29 | 25.20 |
| LightGBM | full_v1 | 14.25 M | 17.79 | 24.19 |
| XGBoost | baseline_v1 | 15.32 M | 18.24 | 25.08 |
| XGBoost | full_v1 | 14.88 M | 18.20 | 24.72 |

Diebold–Mariano, baseline against full: on **absolute** error p = 0.079 (LightGBM) and 0.89 (XGBoost), not significant at 5%; on
**squared** error p = 0.003 and 0.30. For LightGBM the extra features reduce the large errors (RMSE −4.0%) and a little the typical
day's error (MAE −2.8%); for XGBoost the difference is not distinguishable from luck (RMSE −1.4%, MAE −0.2%). Even `baseline_v1` improves on the
first feature set of Phase 6 (LightGBM MAE 18.3 against 19.5). The stability-selected set `selected_v1` is deliberately **not** scored on this
window: it was chosen with data that includes these days, so scoring it here would be circular. It is judged once, on the locked period, with the others.

## 8. Limits

* Only calendar, price-history and market-result features exist; the strongest driver of the hourly price shape (residual
  load: load minus wind and solar forecasts) is not collected (Phase 1 gaps), so the 22–35% of residual variance in Phase 4
  is still unexplained.
* Features are hourly; the 15-minute shape inside an hour is handled by the reconciliation of Phase 5.
* The XBID features are valid at the earliest window (W1); later windows know more.
* When the day-ahead file is late (7.7% of days at the IDA1 trigger), the base-price features are missing for that run; the
  model treats them as missing and the run is flagged (Phase 5 late policy), which has not been evaluated yet.
* The selection is by permutation importance with one learner; another learner or criterion could keep other features.
