# Phase 5 — Problem formulation per target

Status: built and tested, 2026-10-11. Part of the programme in `docs/ML_Forecasting_Plan.pdf`.
Decisions: `config/forecast_formulation.yaml` (one block per catalogue target). Checker:
`common_layer/configuration/forecast_formulation.py`. Code: `common_layer/forecast_formulation/` (transform, target builders,
hourly/15-minute reconciliation, evidence functions). Evidence tool: `tools/run_formulation_evidence.py`.
Tests: `tests/test_forecast_formulation.py` (21).

## 1. Result in one paragraph

Each of the nine catalogue targets now has a written formulation: what is modelled (the price level, or the **spread to the
day-ahead price** for the intraday markets), a transform that allows negative prices (asinh, with a scale chosen from
data), one direct multi-hour forecast per delivery day (no recursion), the output the decision needs, and an information
set in which **every input carries the time it is published and is checked against the target's as-of time** on ordinary,
Saturday and both daylight-saving days. The check found two things the plan had not: the OMIE day-ahead *file* appears
about 40 minutes later than the rulebook says and is late on a few percent of days, and the settled imbalance price is
published about four days late, not one.

## 2. The formulation of each target

| Target | Modelled quantity | Transform (challenger) | Output now → needed | Gate (as-of) |
|---|---|---|---|---|
| Day-ahead price | level | asinh, scale 20 (none) | point → joint scenarios | D-1 11:00 |
| IDA1, IDA2, IDA3 | spread to the day-ahead price | asinh, scale 5 (none) | point → joint scenarios | D-1 14:10, D-1 21:10, D 09:10 |
| Continuous market | spread to the day-ahead price | asinh, scale 5 (none) | point → quantiles | W1 D-1 18:30 … W6 D 12:00 |
| aFRR band price | level, one symmetric series | asinh, scale 5 (none) | point → quantiles | D-1 14:00 (assumed) |
| mFRR price | level (activation energy price) | asinh, scale 20 (none) | point → quantiles | D-1 14:00 (assumed) |
| PV weather, inflow | level | none | point → quantiles | D-1 11:00 |

Common to all: **direct** forecasting (the origin is fixed, so the day's hours are forecast in one pass and no predicted
price is fed back), one pooled model with the hour as a feature, evaluation in price units whatever the training scale.

## 3. Evidence for the choices

**Transform.** Elasticity of the day-over-day change of the day-ahead price with respect to the price level (0 = equally
variable at every level):

| asinh scale (EUR/MWh) | none | 5 | 10 | 20 | 35 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| Elasticity | 0.58 | −0.15 | −0.09 | **0.00** | 0.11 | 0.19 | 0.35 |

For the intraday spreads (development data, 30-day blocks) asinh at scale 5 reduces the excess kurtosis from 14.3 to 0.4
(IDA1), 5.2 to −0.4 (IDA2), 3.7 to −0.7 (IDA3) and 5.0 to −0.8 (continuous market) and moves the level-dependence toward
zero (IDA1 −0.49 → −0.23, IDA2 −0.19 → −0.07, continuous −0.30 → −0.12). The aFRR price has skew 3.3 raw and 0.4 at scale 5;
the mFRR price has excess kurtosis 492 raw and −0.1 at scale 20. The untransformed series stays as the challenger because the
trading decision cares about euros: a model trained on asinh is judged on the price (Phase 6), and its back-transformed
point forecast is a median-type value (`retransform_mean` gives the smeared mean when the expected price is needed).

**One whole-day shift versus joint scenarios.** The optimiser's stochastic mode builds the day-ahead scenarios as the point
forecast plus a **common offset for every hour** (±0.75 and ±1.5 times the model's MAE). Measured on the residual of the day-ahead
price after removing its calendar profile:

| Sample | One common daily shift explains | First three day-shape factors explain |
|---|---|---|
| All years | 78.5% | 92.9% |
| 2024 onward | 71.1% | 90.9% |
| 15-minute era | 68.5% | 90.5% |

So the current fan captures about seven tenths of the day's error structure. A fan with a common shift plus two shape factors
captures about nine tenths; this is the target output of the day-ahead forecast (built in the output phase).

**Spread rather than level for intraday.** The day-ahead price explains 97% (IDA1), 94% (IDA2), 92% (IDA3) and 90% (continuous
market) of the intraday price's variance; the spread's standard deviation is 9.5, 13.9, 16.8 and 17.3 EUR/MWh against level
standard deviations of 54–58. Within the auction day the spread is persistent from hour to hour (0.59–0.73) but not from day
to day (0.02–0.26).

**Reserve prices are levels.** They are in EUR/MW/h, so a "spread to a EUR/MWh price" is meaningless. aFRR up and down are the
same series (correlation 0.998); the day-ahead price explains 0.3% of the aFRR level, so its own history carries the signal.

**Inside the hour.** In the 15-minute era the four quarters of an hour move systematically (hour 9: about +18, +5, −6 and −18
EUR/MWh around the hourly mean; hour 19: −17, −5, +6, +16). Giving every quarter its hour's value costs a mean absolute
error of 4.4 EUR/MWh (RMSE 8.4), concentrated in the ramp hours 8–10 and 17–20. The plan: an hourly model plus a learned
within-hour shape, reconciled so the quarters average to the hourly forecast (`coherence.reconcile`, shape-preserving
and idempotent), because the long history is hourly.

## 4. Information sets: written down and tested

For every input: the latest delivery day it may use (`latest_day_offset`) and when data for a delivery day is published.
`availability_table` computes the margin to the as-of time on four sample days; a check that cannot fail would be useless,
so the tests also break the configuration and require the checks to fail (wrong publication time, too recent a day, a
late input without a policy, a mixed-unit spread, the forbidden same-auction previous hour).

| Series | Published (typical) | Evidence | Late tail |
|---|---|---|---|
| Day-ahead result | D-1 13:30 | 1,682 original files 2021–2026: issued median 13:29, 90% by 14:01, 95% by 14:37, 99% by 18:28 | 7.7% after 14:10; 3.9% after 15:00 |
| IDA1, IDA2, IDA3 | D-1 15:30, D-1 22:30, D 10:30 | rulebook | |
| Continuous-market summary | D+1 04:30 | 812 files: median 02:21, 95% by 04:23 | |
| aFRR band result | D-1 16:00 | assumed from the market sequence; consistent with every load so far | |
| mFRR series | D+1 12:00 | assumed; consistent with every load so far | |
| **Imbalance price** | **D+4 12:00** | **empirical bound: HTTP 500 for days 8–11 October on 11 October; published for 7 October** | |

Consequences already in the configuration: the day-ahead target may use imbalance prices only up to D-6; the
continuous-market target cannot use yesterday's price at all (the summary is published the next morning); a gate that needs
the day-ahead result (IDA1, aFRR, mFRR) has a written **late policy** (wait for the file, then forecast the day-ahead price and
flag the run). The tightest margins are 30 minutes (aFRR and mFRR gates against the day-ahead file) and 40 (IDA1).
Findings F2, F3, F4 and F5 of the Phase 0 audit are closed here: the actual day-ahead result is the base price of every
intraday target, earlier auctions of the same day are inputs for later ones, the previous hour of the same auction is forbidden,
and yesterday's band price is an input of the aFRR target.

## 5. Limits

* Publication times marked *assumed* (aFRR, mFRR, weather forecast, fundamentals) are conservative bounds, not
  measurements; the daily pipeline log will refine them. The imbalance bound is the only one found wrong so far.
* The fixed XBID information set is the one valid at the earliest window (W1); later windows know more, which a
  window-specific model could exploit.
* The scenario structure (common shift plus two shape factors) is specified, not yet built: the optimiser's fan is unchanged.
* Weather, inflow and fundamentals are formulated but have no data yet (Phase 1 gaps).
