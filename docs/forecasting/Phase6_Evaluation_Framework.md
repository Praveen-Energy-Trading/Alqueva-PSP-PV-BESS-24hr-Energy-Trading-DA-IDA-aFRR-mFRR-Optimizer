# Phase 6 — Evaluation framework

Status: built and tested, 2026-10-11. Part of the programme in `docs/ML_Forecasting_Plan.pdf`.
Code: `common_layer/forecast_evaluation/`, settings `config/evaluation.yaml`, command `tools/evaluate_forecaster.py`.
Tests: `tests/test_forecast_evaluation.py` (15).

## 1. Result in one paragraph

One command scores any forecaster of any target the same way and produces the same report: rolling-origin backtests with a
retraining schedule and a gap, SSE, MAE and RMSE in price units, a Diebold–Mariano comparison between forecasters, a locked
final test period that can be scored once per forecaster, and automated leakage tests. By agreement the model family is the
**five tree algorithms** (decision tree, random forest, gradient boosting, XGBoost, LightGBM) and the metrics are **SSE, MAE
and RMSE**; no naive or regression benchmark, no probabilistic score and no profit-based decision value is part of the framework.

## 2. How a run works

* **Folds** (`folds.py`): a model is refitted every 30 delivery days, as it would be live. Its training targets end two days
  before the first day it predicts (`gap_days` 1 plus the day itself) and cover the latest 730 days (rolling window).
* **As-of view** (`asof.py`): a forecaster never sees the raw data. It receives, for each delivery day D, the data its
  information set (Phase 5) allows, with everything later set to missing and all other series removed, plus labels of days
  already known at fit time. Asking for the label or view of a day after the training window raises an error.
* **Metrics** (`metrics.py`): SSE (sum of squared errors), MAE and RMSE of price forecasts. Spread forecasts are converted to
  prices with the day-ahead price known at the gate, so every target is scored in EUR/MWh (EUR/MW/h for aFRR). SSE and RMSE
  rank forecasters identically on the same hours; compare SSE only between forecasters scored on the same hours.
  The report also gives the scores on the hours **all** forecasters predicted, so one that skips hard hours cannot look better.
* **Comparison** (`dm.py`): Diebold–Mariano on one loss difference per delivery day (24 hourly errors of a day are not
  independent), Newey–West variance with 7 lags and the small-sample correction. It replaces the Wilcoxon test on at most four
  fold MAEs of the old comparison (finding F6). Reference model: the decision tree.
* **Locked period** (`guards.py`): 2025-10-12 to 2026-10-11 (the latest year, one full seasonal cycle in the 15-minute
  era). Development scoring must end before it; a `--stage final` run must cover exactly this window, is logged, and is allowed
  once per forecaster and target. `--lock` freezes a digest of the period's outcomes; a final run fails if the data changed since.
* **Leakage test** (`perturbation_check`): replace everything a forecaster may not know with random numbers, refit and
  predict again; the answer must not change. All five tree models pass; a deliberately cheating forecaster (it reaches
  through its fit context into unmasked data) is caught. Forecasters get a data-free interface, never the data object.
* **Reproducibility:** the report contains no clock time; the same code, configuration and data give a byte-identical
  `report.md` (tested). It prints a configuration digest, a data digest, the feature set and its version, and the code revision.

## 3. First measured result (untuned, day-ahead price, development window)

2023-01-01 to 2025-10-11, 1,015 delivery days, 24,359 hours, 34 folds, the first, smaller feature set:

| Model | SSE | MAE | RMSE | Diebold–Mariano vs decision tree |
|---|---|---|---|---|
| Decision tree | 19.44 M | 20.49 | 28.25 | reference |
| Random forest | **16.31 M** | **18.90** | **25.87** | −9.6 (p < 0.001) |
| Gradient boosting | 16.76 M | 19.38 | 26.23 | −5.0 (p < 0.001) |
| XGBoost | 16.81 M | 19.41 | 26.27 | −4.7 (p < 0.001) |
| LightGBM | 17.12 M | 19.52 | 26.51 | −4.3 (p < 0.001) |

All four ensembles are significantly better than the single tree, and random forest is best on all three metrics. For scale:
before the naive forecasters were removed from the framework, "same hour yesterday" scored MAE 19.0 and RMSE 28.3 on the same
window. Untuned, the trees barely improve its MAE (random forest by 0.1) and win mainly on RMSE. Tuning, richer features
(Phase 7) and the right training window (Phase 9) are where the gain has to come from. These models use fixed default
settings and nothing was tuned on this window.

## 4. Limits

* Without a naive or regression benchmark the framework ranks the tree models against each other; it cannot, by itself,
  say that a model is better than a trivial forecast. The measurement above is the only reference.
* SSE, MAE and RMSE score point forecasts only. The scenario and quantile outputs that Phase 5 requires for the day-ahead price
  have no score here; if they are built, a probabilistic metric (for example pinball loss) will be needed.
* No profit-based decision value: a model that wins on MAE can still lose on trading profit.
* Evaluation is hourly (the 15-minute day-ahead price enters as the mean of its quarters).
* The XBID target is scored with the information valid at the earliest window (W1).
