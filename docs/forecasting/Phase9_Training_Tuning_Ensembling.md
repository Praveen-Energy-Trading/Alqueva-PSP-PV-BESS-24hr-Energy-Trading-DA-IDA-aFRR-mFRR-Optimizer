# Phase 9 — Training, tuning and ensembling

Status: built, tested and run on the day-ahead price (development window only), 2026-10-11. Part of the programme in
`docs/ML_Forecasting_Plan.pdf`.
Code: `common_layer/forecast_evaluation/tuning.py`, `ensemble.py` (and `tree_models.py`, `runner.py`). Settings:
`config/tuning.yaml`. Tools: `tools/evaluate_forecaster.py --tuned --ensembles`, `tools/run_schedule_comparison.py`.
Tests: `tests/test_tuning_ensembles.py` (12). Raw outputs: `runtime/analysis/phase9/` (not committed).

## 1. Result in one paragraph

On the day-ahead price, over the 1,015 development days (2023-01-01 to 2025-10-11), three things were measured, each against
a stated reference with the Diebold–Mariano test (daily losses, HAC 7 lags, small-sample correction).
**(a) Nested tuning helps the boosted models and costs time:** LightGBM improves from MAE 18.29 to 17.43 (p < 0.001) and
XGBoost from 18.24 to 17.72 (p < 0.001); the random forest and gradient boosting gain a little (not significant). The tuned run
took 7.7 hours while sharing the machine with two other jobs, and 38 minutes when repeated alone (the untuned run, which
shared the machine, took 21 minutes; the two times are not directly comparable).
**(b) Combinations are as good as the best single model, not clearly better:** against the best tuned model the simple
and weighted averages differ by −0.08 to +0.12 MAE with p between 0.22 and 0.99. Against the best untuned model the
median and the trimmed mean win by 0.15 (p 0.03–0.04), which is not convincing after five comparisons.
**(c) A longer calibration window helps more than anything else tried:** with LightGBM at its defaults, an expanding window
gives MAE 17.56 against 18.29 for the 730-day reference (p < 0.001); how often the model is refitted (7, 30 or 90 days)
makes no detectable difference.

## 2. Method

**Tuning is nested and time-aware.** For every outer fold of the rolling backtest only that fold's training window is used.
Its latest 90 days form the inner validation set, separated from the inner training days by a 1-day gap (the same gap as
the outer folds). Twelve settings are drawn at random from the space in `config/tuning.yaml`; the defaults are always the
first, so a tuned model is never worse than the default on its own validation set (tested). Each is fitted on the inner
training days and scored by MAE in price units on the inner validation days; the best is refitted on the whole training
window. The search is repeated every 180 days and the choice reused in between. Nothing from the outer test days, and
nothing after the fold's training end, enters the choice (checked by the perturbation test, which passes for a tuned
model). Loss function (squared, absolute, Huber) and objective are searched as settings.

**Combinations** use the out-of-sample predictions of the five models: mean, median, trimmed mean (drop the highest and
lowest of five), inverse-error weights, and the mean of the three models with the lowest recent error. Weights come from the
28-day trailing daily MAE of days up to D−2, because the errors of the two most recent days are not known when day D is
forecast (tested: changing the errors of D−1 and D leaves the weights unchanged; changing D−2 does change them).

**Calibration window and refit frequency:** LightGBM at its defaults, six combinations on the same delivery days, each
compared with 730 days refitted every 30 days.

## 3. Results (day-ahead price, development window)

MAE and RMSE in €/MWh, SSE in (€/MWh)²; 24,359 hours, 34 outer folds, 1,015 days.

| Model | Untuned MAE | Untuned RMSE | Tuned MAE | Tuned RMSE | Tuned vs untuned (DM) |
|---|---|---|---|---|---|
| Decision tree | 20.28 | 28.20 | 19.82 | 27.42 | −2.52, p 0.012 |
| Random forest | 18.11 | 24.85 | 18.06 | 24.77 | −0.74, p 0.46 |
| Gradient boosting | 17.96 | 24.72 | 17.77 | 24.60 | −1.38, p 0.17 |
| XGBoost | 18.24 | 25.08 | 17.72 | 24.66 | −3.59, p 0.0003 |
| LightGBM | 18.29 | 25.20 | **17.43** | 24.38 | −4.69, p < 0.0001 |

Combinations (against the best single model of the same group):

| Combination | Untuned MAE | vs gradient boosting (DM p) | Tuned MAE | vs tuned LightGBM (DM p) |
|---|---|---|---|---|
| Mean | 17.93 | 0.71 | 17.55 | 0.22 |
| Median | 17.81 | 0.030 | 17.43 | 0.98 |
| Trimmed mean | 17.81 | 0.039 | 17.43 | 0.99 |
| Inverse error | 17.91 | 0.53 | 17.53 | 0.33 |
| Top 3 mean | 17.86 | 0.13 | 17.36 | 0.32 |

Calibration window and refit frequency (LightGBM, defaults; DM against 730 days / 30 days):

| Window (days) | Refit every (days) | Folds | MAE | RMSE | DM | p |
|---|---|---|---|---|---|---|
| 365 | 30 | 34 | 19.08 | 25.98 | +3.73 | < 0.001 |
| 730 (reference) | 30 | 34 | 18.29 | 25.20 | | |
| 1095 | 30 | 34 | 17.75 | 24.49 | −3.41 | 0.001 |
| expanding | 30 | 34 | 17.56 | 24.31 | −4.22 | < 0.001 |
| 730 | 7 | 145 | 18.12 | 25.24 | −1.02 | 0.31 |
| 730 | 90 | 12 | 18.25 | 25.06 | −0.32 | 0.75 |

## 4. What this means

* **Tuning is worth doing for the boosted models and not for the forests.** The gain for LightGBM (0.86 MAE, 4.7%) is
  real, but the search is expensive; the cost grows with the number of targets and models.
* **Combinations are an insurance, not an improvement.** The tuned combinations are within 0.1 MAE of the best single model
  and cannot be told apart from it; their value is that they do not depend on picking the right model in advance. Simple
  averages did as well as error-weighted ones, as expected for five models with similar errors. The significant gains of
  the untuned median over gradient boosting (p 0.03–0.04) come from comparing five combinations with one model and are
  not adjusted for that.
* **More history beats more tuning here.** Going from a 730-day window to an expanding one improves LightGBM by the same
  amount as tuning it (0.73 against 0.86 MAE). This agrees with the Phase 4 finding that trees cannot extrapolate levels:
  a window that contains more price regimes helps, a short one forgets them. Refitting more often does not help, so 30
  days is kept, and a 90-day refit would be a cheaper alternative if time matters.
* The window result was obtained with defaults, and the tuning with the 730-day window. Whether the two gains add up
  (tuned LightGBM on an expanding window) is **not tested**.

## 5. Limits

* One target (day-ahead price) and the development window only. The locked test period has not been touched; IDA, aFRR and
  mFRR targets have not been tuned.
* Twelve random draws from a large space is a light search; a different seed or more draws could pick other settings.
  The search is seeded (`config/tuning.yaml`), and a repeat of the whole tuned run reproduced every score to the last digit.
* The significance tests are not adjusted for the number of comparisons made in this phase (five models, five
  combinations, six schedules).
* The first run did not save its tuning log because of a defect in the tool that printed it (fixed: it is now written to
  `tuning_log.csv`). The repeat run produced the log: six tuning rounds per model, and in all 30 the chosen setting beat the
  defaults on the inner validation, by 1.8% (random forest) to 7.3% (XGBoost) on average. These inner gains are larger
  than the outer gains in Section 3 (LightGBM 4.7%, XGBoost 2.8%, gradient boosting 1.1%, random forest 0.3%), as expected
  when the best of 12 draws is judged on the same block that chose it. The loss function chosen varies by round:
  LightGBM took the absolute-error objective in 4 of 6 rounds, XGBoost in 3 of 6, gradient boosting chose each of
  squared, absolute and Huber twice. The log is not committed (it is in `runtime/`).
* The expanding window grows the training time with the history; at the current size (about three years) it is acceptable.

## 6. Follow-up (2026-10-12): the stronger tuning and the other targets

The tuning of this phase was extended in `docs/Model_Development_Plan.pdf` (stages 1 to 10): 40 trials per algorithm with a seeded
Optuna search, three inner validation blocks, a random-search control and a numerical guard. The results are in
`model_development/README.md` and `docs/Model_Development_Results.pdf`. In short, for the day-ahead price, tuned LightGBM is the
champion (MAE 17.38 against 19.57 for the decision tree; nine of ten pairs of algorithms differ significantly); Optuna matches
random search with about a third fewer fits; and the gain seen on the validation blocks (2.5% to 8.4%) is larger than the gain on
the days that followed (0.4% to 5.1%). The 12-trial run of Section 3 is superseded by that experiment; its conclusions on
combinations and on the calibration window stand, but the combinations were measured against tuned LightGBM with 12 trials, not
against the 40-trial champion.
