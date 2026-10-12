# Model development: training, tuning and selection

The plan is in [`docs/Model_Development_Plan.pdf`](../docs/Model_Development_Plan.pdf); the results so far are in
[`docs/Model_Development_Results.pdf`](../docs/Model_Development_Results.pdf). This folder holds the evidence: tuning
logs and figures, the comparison of the tuned algorithms, the champion decisions and the one-table summary
([`RESULTS.md`](RESULTS.md)). Code lives in `common_layer/model_development/` and `common_layer/forecast_evaluation/`.

| Stage | What | Status (2026-10-12) |
|---|---|---|
| 1 | Experiment registry: every run recorded with its scores, digests and code version; repeats are checked for identical scores | done |
| 2 | Fair, stronger tuning (equal budget, Optuna, three inner validation blocks, parallel block fits) | done; run on every target |
| 3 | Tuning evidence (every trial logged, figures) | done for the day-ahead price ([`tuning_evidence.md`](tuning_evidence.md)) |
| 4 | Algorithm comparison (pairwise Diebold-Mariano with Holm correction, win rates) | done for every target ([`algorithm_comparison.md`](algorithm_comparison.md), [`rollout/`](rollout)) |
| 5 | Written selection policy (`config/model_selection.yaml`, fixed before the results) | applied: a champion decision exists for six targets ([`selection/`](selection)) |
| 6 | Locked test of the champion (preflight checks, one logged look, report) | **not run**: waits for you (see below) |
| 7 | Champion registry and promotion rule (`config/champion_policy.yaml`) | built; **no champion registered**: waits for you |
| 8 | Monitoring and retraining (`config/monitoring.yaml`) | built; needs a registered champion |
| 9 | Roll-out to all price targets (`config/rollout.yaml`) | done for six of seven targets; mFRR deferred |
| 10 | Replace the old fixed-settings selection (`config/live_models.yaml`) | adapter and wiring done, every target still `legacy`; nothing deleted |

## Results

| target | treatment | development days | champion | MAE of the champion | better than the decision tree by |
|---|---|---|---|---|---|
| day-ahead price | standard | 1,015 | LightGBM | 17.38 | 11.2% |
| aFRR capacity (up) | standard | 1,014 | mean of random forest, gradient boosting, XGBoost, LightGBM | 13.49 | 4.1% (best single model) |
| IDA1 | limited evidence | 293 | mean of the same four | 5.01 | 4.7% |
| IDA2 | limited evidence | 298 | mean of the same four | 8.48 | 1.8% |
| IDA3 | limited evidence | 300 | mean of all five | 10.17 | 2.4% |
| XBID | limited evidence | 303 | mean of all five | 12.09 | 0.8% |
| mFRR | deferred | none | none | | |

MAE is in EUR/MWh (EUR/MW/h for aFRR capacity). All gains are measured against the decision tree, the reference of the
framework, not against a naive benchmark. They are the largest where the target is most predictable (day-ahead) and small
for the intraday spreads.

What the experiments showed:

1. **Day-ahead: LightGBM is clearly the best.** Nine of the ten pairs of algorithms differ significantly after the Holm
   correction, and LightGBM beats each of the others (decision tree 19.57, random forest 18.02, XGBoost 17.68, gradient
   boosting 17.64, LightGBM 17.38).
2. **Averaging helps only where the models are genuinely tied, and once clearly.** For aFRR capacity the average of four
   algorithms is significantly better than the best single model (13.49 against 13.56, p = 0.015). For the intraday targets
   the algorithms cannot be told apart, so the policy averages them; for IDA3 and XBID that includes the decision tree.
   With 300 days the data cannot separate them, which is the cost of the limited-evidence treatment.
3. **Optuna (TPE) converges about as well as random search but spends about a third less.** After 40 trials the best
   setting reaches 0.916 of the default error for LightGBM with TPE and 0.919 with random search (XGBoost 0.921 and
   0.923; gradient boosting 0.933 and 0.931). The pruner stopped 34% to 40% of the trials after one block.
4. **The gain seen when choosing a setting is larger than the gain afterwards.** On the validation blocks, tuning
   improved the error by 2.5% (random forest) to 8.4% (LightGBM); on the days that followed, by 0.4% to 5.1%. The chosen
   settings also change from round to round (the most frequent value of a setting was chosen in 33% to 83% of the
   rounds), so a single tuned setting should not be read as "the" best one.
5. **No single setting dominates.** In the random-search trials the share of the variance explained by one setting is
   at most 0.36 (tree depth for the decision tree), and 0.10 to 0.22 for the most important setting of the boosted models
   and the forest (the objective for XGBoost, the learning rate for LightGBM and gradient boosting).
6. **A numerical failure was found and guarded.** One tuned XGBoost in the random-search control used the pseudo-Huber
   objective and predicted prices up to 5e21 for 30 days of April 2023. Forecasts are now limited to the range of the
   training outcomes plus a margin of one unit (in the transformed space), in training, in the tuning validation and in
   the served champion (`tests/test_prediction_guard.py`). The Optuna runs never triggered it: the guarded rerun of the
   day-ahead experiment is bit-identical to the first. The random-search control still shows a bad XGBoost row (MAE 67),
   because the setting itself is unstable, so the control's XGBoost outer score is not used.
7. **Repeats reproduce.** The Optuna day-ahead experiment, run twice, gave identical predictions (registry: reproduced).

## Decisions that are yours

None of these was taken overnight; each is a one-time or outward-facing step.

1. **Freeze the locked outcomes** per target: `python tools/evaluate_forecaster.py --target <t> --lock`. The window ends on
   2026-10-11, so the last days may still be revised. Every other preflight check passes except that the code has
   uncommitted changes.
2. **IDA1's locked window has too many missing days:** 321 of 365 have outcomes, against 96% during development. They are
   scattered (1 to 4 per month, one run of 6 days in August 2026). If they are empty auctions the floor should be relaxed;
   if they are data that was not ingested they should be filled first. The preflight refuses IDA1 until this is settled.
3. **The one locked test per target:** `python tools/run_locked_test.py --target <t> --confirm-locked`. Each algorithm can
   be scored once.
4. **Register the champions:** `python tools/champion.py init --decision model_development/selection/<t>.json`, then fit
   (`champion.py fit`), calibrate the monitoring (`monitor_champion.py calibrate`).
5. **Switch live targets** in `config/live_models.yaml`, after `tools/compare_old_new.py` has recorded the champion against
   the legacy forecaster, and only then delete the legacy code.
6. **mFRR** stays deferred until the windows are rolled forward with a new locked period.

## Stage 1: the experiment registry

```
python tools/evaluate_forecaster.py --target da_price --tuned        # a run is recorded automatically
python tools/model_registry.py list --target da_price                # all experiments
python tools/model_registry.py show <run_key>                        # scores, tests and tuning choices of one
python tools/model_registry.py check                                 # experiments repeated: same scores every time?
```

An experiment is identified by what defines it (target, stage, test window, feature set, schedule, configuration, data
and tuning digests, forecasters). Running it again is a new attempt of the same experiment, and the registry compares
the scores of all attempts. The file is `runtime/model_registry/registry.db` (not committed). `check` reports the random-search
control as not reproduced on purpose: its second attempt ran with the guard of finding 6, which changed the unstable
XGBoost row.
