# Model development: training, tuning and selection

The plan is in [`docs/Model_Development_Plan.pdf`](../docs/Model_Development_Plan.pdf). This folder will hold the
evidence for it (tuning logs, figures, the comparison of the tuned algorithms and the selection rule). Code lives in
`common_layer/model_development/` and `common_layer/forecast_evaluation/`.

| Stage | What | Status |
|---|---|---|
| 1 | Experiment registry: every run recorded with its scores, digests and code version; repeats are checked for identical scores | built and tested |
| 2 | Fair, stronger tuning (equal budget, Optuna, several inner validation blocks, parallel block fits) | built and tested; day-ahead experiments running |
| 3 | Tuning evidence (every trial logged, figures) | tools built and tested; figures to be generated from the stage 2 runs |
| 4 | Algorithm comparison (pairwise Diebold-Mariano with Holm correction, win rates) | tool built and tested; to be run on the stage 2 predictions |
| 5 | Written selection policy (`config/model_selection.yaml`, fixed before the stage 2 results) | built and tested; to be applied to the stage 2 predictions |
| 6 | Locked test of the champion (preflight checks, one logged look, report) | built and tested; not run: needs the final champion and the frozen outcomes |
| 7 | Champion registry and promotion rule (`config/champion_policy.yaml`) | built and tested; no champion registered yet (waits for the final decision) |
| 8 | Monitoring and retraining (`config/monitoring.yaml`) | built and tested; waits for a registered champion to monitor |
| 9 | Roll-out to all price targets (`config/rollout.yaml`: standard for day-ahead and aFRR capacity, limited evidence for IDA1-3 and XBID, mFRR deferred) | tooling built and tested; runs wait for the stage 2 experiments |
| 10 | Replace the old fixed-settings selection (`config/live_models.yaml`: every target starts as `legacy`; switch per target after `tools/compare_old_new.py`; delete the old code last) | adapter, wiring and comparison built and tested; no target switched; nothing deleted |

## Stage 1: the experiment registry

```
python tools/evaluate_forecaster.py --target da_price --tuned        # a run is recorded automatically
python tools/model_registry.py list --target da_price                # all experiments
python tools/model_registry.py show <run_key>                        # scores, tests and tuning choices of one
python tools/model_registry.py check                                 # experiments repeated: same scores every time?
```

An experiment is identified by what defines it (target, stage, test window, feature set, schedule, configuration, data
and tuning digests, forecasters). Running it again is a new attempt of the same experiment, and the registry compares
the scores of all attempts. The file is `runtime/model_registry/registry.db` (not committed).
