# Repository audit: what is necessary and what can go

Audit of every tracked file (577), 2026-10-12. Nothing has been deleted; this document is the basis for deciding.

## How it was done

1. **Inventory.** Every tracked file with its size, last change in git and kind (core library, phase code, tests, tools, dashboard,
   configuration, documents, results).
2. **Reachability.** The imports of every Python file were parsed. A module is *reachable* if it can be reached from a real entry
   point: `run_production.py`, the `run_*.py` of each phase, the dashboard (app and pages) and the `tools/`. Tests do not count
   as use. Every non-Python file was searched for mentions in code, configuration, documents and tests.
3. **Judgement.** Each unreached or unmentioned file was opened and read. Dead code, stale copies, duplicates and generated
   output were separated from files that are only indirectly used.
4. **Safety net.** The full test suite passes (607 passed on this PC), so any deletion below can be checked by rerunning it.

Limits: a static scan cannot see imports built from strings. The ones that exist (the tools load the legacy forecasters by name)
were checked by hand.

## Result in one table

| Class | Files | Size | What it means |
|---|---|---|---|
| 1. Delete now | 15 | 1.6 MB | Dead, empty, stale or unused; nothing depends on them |
| 2. Built but unused | 13 | 0.1 MB | Working, tested code that no part of the pipeline calls; your call |
| 3. Legacy, deletable after the switch | 20 | 3.5 MB | The old forecasters' training scripts and data for day-ahead, IDA1-3 and XBID; replaced only once the champions are switched on |
| 4. Legacy, must stay | 20 | 5.7 MB | The same kind of files for PV, inflow, aFRR, mFRR and the 15-minute day-ahead forecast; the new framework does not cover these |
| 5. Optional | 15 | 1.1 MB | Deployment files, notes and generated output; depends on how you use the repository |
| 6. Necessary | 494 | 6.1 MB | The pipeline, the new framework, the tests, configuration, documents and results |

## Class 1: delete now (15 files)

| File(s) | Why |
|---|---|
| `tempCodeRunnerFile.py` | Editor leftover (3 lines), imported by nothing |
| `phase_2_da_day_ahead_bidding/market_data_loaders/__init__.py` | An empty package; the folder holds nothing else |
| `phase_5b_afrr_activation_response/afrr_activation_tracking/` (2 files) and `phase_5c_mfrr_activation_response/mfrr_activation_tracking/` (2 files) | 16-line loggers that nothing imports; the activation handlers do their own logging |
| `common_layer/gate_scheduler/` (3 files) | A wall-clock scheduler for unattended operation that no runner calls and no test covers; the demo runs the phases by hand. Only the README and a package docstring mention it |
| `.claude/launch.json` | A local preview setting with a path on your own machine; belongs to the editor, not the repository |
| `docs/Mathematical Modeling — ....docx` and `.pdf` | July snapshots of the same document as `docs/mathematical_modeling.md`, which is the live copy and was updated for the renumbering of the phases on 2026-10-11 |
| `docs/code_structure.docx` and `.pdf` | A July snapshot that still describes a "15-phase" structure; stale, and the README tree is now the live description |
| `tests/fixtures/raw/da_20261001.txt` | A saved market file that no test opens |

Also to update when deleting: the README tree entry for `gate_scheduler/` and the package docstring in `common_layer/__init__.py`.

## Class 2: built but unused (13 files; your call)

These work and are tested, but nothing in the pipeline calls them.

| File | What it is | If deleted |
|---|---|---|
| `common_layer/feature_engineering/store.py` | A feature store (features saved for reuse); the builder computes features on the fly instead | Remove one test in `tests/test_feature_engineering.py` |
| `common_layer/data_quality/imputation.py` | Written fill rules for gaps (comment: "used when features are built") | Remove tests in `tests/test_data_quality.py` and the `imputation:` section of `config/data_quality.yaml` |
| `common_layer/data_quality/provenance.py` | Labels rows live or synthetic | Same; and the `provenance:` section of the same config. Note that this is the kind of label the artificial IDA1 fill could use |
| `common_layer/forecast_formulation/targets.py` | Builds a target and converts a forecast back to a price; the as-of view does this now | Remove tests in `tests/test_forecast_formulation.py` |
| `common_layer/physical_plant_models/reservoir_activation_checker.py` | A reservoir safety check for activations. It is documented (README, mathematical model) and tested, but **no runner calls it** | Remove 2 tests in `tests/test_reserve_realtime_delivery.py`, a README line and a section of `docs/mathematical_modeling.md`. Better: decide whether it should be wired into the real-time run; it reads like a safety requirement, not clutter |
| `common_layer/configuration/forecast_catalogue.py`, `forecast_data_inventory.py`, `config/forecast_catalogue.yaml`, `config/forecast_data_inventory.yaml`, `tests/test_forecast_catalogue.py`, `tests/test_forecast_data_inventory.py`, `docs/forecasting/Phase0_Forecast_Catalogue.md`, `Phase1_Data_Inventory.md` | The generators and sources of the Phase 0 and Phase 1 documents of the forecasting programme | Keep if you want to regenerate those two documents; otherwise they are history |

My recommendation: delete `store.py`, `imputation.py`, `provenance.py` and `targets.py` with their tests (4 files plus test and
config sections, no behaviour lost). Decide about the reservoir checker separately, and keep the Phase 0 and 1 generators.

## Class 3: legacy, deletable after the switch (20 files, 3.5 MB)

For day-ahead (hourly), IDA1, IDA2, IDA3 and XBID, the new framework replaces the old selection. Each of the five has four
legacy files: the offline evaluation script (`*_train_val_test.py`), its report (`*_evaluation_report.md`), the saved selection
(`*_selected_model.json`) and the training workbook (`*_training_data_*.xlsx`). They are still used today: the legacy
forecasters read them, and **the dashboard's ML Forecasting page reads the selection files and workbooks directly**
(`dashboard/data.py`). So deletion has three prerequisites, in this order:

1. The champions are registered and the targets are switched (`config/live_models.yaml`, mode `registry`).
2. The dashboard page is repointed to the registry and `model_development/` results.
3. The legacy model code inside the five forecaster modules is removed, and `tests/test_model_selection_metrics.py` and the
   legacy parts of the stochastic IDA tests are rewritten.

Then the 20 files go, in a separate commit so that history can restore them.

## Class 4: legacy, must stay (20 files, 5.7 MB)

The PV and reservoir-inflow forecasters (not price targets), the aFRR and mFRR price forecasters (aFRR returns an up and a
down price and the framework models only up; the legacy mFRR forecast is the capacity price while the framework's target is
the activation price) and the 15-minute day-ahead forecaster are outside the new framework. Their scripts, reports, saved
selections and workbooks (`pv_`, `inflow_`, `afrr_`, `mfrr_` and `da_*_isp_`) stay. So does `ml_train_val_test_common.py`, which
they share, and `tools/rebuild_real_market_history.py`, which rebuilds some of the workbooks.

## Class 5: optional (15 files)

| File(s) | Comment |
|---|---|
| `.github/workflows/tests.yml`, `data-pipeline.yml` | Now manual-only, because you run tests on your PC. Delete them if GitHub will never run tests |
| `Dockerfile`, `.dockerignore` | A container that runs the test suite; mentioned nowhere else. Keep only if you use Docker |
| `docs/Reserve_Allocation_Interview_Answer.md`, `docs/aFRR_mFRR_Reserve_Sizing_Concept.md` | Interview and concept notes; not part of the pipeline, useful to you |
| `docs/results/backtest_2025-10-01_365d.xlsx` | A saved backtest output, referenced once |
| `figures/output/*.png` (8 files) | Generated by the production run and shown on the dashboard; regenerable |

## Class 6: necessary (494 files)

| Area | Files | Role |
|---|---|---|
| `common_layer/` (without class 1 and 2) | 100 | Configuration loaders, data layer, data quality, database, optimisation model, plant models, forecast evaluation, feature engineering, market analysis, model development, utilities |
| `model_development/` | 100 | Generated results of the ten stages: decisions, evidence tables and figures, comparisons, the summary |
| `tests/` | 66 | The 607-test suite and its saved market files |
| `docs/` | 26 | The plans and results (LaTeX and PDF), the phase documents, the mathematical model, the architecture diagram |
| `config/` | 23 | Every configuration file is read by code (checked file by file) |
| `tools/` | 19 | The command-line tools; all are documented in the README |
| phase folders | 137 | The daily pipeline runners, forecasters, bid formatters, settlement and analytics, backtesting |
| `dashboard/` | 16 | The Streamlit app and pages; every module is reached from the app or a page |
| root | 5 + 2 | `README.md`, `run_production.py`, `requirements.txt`, `PHASE-1-SPECIFICATION.md` (the original specification the code is checked against; it could move to `docs/`), `.gitignore`, `.streamlit/config.toml`, `figures/__init__.py` |

Other observations (not deletions):
- `phase_7_backtesting_and_validation/run_risk_comparison.py` is mentioned nowhere but is a real experiment script (CVaR against
  expected value on real prices). Keep; it deserves a line in the README.
- Several live modules have no test of their own (the position, reserve and audit stores, the reserve and IDA re-optimisers,
  the scenario generator, parts of the plant models). They run inside the end-to-end tests, so this is a coverage gap, not clutter.

## Repository size

The `.git` folder is 228 MB, of which a single 97 MB `.rar` archive of the whole project is the biggest part. It was committed on
2026-10-02 and removed the same day, so it is in history only. It cannot be removed without rewriting history (a force push),
which I do not recommend unless the size matters to you; the files in this audit total about 18 MB.

## Proposed deletion plan

| Step | What | Check |
|---|---|---|
| A | Class 1 (15 files) and the README and docstring lines | Full test suite |
| B | Class 2, the four I recommend deleting, with their tests and config sections | Full test suite |
| C | Decision on the reservoir checker (wire in, or delete) | Real-time tests |
| D | After the switch: repoint the dashboard page, remove the legacy code for the five switched targets, delete class 3 | Full test suite, dashboard start |

Each step is one commit, staged by explicit file names.
