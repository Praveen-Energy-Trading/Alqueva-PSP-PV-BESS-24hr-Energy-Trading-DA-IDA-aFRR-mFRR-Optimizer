"""
forecast_evaluation — Phase 6: one contract that scores any forecaster of any target the same way.

    config.py       loads and validates config/evaluation.yaml
    asof.py         the as-of view: a forecaster only ever sees data its information set allows
    folds.py        rolling-origin folds with a retraining schedule and a gap
    metrics.py      SSE, MAE and RMSE, overall and by hour, month and period
    dm.py           Diebold-Mariano test with a HAC variance (finding F6)
    forecasters.py  the forecaster interface, what a forecaster is given, shared lag features
    tree_models.py  decision tree, random forest, gradient boosting, XGBoost, LightGBM
    backtest.py     run a forecaster over the folds and return predictions and metrics
    guards.py       the locked final test period and the leakage tests
    report.py       the report (Markdown, CSV) with the digests that make a run reproducible

Point metrics only: SSE, MAE and RMSE are the agreed set; probabilistic scores and profit-based decision
value are not part of this framework.
"""
