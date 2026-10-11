"""
forecast_formulation — Phase 5: how each target is posed as a forecasting problem.

    transforms.py   variance-stabilising transform that allows negative prices (inverse hyperbolic sine)
    coherence.py    hourly and 15-minute forecasts that agree with each other
    targets.py      builds each target (level or spread to the day-ahead price) from the clean zone, and
                    turns a forecast of the modelled quantity back into a price
    evidence.py     the measurements behind each formulation choice (written up in
                    docs/forecasting/Phase5_Problem_Formulation.md)

The decisions themselves live in config/forecast_formulation.yaml, validated against the forecast catalogue,
the data inventory and the publication times by common_layer/configuration/forecast_formulation.py.
"""
