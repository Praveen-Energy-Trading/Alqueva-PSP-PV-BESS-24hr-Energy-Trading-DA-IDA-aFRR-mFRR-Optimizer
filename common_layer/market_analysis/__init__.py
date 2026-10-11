"""
market_analysis — Phase 4: exploratory analysis and market understanding.

    config.py     loads config/market_analysis.yaml
    panel.py      aligned hourly UTC panel of all price series from the clean zone, plus calendar features
    seasonality.py  daily / weekly / annual structure (profiles and variance shares)
    dependence.py   autocorrelation, volatility clustering, stationarity tests
    breaks.py       structural breaks: tests at known events and data-driven changepoints
    extremes.py     spikes, negative prices, tails, solar cannibalisation, heteroscedasticity
    relationships.py  links between day-ahead, intraday, continuous market, reserves, imbalance
    sample_size.py  effective sample size and which history is still representative
    figures.py      the figures of the findings note
    run.py          run everything, write tables and figures

Every function takes plain pandas objects and returns plain pandas objects, so each can be tested on
synthetic series with a known answer (tests/test_market_analysis.py).
"""
