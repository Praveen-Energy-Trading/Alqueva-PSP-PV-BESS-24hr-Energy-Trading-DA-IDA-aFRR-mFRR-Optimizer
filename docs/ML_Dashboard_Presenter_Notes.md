# ML Lifecycle dashboard: presenter notes

What to say on each page, the likely follow-up question with its answer, and the limits of each page. These used to sit inside the dashboard; they are here so that the dashboard shows only the project. Written in the first person for reading aloud.

## ml_1_data: Data and quality

**What to say.** I treat data work as the first model risk. The raw zone is immutable and checksummed, the clean zone is rebuilt from it, a daily quality score is computed per dataset, and then a curation step removes data errors and fills every missing hour by the rule of its series, flagging each filled value. I also handle time properly: 23 and 25 hour days, a change of resolution in the market in October 2025, and sources that publish late.

**Likely question.** Missing data is filled: how do you know the fill is not hiding a problem?

**Answer.** Every filled value is flagged and counted per series, so the share is visible: for example about 7.5% of the IDA1 hours, mostly empty auctions, which I fill with the day-ahead price plus the typical spread of that hour. Real spikes and negative prices are kept; I only remove values that are outside their legal range, days that repeat or stay flat, and unit-size jumps. The model evaluation and the locked test score the published values only, so the filling cannot flatter the results; it keeps the live pipeline running when a feed has a gap.

**Limits.** Quality is measured on what the sources publish; it cannot tell whether a published value is correct. The pack holds a snapshot of the quality database, not live data.

## ml_2_market: Market analysis

**What to say.** Before choosing a model I looked at the data: the level moves slowly and drives most of the variance, the daily shape has changed because of solar, and the intraday prices are the day-ahead price plus a small spread. Each of those findings became a modelling decision: recent-level features, hour and month features, a rolling training window, and a spread target for the intraday markets.

**Likely question.** How do you know the rule changes did not break your series?

**Answer.** I tested for structural breaks around each documented event. The level moved mostly with fuel and hydrology, not with the rules, so the training window is rolling and recent rather than the whole history. The regime shading above is what I use to report results by period.

**Limits.** Daily means hide intra-day detail; the full hourly analysis (distribution, spikes, negative hours) is in the report figures. The pack holds a snapshot of the clean zone.

## ml_3_formulation: Problem formulation

**What to say.** I wrote down for every target what it forecasts, at what time the forecast is needed and which series exist by then, in one file. The evaluation code reads that file, so a model physically cannot see data it would not have had at the time. Intraday prices are modelled as a spread to the day-ahead price, and prices are transformed with asinh so that spikes and negative prices are handled.

**Likely question.** Why not a logarithm or a plain model on raw prices?

**Answer.** Prices go negative, so a logarithm is not defined, and raw prices let the largest spikes dominate the loss. The inverse hyperbolic sine behaves like a logarithm for large values and like a straight line near zero, and I chose its scale by measuring the dependence of the daily change on the price level, not by hand.

**Limits.** Publication times are rulebook times or measured from the data (day-ahead files: median 13:29, 95% by 14:37); the late-publication policy is in the formulation file. Fundamentals (load, wind, solar) are defined but not collected, so no model uses them yet.

## ml_4_features: Features

**What to say.** Features are a catalogue, not code scattered over scripts. Each one says what it reads, and the system derives from that whether a target may use it at its forecast time. A single builder produces the features for training, evaluation and live use, and a test proves that replacing unknowable data with noise changes nothing.

**Likely question.** How do you know you have no data leakage?

**Answer.** Three independent protections: the information set is written per target and enforced by the as-of view, the feature builder reads only through that view, and an automated perturbation test randomises everything that must not be known and checks that the forecast does not move. I also tried to cheat on purpose, and the check caught it.

**Limits.** The catalogue holds 77 features, of which 9 (load, wind, solar, gas, carbon, hydro, weather forecasts) are defined but not collected. Stability selection is run on development data only and gives small sets; the champions use the baseline sets.

## ml_5_evaluation: Evaluation design

**What to say.** I fixed the scoring contract first: a rolling origin that refits as the live system would, a gap so training never touches the day before the first forecast, and a final year that is locked and scored once. Differences between models are tested on daily losses with a variance that allows for autocorrelation, not read off a single number.

**Likely question.** Why not cross-validation with random folds?

**Answer.** Prices are a time series: a random fold would let the model train on days after the ones it forecasts, which is leakage and inflates accuracy. A rolling origin trains only on the past, like production. The locked year then gives one clean estimate that nothing was tuned on.

**Limits.** The test assumes daily loss differences have a stable variance; with about 300 days (the intraday targets) only larger differences can be established. Tests are not adjusted across targets.

## ml_6_tuning: Tuning lab

**What to say.** Tuning is nested and time-aware: inside each refit the search only sees that refit's past, scored on three later blocks, with the same trial budget for every algorithm and the defaults always included. I keep every trial so I can show the search, not just the winner.

**Likely question.** Is tuning worth it, and did Optuna beat a random search?

**Answer.** Tuning helped the boosted models clearly (LightGBM 4.9% lower error, XGBoost 3.0%, both significant), gradient boosting a little (1.8%, borderline) and the random forest not significantly (0.5%). Optuna matched random search in quality while skipping about a third of the fits. And the gain I saw while choosing was larger than the gain afterwards, which is why the final number comes from the locked year.

**Limits.** Day-ahead has the full comparison with the random control; the other targets show their Optuna runs only. The control's XGBoost row is unreliable (the unstable setting), so it is used for the search behaviour, not for outer scores.

## ml_7_arena: Model arena

**What to say.** I compare algorithms by their out-of-sample errors on the same days, never by their settings: each is tuned with the same budget and then scored the same way. For every pair I test whether the difference is real, and I correct for the number of pairs, because with ten pairs one will look significant by chance.

**Likely question.** How do you compare a decision tree and LightGBM when they have different hyperparameters?

**Answer.** The hyperparameters are internal to each algorithm. What I compare is the forecast error on days the model has not seen, after each algorithm has been tuned fairly. The Diebold-Mariano test then says whether the gap is larger than day-to-day luck, and the Holm correction keeps the many pairwise tests honest.

**Limits.** For the intraday targets (about 300 days) most pairs cannot be separated, so the comparison mostly shows ties; ties are resolved by the selection rule on the next page, not by picking the lowest number.

## ml_8_decision: Decision and locked test

**What to say.** The choice of champion is not a judgement call: a written rule, fixed before the results, is applied by code, and the decision file lists every test and reason. Then the champion is scored once on a locked year. I report the gain over the reference on the same hours, because the raw error depends on the price level of the year.

**Likely question.** What if the locked-year result had been bad?

**Answer.** I would have reported it as it is. The test cannot change the decision, and every algorithm can be scored on the locked year once, so there is no repeating until it looks good. For day-ahead the gain over the reference was 11% in development and 6% on the locked year: smaller, as expected after choosing the best of many, and still highly significant. That gap is exactly why the locked year exists. I also compare with naive forecasts that need no model. The strongest for day-ahead is yesterday's price: the champion is 5.7% better on the locked year (p = 0.01). For the intraday auctions the champions are only 0.4% to 1.9% better than simply assuming the day-ahead price, which I report as it is: with these inputs the intraday spread is largely unpredictable.

**Limits.** The intraday targets use limited evidence (about 300 development days). IDA1's locked window includes 44 artificial days (a demonstration choice), flagged above with the scores on real days only. The reference of the selection rule is a decision tree; the naive forecasts are reported next to it and do not take part in the choice.

## ml_9_run: Run it: registry, monitoring, live

**What to say.** A model is not finished when it is accurate. Every run is recorded and reproducible; a champion is a versioned, checksummed artifact; a challenger replaces it only under a written rule; the champion's error is watched against what was normal in development; and the live switch is a one-line change that falls back safely.

**Likely question.** How would you know the model has stopped working?

**Answer.** The rolling error is compared with a threshold taken from its own development history. In the real locked year the day-ahead champion's rolling error crosses its threshold at the end of September 2026, and the monitor flags it. I can also simulate a market change here to show the flag firing on demand. The response is a review and a re-tune, never an automatic switch.

**Limits.** No champion is registered yet and no live forecast is switched: those are deliberate steps. The monitoring page replays the locked year; it is not live monitoring. The registry snapshot is committed with the pack, the registry itself is a local database.

