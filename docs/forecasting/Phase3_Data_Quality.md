# Phase 3 — Data quality and cleaning

Status: built and tested, 2026-10-11. Part of the programme in `docs/ML_Forecasting_Plan.pdf`.
Code: `common_layer/data_quality/`. Rules: `config/data_quality.yaml`. Tests: `tests/test_data_quality.py`.
Command line: `tools/run_data_quality.py`.

## 1. Result in one paragraph

Every dataset in the clean zone now gets automated checks, a daily score from 0 to 100, a Markdown report and a
**pass/fail gate** that a model run must clear (`runner.require_green()` raises if the gate is red, missing or
more than 26 hours old). Price spikes, negative prices and prices at the market limit are counted and flagged as
**information**, never penalised and never removed. Imputation rules are written per variable and can only look
backwards in time. Every row carries a live/synthetic label and synthetic rows are excluded from training and
testing. **The first full-history report is not produced yet:** the history backfill is still running, and the
thresholds below are starting values to be reviewed against that report.

## 2. Checks

| Check | What it looks at | Severity | Score penalty |
|---|---|---|---|
| `missing_day` | delivery days without rows since the dataset's expected start, per session for IDA; the cause comes from the load log (`not_published`, `empty_publication`, `not_attempted`) | warning | 100 (day scores 0) |
| contract flags | warnings attached when the day was loaded (`range`, `missing_all`, `below_floor`, `regime_resolution`) | warning | 10 / 5 / 10 / 10 |
| `flat_run` | an auction price at one value for a whole day (24 h) | warning | 25 |
| `duplicate_day` | at least 98 % of a day's periods equal the previous day's: the same (stale) file served again | warning | 25 |
| `dst_copy_suspect` | quarter-hour 25-hour day whose repeated hour is a plain copy of the first | warning | 25 |
| `cross_mismatch` | the REN files carry the day-ahead price; it must equal the OMIE value in the same period (hourly OMIE values are matched to their four quarters) | warning | 20 |
| `level_jump` | day median 20 times above or below the trailing 30-day median (unit or scaling errors) | warning | 20 |
| `spike_up` / `spike_down` | trailing 14-day robust z-score above 6 and at least 30 EUR/MWh away; uses only earlier values, so it could be a model input | info | 0 |
| `negative_price`, `at_limit` | counted per day; `at_limit` for day-ahead only | info | 0 |
| freshness | newest delivery day against each dataset's allowed lag | gate | — |
| raw checksums | every raw file against the manifest | gate | — |

## 3. What the first real data changed

Running the checks on the days already loaded (2020 to 2022 day-ahead plus the DST days) showed two rules that
looked sensible but were wrong, and one real anomaly:

* **Identical hours are not a stuck feed.** The Portuguese day-ahead price sat at exactly 1.95 EUR/MWh for 6 to 9
  hours on several days in 2020–2021. A "6 identical hours" rule fired on genuine prices. The rule is now a whole
  constant day, and a new `duplicate_day` rule targets the real failure (a stale file).
* **One equal hour is not a DST copy.** On the 25-hour day 2020-10-25, hour 3 and the repeated hour are both 1.95.
  With hourly data a single equal value proves nothing, so the check applies to quarter-hour days only, where four
  equal quarters in a row are evidence. It is tested on the real 2025-10-26 file (no flag) and on a forged copy
  (flag).
* **Aggregate constants are normal for reserves.** The aFRR band price is legitimately constant for hours and days,
  so the stuck-feed checks apply to auction prices only.
* **A real anomaly.** The day-ahead file of 2022-02-20 shows a Spain-to-Portugal flow of up to 5,085 MW, above any
  plausible interconnector level. It is flagged as a warning on that day and left unchanged.

## 4. Score and gate

Daily score = 100 minus the penalties above, floored at 0, per (dataset, session, delivery day). The gate is green
only if, for every dataset (per session for IDA):

* missing days are within the limit (2 %; IDA 6 %, XBID 5 %, because an auction can have no result on a day);
* the mean score of present days is at least 95 and at most 2 % of days score below 70;
* no feed is stale;
* all raw-zone checksums verify.

`tools/run_data_quality.py` writes `runtime/data/quality/dq.db` (issues and daily scores),
`runtime/data/quality/gate.json` and `runtime/reports/data_quality_<date>.md`, and exits with code 1 when red.
`tools/run_data_quality.py --require-green` only checks the stored result.

## 5. Imputation rules (as of the time of use)

Written per variable in `config/data_quality.yaml`; a validator refuses a price column without a written rule.

| Variable | Rule | Reason |
|---|---|---|
| day-ahead and intraday auction prices | none | a gap on a published day is an error; an unfilled auction means no clearing |
| XBID Portuguese mean price | none | no Portuguese trade is information (`pt_traded = 0`) |
| aFRR adjusted up/down price | carry forward, at most 4 periods | set per block, rarely changes |
| mFRR programmed-activation price | value exactly 24 h earlier | isolated gaps only |
| imbalance prices | none | settlement prices are not estimated |

Every filled value is marked in `<column>__imputed`. A test confirms that changing later values never changes
earlier filled values. The rules are applied when features are built (Phase 5); the clean zone is never filled.

## 6. Provenance

Market rows are `live` by construction (each traces to a downloaded file through its raw checksum). PV weather and
inflow, which come from static files, are labelled by date: real until 2026-06-30, `synthetic` after
(Phase 1 finding D1). `provenance.trainable` keeps live rows only. The weather and inflow data are not in the clean
zone yet, so the labelling is configured and tested but not yet applied to them.

## 7. Limits and next steps

* Thresholds (spike z-score, 24-hour constant day, 98 % duplicate share, mismatch tolerances, gate limits) were set
  before seeing the full history. Review them against the first full report; do not tune them to make the gate green.
* The first full-history run waits for the backfill; the stale-feed rule then uses the real newest day.
* Intraday auctions with no result are expected gaps. The gate allows a share of them but does not yet distinguish
  "source says no data" from "never attempted"; the report shows the cause, the gate does not use it.
* Spike flags use a trailing 14-day window and will fire often in 2021–2022 when the price level rose fast;
  that is information for regime analysis in Phase 4, not an error.
* The gate is not yet wired into any training script because no training script exists. Phase 7 onward must call
  `runner.require_green()` first.
