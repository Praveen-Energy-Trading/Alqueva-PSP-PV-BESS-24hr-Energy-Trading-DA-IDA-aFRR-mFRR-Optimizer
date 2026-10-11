# Phase 2 — Data layer

Status: built and tested offline, 2026-10-11. Part of the programme in `docs/ML_Forecasting_Plan.pdf`.
Code: `common_layer/data_layer/`. Tests: `tests/test_data_layer.py`. Command line: `tools/ingest_market_data.py`.

## 1. Result in one paragraph

Market data now flows through three zones: **raw** (the downloaded bytes, never edited), **clean** (validated,
typed tables with lineage) and, later, **feature** (Phase 3). Every delivery day is stored with its **true
periods** (23, 24 or 25 hours; 92, 96 or 100 quarter-hours) and the UTC start of each period. The fixed 24/96-slot
view the optimiser needs is made in one documented place. **Exit criterion met on the saved real files:** the
clean zone was deleted and rebuilt from the raw zone alone, and every dataset came out bit-identical
(`test_exit_criterion_clean_zone_rebuilds_exactly_from_raw`). No live download and no history backfill has been
run yet (section 7).

## 2. Zones

| Zone | Location (under `runtime/data/`, not in git) | Content | Rule |
|---|---|---|---|
| Raw | `raw/<source>/<year>/<date>__vNNN__<sha10>.<ext>` and `raw/manifest.jsonl` | the file as downloaded | written once, never changed or deleted; SHA-256, size, URL, HTTP status and retrieval time (UTC) in the manifest |
| Clean | `clean/market_clean.db` (SQLite) | one table per dataset | written per delivery day in one transaction, replace-the-day, so repeating a load gives the same table |
| Load log | `load_log.jsonl` | one line per (source, day) attempt | outcome, reason, ingest id, checksum |

Because `runtime/` is not in git, the raw zone should be backed up outside the repository once it holds the
full history; it is the only copy of what the sources published at that time.

## 3. Versions and "as of" reads

If a source file for a delivery day changes (OMIE re-issues files; the saved 2025-10-26 file was issued on
2026-01-26), the new bytes are stored as **version n+1**; identical bytes only add an "unchanged" line. Older
versions stay. `rebuild_clean(..., as_of_utc=t)` rebuilds what was known at time `t`
(`test_rebuild_as_of_gives_the_state_known_at_that_time`). Each clean row stores `raw_sha256`, `raw_version`,
`parser_version`, `ingest_id` and `quality_flags`, so every number can be traced to its file.

## 4. Time handling

* Delivery days follow Europe/Madrid (CET/CEST); every period carries `start_utc`.
* DST days keep their true length: 2025-03-30 has 23 hourly rows, 2025-10-26 has 100 quarter-hours with two
  distinct "hour 3" slots, 2026-03-29 has 92 quarter-hours. Nothing is invented or averaged in the clean zone.
* `calendar.to_model_grid` is the one place that maps a true day onto the optimiser's 24/96 slots, with the same
  policy as the old pipeline (spring: fill the missing hour from its neighbours; autumn: average the two hour-3
  slots). The old loaders repeat this policy inline; they are not changed in this phase.
* `config/market_regimes.yaml` dates the rule changes (three intraday auctions, MARI, 15-minute settlement,
  15-minute day-ahead, price floor change, platform deadlines). `regime_at(day)` gives the resolution, number of
  intraday sessions and price floor in force. Two entries (Iberian exception dates, blackout effect) are marked
  `to_verify`.

## 5. Datasets and what the parsers keep

| Dataset | Source | Kept (beyond the old loaders) |
|---|---|---|
| `da_price` | OMIE day-ahead file | ES and PT price, purchase and sale volumes, Spain–Portugal flows, issue time |
| `ida_price` | OMIE `marginalpibcpt`, sessions 1–3 | ES and PT price per session |
| `xbid_price` | OMIE `precios_pibcic` | PT max, min, volume-weighted mean, ES mean, `pt_traded` flag |
| `afrr_price` | REN `BaFRRPreco` | initial and adjusted up/down prices, DA reference |
| `mfrr_price` | REN `MFRRPreco` | programmed-activation price, direct-activation prices (empty so far), DA reference |
| `imbalance_price` | REN `DesvioPreco` | short and long prices, DA reference |

The XBID file writes 0 in the Portuguese columns when there were no Portuguese trades. The parser turns these
into `pt_traded = 0` with empty prices; the old loader's synthetic filling is a modelling choice and belongs in
the feature zone, not here.

## 6. Data contracts

`contracts.validate_day` runs before a day enters the clean zone.

| Rule | Severity | Meaning |
|---|---|---|
| structure | error | required columns, no duplicate keys, rows exist |
| calendar | error | period count equals the true count for that day and resolution; indices 1..N |
| time | error | `start_utc` equals the calendar |
| missing | error / warning | NaN in a required column is an error; in an optional column a warning |
| range | warning | outside a plausibility band (catches unit errors, not unusual real prices) |
| below_floor | warning | DA price below the floor in force that day |
| regime_resolution | warning | DA resolution differs from the regime |

Errors reject the day (logged with the reason); warnings are kept as `quality_flags` on the rows. `freshness_issues`
compares each dataset's newest day with its allowed lag.

## 7. Ingestion

* `ingest_day`: download, check that the response parses as the expected file (an HTML error page with HTTP 200 is
  rejected and not stored), store raw, validate, write clean, log.
* `Fetcher`: User-Agent, 1 s between requests, retries with backoff for timeouts and HTTP 429/5xx, 404 or empty
  body reported as "not published".
* `ingest_range`: skips days already in the raw zone (safe to re-run); `refresh=True` re-checks the source.
* IDA files before 2024-06-13 belong to the old six-session regime and are not loaded under sessions 1–3.
* The command-line tool also offers `--rebuild` (no network) and `--verify` (checksums).

**Not done, needs your approval:** a first live run on a short window including 2025-10-26, and then the history
backfill (about 10 000 requests at one per second, roughly three hours, from 2019 for aFRR and 2020 for OMIE).

## 8. What this changes for the existing code

Nothing yet. The optimiser and the old forecasters still read their own loaders. Phase 3 (features) and later
phases read from the clean zone; the old loaders are replaced only when the new path matches them.

## 9. Limits

* Tested with real saved files for three DST days and for each source, but not against live responses.
* The contract ranges are plausibility bands set from experience, not from the full history; they should be
  reviewed once the backfill exists.
* REN endpoints are an internal website API with no published terms or service level (Phase 1); a changed field
  name would make parsing fail loudly (rejected day), not silently.
* Weather, inflow and fundamentals are not part of this phase; they wait for the source decisions in Phase 1.
