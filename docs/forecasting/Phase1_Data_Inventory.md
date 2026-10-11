# Phase 1 — Data inventory and access

Status: draft for review, 2026-10-11. Part of the programme in `docs/ML_Forecasting_Plan.pdf`.
Machine-readable version: `config/forecast_data_inventory.yaml` (validated by `tests/test_forecast_data_inventory.py`).
Depends on the forecast catalogue (`config/forecast_catalogue.yaml`, Phase 0).

## 1. Result in one paragraph

Seven of the nine forecasts rest on market-price files that have a working live feed (OMIE and REN).
The two physical forecasts do not: **PV weather and reservoir inflow have no live feed, their real data end
on 30 June 2026, and everything after is synthetic or climatology.** No fundamentals (load, wind and solar
forecasts, gas, carbon, reservoir levels) are collected at all, no source keeps what was published at the
time (vintages), and the licence terms of the OMIE and REN data have not been verified. Free candidates exist
for most gaps.

## 2. Sources in use

| Source | Content | Access | Live feed | History | Licence |
|---|---|---|---|---|---|
| OMIE day-ahead | PT/ES DA prices, hourly, 15-min from 2025-10-01 | file download, no key | yes | 2020-01-01 to 2026-10-10 | **unverified** (read "Aviso legal" on omie.es) |
| OMIE IDA1/2/3 | intraday auction prices | file download, no key | yes | 2024-06-13 to 2026-10-09 | **unverified** |
| OMIE continuous (XBID) | weighted mean per hour (MedioPT) | file download, no key | yes | 2024-06-13 to 2026-10-09 | **unverified**; order-book depth needs a paid EPEX SPOT subscription |
| REN market API | aFRR band price, mFRR price series, imbalance price | JSON, static key copied from REN's website | yes | aFRR 2019-01-01, mFRR 2024-11-27 to 2026-10-09 | **unverified**; internal website API, no published terms or service level |
| PV weather file | GHI, ambient temperature, hourly | static Excel file | **no** | 2015-01-01 to 2026-06-30 real, then synthetic | unknown origin |
| Inflow file | daily inflow, m3/h | static Excel file | **no** | 2015-01-01 to 2026-06-30, then climatology | unknown origin |

## 3. Candidates and gaps

| Need | Candidate | Access | Licence and limits | Notes |
|---|---|---|---|---|
| Load, wind, solar forecasts, generation (PT/ES) | **ENTSO-E Transparency Platform** | API, free token (register, then email transparency@entsoe.eu) | free of charge by EU Regulation 543/2013; open licence for TSO data; 400 requests/min | forecasts stored as published; per-item coverage to audit |
| Spanish demand, wind, solar forecasts | **REE ESIOS** | API, personal token by email to REE | terms to read at token request | Iberia is coupled, so Spanish fundamentals matter for Portuguese prices |
| Portuguese consumption, production | **REN Data Hub** | JSON API (electricity endpoints to confirm) | described as open data | current-day values are provisional |
| Irradiance history (replace the unverified file) | **NASA POWER**, **PVGIS**, **CAMS** | API; CAMS needs a free account | NASA CC BY 4.0; PVGIS and CAMS terms to read; CAMS 100 requests/day | reanalysis or satellite, not a plant sensor |
| Weather forecasts with issue time | not chosen | to decide | Open-Meteo free tier is non-commercial only (CC BY 4.0, 10,000 calls/day); archived forecast runs not confirmed | needed for point-in-time tests |
| Gas (TTF) and carbon (EUA) prices | not chosen | to decide | exchange data is often licensed | a daily series is enough |
| Reservoir level, flows, rainfall | **EDIA** daily bulletins (PDF), **SNIRH/APA** | PDF scraping; SNIRH API not confirmed | no reuse terms found for EDIA; SNIRH metadata mirrors show CC BY-NC 4.0 | inflow can only be derived from volume change and releases |

Recommendation for the use made here (research and portfolio, not a commercial service): the free tiers
above are acceptable, but the licence of every source is recorded so a commercial use can be checked later.

## 4. Data dictionary (stored files)

All dates are delivery dates (local market calendar). Prices are EUR/MWh unless stated; "source" is the
provenance label (live, synthetic, climatology).

| File (sheet) | Columns | Resolution | Time convention |
|---|---|---|---|
| `da_training_data_2020_2026.xlsx` | Date, Hour 1-24, `price_DA_PT_EUR_MWh`, `price_DA_ES_EUR_MWh`, source | hourly | 24 slots per day |
| `da_training_data_isp_2025_2026.xlsx` | Date, ISP 1-96, `price_DA_PT_EUR_MWh`, source | 15 minutes | 96 slots per day |
| `ida{1,2,3}_training_data_2024_2025.xlsx` | Date, Hour, `price_DA_PT_EUR_MWh`, `price_IDA_PT_EUR_MWh`, `spread_EUR_MWh` (= IDA minus DA), source | hourly | 24 slots per day (IDA3 covers hours 12-24 only) |
| `xbid_training_data_2024_2025.xlsx` | Date, Hour, `price_DA_PT_EUR_MWh`, `price_XBID_PT_EUR_MWh`, `spread_EUR_MWh`, source | hourly | 24 slots per day |
| `afrr_training_data_2019_2025.xlsx` | Date, Hour, `price_DA_PT_EUR_MWh`, `cap_up_EUR_MW`, `cap_dn_EUR_MW`, source | hourly | band price per MW per hour, 15-min values averaged |
| `mfrr_training_data_2024_2025.xlsx` | same columns as aFRR | hourly | the series is the activation energy price, not a capacity price |
| `pv_training_data_from_2015.xlsx` | Date, Hour, `GHI` (W/m2), `T_amb` (C), source | hourly | 24 slots per day |
| `inflow_training_data_from_2015.xlsx` | Date, `inflow_m3h` (m3/h), source | daily | one value per day |

The file names carry stale year ranges; the real spans are in the tables above.

## 5. Lineage map: from source to decision

| Source | Loader | Stored file | Forecaster | Catalogue target | Decision |
|---|---|---|---|---|---|
| OMIE DA | `omie_da_price_loader.py` | DA files | `da_price_forecaster.py` | `da_price` | day-ahead bid |
| OMIE DA | same | DA files (as feature `da_price`) | all intraday and reserve forecasters | `ida*_price`, `xbid_price`, `afrr_cap_price`, `mfrr_price` | intraday and reserve decisions |
| OMIE IDA | `omie_ida*_price_loader.py` | IDA files | `ida*_price_forecaster.py` | `ida1/2/3_price` | IDA re-optimisation |
| OMIE continuous | `xbid_price_loader.py` | XBID file | `xbid_price_forecaster.py` | `xbid_price` | XBID windows |
| REN aFRR | `picasso_afrr_price_loader.py` | aFRR file | `afrr_price_forecaster.py` | `afrr_cap_price` | aFRR offer |
| REN mFRR | `mari_mfrr_price_loader.py` | mFRR file | `mfrr_price_forecaster.py` | `mfrr_price` | no revenue consumer today |
| (none) static file | `_fill_gaps` (synthetic) | PV file | `pv_power_forecaster.py` | `pv_weather` | PV availability |
| (none) static file | `_fill_gaps` (climatology) | inflow file | `reservoir_inflow_forecaster.py` | `reservoir_inflow` | water balance |
| REN imbalance | `ren_imbalance_price_loader.py` | none (read on demand) | none | not forecast | settlement only |

## 6. Findings of the data audit

| # | Severity | Finding |
|---|---|---|
| D1 | **high** | PV weather and inflow have no live feed. Real data end on 2026-06-30 and every later row is synthetic (PV: 257 days to 2027-03-14) or climatology (inflow: 174 rows). Recent training history and the PV "actuals" are not real. |
| D2 | medium | PV weather provenance is unknown. The label PLANT_SENSOR cannot be literal before mid-2022 (the floating PV pilot was inaugurated in July 2022); July daily irradiance varies by only 11%. Replace or validate against NASA POWER or PVGIS/CAMS and archive weather forecasts with their issue time. |
| D3 | medium | REN data come from the website's internal API with a static key and no published terms or service level. Build a monitored loader that stores the raw response. |
| D4 | medium | No fundamentals are collected: load, wind and solar forecasts, gas, carbon, reservoir levels. |
| D5 | medium | The stored files hold exactly 24 (or 96) slots on every day, including the daylight-saving days that have 23 or 25 hours, and the raw downloads are not kept, so the fix requires a rebuild from the source. Some files also contain synthetic rows (Phase 0, F7). |
| D6 | medium | No source keeps vintages (what was published at the time), and our own forecasts are not archived, so point-in-time testing of features and of past forecasts is not yet possible. |

## 7. Decisions and actions for you

1. **Licences.** Read the "Aviso legal" on omie.es and REN's market-site terms (I could not retrieve either online) and tell me what they allow, or confirm the use is non-commercial research.
2. **Tokens.** Request an ENTSO-E token (register, then email transparency@entsoe.eu) and, if you want Spanish fundamentals, an ESIOS token (email to REE). Free; I will not do this for you.
3. **Weather.** Choose how to replace the PV weather history: NASA POWER (open licence, simplest), PVGIS/CAMS (better satellite quality, CAMS needs an account), or both.
4. **Weather forecasts with issue time.** No source is confirmed; decide whether to start archiving forecasts from now (Phase 2) even if the history will be short.
5. **Inflow.** Decide whether to keep the research dataset as is (stale after June 2026) or build a derived inflow from EDIA/SNIRH data.

## 8. Exit criterion

"Every input has an owner, a timestamp rule, a licence note and a quality note": met in
`config/forecast_data_inventory.yaml` (checked by test). Open items are the licence terms, the tokens and the
weather source choice above; they are inputs to Phase 2 (ingestion), not blockers for starting it.
