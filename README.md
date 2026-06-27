# ⚡ Alqueva PSP + PV + BESS — 24-Hour Energy Trading Optimizer

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Solver-CPLEX%2022.1-brightgreen?style=for-the-badge&logo=ibm&logoColor=white"/>
  <img src="https://img.shields.io/badge/Fallback-HiGHS%20%7C%20CBC-green?style=for-the-badge"/>
  <br/>
  <img src="https://img.shields.io/badge/Optimisation-MILP-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Markets-DA%20%7C%20IDA%20%7C%20XBID%20%7C%20aFRR%20%7C%20mFRR-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Grid-MIBEL%20%2F%20OMIE-red?style=for-the-badge"/>
</p>

<p align="center">
  <b>Production-grade 24-hour MILP trading optimizer for the Alqueva hybrid energy plant (Portugal / MIBEL)</b><br/>
  Pumped-Storage Hydro ,  Floating PV ,  Battery Storage ,  DA / IDA / XBID / aFRR / mFRR ,  Full Settlement & Analytics
</p>

<p align="center">
  <code>15 pipeline phases</code> , 
  <code>16 entry points</code> , 
  <code>9 production figures</code> , 
  <code>5-sheet Excel report</code> , 
  <code>4 YAML configs</code> , 
  <code>1 shared MILP model</code>
</p>

---

## 🗺️ Pipeline Architecture

<p align="center">
  <img src="docs/pipeline_architecture.png" alt="Alqueva 24-Hour Trading Pipeline" width="840"/>
</p>

---

## 🏭 Plant

| Asset | Specification |
|-------|--------------|
| **PSP — Pumped Storage** | 4 × reversible Francis units, 129.6 MW turbine / 111.6 MW pump each → **518.4 MW generation / 446.4 MW pumping** |
| **PV — Floating Solar** | 5 MWp, commissioned 2022, temperature derate, annual degradation model |
| **BESS — Battery** | 1 MW / 2 MWh, SOC 10 %–95 %, η_charge = η_discharge = 0.90 |
| **Upper Reservoir** | Alqueva, 3,150 hm³ usable, head range **54.7–73.0 m** |
| **Lower Reservoir** | Pedrógão, 54 hm³ usable, binding constraint on long pumping sequences |

> **Sign convention:** generation / discharge = **+** ,  pumping / charging = **−**

---

## 📈 Market Coverage

| Gate | Exchange | Gate Close (CET) | Hours in Scope |
|------|----------|-----------------|----------------|
| **Day-Ahead (DA)** | OMIE | D-1 12:00 | H1–H24, all 24 hours |
| **IDA1** | OMIE SIDC | D-1 15:00 | H1–H24 |
| **IDA2** | OMIE SIDC | D-1 22:00 | H3–H24 |
| **IDA3** | OMIE SIDC | D 10:00 | H12–H24 &nbsp;(H1–H11 frozen) |
| **XBID** | SIDC continuous | H-1 rolling | Open hours only |
| **aFRR** | PICASSO | DA + 1 h | Symmetric ±MW, FAT = 5 min, cap ≤ 250 EUR/MW |
| **mFRR** | MARI | DA + 1 h | Symmetric ±MW, FAT = 12.5 min |
| **Imbalance** | REN | Post-delivery | Long → DA×0.85, Short → DA×1.20 |

> [!NOTE]
> **Regulatory dates hard-coded in `config/market.yaml`:**
> SIDC 6→3 sessions from **13 Jun 2024**, ISP 15-min (96/day) from **19 Mar 2025**, PICASSO harmonised **4 Dec 2024**, MARI/REN joined **27 Nov 2024**, FCR is mandatory & non-remunerated — modelled as reserved headroom only, never a market gate.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run full pipeline for tomorrow (AUTO mode, synthetic prices — no CPLEX needed)
python run_production.py

# 3. Run for a specific date
python run_production.py --date 2026-06-28

# 4. Backtest mode — fully automated, no live APIs
python run_production.py --date 2026-06-28 --auto --synthetic

# 5. Resume from a specific phase after a crash
python run_production.py --date 2026-06-28 --from-phase realtime

# 6. Run only selected phases
python run_production.py --date 2026-06-28 --only da,afrr,mfrr

# 7. Validate config and imports without executing anything
python run_production.py --dry-run
```

> [!TIP]
> **No CPLEX licence?** The pipeline auto-selects **HiGHS** (free, bundled via `highspy`) or CBC. Set solver preference in `config/solver.yaml`. Everything works out of the box.

---

## 📋 Phase Reference

| Phase | Entry Point | What It Does |
|-------|------------|--------------|
| **Phase 1, DA** | `run_da.py` | MILP solve → physical check → risk check → trader approval → position save |
| **Phase 2A, IDA1** | `run_ida1.py` | Re-optimise H1–H24, no-churn threshold, SIDC delta bids |
| **Phase 2B, IDA2** | `run_ida2.py` | H1–H2 frozen, re-optimise H3–H24 |
| **Phase 2C, IDA3** | `run_ida3.py` | H1–H11 frozen, re-optimise H12–H24 |
| **Phase 2D, XBID** | `run_xbid.py` | Continuous intraday, per-order caps, H-1 rolling |
| **Phase 3A, aFRR** | `run_afrr.py` | Capacity offers, PICASSO, FAT 5 min, eff_isp_h = 0.2083 h |
| **Phase 3B, mFRR** | `run_mfrr.py` | Capacity offers, MARI, FAT 12.5 min, eff_isp_h = 0.1458 h |
| **Phase 4A, Real-Time** | `run_realtime.py` | 96 ISPs/day, PSP + BESS setpoints, REN telemetry feed |
| **Phase 4B, aFRR Act.** | `run_afrr_activation.py` | TSO activation → ramp-corrected energy, min hold 2 ISPs |
| **Phase 4C, mFRR Act.** | `run_mfrr_activation.py` | TSO activation → ramp-corrected energy, min hold 3 ISPs |
| **Phase 5A, Energy** | `run_energy_settlement.py` | DA + IDA delta per gate, OMIE prices, no double-counting |
| **Phase 5B, Reserve** | `run_reserve_settlement.py` | Capacity (hourly) + activation, eff_isp_h ramp-corrected, PICASSO + MARI |
| **Phase 5C, Imbalance** | `run_imbalance_settlement.py` | Long→DA×0.85, Short→DA×1.20, REN imbalance prices |
| **Phase 5D, Analytics** | `run_analytics.py` | Daily P&L, KPIs, 5-sheet Excel report, 9 production figures |
| **Phase 6, Backtest** | `run_backtest.py` | Historical replay, forecast validation, MILP quality, portfolio risk |

---

## 🏗️ Project Structure

```
Alqueva-PSP-PV-BESS-24hr-Energy-Trading-DA-IDA-aFRR-mFRR-Optimizer/
│
├── run_production.py                                    # ◀ Master orchestrator — all 15 phases
│
├── common_layer/                                        # Shared foundation — imported by every phase
│   ├── configuration/
│   │   ├── config_loader.py                            #   load_config() → AppConfig
│   │   ├── market_config.py                            #   MarketConfig dataclass
│   │   ├── plant_config.py                             #   PlantConfig dataclass
│   │   └── solver_config.py                            #   SolverConfig dataclass
│   │
│   ├── optimisation_model/
│   │   ├── core_milp_builder.py                        #   CoreModelMeta, build_milp() — ONE model for all gates
│   │   ├── core_milp_solver.py                         #   solve_milp(), SolveError — CPLEX → HiGHS → CBC
│   │   ├── ida_reoptimiser.py                          #   optimise_ida() — freeze hours, re-solve
│   │   ├── reserve_offer_builder.py                    #   build_afrr_offers(), build_mfrr_offers()
│   │   ├── activation_ramp_tracker.py                  #   ramp-corrected eff_isp_h for settlement
│   │   └── reserve_activation.py                       #   reserve activation helpers
│   │
│   ├── physical_plant_models/
│   │   ├── psp_turbine_pump_model.py                   #   PSPModel, UnitDispatch (4 Francis units)
│   │   ├── pv_production_model.py                      #   PVModel (5 MWp, temperature derate, degradation)
│   │   ├── bess_model.py                               #   BESSModel, BESSDispatch (1 MW / 2 MWh)
│   │   ├── reservoir_model.py                          #   ReservoirModel (Alqueva + Pedrógão)
│   │   ├── fcr_headroom_model.py                       #   FCRHeadroomModel (reserved — never sold)
│   │   └── reservoir_activation_checker.py             #   validates long-pumping sequences
│   │
│   ├── database/
│   │   ├── position_store.py                           #   PositionStore → runtime/db/positions.db
│   │   ├── reserve_store.py                            #   ReserveStore  → runtime/db/reserve.db
│   │   ├── realtime_store.py                           #   DeliveryStore, ActivationStore → realtime.db
│   │   ├── component_store.py                          #   ComponentStore → runtime/components/<date>.json
│   │   ├── audit_store.py                              #   AuditStore (read-only) → audit_<date>.jsonl
│   │   └── schema_validator.py                         #   input schema validation
│   │
│   ├── gate_scheduler/
│   │   ├── gate_scheduler.py                           #   GateScheduler — CET gate-time resolver
│   │   └── gate_trigger_spec.py                        #   trigger definitions from market.yaml
│   │
│   └── utilities/
│       ├── audit_logger.py                             #   AuditLogger — append-only JSONL trail
│       ├── timezone_utils.py                           #   CET (OMIE) ↔ WET/CET (plant) conversions
│       ├── date_utils.py                               #   delivery date parsing, D-1 calculations
│       └── logging_utils.py                            #   phase-prefixed logger setup
│
├── phase_1_da_day_ahead_bidding/                        # ── Phase 1, Day-Ahead ──────────────────
│   ├── run_da.py                                       #   entry point
│   ├── da_price_pv_inflow_forecasting/
│   │   ├── da_price_forecaster.py                      #   ML DA price forecast
│   │   ├── da_price_train_val_test.py
│   │   ├── pv_power_forecaster.py                      #   PV production forecast
│   │   ├── pv_train_val_test.py
│   │   ├── reservoir_inflow_forecaster.py              #   natural inflow forecast
│   │   ├── reservoir_inflow_train_val_test.py
│   │   ├── omie_da_price_loader.py                     #   live OMIE DA price loader
│   │   └── ml_train_val_test_common.py                 #   shared ML utilities
│   ├── da_bid_formatting/
│   │   ├── da_bid_formatter.py                         #   formats OMIE bid payload
│   │   └── da_bid_checker.py                           #   physical bid validation
│   ├── risk_and_bid_validation/
│   │   └── pre_trade_risk_checker.py                   #   pre-trade risk limits
│   └── trader_approval/
│       └── trader_approval_prompt.py                   #   [A]/[R] interactive prompt
│
├── phase_2a_ida1_intraday_auction_1/                   # ── Phase 2A, IDA1 ──────────────────────
│   ├── run_ida1.py                                     #   entry point
│   ├── ida1_price_forecasting/
│   │   ├── ida1_price_forecaster.py
│   │   ├── ida1_price_train_val_test.py
│   │   └── omie_ida1_price_loader.py
│   ├── ida1_milp_reoptimiser/
│   │   └── ida1_reoptimiser.py                         #   freeze DA, re-solve H1–H24
│   └── ida1_bid_formatting/
│       └── ida1_bid_formatter.py                       #   SIDC delta bid payload
│
├── phase_2b_ida2_intraday_auction_2/                   # ── Phase 2B, IDA2 ──────────────────────
│   ├── run_ida2.py
│   ├── ida2_price_forecasting/
│   │   ├── ida2_price_forecaster.py
│   │   ├── ida2_price_train_val_test.py
│   │   └── omie_ida2_price_loader.py
│   ├── ida2_milp_reoptimiser/
│   │   └── ida2_reoptimiser.py                         #   freeze H1–H2, re-solve H3–H24
│   └── ida2_bid_formatting/
│       └── ida2_bid_formatter.py
│
├── phase_2c_ida3_intraday_auction_3/                   # ── Phase 2C, IDA3 ──────────────────────
│   ├── run_ida3.py
│   ├── ida3_price_forecasting/
│   │   ├── ida3_price_forecaster.py
│   │   ├── ida3_price_train_val_test.py
│   │   └── omie_ida3_price_loader.py
│   ├── ida3_milp_reoptimiser/
│   │   └── ida3_reoptimiser.py                         #   freeze H1–H11, re-solve H12–H24
│   └── ida3_bid_formatting/
│       └── ida3_bid_formatter.py
│
├── phase_2d_xbid_continuous_intraday/                  # ── Phase 2D, XBID ──────────────────────
│   ├── run_xbid.py
│   ├── xbid_price_forecasting/
│   │   ├── xbid_price_forecaster.py
│   │   ├── xbid_price_train_val_test.py
│   │   ├── xbid_price_loader.py
│   │   └── create_xbid_training_data.py
│   ├── xbid_milp_optimiser/
│   │   └── xbid_optimiser.py                           #   per-order caps, H-1 rolling
│   └── xbid_bid_formatting/
│       └── xbid_bid_formatter.py
│
├── phase_3a_afrr_automatic_frequency_reserve/          # ── Phase 3A, aFRR ──────────────────────
│   ├── run_afrr.py
│   ├── afrr_price_forecasting/
│   │   ├── afrr_price_forecaster.py
│   │   ├── afrr_price_train_val_test.py
│   │   ├── picasso_afrr_price_loader.py                #   PICASSO live price loader
│   │   └── create_afrr_training_data.py
│   └── afrr_reserve_offer_builder/
│       ├── afrr_offer_builder.py                       #   headroom → symmetric up/dn offers
│       └── afrr_offer_checker.py                       #   FAT deliverability, cap ≤ 250 EUR/MW
│
├── phase_3b_mfrr_manual_frequency_reserve/             # ── Phase 3B, mFRR ──────────────────────
│   ├── run_mfrr.py
│   ├── mfrr_price_forecasting/
│   │   ├── mfrr_price_forecaster.py
│   │   ├── mfrr_price_train_val_test.py
│   │   ├── mari_mfrr_price_loader.py                   #   MARI live price loader
│   │   └── create_mfrr_training_data.py
│   └── mfrr_reserve_offer_builder/
│       ├── mfrr_offer_builder.py
│       └── mfrr_offer_checker.py
│
├── phase_4a_isp_real_time_dispatch/                    # ── Phase 4A, Real-Time ─────────────────
│   ├── run_realtime.py
│   ├── isp_setpoint_dispatch/
│   │   ├── psp_setpoint_dispatcher.py                  #   PSP unit setpoints per ISP
│   │   └── bess_setpoint_dispatcher.py                 #   BESS setpoints per ISP
│   ├── isp_activation_tracking/
│   │   └── isp_position_tracker.py                     #   scheduled vs actual per ISP
│   └── telemetry/
│       └── ren_isp_signal_loader.py                    #   REN telemetry feed
│
├── phase_4b_afrr_activation_response/                  # ── Phase 4B, aFRR Activation ───────────
│   ├── run_afrr_activation.py
│   ├── afrr_setpoint_dispatch/
│   │   └── afrr_activation_handler.py                  #   ramp, min hold 2 ISPs, eff_isp_h
│   └── afrr_activation_tracking/
│       └── afrr_activation_logger.py
│
├── phase_4c_mfrr_activation_response/                  # ── Phase 4C, mFRR Activation ───────────
│   ├── run_mfrr_activation.py
│   ├── mfrr_setpoint_dispatch/
│   │   └── mfrr_activation_handler.py                  #   ramp, min hold 3 ISPs, eff_isp_h
│   └── mfrr_activation_tracking/
│       └── mfrr_activation_logger.py
│
├── phase_5a_da_ida_settlement/                         # ── Phase 5A, Energy Settlement ─────────
│   ├── run_energy_settlement.py
│   └── energy_settlement_calculation/
│       ├── da_settlement_calculator.py                 #   DA volume × OMIE settlement price
│       ├── ida_settlement_calculator.py                #   IDA delta per gate, no double-counting
│       └── omie_settlement_price_loader.py             #   final OMIE settlement prices
│
├── phase_5b_reserve_settlement/                        # ── Phase 5B, Reserve Settlement ────────
│   ├── run_reserve_settlement.py
│   └── reserve_settlement_calculation/
│       ├── afrr_settlement_calculator.py               #   capacity (hourly) + activation, eff_isp_h
│       ├── mfrr_settlement_calculator.py               #   reuses generic settle_reserve()
│       └── ren_reserve_settlement_loader.py            #   PICASSO + MARI invoice loader
│
├── phase_5c_imbalance_settlement/                      # ── Phase 5C, Imbalance Settlement ──────
│   ├── run_imbalance_settlement.py
│   ├── imbalance_price_and_volume/
│   │   ├── imbalance_volume_calculator.py              #   actual − scheduled per ISP
│   │   └── ren_imbalance_price_loader.py               #   REN post-delivery prices
│   └── imbalance_settlement_calculation/
│       └── imbalance_settlement_calculator.py          #   long→DA×0.85, short→DA×1.20
│
├── phase_5d_analytics_and_reporting/                   # ── Phase 5D, Analytics ─────────────────
│   ├── run_analytics.py
│   ├── analytics_and_kpis/
│   │   ├── daily_pnl_calculator.py                     #   compute_daily_pnl()
│   │   ├── revenue_breakdown_analyzer.py               #   revenue_shares() by market stream
│   │   ├── kpi_reporter.py                             #   compute_kpis() — 10 KPI sections
│   │   └── operational_analytics.py
│   └── daily_excel_reports/
│       ├── daily_report_exporter.py                    #   export_daily_report() → .xlsx
│       ├── dispatch_sheet_builder.py                   #   Dispatch_Hourly (94 cols)
│       └── summary_kpi_builder.py                      #   Summary_KPIs sheet
│
├── phase_6_backtesting_and_validation/                 # ── Phase 6, Backtesting ────────────────
│   ├── run_backtest.py
│   ├── backtest_engine/
│   │   ├── backtest_runner.py                          #   historical date-range replay
│   │   └── historical_data_loader.py                   #   load historical prices / inflows
│   ├── forecast_and_model_validation/
│   │   ├── price_forecast_validator.py                 #   DA / IDA price forecast accuracy
│   │   ├── pv_forecast_validator.py                    #   PV production forecast accuracy
│   │   └── milp_solution_quality_checker.py            #   MIP gap, feasibility, solve time
│   ├── risk_analytics/
│   │   └── portfolio_risk_metrics.py                   #   VaR, CVaR, revenue volatility
│   └── backtest_excel_reports/
│       └── backtest_report_exporter.py
│
├── figures/                                             # ── 9 Production Figures ─────────────────
│   └── __init__.py                                     #   generate(date) → all 9 figures at 600 DPI
│
├── config/                                              # ── Configuration ────────────────────────
│   ├── market.yaml                                     #   gate times, IDA dates, ISP, FAT, bid limits
│   ├── plant.yaml                                      #   PSP / PV / BESS / reservoir, head model
│   ├── solver.yaml                                     #   CPLEX → HiGHS → CBC, MIP gap, time limits
│   └── run.yaml                                        #   date, mode, data source, phase flags
│
├── tests/                                               # ── Test Suite (pytest) ──────────────────
│   ├── conftest.py                                     #   shared fixtures
│   ├── test_e2e_chain.py                               #   end-to-end pipeline chain
│   ├── test_milp_physics.py                            #   MILP constraint physics
│   ├── test_ida_frozen.py                              #   IDA hour-freezing logic
│   ├── test_settlement.py                              #   settlement calculations
│   ├── test_reserve_checker.py                         #   reserve offer validation
│   ├── test_reserve_market_deep.py                     #   deep reserve market tests
│   ├── test_reserve_hidden_invariants.py               #   PR-11 no-double-selling invariants
│   ├── test_reserve_realtime_delivery.py               #   activation + delivery chain
│   ├── test_checker_negative.py                        #   negative / edge-case physical checks
│   └── test_bug_regressions.py                         #   regression suite
│
├── runtime/                                             # ── Auto-created at first run ────────────
│   ├── db/
│   │   ├── positions.db                                #   SQLite — energy positions (DA, IDA, XBID)
│   │   ├── reserve.db                                  #   SQLite — aFRR / mFRR capacity offers
│   │   └── realtime.db                                 #   SQLite — per-ISP delivery & activations
│   ├── audit/
│   │   └── audit_YYYY-MM-DD.jsonl                      #   append-only audit trail (one JSON per event)
│   ├── components/
│   │   └── components_YYYY-MM-DD.json                  #   per-component DA dispatch results
│   └── reports/
│       └── daily_report_YYYY-MM-DD.xlsx                #   5-sheet Excel report
│
├── docs/                                                # ── Architecture Diagrams ────────────────
│   ├── pipeline_architecture.png                       #   pipeline diagram (2× retina, 940 px)
│   └── pipeline_architecture.svg                       #   same diagram as scalable SVG
│
└── requirements.txt                                     # pip install -r requirements.txt
```

---

## 📊 Outputs

### 9 Production Figures — `figures/output/`, 600 DPI

| # | Filename | Description |
|---|----------|-------------|
| 1 | `fig01_dispatch_profile.png` | DA net position (MWh) + DA price (EUR/MWh) — bar + line overlay |
| 2 | `fig02_soc_trajectory.png` | BESS state of charge (% of 2 MWh), 10 %/95 % bounds, step plot |
| 3 | `fig03_revenue_waterfall.png` | Revenue by stream — DA, IDA+XBID, aFRR, mFRR, Imbalance — stacked bar |
| 4 | `fig04_reserve_capacity.png` | aFRR + mFRR capacity offered (MW up/dn per hour), dual subplots |
| 5 | `fig05_gate_position_comparison.png` | Position evolution DA → IDA1 → IDA2 → IDA3 → XBID — line plot |
| 6 | `fig06_intraday_reoptimisation.png` | DA vs final committed position, IDA+XBID delta bar overlay |
| 7 | `fig07_psp_dispatch.png` | PSP turbine / pump MW schedule vs DA price — bar + line |
| 8 | `fig08_pv_bess_flow.png` | PV disposition (used / to-BESS / curtailed) + BESS power — dual subplots |
| 9 | `ops_board.png` | 3×3 operations dashboard — dispatch, SoC, KPIs, positions, aFRR, mFRR, P&L |

### 5-Sheet Excel Report — `runtime/reports/daily_report_<date>.xlsx`

| Sheet | Cols | Contents |
|-------|------|----------|
| **Dispatch_Hourly** | 94 | Hour-by-hour schedule — PSP units, BESS, PV, reservoir, head, all gate positions |
| **ISP_Activation** | — | Per-ISP (15 min) aFRR / mFRR activated MW with ramp-corrected energy |
| **Gate_Decisions** | — | DA → IDA1 → IDA2 → IDA3 → XBID position evolution and P&L per gate |
| **Summary_KPIs** | — | 10 KPI sections — revenue, dispatch, reserves, imbalance, solver quality, risk |
| **Glossary** | — | Variable definitions, units, and specification cross-references |

---

## ✅ Design Principles

| Principle | Implementation |
|-----------|----------------|
| **One MILP, all gates** | `core_milp_builder.py` owns all physics; gates change only price inputs and frozen-hour mask |
| **No double-selling** | `reserve_offer_builder.py` subtracts committed `p_net` before computing headroom (PR-11) |
| **Audit trail on every action** | `AuditLogger` writes one JSONL record per solve / position save / approval / submission |
| **Physical validation first** | Each phase runs a checker (mode exclusivity, SOC, reservoir, FAT) before any market call |
| **Timezone-correct** | Gate times in CET (Madrid), plant timestamps in WET/CET (Lisbon), DST handled automatically |
| **Solver resilience** | CPLEX → HiGHS → CBC fallback, `SolveError` raised if no feasible solution within time limit |
| **Fully restartable** | `--from-phase` resumes any run from any phase without re-running earlier phases |
