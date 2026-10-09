# aFRR / mFRR Reserve-Sizing Concept — Reference Document

**Plant**: Alqueva PSP + PV + BESS
**Delivery date (real run)**: 2026-10-10
**Source**: full pipeline run (`run_production.py`), real console output. Rules are
cited from the Portuguese grid-operator rulebook (MPGGS, ERSE Directive 9/2025).

For the market rules and the interview-style answer, see
[Reserve_Allocation_Interview_Answer.md](Reserve_Allocation_Interview_Answer.md).

---

## 1. The Concept — Step by Step

Reserve capacity is offered from the headroom **left after the energy position is
committed**: energy first, then reserves from what remains. This makes the
"no MW sold twice" rule true by construction, and it matches the real order of
the day (MPGGS Art. 80(3): DA → aFRR band → mFRR band → intraday).

All offers are per 15-minute settlement period (ISP), 96 per day, which is the
real product resolution (MPGGS Art. 142, 182).

**Step 1 — Physical envelope** (`plant.yaml`)
```
Gen cap  = 524.4 MW   (4 turbines + PV + BESS)
Pump cap = 447.4 MW   (4 pumps + BESS)
```

**Step 2 — FCR subtracted first** (mandatory, not paid)
```
Effective Gen cap  = 524.4 - 5.0 (FCR) = 519.4 MW
Effective Pump cap = 447.4 - 5.0 (FCR) = 442.4 MW
```
The 5 MW is an assumption of this model. The real FCR share is set yearly by the
grid operator from previous-year energy production (MPGGS Art. 68).

**Step 3 — Energy commitment** (DA, then IDA) fixes N MW for each period
```
N > 0  -> generating
N < 0  -> pumping
```

**Step 4 — Headroom left for reserves**
```
Up headroom = Gen_cap - N
Dn headroom = N + Pump_cap
```

**Step 5 — aFRR claims first** (faster product, 5 min FAT)
```
aFRR = min( headroom , FAT-deliverable ramp , market cap )      (0 while the plant is idle)
```
- Market cap = `afrr.max_offer_up/dn_mw` = **105 MW** per direction = 20% of the
  524.4 MW plant. An assumption, checked against REN's own report: the
  secondary-reserve band averaged about 280 MW (2018-2019), so 105 MW is about
  37% of it, while hydro including pumping supplied 40-48% of the secondary
  energy. The real need is set by the grid operator (Art. 145).
- **Idle rule** (`afrr.require_synchronised_unit`): aFRR is automatic and
  5 minutes, so it needs a unit that is already running. When the schedule is
  idle (|net| < 1 MW) no aFRR is offered. The 57 MW per-unit minimum stable load
  is not modelled.
- Mode-switch rule: within 5 min a pump-to-generation switch is not guaranteed
  safe, so aFRR up cannot count the generation side when the plant is pumping
  (and the reverse).
- Up and down are separate offers; the real aFRR band is not symmetric (Art. 142).

**Step 6 — mFRR claims what is left** (slower product, 12.5 min FAT)
```
mFRR = min( 100% x (headroom - aFRR) , FAT-deliverable ramp , mFRR cap )
```
- Mode switch **is** allowed (12.5 min >= 8 min safety threshold), so mFRR can reach
  headroom aFRR could not touch, and it can be offered from standstill.
- mFRR cap = `mfrr.max_offer_up/dn_mw` = **74 MW** per direction = 14% of the plant.
- 100% of the leftover is offered because mFRR is a legal obligation for large
  (Type D) generators: unused capacity must be offered as energy (ROR Art. 49(5)(c)).
  Before 2026-10 the model offered only 20% and shared the aFRR cap.

**Step 7 — Pricing**
Each product bids its own ML-forecast capacity price, capped at REN's
EUR 250/MW ceiling. The real market pays every accepted MW the **last accepted
price** (pay-as-clear, Art. 151).

**mFRR capacity has two settings** (`mfrr.capacity_payment`):
- `true` (**default**, a **future scenario, not today's rules**): the daily mFRR
  band that MPGGS Chapter XVI requires by 1 Apr 2027 (Art. 453). No price exists
  yet, so it is paid at the **real aFRR band price** (about EUR 9.7/MW/h up, 8.8
  down over Oct 2025 - Sep 2026). Label any result "future scenario".
- `false` (strict today's rules): energy-only, band = EUR 0. Today the paid mFRR
  band is the quarterly/monthly ERSE auction, won by consumers, and plants under a
  mandatory mFRR duty are excluded (MPGGS Art. 257).

mFRR activation energy is earned in both cases.

**One-line summary**:
energy first -> FCR reserved -> aFRR takes first bite of the headroom (105 MW cap, none
while idle) -> mFRR takes all that is left (74 MW cap, a paid future-band scenario by default, EUR 0 if switched off). Strict priority cascade, no
double-counting. The intraday auctions (IDA1/2/3, XBID) then re-optimise with the
sold reserve MW kept free.

---

## 2. Worked Examples (real run, 2026-10-10)

**Period 1 — idle, N = 0 MW**
```
aFRR:  0 MW up, 0 MW down            (idle rule: no running unit)
mFRR:  74.0 MW up, 74.0 MW down      (allowed from standstill, 12.5 min FAT)
```

**Period 40 — pumping, N = -423.4 MW**
```
Up headroom  = 519.4 - (-423.4)   = 942.8 MW
Dn headroom  = -423.4 + 442.4     =  19.0 MW

aFRR:  Up = min(942.8, ..., 105) = 105.0 MW  (market cap binds)
       Dn = 19.0 MW                          (all of the headroom)
mFRR:  Up leftover = 942.8 - 105.0 = 837.8 MW x 100% -> capped at 74.0 MW
       Dn leftover = 19.0 - 19.0  = 0.0 MW   -> 0.0 MW
```

**Period 80 — generating, N = +454.1 MW**
```
Up headroom  = 519.4 - 454.1      =  65.3 MW
Dn headroom  = 454.1 + 442.4      = 896.5 MW

aFRR:  Up = 65.3 MW                          (all of the headroom)
       Dn = min(896.5, ..., 105) = 105.0 MW  (market cap binds)
mFRR:  Up leftover = 0.0 MW      -> 0.0 MW
       Dn leftover = 896.5 - 105.0 = 791.5 MW x 100% -> capped at 74.0 MW
```

Every number in the console output traces back to this chain.

---

## 3. Why 12.5 min FAT Matters (but does not set the MW number)

The mFRR FAT (12.5 min) does two separate jobs:

1. **Mode-switch permission**: 12.5 min >= 8 min threshold, so mFRR may count
   headroom that needs a pump-to-turbine switch. aFRR (5 min FAT) may not.
2. **Ramp-capacity check** (a ceiling, not the binding constraint here):
   ```
   FAT-deliverable = ramp_rate x n_units x FAT_min
                   = 25 MW/min x 4 x 12.5 min = 1,250 MW
   ```
   Far larger than the offers above, so ramp speed never binds. The size caps
   (105 MW aFRR, 74 MW mFRR) and the remaining headroom do.

---

## 4. What is a rule and what is our assumption

| Item | Status |
|---|---|
| Order DA → aFRR → mFRR → intraday | Rulebook, Art. 80(3) |
| 15-min product, 1 MW minimum, up and down separate | Rulebook, Art. 142 |
| Pay-as-clear (last accepted offer) | Rulebook, Art. 151, 191 |
| Need is inelastic, set by the grid operator | Rulebook, Art. 145, 185 |
| Clock hours of the aFRR / mFRR band markets | Set in a separate notice ("Aviso do GGS"); **estimate** here |
| 105 MW aFRR cap, 74 MW mFRR cap | **Assumption** (replica allocation 65 / 20 / 14 / 1; aFRR checked against REN's observed ~280 MW band) |
| mFRR offers all leftover headroom | Follows the legal obligation for large generators (ROR Art. 49(5)(c)) |
| No aFRR while idle | **Modelling rule** (needs a running unit); 57 MW minimum stable load not modelled |
| 5 MW FCR hold-back | **Assumption** |
| mFRR capacity paid at the aFRR band price | **Scenario** for the daily band required from Apr 2027; switch `mfrr.capacity_payment` to false for today's rules (EUR 0) |

---

## 5. Full 96-Period Output (real run)

The complete tables are printed by the pipeline (phases 2 and 3) and saved in
`runtime/logs/pipeline_2026-10-10.log`. Summary of the 10 Oct 2026 run with the
65 / 20 / 14 / 1 allocation:

| Plant mode | Share of the day | aFRR up / down (MW) | mFRR up / down (MW) |
|---|---|---|---|
| Turbining | 43% | 92 / 105 | 2 / 74 |
| Pumping | 38% | 105 / 19 | 74 / 2 |
| Idle | 20% | 0 / 0 | 74 / 74 |
| **Daily average** | | **78 / 52** | **43 / 47** |

| | Result |
|---|---|
| Expected aFRR capacity revenue | EUR 59,366 |
| Expected mFRR capacity revenue | EUR 42,384 (future daily-band scenario, paid at the aFRR band price; EUR 0 if `capacity_payment` is false) |

Why these shapes: while pumping near full draw there is almost no room to pump
more (aFRR down only 19 MW); while turbining near full load there is almost no room
to generate more (mFRR up only 2 MW); the idle rule removes aFRR while no unit runs.

## 6. Reading the Pattern Across the Day

- **Pumping periods** (negative N): aFRR up is large (the plant can stop pumping)
  and aFRR down is small (the pumps are already near their limit). mFRR then
  offers up, not down.
- **Generating periods** (positive N): the picture flips. aFRR down is large (room
  to pump instead), aFRR up is small, and mFRR offers down.
- **Idle periods** (N near zero): no unit is running, so no aFRR is offered. mFRR
  can still be offered from standstill (12.5 min is enough to start a unit), so it
  takes its 74 MW cap in both directions.

The Up/Dn split follows the plant's pump/generate mode, and aFRR always claims
before mFRR.
