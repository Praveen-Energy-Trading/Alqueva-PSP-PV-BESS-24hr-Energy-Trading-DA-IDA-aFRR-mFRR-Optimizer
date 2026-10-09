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
aFRR = min( headroom , FAT-deliverable ramp , market cap )
```
- Market cap = `afrr.max_offer_up/dn_mw` = **95 MW** per direction. This is an
  assumption: about 50% of an estimated ~190 MW national need (ENTSO-E sizing
  rule). The real need is set by the grid operator (Art. 145) and is not
  published in the data this project reads.
- Mode-switch rule: within 5 min a pump-to-generation switch is not guaranteed
  safe, so aFRR up cannot count the generation side when the plant is pumping
  (and the reverse).
- Up and down are separate offers; the real aFRR band is not symmetric (Art. 142).

**Step 6 — mFRR claims what is left** (slower product, 12.5 min FAT)
```
mFRR = min( 20% x (headroom - aFRR) , FAT-deliverable ramp , market cap )
```
- Mode switch **is** allowed (12.5 min >= 8 min safety threshold), so mFRR can reach
  headroom aFRR could not touch.
- The 20% is this model's own risk margin, **not** a rulebook figure (see
  `config/market.yaml`, `mfrr.max_offer_fraction`).

**Step 7 — Pricing**
Each product bids its own ML-forecast capacity price, capped at REN's
EUR 250/MW ceiling. The real market pays every accepted MW the **last accepted
price** (pay-as-clear, Art. 151). **mFRR capacity is not counted as paid in this model**
(`mfrr.capacity_payment: false`). That is a conservative modelling choice, not a
rule: a paid mFRR capacity market exists, but today it is the ERSE auction for a
whole quarter or month (not daily), upward only, and the latest notice limits
generation and storage to at most 25% of the awarded volume (see Section 4 and
the interview-answer note). The new daily mFRR band only has to start by
1 Apr 2027 (Art. 453). mFRR activation energy is still earned.

**One-line summary**:
energy first -> FCR reserved -> aFRR takes first bite of the headroom (capped) ->
mFRR takes 20% of what is left (capped, energy-only). Strict priority cascade, no
double-counting. The intraday auctions (IDA1/2/3, XBID) then re-optimise with the
sold reserve MW kept free.

---

## 2. Worked Examples (real run, 2026-10-10)

**Period 40 — pumping, N = -423.4 MW**
```
Up headroom  = 519.4 - (-423.4)   = 942.8 MW
Dn headroom  = -423.4 + 442.4     =  19.0 MW

aFRR:  Up = min(942.8, ..., 95)  = 95.0 MW   (market cap binds)
       Dn = 19.0 MW                          (all of the headroom)
mFRR:  Up leftover = 942.8 - 95.0 = 847.8 MW x 20% = 169.6 -> capped at 95.0 MW
       Dn leftover = 19.0 - 19.0  = 0.0 MW   -> 0.0 MW
```

**Period 80 — generating, N = +454.1 MW**
```
Up headroom  = 519.4 - 454.1      =  65.3 MW
Dn headroom  = 454.1 + 442.4      = 896.5 MW

aFRR:  Up = 65.3 MW                          (all of the headroom)
       Dn = min(896.5, ..., 95)  = 95.0 MW   (market cap binds)
mFRR:  Up leftover = 0.0 MW      -> 0.0 MW
       Dn leftover = 896.5 - 95.0 = 801.5 MW x 20% = 160.3 -> capped at 95.0 MW
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
   Far larger than the offers above, so ramp speed never binds. The market cap
   and the 20% rule do.

---

## 4. What is a rule and what is our assumption

| Item | Status |
|---|---|
| Order DA → aFRR → mFRR → intraday | Rulebook, Art. 80(3) |
| 15-min product, 1 MW minimum, up and down separate | Rulebook, Art. 142 |
| Pay-as-clear (last accepted offer) | Rulebook, Art. 151, 191 |
| Need is inelastic, set by the grid operator | Rulebook, Art. 145, 185 |
| Clock hours of the aFRR / mFRR band markets | Set in a separate notice ("Aviso do GGS"); **estimate** here |
| 95 MW aFRR cap | **Assumption** (share of an estimated national need) |
| 20% of leftover for mFRR | **Assumption** (own risk margin) |
| 5 MW FCR hold-back | **Assumption** |
| mFRR energy-only | **Modelling choice**. Paid capacity exists via quarterly / monthly ERSE auctions, with limits that favour consumers; not modelled |

---

## 5. Full 96-Period Output (real run)

The complete tables are printed by the pipeline (phases 2 and 3) and saved in
`runtime/logs/pipeline_2026-10-10.log`. Summary of this run:

| | Result |
|---|---|
| Expected aFRR capacity revenue | EUR 74,620 |
| Expected mFRR capacity revenue | EUR 0 (energy-only) |
| aFRR offer | 95 MW up / 95 MW down in most periods; 18-19 MW down while pumping near full load; 65 MW up at the peak-generation period |
| mFRR offer | Up: 85 MW when idle, 95 MW when pumping. Down: 70 MW when idle, 95 MW when generating |

## 6. Reading the Pattern Across the Day

- **Pumping periods** (negative N): aFRR up is large (the plant can stop pumping)
  and aFRR down is small (the pumps are already near their limit). mFRR then
  offers up, not down.
- **Generating periods** (positive N): the picture flips. aFRR down is large (room
  to pump instead), aFRR up is small, and mFRR offers down.
- **Idle periods** (N near zero): almost the whole envelope is free, so aFRR hits
  the 95 MW cap in both directions and mFRR offers the 20% rule's result
  (about 85 MW up and 70 MW down).

The Up/Dn split follows the plant's pump/generate mode, and aFRR always claims
before mFRR.
