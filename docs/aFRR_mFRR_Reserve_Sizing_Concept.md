# aFRR / mFRR Reserve-Sizing Concept — Reference Document

**Plant**: Alqueva PSP + PV + BESS
**Delivery date (real run)**: 2026-08-06
**Source**: fresh, isolated pipeline run — `run_afrr.py` and `run_mfrr.py`, real console output, not estimated

---

## 1. The Concept — Step by Step

Reserve capacity is offered from the headroom **left after the energy position is
committed**: energy first, then reserves from what remains. This makes the
"no MW sold twice" rule true by construction.

**Step 1 — Physical envelope** (`plant.yaml`)
```
Gen cap  = 524.4 MW   (4 turbines + PV + BESS)
Pump cap = 447.4 MW   (4 pumps + BESS)
```

**Step 2 — FCR subtracted first** (mandatory, non-remunerated, INV-7)
```
Effective Gen cap  = 524.4 - 5.0 (FCR) = 519.4 MW
Effective Pump cap = 447.4 - 5.0 (FCR) = 442.4 MW
```

**Step 3 — Energy commitment** (DA/IDA) locks in N MW for the hour
```
N > 0  -> generating
N < 0  -> pumping
```

**Step 4 — Headroom left for reserves**
```
Up headroom = Gen_cap - N
Dn headroom = N + Pump_cap
```

**Step 5 — aFRR claims first** (higher priority: faster FAT = 5 min, higher value)
- Capped by FAT-deliverable ramp, market cap (`max_offer_up/dn_mw`), and the
  mode-switch rule: within 5 min, a pump-to-generation mode switch is **not**
  guaranteed safe, so aFRR up-offers cannot cross the pump/generation boundary.

**Step 6 — mFRR claims what's left** (lower priority: slower FAT = 12.5 min)
```
leftover = total headroom - aFRR's claim
```
- Mode switch **is** allowed (12.5 min >= 8 min safety threshold), so mFRR can
  reach generation-side headroom aFRR could not touch.
- Capped at **20% of that leftover** — a conservative safety margin, our own
  design choice, not a disclosed REN rule (see Section 4).

**Step 7 — Pricing**
Each product bids its own ML-forecasted €/MW capacity price, capped at REN's
€250/MW technical ceiling.

**One-line summary**:
energy first -> FCR reserved -> aFRR takes first bite of leftover headroom ->
mFRR takes 20% of whatever's left after that. Strict priority cascade, no
double-counting, REN-consistent sequencing (MPGGS Art. 80(3): DA -> aFRR -> mFRR).

---

## 2. Worked Example — H01 (real run)

Energy commitment: **N = -425.9 MW** (heavy pumping)

```
Up headroom  = 519.4 - (-425.9)      = 945.3 MW
Dn headroom  = -425.9 + 442.4        =  16.5 MW

aFRR claims:
  Up = 425.9 MW   (stops pumping only — no mode switch within 5 min FAT)
  Dn =  16.5 MW   (all of it)

mFRR leftover:
  Up leftover = 945.3 - 425.9 = 519.4 MW  x 20% = 103.9 MW  <- matches table exactly
  Dn leftover =  16.5 -  16.5 =   0.0 MW  ->  0.0 MW

Prices bid:
  aFRR CapUp = 24.6 EUR/MW
  mFRR CapUp =  9.0 EUR/MW   (cheaper — lower priority product)
```

Every number in the real console output for H01 traces back to this exact chain.

---

## 3. Why 12.5 min FAT Matters (but doesn't set the MW number)

The mFRR FAT (12.5 min) does two separate jobs:

1. **Mode-switch permission**: 12.5 min >= 8 min threshold -> mFRR is allowed to
   count generation-side headroom (pump -> turbine switch) as deliverable.
   aFRR (5 min FAT) is not allowed this.
2. **Ramp-capacity check** (a ceiling, not the binding constraint here):
   ```
   FAT-deliverable = ramp_rate x n_units x FAT_min
                    = 25 MW/min x 4 x 12.5 min = 1,250 MW
   ```
   Far bigger than the 103.9 MW headroom-based offer, so ramp speed never
   binds — the 20%-of-headroom rule does.

**Real role of 12.5 min**: it *unlocks* generation-side headroom for mFRR; it
does not itself compute the 103.9 MW figure — headroom sizing does that.

---

## 4. Why the 20% mFRR Cap Exists (not a REN rule)

`market.yaml` comment: `max_offer_fraction: 0.20  # cap offer at 20% of available headroom`

No REN/ENTSO-E document specifies this exact fraction for this plant. It exists
because:

1. **Safety buffer** — pledging 100% of leftover headroom leaves zero margin
   for forecast error or real-time deviation.
2. **Reflects mFRR's real-world low priority** — BSPs don't max out every
   product; they hold back capacity for re-dispatch and intraday flexibility.
3. **Conservative modeling default** — absent a published REN cap, we
   under-offer rather than over-promise.

**Interview-honest framing**: this is our own engineering judgment call,
explicitly labeled ESTIMATE — not presented as a real regulatory rule.

---

## 5. Full 24-Hour Real Simulation Output

### aFRR Capacity Offer — 2026-08-06 (real run)
```
Band   : 49.800 - 50.200 Hz   (nominal 50.000 Hz)
FAT    : 5 min   |   Platform: NATIONAL
Cap ceiling: 250 EUR/MW

Hour   Energy MW    Up MW    Dn MW  CapUp EUR/MW  CapDn EUR/MW
------------------------------------------------------------
H01      -425.9    425.9     16.5        24.6        17.3
H02      -426.0    426.0     16.4        24.2        17.6
H03      -426.1    426.1     16.3        23.9        18.0
H04      -426.2    426.2     16.2        23.6        18.4
H05      -426.3    426.3     16.1        23.6        18.7
H06      -216.4    216.4    226.0        23.7        19.1
H07      +427.9     91.5    447.4        24.0        19.4
H08      +427.6     91.8    447.4        24.4        19.6
H09      +427.6     91.8    447.4        24.9        19.7
H10      +427.8     91.6    447.4        25.4        19.8
H11      +428.0     91.4    447.4        26.0        19.7
H12        +1.9    500.0    444.3        26.6        19.5
H13      -215.0    215.0    227.4        27.1        19.2
H14      -360.0    360.0     82.4        27.6        18.9
H15      -214.5    214.5    227.9        27.9        18.6
H16      -214.4    214.4    228.0        28.1        18.2
H17        +2.3    500.0    444.7        28.2        17.8
H18      +429.2     90.2    447.4        28.1        17.4
H19      +428.4     91.0    447.4        27.8        17.1
H20      +488.3     31.1    447.4        27.4        16.9
H21      +448.3     71.1    447.4        26.9        16.8
H22      +425.8     93.6    447.4        26.4        16.8
H23      -217.6    217.6    224.8        25.8        16.9
H24      -425.7    425.7     16.7        25.2        17.0
------------------------------------------------------------
Expected aFRR capacity revenue:   272,171.83 EUR
```

### mFRR Capacity Offer — 2026-08-06 (real run)
```
Band   : 49.800 - 50.200 Hz   (nominal 50.000 Hz)
FAT    : 12.5 min   |   Platform: MARI
Sizing : <= 20% of headroom AFTER aFRR

Hour   Energy MW  aFRRup MW  mFRRup MW  mFRRdn MW  CapUp EUR/MW
------------------------------------------------------------
H01      -425.9      425.9      103.9        0.0         9.0
H02      -426.0      426.0      103.9        0.0         8.6
H03      -426.1      426.1      103.9        0.0         8.5
H04      -426.2      426.2      103.9        0.0         8.6
H05      -426.3      426.3      103.9        0.0         8.9
H06      -216.4      216.4      103.9        0.0         8.1
H07      +427.9       91.5        0.0       84.6         9.1
H08      +427.6       91.8        0.0       84.5         9.3
H09      +427.6       91.8        0.0       84.5        10.4
H10      +427.8       91.6        0.0       84.6        10.4
H11      +428.0       91.4        0.0       84.6        11.1
H12        +1.9      500.0        3.5        0.0        11.5
H13      -215.0      215.0      103.9        0.0        11.9
H14      -360.0      360.0      103.9        0.0        11.8
H15      -214.5      214.5      103.9        0.0        11.7
H16      -214.4      214.4      103.9        0.0        11.0
H17        +2.3      500.0        3.4        0.0        10.1
H18      +429.2       90.2        0.0       84.8         9.7
H19      +428.4       91.0        0.0       84.7         9.6
H20      +488.3       31.1        0.0       96.7         9.1
H21      +448.3       71.1        0.0       88.7         7.6
H22      +425.8       93.6        0.0       84.2         7.1
H23      -217.6      217.6      103.9        0.0         8.6
H24      -425.7      425.7      103.9        0.0         8.8
------------------------------------------------------------
Expected mFRR capacity revenue:    17,657.63 EUR
```

---

## 6. Reading the Pattern Across the Day

- **Pumping hours** (H01-H05, H13-H16, H23-H24, negative Energy MW): aFRR Up
  is large (stops pumping), aFRR Dn is small (pump already near its cap).
  mFRR gets a fixed ~103.9 MW Up (20% of the generation-side leftover), Dn = 0.
- **Generating hours** (H07-H11, H18-H22, positive Energy MW): the picture
  flips — aFRR Dn is large (~447.4 MW, room to pump instead), aFRR Up is
  small. mFRR then gets Dn instead of Up, Up = 0.
- **Near-idle hours** (H12, H17, Energy MW near zero): almost the full
  envelope is free — aFRR claims up to 500 MW, mFRR gets only the small
  remainder (3.5 / 3.4 MW) since aFRR has already used almost everything.

This day-shape confirms the cascade logic holds consistently across all 24
hours, not just the H01 example — aFRR always claims first, mFRR always gets
the 20% remainder, and the Up/Dn split flips with the plant's pump/generate
mode.
