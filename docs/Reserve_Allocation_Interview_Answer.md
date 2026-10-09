# How much capacity goes to DA, aFRR and mFRR? — interview answer

Source for every rule below: the Portuguese grid-operator rulebook (MPGGS, ERSE
Directive 9/2025 of 11 Sep 2025). Article numbers are given so a claim can be
checked. Anything that is our own modelling choice is labelled **assumption**.

## The 30-second answer

> There is no fixed percentage. Day-ahead energy is committed first, at the
> 12:00 CET day-ahead gate. The grid operator then opens the aFRR and mFRR band
> markets for the next day, and I offer the headroom the energy schedule has
> left, up and down separately, to whichever product pays more per MW. The
> grid operator sets the national need, so my share depends on where my offer
> falls in the merit order. In my replica I fixed the split of the 524 MW plant at
> 65% day-ahead energy, 20% aFRR, 14% mFRR and 1% FCR. That is an assumption,
> checked against REN's observed secondary-reserve band of about 280 MW, not a
> published percentage. In real operation it changes every period.

## The order of the day (Art. 80(3))

1. **DA auction** — gate closes 12:00 CET on D-1.
2. **PDVD** — the grid operator checks the DA result for grid security (Art. 81).
3. **aFRR band market** (Art. 82) — then
4. **mFRR band market** (Art. 83).
5. **Intraday**: IDA1 15:00, IDA2 22:00 (D-1), IDA3 10:00 (D), then XBID.

The clock hours of steps 3–4 are set in a separate grid-operator notice
("Aviso do GGS"), not in the rulebook. The pipeline runs in exactly this order,
and the intraday stages keep the reserve MW already sold free.

## Who decides how much — three layers

| Layer | Who decides | Rule |
|---|---|---|
| **FCR** | Grid operator, yearly | Mandatory and **not paid**. Portugal's share of the European FCR is set by its previous-year energy production (Art. 68). Our model holds back 5 MW = 1% (**assumption**). |
| **System need for aFRR / mFRR** | Grid operator, per 15 min | aFRR: sized to cover 99% of historical frequency deviations, up and down separately (Art. 145). mFRR up: largest single loss + 2% of load + 10% of wind + 5% of solar (Art. 70). Both are **inelastic** (fixed, not price-dependent). |
| **My share** | Auction | Offers in €/MW per 15 min, minimum 1 MW. Cheapest offers are taken until the need is met (±5%), and every accepted MW is paid the **last accepted price** (Art. 149, 151). The contract is firm, with penalties for not delivering. |

## What limits my offer — physical headroom

- Up headroom = maximum output − energy schedule.
- Down headroom = energy schedule − minimum (for a pumped-storage plant, down can
  mean pumping more).
- The schedule from DA and intraday must leave the band deliverable (Art. 146(11)).
- aFRR must start within 30 s and finish within 5 min (Art. 69), so it cannot
  count on switching a unit from pumping to generating. mFRR has 12.5 min, so it can.
- The same MW cannot be sold twice, to energy or to two reserve products.

## What our model does

| Item | Setting | Status |
|---|---|---|
| Order | DA → aFRR → mFRR → IDA1/2/3 → XBID | Matches Art. 80(3) |
| aFRR offer | Headroom left after DA, capped at 105 MW per direction (20% of the plant), none while idle | **Assumption** (about 37% of REN's observed ~280 MW band) |
| mFRR offer | All headroom left after aFRR, capped at 74 MW per direction (14% of the plant) | Offering all of it follows the legal obligation for large generators (ROR Art. 49(5)(c)); the 74 MW size is an **assumption** |
| mFRR payment | Default: **future daily-band scenario**, band paid at the aFRR band price. Switch off for today's rules (energy-only, band EUR 0) | Today's paid band is the quarterly/monthly auction won by consumers; the daily band only has to start by 1 Apr 2027 (Art. 453). See the mFRR section below. |
| Energy first, reserves second | DA solved before reserves | Matches the market sequence, because the DA schedule is fixed before the band result is known |

## Follow-up questions to expect

**"Would a real desk do it that way?"** Largely yes, because the market forces
the sequence. But a desk would also leave headroom in its DA bid when it expects
reserve prices to beat the energy spread. Our model has a switch for this
(`dynamic_allocation_enabled`), which is off by default.

**"What share of the plant is reserved on a typical day?"** The replica caps are
20% aFRR and 14% mFRR per direction, but the offered MW changes every 15 minutes
with the schedule. On a real run (10 Oct 2026): turbining 43% of the day (aFRR
92 up / 105 down, mFRR 2 up / 74 down), pumping 38% (aFRR 105 up / 19 down, mFRR
74 up / 2 down), idle 20% (no aFRR, mFRR 74 / 74). The day averages about 78 MW
up and 52 MW down for aFRR. Reserve income stays small next to energy income.

**"Why no aFRR when the plant is idle?"** aFRR is an automatic 5-minute service,
so it needs a unit that is already running. mFRR has 12.5 minutes, enough to
start a unit, so it can still be offered from standstill.

**"Does Alqueva have to offer reserve?"** Units with a legal aFRR obligation
must offer all their feasible power (Art. 144(4)). I could not confirm from
public sources whether Alqueva is one of them, so I model it as voluntary.

**"Your model shows mFRR capacity revenue. Is that real?"** It is a labelled
**future scenario**, not today's rules. Portugal's rulebook requires a daily mFRR
band (D+1, 15 minutes, pay-as-clear, open to any approved provider) by 1 April
2027 (MPGGS Chapter XVI, Art. 453). No price exists yet, so I pay it at the real
aFRR band price (about EUR 9.7/MW per hour up, 8.8 down). Today the paid mFRR
band is the ERSE auction for a quarter or month, upward only, as a firm
commitment for every 15 minutes. In the Q4-2026 auction 13 industrial consumers
took 165 of 180 MW at the ceiling price of EUR 10/MW per quarter-hour, at least
75% of the awarded volume must come from consumption, and production units under
a mandatory-participation duty are excluded. So for a plant like Alqueva, today's
real mFRR capacity income is about zero, and a switch in the config
(`mfrr.capacity_payment: false`) shows that case.

**"What do companies actually do in mFRR today?"** Two routes (details in the
section below): industrial consumers sell the mFRR **band** through demand
response in the ERSE auctions, and everyone else (generators, storage,
aggregators) earns from mFRR **energy** bids when the grid operator calls them,
through the MARI platform.

## Real mFRR band auction results (ERSE / REN, published on OMIP)

| Auction | Product | Offered | Awarded | Price (EUR/MW per 15 min) | Winners |
|---|---|---|---|---|---|
| 7th (Dec 2025) | Year 2026 | 300 MW | 300 MW (100%) | 10.49 | 30 consumption installations |
| 7th | Q1 2026 | 150 MW | 132 MW (88%) | 10.00 | 11 |
| 7th | Jan / Feb 2026 | 150 MW | 22 MW (15%) each | 9.00 | 2 each |
| 7th | Mar 2026 | 100 MW | 26 MW (26%) | 9.00 | 2 |
| 9th (Sep 2026) | Q4 2026 | 180 MW | 165 MW (92%) | 10.00 | 13 |
| 9th | Oct 2026 | 150 MW | 6 MW (4%) | 8.00 | 3 |
| 9th | Nov / Dec 2026 | 150 MW | 3 MW / 2 MW | 8.00 | 2 each |

What this shows:
- **Every winner is a consumption installation** (industrial demand response). The
  results report no generator or storage winner.
- **Long contracts sell out, short ones do not.** The year and quarter lots fill
  at or near the price ceiling; the monthly lots mostly stay empty.
- **Price is about EUR 32-42/MW per hour**, set by the regulator's ceiling.
- For Q2 2026 the grid operator said its need was already met by the annual lot
  (ERSE statement, 17 Mar 2026).

## What companies do in real time today

1. **Industrial consumers**: sell the mFRR band (the auctions above) and commit
   to be curtailed when called. This is where the band money goes.
2. **Generators, storage and aggregators**: no band payment, so they earn from
   mFRR **energy** offers. The grid operator calls them in merit order through
   the European MARI platform (Portugal joined on 27 Nov 2024) and they are paid
   for the energy delivered. Independent aggregators have started to bring in
   renewable plants (for example IGNIS Energia).
3. **Large hydro and pumped storage**: I found no public data on what the EDP
   Alqueva plant offers in mFRR, so I make no claim about it.

## Things I did not find (say so if asked)

- No public figure for how much of Alqueva's capacity is committed to reserves.
- The clock hours of the aFRR and mFRR band markets (they are in a notice that is
  not in the rulebook).
- Alqueva's certified reserve capacity, if any.

## Sources

- [MPGGS rulebook, Sep 2025 (ERSE)](https://www.erse.pt/media/q10chfti/mpggs_articulado-250911.pdf)
- [ERSE workshop on the new aFRR and mFRR products](https://www.erse.pt/media/bexnjafq/erse-workshop_afrr.pdf)
- [Results of the 9th BmFRR auction (OMIP / REN)](https://www.omip.pt/sites/default/files/2026-09/9o-leilao-bmfrr-comunicacao-resultados_site-ren_4.pdf)
- [Results of the 7th BmFRR auction (OMIP / REN, zip)](https://www.omip.pt/sites/default/files/2025-12/7o-leilao-bmfrr_4.zip)
- [ERSE statement on Q2 2026 mFRR band need](https://www.erse.pt/media/iksjiiv5/comunicado_bmfrr_2trimestre.pdf)
- [REN joins MARI (Artelys)](https://www.artelys.com/news/powering-progress-artelys-algorithms-enable-ren-integration-into-mari-platform/)
- [ERSE notice, 8th BmFRR auction (Q3 2026)](https://www.erse.pt/media/eodblc2l/convocatoria_08_leilao_bmfrr_20260609.pdf)
- [ERSE notice, 9th BmFRR auction, 25 Sep 2026](https://www.erse.pt/media/eq5bfo4o/convocatoria_09_leilao_bmfrr_20260908_final.pdf)
