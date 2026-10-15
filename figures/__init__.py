"""figures - the figures of the pipeline, written as SVG into figures/output/.

Eight figures, each one idea, in the order a reader needs them:

    01_trading_day         when each market decides, from the day before delivery to the delivery itself
    02_day_plan            what the optimiser decided for the solved day: price, and the plan it makes of it
    03_capacity_stack      how every MW is shared between energy and the reserves held back for the grid
    04_water_and_battery   the reservoirs and the battery staying inside their limits
    05_efficiency_surface  the physics the optimiser works on: turbine efficiency against flow and head
    06_forecast_skill      how good the price forecasts are, on a locked test period
    07_revenue_sources     where a year of revenue comes from
    08_risk                how bad a bad day can be

Every number is read from the run's own outputs (component store, position store, daily report, backtest report, model decisions); nothing is
illustrative. The text of the figures is converted to paths, so they look the same everywhere.

    figures.generate(date)      called by run_production.py after every run; a figure whose data is missing is skipped, never an error
"""
from __future__ import annotations

import traceback
from pathlib import Path

from . import _data as D
from . import _style as S

OUT = S.OUT


def generate(date: str | None = None) -> list[Path]:
    """Write all figures; returns the files written. The day figures use `date` (default: the newest solved day with a report)."""
    from . import backtest_figures as B
    from . import day_figures as F
    from . import pipeline_figures as P

    written: list[Path] = []

    def run(name: str, fn, *args) -> None:
        try:
            written.append(fn(*args))
        except Exception:                                   # a missing input must never stop the pipeline
            print(f"[figures] {name} skipped:")
            traceback.print_exc(limit=2)

    S.OUT.mkdir(exist_ok=True)
    run("01_trading_day", P.trading_day)
    day_date = date if date and (D.RUNTIME / "components" / f"components_{date}.json").exists() else D.latest_date()
    if day_date:
        try:
            day, plant = D.load_day(day_date), D.plant()
        except Exception:
            print("[figures] the solved day could not be read:")
            traceback.print_exc(limit=2)
            day = plant = None
        if day is not None:
            run("02_day_plan", F.day_plan, day, plant)
            if day.hourly is not None:
                run("03_capacity_stack", F.capacity_stack, day, plant)
            run("04_water_and_battery", F.water_and_battery, day, plant)
            run("05_efficiency_surface", F.efficiency_surface, day, plant)
    skill = D.forecast_skill()
    if skill:
        run("06_forecast_skill", P.forecast_skill, skill)
    bt = D.backtest()
    if bt is not None:
        run("07_revenue_sources", B.revenue_sources, bt)
        run("08_risk", B.risk, bt)
    return written
