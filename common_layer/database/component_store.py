"""
component_store.py — persist per-component DA dispatch results to JSON.

Saves the rich GateResults decomposition (per-unit PSP, BESS, PV, reservoir,
efficiency, water flows) that the MILP computes but PositionStore discards.
Also stores natural inflow and solver metrics for the analytics Excel exporter.

File: runtime/components/components_<date>.json
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Dict, Any, Optional


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir))


def _path(delivery_date: str) -> str:
    d = os.path.join(_repo_root(), "runtime", "components")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"components_{delivery_date}.json")


class ComponentStore:
    """Save and load per-component hourly dispatch data for a delivery date."""

    def save(
        self,
        delivery_date: str,
        psp_schedule: Dict[int, dict],
        bess_schedule: Dict[int, dict],
        pv_schedule: Dict[int, dict],
        reservoir_trajectory: Dict[int, dict],
        efficiency_per_hour: Dict[int, dict],
        inflow_m3h: Dict[int, float],
        solver_metrics: Optional[dict] = None,
        initial_state: Optional[dict] = None,
    ) -> None:
        payload = {
            "delivery_date": delivery_date,
            "psp_schedule":          {str(h): v for h, v in psp_schedule.items()},
            "bess_schedule":         {str(h): v for h, v in bess_schedule.items()},
            "pv_schedule":           {str(h): v for h, v in pv_schedule.items()},
            "reservoir_trajectory":  {str(h): v for h, v in reservoir_trajectory.items()},
            "efficiency_per_hour":   {str(h): v for h, v in efficiency_per_hour.items()},
            "inflow_m3h":            {str(h): v for h, v in inflow_m3h.items()},
            "solver_metrics":        solver_metrics or {},
            "initial_state":         initial_state or {},
        }
        with open(_path(delivery_date), "w") as f:
            json.dump(payload, f, indent=2)

    def load(self, delivery_date: str) -> Optional[dict]:
        p = _path(delivery_date)
        if not os.path.exists(p):
            return None
        with open(p) as f:
            raw = json.load(f)
        # Re-key hour strings back to int
        for key in ("psp_schedule", "bess_schedule", "pv_schedule",
                    "reservoir_trajectory", "efficiency_per_hour", "inflow_m3h"):
            if key in raw:
                raw[key] = {int(h): v for h, v in raw[key].items()}
        return raw

    def load_initial_state(self, delivery_date: str) -> dict:
        """Return the initial_state dict saved with this date, or empty dict."""
        raw = self.load(delivery_date)
        return raw.get("initial_state", {}) if raw else {}

    def load_chained_initial_state(self, delivery_date: str, bess_capacity_mwh: float,
                                    default_state: dict, reservoir_bounds: dict | None = None) -> dict:
        """Real day-to-day state continuity: reservoir levels and BESS SOC
        carry over from the PREVIOUS calendar day's actual solved ending
        state (its last-hour reservoir_trajectory/bess_schedule), instead of
        resetting to a static config constant every day. Without this, the
        solver has no incentive to ever recharge the BESS -- SOC just resets
        to the config default each morning regardless of where it actually
        ended, so any same-day discharge is a free lunch it never needs to
        pay back. Falls back to default_state when no previous day's
        ComponentStore record exists (first run, gaps in history) or that
        record is incomplete.

        Clamped to the same feasible bounds core_milp_builder.py's v_up/v_low
        constraints enforce (reservoir_bounds: upper_min_hm3, upper_usable_hm3,
        lower_min_hm3, lower_capacity_hm3) -- the solved ending value can sit
        fractionally outside those bounds (solver tolerance), which would
        otherwise make the very next day's own DA bid infeasible."""
        prev_date = (
            _dt.datetime.strptime(delivery_date, "%Y-%m-%d").date() - _dt.timedelta(days=1)
        ).isoformat()
        prev = self.load(prev_date)
        if prev is None:
            return dict(default_state)
        traj = prev.get("reservoir_trajectory") or {}
        bess_sched = prev.get("bess_schedule") or {}
        if not traj or not bess_sched or bess_capacity_mwh <= 0:
            return dict(default_state)
        last_traj_hour = max(traj)
        last_bess_hour = max(bess_sched)
        soc_frac = bess_sched[last_bess_hour]["soc_mwh"] / bess_capacity_mwh

        upper_hm3 = traj[last_traj_hour]["upper_hm3"]
        lower_hm3 = traj[last_traj_hour]["lower_hm3"]
        if reservoir_bounds:
            upper_hm3 = max(reservoir_bounds["upper_min_hm3"],
                             min(reservoir_bounds["upper_usable_hm3"], upper_hm3))
            lower_hm3 = max(reservoir_bounds["lower_min_hm3"],
                             min(reservoir_bounds["lower_capacity_hm3"], lower_hm3))

        return {
            "upper_reservoir_hm3": upper_hm3,
            "lower_reservoir_hm3": lower_hm3,
            "bess_soc_frac": max(0.0, min(1.0, soc_frac)),
        }
