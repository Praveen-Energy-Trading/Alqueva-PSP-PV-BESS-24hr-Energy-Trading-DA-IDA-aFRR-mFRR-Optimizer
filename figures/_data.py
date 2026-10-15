"""_data.py - everything the figures read, in one place. Every number comes from the run's own outputs (component store, position store, daily report,
backtest report, model decisions); nothing is made up or recomputed away from the solver."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / "runtime"
N_ISP = 96


def plant() -> dict:
    p = yaml.safe_load((ROOT / "config" / "plant.yaml").read_text(encoding="utf-8"))
    psp, bess, pv, res = p["psp"], p["bess"], p["pv"], p["reservoir"]
    n = psp["n_units"]
    p["derived"] = {
        "gen_cap_mw": n * psp["p_turbine_max_mw"] + pv["peak_capacity_mw"] + bess["power_mw"],     # all turbines + PV + battery discharge
        "pump_cap_mw": n * psp["p_pump_max_mw"] + bess["power_mw"],                                  # all pumps + battery charge
        "turbine_cap_mw": n * psp["p_turbine_max_mw"], "pump_only_cap_mw": n * psp["p_pump_max_mw"],
        "upper_max_hm3": res["upper_usable_hm3"], "upper_min_hm3": res["upper_min_hm3"],
        "bess_min_mwh": bess["soc_min_frac"] * bess["capacity_mwh"], "bess_max_mwh": bess["soc_max_frac"] * bess["capacity_mwh"],
        "fcr_mw": p["fcr"]["mandatory_headroom_mw"],
    }
    return p


@dataclass
class Day:
    date: str
    isp: np.ndarray = field(default_factory=lambda: np.arange(1, N_ISP + 1))
    price: np.ndarray | None = None          # DA price per ISP, EUR/MWh
    turbine: np.ndarray | None = None        # MW, per ISP
    pump: np.ndarray | None = None           # MW, positive number
    pv_used: np.ndarray | None = None
    pv_available: np.ndarray | None = None
    bess_charge: np.ndarray | None = None    # grid + PV charging, MW
    bess_discharge: np.ndarray | None = None
    bess_soc: np.ndarray | None = None       # MWh
    upper: np.ndarray | None = None          # hm3
    lower: np.ndarray | None = None
    head: np.ndarray | None = None           # m
    spill: np.ndarray | None = None
    units_q_turb: np.ndarray | None = None   # (isp, unit) m3/h
    eta_turbine: np.ndarray | None = None    # power-weighted turbine efficiency per ISP (0 when idle)
    hourly: pd.DataFrame | None = None       # the report's 24-row dispatch sheet
    initial: dict = field(default_factory=dict)
    solver: dict = field(default_factory=dict)

    @property
    def net(self) -> np.ndarray:
        """Net MW the plant puts on the grid in each ISP (generation positive)."""
        return self.turbine - self.pump + self.pv_used + self.bess_discharge - self.bess_charge


def _series(block: dict, key: str) -> np.ndarray:
    return np.array([float(block[str(i)][key]) for i in range(1, N_ISP + 1)])


def latest_date() -> str | None:
    """The newest delivery date that has both a solved day (component store) and a daily report."""
    comps = {p.stem.replace("components_", "") for p in (RUNTIME / "components").glob("components_*.json")}
    reps = {p.stem.replace("daily_report_", "") for p in (RUNTIME / "reports").glob("daily_report_*.xlsx")}
    both = sorted(comps & reps)
    return both[-1] if both else None


def load_day(date: str) -> Day:
    comp = json.loads((RUNTIME / "components" / f"components_{date}.json").read_text(encoding="utf-8"))
    d = Day(date=date)
    psp, bess, pv, res = comp["psp_schedule"], comp["bess_schedule"], comp["pv_schedule"], comp["reservoir_trajectory"]
    d.turbine, d.pump = _series(psp, "turbine_mw"), _series(psp, "pump_mw")
    d.pv_used, d.pv_available = _series(pv, "used_mw"), _series(pv, "available_mw")
    d.bess_charge, d.bess_discharge, d.bess_soc = _series(bess, "total_charge_mw"), _series(bess, "discharge_mw"), _series(bess, "soc_mwh")
    d.upper, d.lower, d.head, d.spill = _series(res, "upper_hm3"), _series(res, "lower_hm3"), _series(res, "head_m"), _series(res, "spill_m3h")
    d.units_q_turb = np.array([psp[str(i)]["units_q_turb"] for i in range(1, N_ISP + 1)], dtype=float)
    d.eta_turbine = _series(comp["efficiency_per_hour"], "eta_trb_pw")
    d.initial, d.solver = comp.get("initial_state", {}), comp.get("solver_metrics", {})
    try:
        con = sqlite3.connect(RUNTIME / "db" / "positions.db")
        rows = pd.read_sql("select hour, price_eur_mwh from positions where delivery_date=? and gate='DA' order by hour", con, params=(date,))
        con.close()
        if len(rows) == N_ISP:
            d.price = rows["price_eur_mwh"].to_numpy(dtype=float)
    except Exception:
        pass
    try:
        raw = pd.read_excel(RUNTIME / "reports" / f"daily_report_{date}.xlsx", sheet_name="Dispatch_Hourly", header=None)
        h = raw.iloc[3:].copy()
        h.columns = list(raw.iloc[2])
        h = h.reset_index(drop=True)
        d.hourly = h.apply(lambda col: pd.to_numeric(col, errors="coerce") if pd.to_numeric(col, errors="coerce").notna().all() else col)
    except Exception:
        pass
    return d


def backtest() -> dict | None:
    """The rolling backtest report: the daily table and the risk sheet."""
    path = RUNTIME / "reports" / "backtest_rolling_365d.xlsx"
    if not path.exists():
        return None
    x = pd.ExcelFile(path)
    daily = x.parse("Backtest")
    daily["date"] = pd.to_datetime(daily["date"])
    risk = x.parse("Risk", header=None)
    rows = {}
    section = None
    for label, value in zip(risk[0], risk[1]):
        if isinstance(label, str) and label.startswith("---"):
            section = label.strip("- ").strip()
        elif isinstance(label, str) and pd.notna(value):
            rows[(section, label)] = value
    return {"daily": daily, "risk": rows}


TARGETS = [("da_price", "Day-ahead"), ("ida1_price", "IDA 1"), ("ida2_price", "IDA 2"), ("ida3_price", "IDA 3"), ("xbid_price", "XBID"),
           ("afrr_cap_price", "aFRR up"), ("afrr_dn_price", "aFRR down")]


def forecast_skill() -> list[dict]:
    """Locked-test error of the chosen model against the reference model, per price target."""
    out = []
    for key, name in TARGETS:
        path = ROOT / "model_development" / "locked_test" / f"{key}.json"
        if not path.exists():
            continue
        j = json.loads(path.read_text(encoding="utf-8"))
        ref = (j.get("reference_scores") or {}).get("mae")
        champ = (j.get("locked_scores") or {}).get("mae")
        ref_rmse = (j.get("reference_scores") or {}).get("rmse")
        champ_rmse = (j.get("locked_scores") or {}).get("rmse")
        if ref is None or champ is None:
            continue
        model = j["champion"]
        label = model.get("model") if model.get("type") == "single" else f"mean of {len(j.get('members') or model.get('members') or [])} models"
        out.append({"key": key, "name": name, "champion": float(champ), "reference": float(ref), "champion_rmse": champ_rmse, "reference_rmse": ref_rmse, "reference_name": j.get("reference", "reference"),
                    "model": label, "days": int((j.get("locked_scores") or {}).get("n", 0) // 24),
                    "p": (j.get("dm_vs_reference") or {}).get("p_value")})
    return out
