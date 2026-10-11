"""
monitoring.py — track the champion's forecasts, flag degradation, plan refits (stage 8 of docs/Model_Development_Plan.pdf).

    log_day         append the forecasts of one delivery day to the log (made from the as-of view only; never replaced)
    read_log        the logged forecasts joined with the outcomes known so far, in the same table format as a backtest
    calibrate       thresholds from the champion's development predictions (what "normal" error looked like)
    assess          the flags and the status for the last window (rule: config/monitoring.yaml)
    plan            is a refit or a re-tuning due?
    build_report    everything above for one target, with recommendations (never an automatic change)
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from common_layer.forecast_evaluation import metrics
from common_layer.forecast_evaluation.asof import AsOfView
from common_layer.forecast_evaluation.dm import daily_losses, diebold_mariano
from common_layer.model_development import champions
from common_layer.model_development.registry import digest_of, session

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_PATH = os.path.join(_REPO, "config", "monitoring.yaml")

SCHEMA = """
CREATE TABLE IF NOT EXISTS forecast_log (
    target TEXT NOT NULL, forecaster TEXT NOT NULL, ts_utc TEXT NOT NULL, local_date INTEGER NOT NULL, local_hour INTEGER,
    y_pred REAL, logged_utc TEXT NOT NULL, champion_version INTEGER, train_end TEXT, PRIMARY KEY (target, forecaster, ts_utc));
CREATE TABLE IF NOT EXISTS monitoring_baselines (
    target TEXT NOT NULL, version INTEGER NOT NULL, calibrated_utc TEXT NOT NULL, baseline_json TEXT NOT NULL,
    PRIMARY KEY (target, version));
"""


class LogError(RuntimeError):
    """A forecast could not be logged (for example, it was already logged)."""


def load_monitoring_config(path: Optional[str] = None, tuning_cfg: Optional[dict] = None, eval_cfg: Optional[dict] = None) -> dict:
    with open(path or DEFAULT_PATH, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    t, r, s = cfg["tracking"], cfg["relative"], cfg["schedule"]
    problems = []
    if t["window_days"] < 7 or not 0 < t["min_days"] <= t["window_days"] or not 0.5 <= t["quantile"] < 1 or not 0 < t["coverage_min"] <= 1:
        problems.append("tracking: window_days >= 7, 0 < min_days <= window_days, quantile in [0.5, 1), coverage_min in (0, 1]")
    if r["window_days"] < 30 or not 0 < r["alpha"] < 1:
        problems.append("relative: window_days >= 30, alpha in (0, 1)")
    if tuning_cfg is not None and s["retune_every_days"] != tuning_cfg["search"]["retune_every_days"]:
        problems.append(f"schedule.retune_every_days {s['retune_every_days']} differs from config/tuning.yaml "
                        f"({tuning_cfg['search']['retune_every_days']})")
    if eval_cfg is not None and s["refit_every_days"] != eval_cfg["schedule"]["retrain_every_days"]:
        problems.append(f"schedule.refit_every_days {s['refit_every_days']} differs from config/evaluation.yaml "
                        f"({eval_cfg['schedule']['retrain_every_days']})")
    if problems:
        raise ValueError("; ".join(problems))
    return cfg


class _Db:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    def __enter__(self):
        self._cm = session(self.db_path)
        con = self._cm.__enter__()
        con.executescript(SCHEMA)
        return con

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


def _now(now: Optional[dt.datetime]) -> str:
    return (now or dt.datetime.now(dt.timezone.utc)).isoformat()


# ------------------------------------------------------------------------------------------------------- log
def log_day(target: str, forecaster: str, fc, view: AsOfView, D: dt.date, version: int, train_end: str, db_path: Optional[str] = None,
            now: Optional[dt.datetime] = None, history_days: int = 400) -> int:
    """Forecast delivery day D from its as-of view and append the prices to the log. Returns the number of rows.

    The log is append-only: logging a day again is an error, so a forecast cannot be changed after the outcome is known."""
    v = view.at(D, history_days)
    rows = v[v["local_date"] == D.toordinal()]
    prices = fc.forecast_prices(v, D)
    stamp = _now(now)
    try:
        with _Db(db_path) as con:
            con.executemany("INSERT INTO forecast_log VALUES (?,?,?,?,?,?,?,?,?)",
                            [(target, forecaster, ts.isoformat(), D.toordinal(), int(h), float(p), stamp, version, train_end)
                             for ts, h, p in zip(rows.index, rows["local_hour"], prices.to_numpy())])
    except sqlite3.IntegrityError as e:
        raise LogError(f"{forecaster} forecasts for {target} {D} are already logged; the log is append-only") from e
    return len(rows)


def read_log(target: str, forecaster: str, view: AsOfView, db_path: Optional[str] = None) -> pd.DataFrame:
    """Logged forecasts with the outcomes known so far (rows without an outcome are left out), indexed by hour (UTC)."""
    with _Db(db_path) as con:
        df = pd.read_sql_query("SELECT ts_utc, local_date, local_hour, y_pred, logged_utc, champion_version FROM forecast_log "
                               "WHERE target=? AND forecaster=? ORDER BY ts_utc", con, params=(target, forecaster))
    if df.empty:
        return pd.DataFrame(columns=["y_true", "y_pred", "local_date", "local_hour", "fold", "logged_utc"])
    df["ts"] = pd.to_datetime(df["ts_utc"], utc=True)
    series = view.spec["series"]
    truth = pd.concat([view.truth_day(dt.date.fromordinal(int(d)))[series] for d in sorted(df["local_date"].unique())])
    out = pd.DataFrame({"y_true": truth.reindex(df["ts"]).to_numpy(), "y_pred": df["y_pred"].to_numpy(), "local_date": df["local_date"].to_numpy(),
                        "local_hour": df["local_hour"].to_numpy(), "fold": df["champion_version"].to_numpy(), "logged_utc": df["logged_utc"].to_numpy()},
                       index=pd.DatetimeIndex(df["ts"]))
    return metrics.add_error_columns(out.dropna(subset=["y_true", "y_pred"]))


def late_days(log: pd.DataFrame) -> int:
    """Days whose forecast was logged on or after the delivery day (not made ahead of time)."""
    if log.empty:
        return 0
    logged = pd.to_datetime(log["logged_utc"], utc=True).dt.date.to_numpy()
    deliv = np.array([dt.date.fromordinal(int(d)) for d in log["local_date"]])
    return int(len({d for d, a, b in zip(log["local_date"], logged, deliv) if a >= b}))


# ---------------------------------------------------------------------------------------------- thresholds
def rolling_stats(pred: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Per delivery day: the mean absolute error and the mean signed error over the `window_days` days ending that day."""
    p = metrics.add_error_columns(pred.dropna(subset=["y_true", "y_pred"])) if "err" not in pred else pred.dropna(subset=["err"])
    g = p.groupby("local_date")["err"]
    daily = pd.DataFrame({"loss": g.apply(lambda e: float(e.abs().mean())), "bias": g.mean()})
    full = daily.reindex(range(int(daily.index.min()), int(daily.index.max()) + 1))
    roll = full.rolling(window_days, min_periods=window_days).mean()
    roll["n"] = full["loss"].notna().astype(float).rolling(window_days, min_periods=1).sum()
    return roll.dropna(subset=["loss"])


def calibrate(dev_pred: pd.DataFrame, cfg: dict) -> dict:
    """What normal looked like: the `quantile` of the rolling mean absolute error and of the size of the rolling bias over the
    development windows (full windows only)."""
    t = cfg["tracking"]
    r = rolling_stats(dev_pred, int(t["window_days"]))
    r = r[r["n"] >= int(t["window_days"])]
    if len(r) < 30:
        raise ValueError(f"only {len(r)} full development windows; too few to set thresholds")
    return {"window_days": int(t["window_days"]), "quantile": float(t["quantile"]), "n_windows": int(len(r)),
            "dev_loss": float(r["loss"].mean()), "loss_threshold": float(r["loss"].quantile(t["quantile"])),
            "bias_threshold": float(r["bias"].abs().quantile(t["quantile"])), "config_digest": digest_of(cfg)}


def dev_predictions(spec: dict, preds: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The champion's development-window predictions from those of its members (one algorithm, or their average)."""
    from common_layer.forecast_evaluation.ensemble import combine
    from common_layer.model_development import comparison as cmp
    from common_layer.model_development.selection import _with_keep_columns
    prep = {m: _with_keep_columns(p) for m, p in cmp.prepare({m: preds[m] for m in spec["members"]}).items()}
    if spec["type"] == "single":
        return prep[spec["members"][0]]
    return combine(prep, spec["method"])


def save_baseline(target: str, version: int, baseline: dict, db_path: Optional[str] = None, now: Optional[dt.datetime] = None) -> None:
    with _Db(db_path) as con:
        con.execute("INSERT OR REPLACE INTO monitoring_baselines VALUES (?,?,?,?)", (target, version, _now(now), json.dumps(baseline, sort_keys=True)))


def load_baseline(target: str, version: int, db_path: Optional[str] = None) -> Optional[dict]:
    with _Db(db_path) as con:
        row = con.execute("SELECT baseline_json FROM monitoring_baselines WHERE target=? AND version=?", (target, version)).fetchone()
    return json.loads(row[0]) if row else None


# -------------------------------------------------------------------------------------------------- assess
def assess(champ_log: pd.DataFrame, ref_log: Optional[pd.DataFrame], baseline: dict, cfg: dict, as_of: dt.date, outcome_days: List[int],
           comparison: dict) -> dict:
    """The flags and the status for the window ending `as_of` (days after it are ignored)."""
    t, rel = cfg["tracking"], cfg["relative"]
    W = int(t["window_days"])
    cutoff = as_of.toordinal()
    window = [d for d in range(cutoff - W + 1, cutoff + 1)]
    have = champ_log[champ_log["local_date"].isin(window)]
    scored = sorted(set(have["local_date"]))
    expected = [d for d in window if d in set(outcome_days)]
    out: Dict = {"as_of": str(as_of), "window_days": W, "scored_days": len(scored), "days_with_outcomes": len(expected),
                 "flags": {}, "thresholds": {k: baseline[k] for k in ("dev_loss", "loss_threshold", "bias_threshold")}}
    if len(scored) < int(t["min_days"]):
        out["status"] = "insufficient_data"
        out["notes"] = [f"{len(scored)} scored days in the window, need {t['min_days']}"]
        return out
    loss = float(daily_losses(have, "absolute").mean())
    bias = float(have["err"].mean())
    out.update(loss=loss, bias=bias, loss_ratio_to_dev=loss / baseline["dev_loss"])
    cover = len(scored) / max(1, len(expected))
    out["coverage"] = cover
    flags = {"level": loss > baseline["loss_threshold"], "bias": abs(bias) > baseline["bias_threshold"], "coverage": cover < t["coverage_min"]}
    notes: List[str] = []
    rel_flag = None
    if ref_log is not None and len(ref_log):
        rd = sorted(set(champ_log["local_date"]) & set(ref_log["local_date"]))
        rd = [d for d in rd if d <= cutoff][-int(rel["window_days"]):]
        if len(rd) >= int(rel["window_days"]):
            a, b = champ_log[champ_log["local_date"].isin(rd)], ref_log[ref_log["local_date"].isin(rd)]
            r = diebold_mariano(a, b, "absolute", comparison["hac_lags"], comparison["small_sample"])      # a = champion, b = reference
            rel_flag = bool(r["p_value"] < rel["alpha"] and r["mean_diff"] > 0)                          # the reference has the smaller loss
            out["relative"] = {"days": len(rd), "statistic": float(r["statistic"]), "p_value": float(r["p_value"]), "mean_diff": float(r["mean_diff"])}
        else:
            notes.append(f"relative check skipped: {len(rd)} common days with the reference, need {rel['window_days']}")
    else:
        notes.append("relative check skipped: no reference forecasts logged")
    flags["relative"] = rel_flag
    out["flags"] = flags
    late = late_days(have)
    if late:
        notes.append(f"{late} day(s) in the window were logged on or after the delivery day (not made ahead of time)")
    if flags["relative"] or (flags["level"] and flags["bias"]):
        out["status"] = "degraded"
    elif flags["level"] or flags["bias"] or flags["coverage"]:
        out["status"] = "warning"
    else:
        out["status"] = "ok"
    out["notes"] = notes
    return out


# ---------------------------------------------------------------------------------------------------- plan
def plan(target: str, as_of: dt.date, cfg: dict, eval_cfg: dict, db_path: Optional[str] = None) -> dict:
    """Is a refit due (every `refit_every_days`), and would it re-tune (settings older than `retune_every_days`)?"""
    champ = champions.active(target, db_path)
    if champ is None:
        raise ValueError(f"{target} has no active champion")
    s = cfg["schedule"]
    art = champions.latest_artifact(target, champ["version"], db_path)
    out: Dict = {"target": target, "as_of": str(as_of), "version": champ["version"]}
    if art is None:
        out.update(action="fit", reason="the active champion has no fitted artifact", refit_due=True, retune_due=bool(champ["spec"]["tuned"]))
        return out
    first = dt.date.fromisoformat(art["first_forecast_day"])
    out["days_since_refit"] = (as_of - first).days
    out["refit_due"] = out["days_since_refit"] >= int(s["refit_every_days"])
    out["next_refit_day"] = str(first + dt.timedelta(days=int(s["refit_every_days"])))
    retune = False
    if champ["spec"]["tuned"] and art["tuned_at"]:
        next_train_end = as_of - dt.timedelta(days=int(champ["spec"]["schedule"]["gap_days"]) + 1)
        out["days_since_tuning"] = (next_train_end - dt.date.fromisoformat(art["tuned_at"])).days
        retune = out["days_since_tuning"] >= int(s["retune_every_days"])
    out["retune_due"] = bool(out["refit_due"] and retune)
    out["action"] = "refit_and_retune" if out["retune_due"] else ("refit" if out["refit_due"] else "none")
    return out


def recommendations(status: dict, schedule: dict, flags_notes: Optional[List[str]] = None) -> List[str]:
    rec: List[str] = []
    if schedule["action"] == "fit":
        rec.append("fit the champion (`tools/champion.py fit`): it has no artifact")
    elif schedule["action"] == "refit_and_retune":
        rec.append("refit and re-tune now: the fit is due and the tuned settings are older than the re-tuning interval")
    elif schedule["action"] == "refit":
        rec.append("refit now (settings are reused): the fit is older than the refit interval")
    if status["status"] == "degraded":
        rec.append("review the champion: run a challenge (config/champion_policy.yaml) and consider a forced re-tuning (`fit --force-retune`)")
    elif status["status"] == "warning":
        raised = [k for k, v in status["flags"].items() if v]
        rec.append(f"watch closely: flags raised: {', '.join(raised)}")
    if status["flags"].get("coverage"):
        rec.append("forecasts are missing for days with outcomes: check the daily forecast job")
    return rec or ["no action needed"]


def build_report(target: str, as_of: dt.date, view: AsOfView, cfg: dict, eval_cfg: dict, db_path: Optional[str] = None) -> dict:
    champ = champions.active(target, db_path)
    if champ is None:
        raise ValueError(f"{target} has no active champion")
    base = load_baseline(target, champ["version"], db_path)
    if base is None:
        raise ValueError(f"no monitoring baseline for {target} version {champ['version']}; calibrate it first")
    clog = read_log(target, "champion", view, db_path)
    rlog = read_log(target, "reference", view, db_path)
    cutoff = as_of.toordinal()
    outcome_days = [d for d in sorted(set(range(cutoff - int(cfg["tracking"]["window_days"]) + 1, cutoff + 1)))
                    if view.truth_day(dt.date.fromordinal(d))[view.spec["series"]].notna().any()]
    status = assess(clog, rlog if len(rlog) else None, base, cfg, as_of, outcome_days, eval_cfg["comparison"])
    sched = plan(target, as_of, cfg, eval_cfg, db_path)
    series = rolling_stats(clog[clog["local_date"] <= cutoff], int(cfg["tracking"]["window_days"])) if len(clog) else pd.DataFrame()
    return {"target": target, "version": champ["version"], "spec": champ["spec"], "status": status, "schedule": sched,
            "recommendations": recommendations(status, sched), "series": series}


# ------------------------------------------------------------------------------------------------ presentation
def plot_rolling(report: dict, baseline: dict, path: str) -> str:
    """Rolling mean absolute error of the champion against its development level and threshold."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ser = report["series"]
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    if len(ser):
        x = [dt.date.fromordinal(int(d)) for d in ser.index]
        ax.plot(x, ser["loss"], color="#1f4e79", lw=1.8, label=f"champion, rolling {baseline['window_days']}-day MAE")
    ax.axhline(baseline["dev_loss"], color="0.5", lw=1, label="development level")
    ax.axhline(baseline["loss_threshold"], color="#c0504d", lw=1.2, ls="--", label=f"threshold ({baseline['quantile']:.0%} of development windows)")
    ax.set_ylabel("MAE")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def to_json(report: dict) -> dict:
    return {k: v for k, v in report.items() if k != "series"}


def render(report: dict, figure_name: Optional[str] = None) -> str:
    s, sc = report["status"], report["schedule"]
    spec = report["spec"]
    what = f"the {spec['method']} of {', '.join(spec['members'])}" if spec["type"] == "ensemble" else spec["members"][0]
    out = [f"# Champion monitoring: {report['target']}", "", "Generated by `tools/monitor_champion.py`; do not edit by hand.", "",
           f"Champion version {report['version']}: {what} ({'tuned' if spec['tuned'] else 'default'} settings). As of {s['as_of']}.", "",
           f"**Status: {s['status']}**", ""]
    if figure_name:
        out += [f"![rolling error]({figure_name})", ""]
    if s["status"] != "insufficient_data":
        t = s["thresholds"]
        out += ["| | value | development level | threshold | flag |", "|---|---|---|---|---|",
                f"| mean absolute error, last {s['window_days']} days | {s['loss']:.3f} | {t['dev_loss']:.3f} | {t['loss_threshold']:.3f} | {s['flags']['level']} |",
                f"| mean error (bias) | {s['bias']:+.3f} | | ±{t['bias_threshold']:.3f} | {s['flags']['bias']} |",
                f"| days scored / days with outcomes | {s['scored_days']} / {s['days_with_outcomes']} | | coverage {s['coverage']:.0%} | {s['flags']['coverage']} |"]
        if "relative" in s:
            r = s["relative"]
            out.append(f"| against the reference, last {r['days']} days | DM statistic {r['statistic']:.2f}, p = {r['p_value']:.3g} | | | {s['flags']['relative']} |")
        out.append("")
    for n in s.get("notes", []):
        out.append(f"- {n}")
    out += ["", "## Schedule", "", f"- action: **{sc['action']}**"]
    for k in ("days_since_refit", "next_refit_day", "days_since_tuning"):
        if k in sc:
            out.append(f"- {k.replace('_', ' ')}: {sc[k]}")
    out += ["", "## Recommendations", ""] + [f"- {r}" for r in report["recommendations"]]
    return "\n".join(out) + "\n"
