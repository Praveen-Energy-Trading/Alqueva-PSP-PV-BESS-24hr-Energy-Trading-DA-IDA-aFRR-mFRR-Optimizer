"""
champions.py — the champion registry and the promotion rule (stage 7 of docs/Model_Development_Plan.pdf).

For every target exactly one champion is ACTIVE (enforced by the database). A champion is a version: the decision that
chose it, its specification (algorithm or average of algorithms, tuned or default settings, feature set, schedule) and,
once fitted, artifact files (serving.py). Changes are made only here, and every change is an event in an append-only
log: initial promotion, promotion after a challenge, rejected challenge, rollback.

Promotion rule: config/champion_policy.yaml. A challenger replaces the champion only if it is significantly and materially
better over the last N days, and no promotion happened recently. The rule is applied to out-of-sample predictions of both
on the same days; nothing here looks at the locked test period.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from common_layer.forecast_evaluation import metrics
from common_layer.forecast_evaluation.dm import daily_losses, diebold_mariano
from common_layer.model_development import registry, selection
from common_layer.model_development.registry import digest_of, session

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
DEFAULT_POLICY = os.path.join(_REPO, "config", "champion_policy.yaml")

SCHEMA = """
CREATE TABLE IF NOT EXISTS champions (
    target TEXT NOT NULL, version INTEGER NOT NULL, status TEXT NOT NULL, spec_json TEXT NOT NULL, decision_json TEXT,
    policy_digest TEXT, source TEXT, promoted_utc TEXT NOT NULL, retired_utc TEXT, notes TEXT, PRIMARY KEY (target, version));
CREATE UNIQUE INDEX IF NOT EXISTS one_active_champion ON champions(target) WHERE status = 'active';
CREATE TABLE IF NOT EXISTS champion_artifacts (
    target TEXT NOT NULL, version INTEGER NOT NULL, role TEXT NOT NULL DEFAULT 'champion', train_end TEXT NOT NULL,
    first_forecast_day TEXT, path TEXT NOT NULL, sha256 TEXT NOT NULL, set_version TEXT, members_json TEXT, tuned_at TEXT,
    created_utc TEXT NOT NULL, PRIMARY KEY (target, version, role, train_end));
CREATE TABLE IF NOT EXISTS champion_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, target TEXT NOT NULL, at_utc TEXT NOT NULL, event TEXT NOT NULL, version INTEGER,
    detail_json TEXT);
"""


def _open(db_path: Optional[str]):
    cm = session(db_path)
    con = cm.__enter__()
    con.executescript(SCHEMA)
    return cm, con


class _Db:
    """Registry session with the champion tables present."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    def __enter__(self):
        self._cm, self.con = _open(self.db_path)
        return self.con

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


def _now(now: Optional[dt.datetime] = None) -> str:
    """Time stamp of a change; `now` lets a replay of history (or a test) set it."""
    return (now or dt.datetime.now(dt.timezone.utc)).isoformat()


# ---------------------------------------------------------------------------------------------------- policy
def load_champion_policy(path: Optional[str] = None) -> dict:
    with open(path or DEFAULT_POLICY, encoding="utf-8") as fh:
        p = yaml.safe_load(fh)["promotion"]
    problems = []
    if p["window_days"] < 30 or not 0 < p["alpha"] < 1 or p["min_improvement"] < 0 or p["min_days_between_promotions"] < 0 \
            or p["loss"] not in ("absolute", "squared"):
        problems.append("promotion: window_days >= 30, alpha in (0,1), min_improvement >= 0, min_days_between_promotions >= 0, "
                        "loss absolute or squared")
    if problems:
        raise ValueError("; ".join(problems))
    return p


# ------------------------------------------------------------------------------------------------ the registry
def spec_from_decision(decision: dict, feature_set: str) -> dict:
    """What a champion is, in terms the serving code needs."""
    if decision.get("status") != "chosen" or decision.get("champion") is None:
        raise ValueError("only a decision with a chosen champion can be promoted")
    c = decision["champion"]
    members = list(c["members"]) if c["type"] == "ensemble" else [c["model"]]
    return {"type": c["type"], "members": members, "method": c.get("method"), "tuned": bool(decision["source"]["tuned"]),
            "feature_set": feature_set, "schedule": decision["source"]["schedule"], "reference": decision["policy"]["reference"],
            "metric": decision["policy"]["metric"]}


def active(target: str, db_path: Optional[str] = None) -> Optional[dict]:
    """The active champion of a target (version, spec, decision, promotion time), or None."""
    with _Db(db_path) as con:
        row = con.execute("SELECT version, spec_json, decision_json, promoted_utc, source, notes FROM champions "
                          "WHERE target=? AND status='active'", (target,)).fetchone()
    if row is None:
        return None
    return {"target": target, "version": row[0], "spec": json.loads(row[1]), "decision": json.loads(row[2]) if row[2] else None,
            "promoted_utc": row[3], "source": row[4], "notes": row[5]}


def _event(con, target: str, event: str, version: Optional[int], detail: dict, now: Optional[dt.datetime] = None) -> None:
    con.execute("INSERT INTO champion_events (target, at_utc, event, version, detail_json) VALUES (?,?,?,?,?)",
                (target, _now(now), event, version, json.dumps(detail, sort_keys=True, default=str)))


def _next_version(con, target: str) -> int:
    return (con.execute("SELECT MAX(version) FROM champions WHERE target=?", (target,)).fetchone()[0] or 0) + 1


def promote_initial(decision: dict, feature_set: str, source: str = "", db_path: Optional[str] = None,
                    now: Optional[dt.datetime] = None) -> dict:
    """Make the chosen champion of a target the active one. Refuses if the target already has an active champion
    (use a challenge to replace it, or a rollback)."""
    spec = spec_from_decision(decision, feature_set)
    target = decision["target"]
    with _Db(db_path) as con:
        if con.execute("SELECT 1 FROM champions WHERE target=? AND status='active'", (target,)).fetchone():
            raise ValueError(f"{target} already has an active champion; replace it through a challenge")
        v = _next_version(con, target)
        con.execute("INSERT INTO champions (target, version, status, spec_json, decision_json, policy_digest, source, promoted_utc) "
                    "VALUES (?,?,?,?,?,?,?,?)", (target, v, "active", json.dumps(spec, sort_keys=True), json.dumps(decision, sort_keys=True, default=str),
                                                 decision["policy_digest"], source, _now(now)))
        _event(con, target, "initial_promotion", v, {"spec": spec, "policy_digest": decision["policy_digest"], "source": source}, now)
    return {"target": target, "version": v, "spec": spec}


def events(target: Optional[str] = None, db_path: Optional[str] = None) -> pd.DataFrame:
    with _Db(db_path) as con:
        q = "SELECT id, target, at_utc, event, version, detail_json FROM champion_events"
        df = pd.read_sql_query(q + (" WHERE target=?" if target else "") + " ORDER BY id", con, params=(target,) if target else ())
    return df


def versions(target: str, db_path: Optional[str] = None) -> pd.DataFrame:
    with _Db(db_path) as con:
        return pd.read_sql_query("SELECT version, status, promoted_utc, retired_utc, source, spec_json FROM champions WHERE target=? "
                                 "ORDER BY version", con, params=(target,))


# --------------------------------------------------------------------------------------------- promotion rule
def evaluate_challenge(champion_pred: pd.DataFrame, challenger_pred: pd.DataFrame, policy: dict, comparison: dict, as_of: dt.date,
                       days_since_promotion: Optional[float]) -> dict:
    """Apply the promotion rule to the out-of-sample predictions of the champion and the challenger.

    Only days up to `as_of` are used (the last `window_days` days both predicted). Returns a dictionary with `promote`, the
    numbers, and a reason for every rule that failed."""
    loss = policy["loss"]
    a, b = metrics.add_error_columns(champion_pred.dropna(subset=["y_true", "y_pred"])), metrics.add_error_columns(challenger_pred.dropna(subset=["y_true", "y_pred"]))
    common = sorted(set(a["local_date"]) & set(b["local_date"]))
    common = [d for d in common if d <= as_of.toordinal()][-int(policy["window_days"]):]
    out: Dict[str, Any] = {"as_of": str(as_of), "window_days": int(policy["window_days"]), "n_days": len(common), "reasons": [],
                           "policy_digest": digest_of(policy), "promote": False}
    if len(common) < int(policy["window_days"]):
        out["reasons"].append(f"not enough evidence: {len(common)} common days up to {as_of}, need {policy['window_days']}")
        return out
    a, b = a[a["local_date"].isin(common)], b[b["local_date"].isin(common)]
    la, lb = float(daily_losses(a, loss).mean()), float(daily_losses(b, loss).mean())
    r = diebold_mariano(b, a, loss, comparison["hac_lags"], comparison["small_sample"])
    improvement = 1.0 - lb / la
    out.update(champion_loss=la, challenger_loss=lb, improvement=improvement, statistic=float(r["statistic"]), p_value=float(r["p_value"]),
               mean_diff=float(r["mean_diff"]))
    if not (r["p_value"] < policy["alpha"] and r["mean_diff"] < 0):
        out["reasons"].append(f"the challenger is not significantly better (DM statistic {r['statistic']:.2f}, p = {r['p_value']:.3g}, "
                              f"alpha {policy['alpha']})")
    if improvement < policy["min_improvement"]:
        out["reasons"].append(f"the improvement of {improvement:.1%} is below the required {policy['min_improvement']:.0%}")
    if days_since_promotion is not None and days_since_promotion < policy["min_days_between_promotions"]:
        out["reasons"].append(f"the last promotion was {days_since_promotion:.0f} days ago (minimum {policy['min_days_between_promotions']})")
    out["promote"] = not out["reasons"]
    if out["promote"]:
        out["reasons"].append(f"the challenger has the smaller {loss} loss by {improvement:.1%} over {len(common)} days "
                              f"(p = {r['p_value']:.3g}), and the cooldown has passed")
    return out


def apply_challenge(target: str, challenger_decision: dict, feature_set: str, evaluation: dict, source: str = "",
                    db_path: Optional[str] = None, now: Optional[dt.datetime] = None) -> dict:
    """Record the result of a challenge: promote the challenger (the old champion is retired) or log the rejection."""
    spec = spec_from_decision(challenger_decision, feature_set)
    with _Db(db_path) as con:
        cur = con.execute("SELECT version FROM champions WHERE target=? AND status='active'", (target,)).fetchone()
        if cur is None:
            raise ValueError(f"{target} has no active champion to challenge")
        if not evaluation["promote"]:
            _event(con, target, "challenge_rejected", cur[0], {"evaluation": evaluation, "challenger": spec}, now)
            return {"promoted": False, "version": cur[0], "reasons": evaluation["reasons"]}
        con.execute("UPDATE champions SET status='retired', retired_utc=? WHERE target=? AND status='active'", (_now(now), target))
        v = _next_version(con, target)
        con.execute("INSERT INTO champions (target, version, status, spec_json, decision_json, policy_digest, source, promoted_utc) "
                    "VALUES (?,?,?,?,?,?,?,?)", (target, v, "active", json.dumps(spec, sort_keys=True),
                                                 json.dumps(challenger_decision, sort_keys=True, default=str),
                                                 challenger_decision["policy_digest"], source, _now(now)))
        _event(con, target, "promotion", v, {"replaced_version": cur[0], "evaluation": evaluation, "spec": spec}, now)
    return {"promoted": True, "version": v, "replaced": cur[0], "reasons": evaluation["reasons"]}


def rollback(target: str, to_version: int, reason: str, db_path: Optional[str] = None, now: Optional[dt.datetime] = None) -> dict:
    """Reactivate an earlier version (the current one is retired). A reason is required."""
    if not reason.strip():
        raise ValueError("a rollback needs a reason")
    with _Db(db_path) as con:
        row = con.execute("SELECT status FROM champions WHERE target=? AND version=?", (target, to_version)).fetchone()
        if row is None:
            raise KeyError(f"{target} has no version {to_version}")
        cur = con.execute("SELECT version FROM champions WHERE target=? AND status='active'", (target,)).fetchone()
        if cur is not None and cur[0] == to_version:
            raise ValueError(f"version {to_version} is already active")
        con.execute("UPDATE champions SET status='retired', retired_utc=? WHERE target=? AND status='active'", (_now(now), target))
        con.execute("UPDATE champions SET status='active', retired_utc=NULL, promoted_utc=? WHERE target=? AND version=?",
                    (_now(now), target, to_version))
        _event(con, target, "rollback", to_version, {"from_version": cur[0] if cur else None, "reason": reason}, now)
    return {"version": to_version, "from_version": cur[0] if cur else None}


def days_since_last_promotion(target: str, as_of: dt.date, db_path: Optional[str] = None) -> Optional[float]:
    """Days from the active champion's promotion to `as_of`, or None if there is no champion."""
    a = active(target, db_path)
    if a is None:
        return None
    return float((as_of - dt.datetime.fromisoformat(a["promoted_utc"]).date()).days)


def register_artifact(target: str, version: int, train_end: dt.date, first_forecast_day: dt.date, path: str, sha256: str,
                      set_version: str, members: List[str], db_path: Optional[str] = None, now: Optional[dt.datetime] = None,
                      role: str = "champion", tuned_at: Optional[dt.date] = None) -> None:
    with _Db(db_path) as con:
        con.execute("INSERT OR REPLACE INTO champion_artifacts VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (target, version, role, str(train_end), str(first_forecast_day), path, sha256, set_version, json.dumps(members),
                     str(tuned_at) if tuned_at else None, _now(now)))
        _event(con, target, "artifact_fitted", version, {"role": role, "train_end": str(train_end),
                                                         "first_forecast_day": str(first_forecast_day), "path": path, "sha256": sha256,
                                                         "tuned_at": str(tuned_at) if tuned_at else None}, now)


def latest_artifact(target: str, version: int, db_path: Optional[str] = None, role: str = "champion") -> Optional[dict]:
    with _Db(db_path) as con:
        row = con.execute("SELECT train_end, first_forecast_day, path, sha256, set_version, members_json, tuned_at FROM champion_artifacts "
                          "WHERE target=? AND version=? AND role=? ORDER BY train_end DESC LIMIT 1", (target, version, role)).fetchone()
    if row is None:
        return None
    return {"train_end": row[0], "first_forecast_day": row[1], "path": row[2], "sha256": row[3], "set_version": row[4],
            "members": json.loads(row[5]), "tuned_at": row[6], "role": role}
