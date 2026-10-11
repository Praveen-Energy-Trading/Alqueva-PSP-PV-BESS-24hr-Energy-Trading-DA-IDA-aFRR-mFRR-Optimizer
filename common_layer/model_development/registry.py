"""
registry.py — the experiment registry (stage 1 of docs/Model_Development_Plan.pdf).

Every evaluation run (and schedule comparison) is written to a local SQLite database with what it measured and what it
depended on, so that any number in a document can be traced to a run and a repeat of an experiment can be checked.

    run_key        identity of an EXPERIMENT: hash of what defines it (kind, target, stage, test window, feature set,
                   schedule, config digest, data digest, tuning digest, forecasters). The code revision is recorded but
                   is not part of the key, so the same experiment run again after a commit is the same experiment.
    attempt        1, 2, ... each time the same experiment is registered again.
    result_digest  hash of the scores (SSE, MAE, RMSE per forecaster). All attempts of one run_key must have the same
                   digest; if not, the experiment is not reproducible and `reproducibility()` says so.

Tables: runs, run_models, run_comparisons, tuning_choices. Default file: runtime/model_registry/registry.db
(override with the `db_path` argument or the MODEL_REGISTRY_DB environment variable).
"""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import os
import sqlite3
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_key TEXT NOT NULL, attempt INTEGER NOT NULL, kind TEXT NOT NULL, target TEXT NOT NULL, stage TEXT,
    window_start TEXT, window_end TEXT, feature_set TEXT, schedule_json TEXT, config_digest TEXT, data_digest TEXT,
    tuning_digest TEXT, code_revision TEXT, result_digest TEXT NOT NULL, created_utc TEXT NOT NULL, source TEXT,
    artifacts_dir TEXT, notes TEXT, PRIMARY KEY (run_key, attempt));
CREATE TABLE IF NOT EXISTS run_models (
    run_key TEXT NOT NULL, attempt INTEGER NOT NULL, forecaster TEXT NOT NULL, role TEXT, tuned INTEGER,
    n REAL, sse REAL, mae REAL, rmse REAL, coverage REAL, PRIMARY KEY (run_key, attempt, forecaster));
CREATE TABLE IF NOT EXISTS run_comparisons (
    run_key TEXT NOT NULL, attempt INTEGER NOT NULL, forecaster TEXT NOT NULL, reference TEXT NOT NULL,
    n_days REAL, mean_diff REAL, statistic REAL, p_value REAL, PRIMARY KEY (run_key, attempt, forecaster, reference));
CREATE TABLE IF NOT EXISTS tuning_choices (
    run_key TEXT NOT NULL, attempt INTEGER NOT NULL, model TEXT NOT NULL, tuned_at TEXT NOT NULL, score REAL,
    default_score REAL, n_trials INTEGER, inner_train_days INTEGER, validation_days INTEGER, params_json TEXT,
    PRIMARY KEY (run_key, attempt, model, tuned_at));
CREATE TABLE IF NOT EXISTS selections (
    target TEXT NOT NULL, created_utc TEXT NOT NULL, policy_digest TEXT, source TEXT, status TEXT, champion_json TEXT,
    decision_json TEXT, PRIMARY KEY (target, created_utc));
CREATE TABLE IF NOT EXISTS tuning_trials (
    run_key TEXT NOT NULL, attempt INTEGER NOT NULL, model TEXT NOT NULL, tuned_at TEXT NOT NULL, trial INTEGER NOT NULL,
    params_json TEXT, score REAL, pruned INTEGER, is_default INTEGER, duplicate INTEGER, selected INTEGER,
    block_scores TEXT, PRIMARY KEY (run_key, attempt, model, tuned_at, trial));
"""


def default_db_path() -> str:
    return os.environ.get("MODEL_REGISTRY_DB") or os.path.join(_REPO, "runtime", "model_registry", "registry.db")


def connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = db_path or default_db_path()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


@contextlib.contextmanager
def session(db_path: Optional[str] = None):
    """Open the registry, commit on success, always close (the sqlite3 context manager alone does not close)."""
    con = connect(db_path)
    try:
        yield con
        con.commit()
    finally:
        con.close()


def digest_of(obj: Any) -> str:
    """Hash of any JSON-able object (dictionary keys sorted), for configuration digests."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _num(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if v != v else v


def _result_digest(models: Iterable[dict]) -> str:
    rows = sorted((m["forecaster"], f"{m['sse']:.9g}", f"{m['mae']:.9g}", f"{m['rmse']:.9g}") for m in models)
    return digest_of(rows)


def register(kind: str, target: str, models: List[dict], *, stage: str = "dev", window: tuple = (None, None),
             feature_set: str = "", schedule: Optional[dict] = None, config_digest: str = "", data_digest: str = "",
             tuning_digest: str = "", code_revision: str = "", comparisons: Optional[List[dict]] = None,
             tuning_choices: Optional[List[dict]] = None, tuning_trials: Optional[List[dict]] = None, source: str = "",
             artifacts_dir: str = "", notes: str = "", db_path: Optional[str] = None) -> Dict[str, Any]:
    """Record one run. `models` rows need forecaster, sse, mae, rmse (and may carry n, coverage, role).

    Returns {"run_key", "attempt", "reproduced"}: `reproduced` is None for the first attempt, True when every attempt
    of this experiment has the same result digest, False otherwise.
    """
    if not models:
        raise ValueError("a run needs at least one forecaster")
    for m in models:
        for k in ("forecaster", "sse", "mae", "rmse"):
            if k not in m:
                raise ValueError(f"model row without {k!r}: {m}")
    schedule = schedule or {}
    key = digest_of({"kind": kind, "target": target, "stage": stage, "window": list(window), "feature_set": feature_set,
                     "schedule": schedule, "config": config_digest, "data": data_digest, "tuning": tuning_digest,
                     "forecasters": sorted(m["forecaster"] for m in models)})[:12]
    result = _result_digest(models)
    with session(db_path) as con:
        last = con.execute("SELECT MAX(attempt) FROM runs WHERE run_key=?", (key,)).fetchone()[0]
        attempt = (last or 0) + 1
        con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (key, attempt, kind, target, stage, window[0], window[1], feature_set,
                     json.dumps(schedule, sort_keys=True, default=str), config_digest, data_digest, tuning_digest,
                     code_revision, result, dt.datetime.now(dt.timezone.utc).isoformat(), source, artifacts_dir, notes))
        for m in models:
            name = m["forecaster"]
            role = m.get("role") or ("ensemble" if name.startswith("ensemble_") else "single")
            con.execute("INSERT INTO run_models VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (key, attempt, name, role, int(name.endswith("_tuned")), _num(m.get("n")), _num(m["sse"]),
                         _num(m["mae"]), _num(m["rmse"]), _num(m.get("coverage"))))
        for c in comparisons or []:
            con.execute("INSERT OR REPLACE INTO run_comparisons VALUES (?,?,?,?,?,?,?,?)",
                        (key, attempt, c["forecaster"], c["reference"], _num(c.get("n_days")), _num(c.get("mean_diff")),
                         _num(c.get("statistic")), _num(c.get("p_value"))))
        for t in tuning_choices or []:
            con.execute("INSERT OR REPLACE INTO tuning_choices VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (key, attempt, t["model"], str(t["tuned_at"]), _num(t.get("score")), _num(t.get("default_score")),
                         int(t["n_trials"]) if _num(t.get("n_trials")) is not None else None,
                         int(t["inner_train_days"]) if _num(t.get("inner_train_days")) is not None else None,
                         int(t["validation_days"]) if _num(t.get("validation_days")) is not None else None,
                         str(t.get("params", ""))))
        for t in tuning_trials or []:
            blocks = [t[c] for c in sorted(c for c in t if c.startswith("block")) if _num(t[c]) is not None]
            con.execute("INSERT OR REPLACE INTO tuning_trials VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                        (key, attempt, t["model"], str(t["tuned_at"]), int(t["trial"]), str(t.get("params", "")), _num(t.get("score")),
                         int(bool(t.get("pruned"))), int(bool(t.get("is_default"))), int(bool(t.get("duplicate"))),
                         int(bool(t.get("selected"))), json.dumps([round(float(b), 6) for b in blocks])))
        digests = [r[0] for r in con.execute("SELECT result_digest FROM runs WHERE run_key=?", (key,))]
    return {"run_key": key, "attempt": attempt, "reproduced": None if attempt == 1 else len(set(digests)) == 1}


def code_state(revision: str) -> str:
    """The git revision, with '+dirty' when code, configuration or tools have uncommitted changes."""
    import subprocess
    try:
        out = subprocess.run(["git", "status", "--porcelain", "--", "common_layer", "config", "tools"], cwd=_REPO,
                             capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return revision
    return revision + ("+dirty" if out else "")


def register_evaluation(res: dict, *, tuning_cfg: Optional[dict] = None, source: str = "evaluate_forecaster",
                        artifacts_dir: str = "", notes: str = "", db_path: Optional[str] = None) -> Dict[str, Any]:
    """Record the result dictionary returned by `forecast_evaluation.runner.evaluate`."""
    meta = res["meta"]
    models = res["overall"].to_dict("records")
    comps = res["comparisons"].to_dict("records") if res.get("comparisons") is not None and len(res["comparisons"]) else []
    log = res.get("tuning_log")
    choices = log.to_dict("records") if log is not None else []
    tt = res.get("tuning_trials")
    trials = tt.to_dict("records") if tt is not None else []
    uses_tuning = bool(choices) or any(m["forecaster"].startswith("ensemble_") for m in models)
    return register("evaluation", meta["target"], models, stage=meta["stage"], window=(meta["test_start"], meta["test_end"]),
                    feature_set=str(meta.get("feature_set", "")),
                    schedule={k: meta.get(k) for k in ("retrain_every_days", "gap_days", "train_window_days")},
                    config_digest=meta["config_digest"], data_digest=meta["data_digest"],
                    tuning_digest=digest_of(tuning_cfg) if (tuning_cfg is not None and uses_tuning) else "",
                    code_revision=code_state(meta.get("code", "")), comparisons=comps, tuning_choices=choices, tuning_trials=trials, source=source,
                    artifacts_dir=artifacts_dir or res.get("paths", {}).get("report", ""), notes=notes, db_path=db_path)


def register_run_dir(path: str, *, tuning_cfg: Optional[dict] = None, notes: str = "backfilled from files",
                     db_path: Optional[str] = None) -> Dict[str, Any]:
    """Record an earlier run from its output folder (run.json, metrics_overall.csv, optional tuning_log.csv).

    The Diebold-Mariano rows are not in those files, so they are not recorded for a backfilled run.
    """
    meta = json.load(open(os.path.join(path, "run.json"), encoding="utf-8"))
    models = pd.read_csv(os.path.join(path, "metrics_overall.csv")).to_dict("records")
    tl = os.path.join(path, "tuning_log.csv")
    choices = pd.read_csv(tl).to_dict("records") if os.path.exists(tl) else []
    tt = os.path.join(path, "tuning_trials.csv")
    trials = pd.read_csv(tt).to_dict("records") if os.path.exists(tt) else []
    uses_tuning = bool(choices) or any(m["forecaster"].endswith("_tuned") or m["forecaster"].startswith("ensemble_") for m in models)
    return register("evaluation", meta["target"], models, stage=meta["stage"], window=(meta["test_start"], meta["test_end"]),
                    feature_set=str(meta.get("feature_set", "")),
                    schedule={k: meta.get(k) for k in ("retrain_every_days", "gap_days", "train_window_days")},
                    config_digest=meta["config_digest"], data_digest=meta["data_digest"],
                    tuning_digest=digest_of(tuning_cfg) if (tuning_cfg is not None and uses_tuning) else "",
                    code_revision=meta.get("code", ""), tuning_choices=choices, tuning_trials=trials, source="backfill", artifacts_dir=path,
                    notes=notes, db_path=db_path)


# ----------------------------------------------------------------------------------------------- queries
def list_runs(target: Optional[str] = None, kind: Optional[str] = None, db_path: Optional[str] = None) -> pd.DataFrame:
    """One row per experiment (latest attempt) with the number of attempts and whether they agree."""
    with session(db_path) as con:
        df = pd.read_sql_query("SELECT * FROM runs", con)
    if df.empty:
        return df
    n = df.groupby("run_key").agg(attempts=("attempt", "max"), distinct_results=("result_digest", "nunique"))
    latest = df.sort_values("attempt").groupby("run_key").tail(1).set_index("run_key")
    out = latest.join(n).reset_index()
    out["reproduced"] = out.apply(lambda r: None if r["attempts"] == 1 else bool(r["distinct_results"] == 1), axis=1)
    if target:
        out = out[out["target"] == target]
    if kind:
        out = out[out["kind"] == kind]
    cols = ["run_key", "kind", "target", "stage", "window_start", "window_end", "attempts", "reproduced", "code_revision",
            "created_utc", "source", "notes"]
    return out[cols].sort_values("created_utc").reset_index(drop=True)


def models_of(run_key: str, attempt: Optional[int] = None, db_path: Optional[str] = None) -> pd.DataFrame:
    """Scores of one run (latest attempt unless given)."""
    with session(db_path) as con:
        a = attempt or con.execute("SELECT MAX(attempt) FROM runs WHERE run_key=?", (run_key,)).fetchone()[0]
        if a is None:
            raise KeyError(f"unknown run {run_key!r}")
        return pd.read_sql_query("SELECT forecaster, role, tuned, n, sse, mae, rmse, coverage FROM run_models "
                                 "WHERE run_key=? AND attempt=? ORDER BY mae", con, params=(run_key, a))


def comparisons_of(run_key: str, attempt: Optional[int] = None, db_path: Optional[str] = None) -> pd.DataFrame:
    with session(db_path) as con:
        a = attempt or con.execute("SELECT MAX(attempt) FROM runs WHERE run_key=?", (run_key,)).fetchone()[0]
        return pd.read_sql_query("SELECT forecaster, reference, n_days, mean_diff, statistic, p_value FROM run_comparisons "
                                 "WHERE run_key=? AND attempt=?", con, params=(run_key, a))


def tuning_choices_of(run_key: str, attempt: Optional[int] = None, db_path: Optional[str] = None) -> pd.DataFrame:
    with session(db_path) as con:
        a = attempt or con.execute("SELECT MAX(attempt) FROM runs WHERE run_key=?", (run_key,)).fetchone()[0]
        return pd.read_sql_query("SELECT model, tuned_at, score, default_score, n_trials, params_json FROM tuning_choices "
                                 "WHERE run_key=? AND attempt=? ORDER BY model, tuned_at", con, params=(run_key, a))


def tuning_trials_of(run_key: str, attempt: Optional[int] = None, db_path: Optional[str] = None) -> pd.DataFrame:
    """Every trial of every tuning round of a run (settings, score, pruned, selected)."""
    with session(db_path) as con:
        a = attempt or con.execute("SELECT MAX(attempt) FROM runs WHERE run_key=?", (run_key,)).fetchone()[0]
        return pd.read_sql_query("SELECT model, tuned_at, trial, params_json, score, pruned, is_default, duplicate, selected, "
                                 "block_scores FROM tuning_trials WHERE run_key=? AND attempt=? ORDER BY model, tuned_at, trial",
                                 con, params=(run_key, a))


def record_selection(decision: dict, source: str = "", db_path: Optional[str] = None) -> None:
    """Store a champion decision (from model_development.selection.select) with the digest of the policy that made it."""
    with session(db_path) as con:
        con.execute("INSERT INTO selections VALUES (?,?,?,?,?,?,?)",
                    (decision["target"], dt.datetime.now(dt.timezone.utc).isoformat(), decision["policy_digest"], source,
                     decision["status"], json.dumps(decision["champion"], sort_keys=True),
                     json.dumps(decision, sort_keys=True, default=str)))


def latest_selection(target: str, db_path: Optional[str] = None) -> Optional[dict]:
    """The most recent recorded decision for a target, or None."""
    with session(db_path) as con:
        row = con.execute("SELECT decision_json FROM selections WHERE target=? ORDER BY created_utc DESC LIMIT 1", (target,)).fetchone()
    return json.loads(row[0]) if row else None


def reproducibility(db_path: Optional[str] = None) -> pd.DataFrame:
    """Experiments that were run more than once, and whether all attempts gave the same scores."""
    runs = list_runs(db_path=db_path)
    if runs.empty:
        return runs
    return runs[runs["attempts"] > 1][["run_key", "target", "kind", "attempts", "reproduced"]].reset_index(drop=True)
