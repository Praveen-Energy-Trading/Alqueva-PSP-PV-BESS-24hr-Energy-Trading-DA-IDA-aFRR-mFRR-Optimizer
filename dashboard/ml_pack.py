"""
ml_pack.py — the demo data pack of the ML Lifecycle pages (model_development/data/).

The ML pages read only committed files, so that the dashboard works on any checkout without runtime/ (the data and run folders are
not committed). This tool compacts what the pages need out of the run folders, the clean zone and the registry:

    python dashboard/ml_pack.py                 build every part that its sources allow; a missing source is skipped and said so

Parts (all small; compressed CSV where large):
    quality_daily.csv.gz        quality score of every dataset and day (data quality database)
    quality_gate.json           the latest gate decision and its reasons
    load_status_monthly.csv     per source and month: days loaded, published empty, not published, rejected (load log)
    raw_manifest_summary.csv    per source: files stored, versions, size, first and last day (raw zone manifest)
    prices_daily.csv.gz         daily mean of every price series (clean zone, local days)
    price_profile.csv           mean price by series, month and hour
    trials.csv.gz               every tuning trial of every target (Optuna and the random-search control)
    tuning_log.csv              the setting chosen in every tuning round
    daily_loss.csv.gz           development window: mean absolute and signed error per day, per tuned algorithm, the champion
                                (an average of algorithms where the decision says so) and the reference
    locked_forecasts.csv.gz     the locked year: outcome, champion forecast and reference forecast per hour and target
    registry_snapshot.json      experiments, reproducibility, selections, champions and their event log
    feature_availability.csv    for every feature and target: available at the target's as-of time, or why not
    facts.json                  counts of tests, modules, tools and configuration files (found by scanning the repository)
    pack_manifest.json          what was built, row counts

The output depends only on its sources, so rerunning gives the same files.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import shutil
import sys
from typing import Dict, List, Optional

import pandas as pd

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, _REPO)

PACK = os.path.join(_REPO, "model_development", "data")
TARGETS = ("da_price", "afrr_cap_price", "afrr_dn_price", "ida1_price", "ida2_price", "ida3_price", "xbid_price", "mfrr_price")


# ------------------------------------------------------------------------------------------------------ pure transforms
def load_status_monthly(records: List[dict]) -> pd.DataFrame:
    """Per source and month, the count of days by their latest real status (a later 'skipped' never overrides a load)."""
    latest: Dict[tuple, str] = {}
    for r in records:
        if r["status"] == "skipped":
            latest.setdefault((r["source"], r["delivery_date"]), "loaded")
            continue
        latest[(r["source"], r["delivery_date"])] = r["status"]
    rows = collections.Counter((s, d[:7], st) for (s, d), st in latest.items())
    df = pd.DataFrame([{"source": s, "month": m, "status": st, "days": n} for (s, m, st), n in rows.items()])
    return df.sort_values(["source", "month", "status"]).reset_index(drop=True) if len(df) else df


def manifest_summary(records: List[dict]) -> pd.DataFrame:
    stored = [r for r in records if r.get("event") == "stored"]
    g = collections.defaultdict(list)
    for r in stored:
        g[r["source"]].append(r)
    rows = []
    for s, rs in sorted(g.items()):
        keys = sorted({str(r["key"]) for r in rs})
        rows.append({"source": s, "files": len(rs), "days": len(keys), "versions_above_1": sum(1 for r in rs if "__v001__" not in r["path"]),
                     "bytes": sum(int(r["size"]) for r in rs), "first_key": keys[0], "last_key": keys[-1]})
    return pd.DataFrame(rows)


def daily_loss(pred: pd.DataFrame, target: str, model: str) -> pd.DataFrame:
    """Mean absolute error, mean signed error and mean squared error per delivery day of a prediction table."""
    p = pred.dropna(subset=["y_true", "y_pred"]).copy()
    p["err"] = p["y_pred"] - p["y_true"]
    g = p.groupby("local_date")["err"]
    out = pd.DataFrame({"mae": g.apply(lambda e: float(e.abs().mean())), "bias": g.mean(), "mse": g.apply(lambda e: float((e ** 2).mean())),
                        "hours": g.size()}).reset_index()
    out.insert(0, "model", model)
    out.insert(0, "target", target)
    out["date"] = [dt.date.fromordinal(int(d)).isoformat() for d in out["local_date"]]
    return out.drop(columns=["local_date"])


def read_run_predictions(run: str, suffix: str) -> Dict[str, pd.DataFrame]:
    from common_layer.model_development import evidence as ev
    return ev.read_predictions(run, suffix)


# ---------------------------------------------------------------------------------------------------------- parts
def _jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def part_quality(out: str, log: List[str]) -> Dict[str, int]:
    import sqlite3
    base = os.path.join(_REPO, "runtime", "data")
    res: Dict[str, int] = {}
    db = os.path.join(base, "quality", "dq.db")
    if os.path.exists(db):
        with sqlite3.connect(db) as con:
            df = pd.read_sql_query("select * from dq_daily order by dataset, session, delivery_date", con)
        df.to_csv(os.path.join(out, "quality_daily.csv.gz"), index=False, float_format="%.4f")
        res["quality_daily"] = len(df)
    else:
        log.append("quality database not found: quality_daily skipped")
    gate = os.path.join(base, "quality", "gate.json")
    if os.path.exists(gate):
        d = json.load(open(gate, encoding="utf-8"))
        d.pop("generated_utc", None)                                  # keep the pack reproducible
        json.dump(d, open(os.path.join(out, "quality_gate.json"), "w", encoding="utf-8", newline="\n"), indent=2, sort_keys=True)
    for name, target in (("curation_summary.csv", "curation_summary.csv"), ("curation_gaps.csv", "curation_gaps.csv.gz")):
        src = os.path.join(base, "quality", name)
        if os.path.exists(src):
            df = pd.read_csv(src)
            df.to_csv(os.path.join(out, target), index=False, float_format="%.4f")
            res[target.split(".")[0]] = len(df)
        else:
            log.append(f"{name} not found (python tools/run_data_quality.py): skipped")
    ll = os.path.join(base, "load_log.jsonl")
    if os.path.exists(ll):
        df = load_status_monthly(_jsonl(ll))
        df.to_csv(os.path.join(out, "load_status_monthly.csv"), index=False)
        res["load_status_monthly"] = len(df)
    mf = os.path.join(base, "raw", "manifest.jsonl")
    if os.path.exists(mf):
        df = manifest_summary(_jsonl(mf))
        df.to_csv(os.path.join(out, "raw_manifest_summary.csv"), index=False)
        res["raw_manifest_summary"] = len(df)
    return res


def part_prices(out: str, log: List[str]) -> Dict[str, int]:
    try:
        from common_layer.data_layer.clean_store import CleanStore
        from common_layer.market_analysis import panel as pnl
        panel = pnl.build_panel(CleanStore())
    except Exception as e:                                           # pragma: no cover
        log.append(f"clean zone not available ({e}): prices skipped")
        return {}
    local = panel.index.tz_convert("Europe/Madrid")
    daily = panel.groupby(local.date).mean()
    daily.index = [d.isoformat() for d in daily.index]
    daily.index.name = "date"
    daily.round(3).to_csv(os.path.join(out, "prices_daily.csv.gz"))
    prof = []
    for s in [c for c in ("da_pt", "ida1_pt", "ida2_pt", "ida3_pt", "xbid_pt", "afrr_up", "mfrr_ap") if c in panel]:
        g = panel[s].groupby([local.month, local.hour]).mean().reset_index()
        g.columns = ["month", "hour", "mean"]
        g.insert(0, "series", s)
        prof.append(g)
    pd.concat(prof).round(3).to_csv(os.path.join(out, "price_profile.csv"), index=False)
    return {"prices_daily": len(daily), "price_profile": sum(len(p) for p in prof)}


def tuned_runs() -> List[tuple]:
    """(target, search, run folder) of every tuned run available."""
    ph9 = os.path.join(_REPO, "runtime", "analysis", "phase9")
    runs = [("da_price", "tpe", os.path.join(ph9, "da_stage2_tpe_v2")), ("da_price", "random", os.path.join(ph9, "da_stage2_random_v2"))]
    for t in TARGETS[1:]:
        runs.append((t, "tpe", os.path.join(_REPO, "runtime", "analysis", "rollout", t, "tuned")))
    return [r for r in runs if os.path.exists(os.path.join(r[2], "tuning_trials.csv"))]


def part_tuning(out: str, log: List[str]) -> Dict[str, int]:
    trials, logs = [], []
    for t, s, d in tuned_runs():
        a = pd.read_csv(os.path.join(d, "tuning_trials.csv"))
        a.insert(0, "search", s)
        a.insert(0, "target", t)
        trials.append(a)
        b = pd.read_csv(os.path.join(d, "tuning_log.csv"))
        b.insert(0, "search", s)
        b.insert(0, "target", t)
        logs.append(b)
    if not trials:
        log.append("no tuned run folders: trials and tuning_log skipped")
        return {}
    t = pd.concat(trials, ignore_index=True)
    t.to_csv(os.path.join(out, "trials.csv.gz"), index=False, float_format="%.5f")
    lg = pd.concat(logs, ignore_index=True)
    lg.to_csv(os.path.join(out, "tuning_log.csv"), index=False, float_format="%.5f")
    return {"trials": len(t), "tuning_log": len(lg)}


def _decision(t: str) -> Optional[dict]:
    p = os.path.join(_REPO, "model_development", "selection", f"{t}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def part_losses(out: str, log: List[str]) -> Dict[str, int]:
    from common_layer.model_development import monitoring
    frames = []
    for t in TARGETS:
        d = _decision(t)
        if d is None or d.get("champion") is None:
            continue
        run = os.path.join(_REPO, d["source"]["run"])
        preds = read_run_predictions(run, d["source"]["suffix"])
        for m, p in preds.items():
            frames.append(daily_loss(p, t, m))
        spec = d["champion"]
        spec = {"type": spec["type"], "members": spec.get("members") or [spec["model"]], "method": spec.get("method")}
        frames.append(daily_loss(monitoring.dev_predictions(spec, preds), t, "champion"))
        if "decision_tree" in preds:
            frames.append(daily_loss(preds["decision_tree"], t, "reference"))
    if not frames:
        log.append("no decisions or run folders: daily_loss skipped")
        return {}
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(out, "daily_loss.csv.gz"), index=False, float_format="%.4f")
    return {"daily_loss": len(df)}


def part_locked(out: str, log: List[str]) -> Dict[str, int]:
    from common_layer.model_development import locked_test as lt
    frames = []
    for t in TARGETS:
        d = _decision(t)
        base = os.path.join(_REPO, "runtime", "evaluation", t, "final")
        if d is None or not os.path.exists(base):
            continue
        champ = lt.champion_name(d)
        cf = os.path.join(base, "champion", f"predictions_{champ}.csv")
        ref_name = d["policy"]["reference"] + ("_tuned" if d["source"]["tuned"] else "")
        rf = [os.path.join(base, "reference", f"predictions_{ref_name}.csv"), os.path.join(base, "champion", f"predictions_{ref_name}.csv")]
        rf = next((p for p in rf if os.path.exists(p)), None)
        if not os.path.exists(cf) or rf is None:
            log.append(f"{t}: locked-test predictions not found; skipped")
            continue
        c = pd.read_csv(cf, index_col=0, parse_dates=True)
        r = pd.read_csv(rf, index_col=0, parse_dates=True)
        df = pd.DataFrame({"target": t, "ts_utc": c.index.strftime("%Y-%m-%dT%H:%M:%SZ"), "y_true": c["y_true"].to_numpy(),
                           "champion": c["y_pred"].to_numpy(), "reference": r["y_pred"].reindex(c.index).to_numpy()})
        frames.append(df)
    if not frames:
        log.append("no locked-test predictions yet: locked_forecasts skipped")
        return {}
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(os.path.join(out, "locked_forecasts.csv.gz"), index=False, float_format="%.3f")
    return {"locked_forecasts": len(df)}


def part_registry(out: str, log: List[str]) -> Dict[str, int]:
    db = os.path.join(_REPO, "runtime", "model_registry", "registry.db")
    if not os.path.exists(db):
        log.append("registry database not found: registry_snapshot skipped")
        return {}
    from common_layer.model_development import champions, registry
    runs = registry.list_runs(db_path=db)
    snap = {"runs": json.loads(runs.drop(columns=["created_utc", "code_revision"], errors="ignore").to_json(orient="records")),
            "reproducibility": json.loads(registry.reproducibility(db_path=db).to_json(orient="records")), "champions": {}, "events": []}
    for t in TARGETS:
        v = champions.versions(t, db_path=db)
        if len(v):
            snap["champions"][t] = json.loads(v.drop(columns=["spec_json"]).to_json(orient="records"))
    ev = champions.events(db_path=db)
    if len(ev):
        snap["events"] = json.loads(ev.drop(columns=["at_utc", "detail_json"]).to_json(orient="records"))
    json.dump(snap, open(os.path.join(out, "registry_snapshot.json"), "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    return {"registry_runs": len(snap["runs"])}


def part_features(out: str, log: List[str]) -> Dict[str, int]:
    try:
        from common_layer.configuration.forecast_catalogue import load_forecast_catalogue
        from common_layer.configuration.forecast_formulation import load_forecast_formulation
        from common_layer.feature_engineering.catalogue import feature_availability, load_feature_catalogue
        df = feature_availability(load_feature_catalogue(), load_forecast_formulation(), load_forecast_catalogue())
    except Exception as e:                                           # pragma: no cover
        log.append(f"feature availability not computed ({e})")
        return {}
    df.to_csv(os.path.join(out, "feature_availability.csv"), index=False, float_format="%.1f")
    return {"feature_availability": len(df)}


def count_test_functions(tests_dir: str) -> int:
    """Test functions and methods (def test_...) found by parsing, before parametrisation."""
    import ast
    n = 0
    for f in sorted(os.listdir(tests_dir)):
        if f.startswith("test_") and f.endswith(".py"):
            tree = ast.parse(open(os.path.join(tests_dir, f), encoding="utf-8").read())
            n += sum(1 for x in ast.walk(tree) if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name.startswith("test_"))
    return n


def part_facts(out: str, log: List[str]) -> Dict[str, int]:
    t = os.path.join(_REPO, "tests")
    facts = {"test_files": sum(1 for f in os.listdir(t) if f.startswith("test_") and f.endswith(".py")),
             "test_functions": count_test_functions(t),
             "framework_modules": len([f for f in os.listdir(os.path.join(_REPO, "common_layer", "model_development")) if f.endswith(".py") and f != "__init__.py"]),
             "tools": len([f for f in os.listdir(os.path.join(_REPO, "tools")) if f.endswith(".py")]),
             "config_files": len([f for f in os.listdir(os.path.join(_REPO, "config")) if f.endswith((".yaml", ".json"))])}
    json.dump(facts, open(os.path.join(out, "facts.json"), "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    return {"facts": len(facts)}


def build(out: str = PACK) -> dict:
    os.makedirs(out, exist_ok=True)
    log: List[str] = []
    counts: Dict[str, int] = {}
    for part in (part_quality, part_prices, part_tuning, part_losses, part_locked, part_registry, part_features, part_facts):
        counts.update(part(out, log))
    manifest = {"parts": counts, "skipped": log}
    json.dump(manifest, open(os.path.join(out, "pack_manifest.json"), "w", encoding="utf-8", newline="\n"), indent=1, sort_keys=True)
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=PACK)
    a = ap.parse_args()
    m = build(a.out)
    print(json.dumps(m, indent=1))
    tot = sum(os.path.getsize(os.path.join(a.out, f)) for f in os.listdir(a.out))
    print(f"pack size: {tot / 1048576:.2f} MB in {a.out}")
