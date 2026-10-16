"""
ml_data.py — what the ML Lifecycle pages read. Pure functions (no Streamlit): they read committed files only, so every page works on a
plain checkout. The demo data pack (model_development/data/, built by dashboard/ml_pack.py) holds what the run folders and the clean
zone would otherwise provide. A missing file gives None or an empty frame, never an exception: a stage that has not run is shown as
"not yet run".
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TARGETS = ("da_price", "afrr_cap_price", "afrr_dn_price", "ida1_price", "ida2_price", "ida3_price", "xbid_price", "mfrr_price")
LABELS = {"pv_weather": "Solar irradiance and temperature (PV)", "reservoir_inflow": "Natural inflow", "da_price": "Day-ahead price", "afrr_cap_price": "aFRR capacity price (up)", "afrr_dn_price": "aFRR capacity price (down)", "ida1_price": "IDA1 price", "ida2_price": "IDA2 price",
          "ida3_price": "IDA3 price", "xbid_price": "XBID price", "mfrr_price": "mFRR price"}
UNITS = {t: "EUR/MWh" for t in TARGETS} | {"afrr_cap_price": "EUR/MW/h", "afrr_dn_price": "EUR/MW/h"}
MODEL_LABELS = {"decision_tree": "Decision tree", "random_forest": "Random forest", "gradient_boosting": "Gradient boosting",
                "xgboost": "XGBoost", "lightgbm": "LightGBM", "champion": "Champion", "reference": "Reference (decision tree)"}
STAGES = ["Experiment registry", "Tuning", "Tuning evidence", "Algorithm comparison", "Selection policy", "Locked test",
          "Champion registry", "Monitoring", "Roll-out", "Live switch"]


def md(root: Path = REPO_ROOT) -> Path:
    return root / "model_development"


def pack_dir(root: Path = REPO_ROOT) -> Path:
    return md(root) / "data"


def label(target: str) -> str:
    return LABELS.get(target, target)


SHORT = {"da_price": "Day-ahead", "afrr_cap_price": "aFRR up", "afrr_dn_price": "aFRR down", "ida1_price": "IDA1", "ida2_price": "IDA2",
         "ida3_price": "IDA3", "xbid_price": "XBID", "mfrr_price": "mFRR"}


def short(target: str) -> str:
    """A short name of a target for chart axes."""
    return SHORT.get(target, label(target))


def model_label(m: str) -> str:
    return MODEL_LABELS.get(m.replace("_tuned", ""), m.replace("_", " "))


# Readable names for what the pages show: sources, datasets, input series, run kinds. Anything not listed gets pretty().
SOURCE_LABELS = {"omie_da": "OMIE day-ahead market", "omie_ida1": "OMIE intraday auction 1 (IDA1)", "omie_ida2": "OMIE intraday auction 2 (IDA2)",
                 "omie_ida3": "OMIE intraday auction 3 (IDA3)", "omie_xbid": "OMIE continuous intraday (XBID)",
                 "ren_afrr": "REN aFRR reserve prices", "ren_imbalance": "REN imbalance prices", "ren_mfrr": "REN mFRR reserve prices"}
DATASET_LABELS = {"da_price": "Day-ahead prices", "ida_price": "Intraday auction prices", "xbid_price": "XBID prices", "afrr_price": "aFRR prices",
                  "mfrr_price": "mFRR prices", "imbalance_price": "Imbalance prices"}
SERIES_LABELS = {"da_result": "Day-ahead price", "da_pt": "Day-ahead price (Portugal)",
                 "ida1_result": "IDA1 price", "ida2_result": "IDA2 price", "ida3_result": "IDA3 price", "ida1_pt": "IDA1 price", "ida2_pt": "IDA2 price",
                 "ida3_pt": "IDA3 price", "xbid_summary": "XBID price", "xbid_pt": "XBID price", "afrr_band_result": "aFRR band price",
                 "afrr_up": "aFRR up price", "afrr_dn": "aFRR down price", "imbalance_price": "Imbalance price", "mfrr_series": "mFRR price", "imb_short": "Imbalance price (short)", "imb_long": "Imbalance price (long)",
                 "da_vol_pt_buy": "Day-ahead purchases (Portugal)", "da_vol_pt_sell": "Day-ahead sales (Portugal)",
                 "mfrr_ap": "mFRR activation price", "weather_forecast": "Weather forecast", "fundamentals_forecast": "Fundamentals forecast",
                 "calendar": "Calendar", "xbid_live": "XBID live trades"}
RUN_LABELS = {"evaluation": "Model evaluation", "schedule_comparison": "Schedule comparison", "dev": "Development", "final": "Locked test",
              "backfill": "Backfilled from files", "evaluate_forecaster": "Evaluation run", "run_locked_test": "Locked-test run"}


def pretty(name) -> str:
    """A code-style name as plain words: underscores become spaces and the first letter is a capital."""
    if str(name) == "market_result":                      # the day-ahead purchases and sales published with the price
        return "Day-ahead volumes"
    s = str(name).replace("_", " ").strip()
    return s[:1].upper() + s[1:]


def source_label(s: str) -> str:
    return SOURCE_LABELS.get(s, pretty(s))


def dataset_label(s: str) -> str:
    return DATASET_LABELS.get(s, pretty(s))


def series_label(s: str) -> str:
    return SERIES_LABELS.get(s, pretty(s))


def run_label(s: str) -> str:
    return RUN_LABELS.get(s, pretty(s))


PHRASES = {"da_price": "day-ahead prices", "ida_price": "intraday auction prices", "xbid_price": "XBID prices", "afrr_price": "aFRR prices",
           "mfrr_price": "mFRR prices", "imbalance_price": "imbalance prices",            "da_vol_pt_buy": "the Portuguese day-ahead purchases", "da_vol_pt_sell": "the Portuguese day-ahead sales",
           "da_pt": "the Portuguese day-ahead price", "ida1_pt": "the IDA1 price", "ida2_pt": "the IDA2 price", "ida3_pt": "the IDA3 price",
           "xbid_pt": "the XBID price", "afrr_up": "the aFRR up price", "afrr_dn": "the aFRR down price", "mfrr_ap": "the mFRR activation price",
           "entsoe_transparency": "ENTSO-E transparency data", "fuel_carbon": "fuel and carbon prices", "reservoir_levels": "reservoir levels",
           "weather_forecast": "weather forecasts", "fundamentals_forecast": "fundamentals forecasts"}


def plain_text(text) -> str:
    """A sentence from the data pack with column and source identifiers replaced by plain words."""
    import re
    s = re.sub(r",? ?read from config/\S+", "", str(text))
    for k in sorted(PHRASES, key=len, reverse=True):
        s = re.sub(r"\b" + re.escape(k) + r"\b", PHRASES[k], s)
    return re.sub(r"\b[a-z]+(?:_[a-z0-9]+)+\b", lambda m: m.group(0).replace("_", " "), s)


def plain_params(p) -> str:
    """A setting dictionary (or its text) as plain words: 'max depth 6, min samples leaf 50'."""
    import ast
    try:
        d = ast.literal_eval(p) if isinstance(p, str) else dict(p)
    except (ValueError, SyntaxError):
        return str(p)
    return ", ".join(f"{str(k).replace('_', ' ')} {v}" for k, v in d.items())


def repair_cards(repairs) -> list:
    """The repaired-days lines of the quality gate ('aFRR prices: 1/2842 days missing (0.0%), filled') as (label, value) pairs for cards:
    the label is the dataset (and session), the value the missing days of all days with their share. A line in another form is kept as it is."""
    import re
    out = []
    for r in repairs:
        text = plain_text(r)
        m = re.match(r"^(.*?): (\d+)/(\d+) days missing \(([\d.]+)%\)", text)
        if not m:
            out.append((text, ""))
            continue
        name, miss, total, share = m.groups()
        name = (name if name.startswith(("aFRR", "mFRR")) else name[:1].upper() + name[1:]).replace(" session ", " · session ")
        out.append((name, f'{int(miss):,} <span style="font-size:13px; font-weight:400; color:#52514e;">of {int(total):,} days · {share}%</span>'))
    return out


def humanize(text: str) -> str:
    """Replaces algorithm ids inside a sentence (for example in a decision reason) by their names."""
    import re
    out = re.sub(r"\b(decision_tree|random_forest|gradient_boosting|xgboost|lightgbm)\b", lambda m: MODEL_LABELS[m.group(1)] if m.group(1) in ("xgboost", "lightgbm") else MODEL_LABELS[m.group(1)].lower(), str(text))
    return out[:1].upper() + out[1:]


# The five algorithms in the order of the project: from the simplest to the most complex. Lists, menus and colours follow it.
MODEL_ORDER = ("decision_tree", "random_forest", "gradient_boosting", "xgboost", "lightgbm")


def model_index(m: str) -> int:
    """Position of an algorithm in MODEL_ORDER (anything else, such as the champion, comes after the five)."""
    base = str(m).replace("_tuned", "")
    return MODEL_ORDER.index(base) if base in MODEL_ORDER else len(MODEL_ORDER)


def ordered_models(models) -> list:
    """The algorithms in the order of the project: decision tree, random forest, gradient boosting, XGBoost, LightGBM."""
    return sorted(set(models), key=lambda m: (model_index(m), str(m)))


# --------------------------------------------------------------------------------------------------------------- files
def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def read_yaml(path: Path) -> dict:
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return {}


def config(name: str, root: Path = REPO_ROOT) -> dict:
    """A configuration file of config/ by name without extension."""
    return read_yaml(root / "config" / f"{name}.yaml")


def pack(name: str, root: Path = REPO_ROOT) -> pd.DataFrame:
    """A table of the demo data pack ('trials', 'daily_loss', ...); empty if it was not built."""
    d = pack_dir(root)
    for ext in (".csv.gz", ".csv"):
        if (d / (name + ext)).exists():
            return read_csv(d / (name + ext))
    return pd.DataFrame()


def pack_json(name: str, root: Path = REPO_ROOT) -> Optional[dict]:
    return read_json(pack_dir(root) / f"{name}.json")


def exploratory(target: str, root: Path = REPO_ROOT) -> Optional[dict]:
    """The exploratory result of a target whose history is too short for a decision (tools/explore_mfrr.py), or None."""
    return read_json(md(root) / "exploratory" / f"{target}.json")


PLANT_INPUTS = [("ghi", "Solar irradiance", "pv_weather"), ("temperature", "Air temperature", "pv_weather"), ("inflow", "Natural inflow", "reservoir_inflow")]


def plant_input(key: str, root: Path = REPO_ROOT) -> Optional[dict]:
    """The held-out evaluation of a plant-input forecast (tools/evaluate_plant_inputs.py), or None."""
    return read_json(md(root) / "plant_inputs" / f"{key}.json")


def weather_eval(key: str, root: Path = REPO_ROOT) -> Optional[dict]:
    """The as-of-decision-time evaluation of the weather forecast against the algorithms (tools/evaluate_weather_forecast.py), or None."""
    return read_json(md(root) / "plant_inputs" / f"{key}_weather.json")


# -------------------------------------------------------------------------------------------------- decisions and results
def decisions(root: Path = REPO_ROOT) -> Dict[str, dict]:
    out = {}
    for t in TARGETS:
        d = read_json(md(root) / "selection" / f"{t}.json")
        if d:
            out[t] = d
    return out


def locked_results(root: Path = REPO_ROOT) -> Dict[str, dict]:
    out = {}
    for t in TARGETS:
        d = read_json(md(root) / "locked_test" / f"{t}.json")
        if d:
            out[t] = d
    return out


def benchmarks(root: Path = REPO_ROOT) -> Dict[str, dict]:
    """The naive-benchmark files (tools/benchmark_naive.py): champion, decision tree and naive forecasts on the same hours."""
    out = {}
    for t in TARGETS:
        d = read_json(md(root) / "benchmark" / f"{t}.json")
        if d:
            out[t] = d
    return out


def strongest_naive_gain(bm: Optional[dict], stage: str) -> Optional[float]:
    """The champion's gain over the STRONGEST naive forecast of a benchmark file (smallest gain, so the naive forecasts cannot be
    chosen to flatter the champion): the seasonal, yesterday and last-week naive forecasts, and for a spread target also the
    day-ahead price with no spread."""
    gains = ((bm or {}).get(stage) or {}).get("gain_of_champion_over", {})
    g = [v for k, v in gains.items() if k != "reference" and v is not None]
    return min(g) if g else None


def champion_text(c: Optional[dict]) -> str:
    if not c:
        return "none"
    return f"{c['method']} of {', '.join(model_label(m) for m in c['members'])}" if c["type"] == "ensemble" else model_label(c["model"])


def champion_short(c: Optional[dict]) -> str:
    """The champion for a table cell: the algorithm, or 'Average of n algorithms'."""
    if not c:
        return "–"
    return f"Average of {len(c['members'])} algorithms" if c["type"] == "ensemble" else model_label(c["model"])


def locked_real_only(target: str, root: Path = REPO_ROOT) -> Optional[dict]:
    """The locked year of a target that has artificial outcome days (IDA1), on the REAL days only: champion and reference MAE, the gain, and the
    Diebold-Mariano test of the champion against the reference (the framework's own function, one loss difference per real day). None for a target without them."""
    lk = locked_results(root).get(target)
    if not lk or not lk.get("synthetic_fill"):
        return None
    ro = lk.get("real_only") or {}
    out = {"mae": ro["champion"]["mae"], "reference_mae": (ro.get("reference") or {}).get("mae"), "days_left_out": len(lk["synthetic_fill"]["days"]), "p_value": None}
    out["gain"] = 1 - out["mae"] / out["reference_mae"] if out["reference_mae"] else None
    lf = pack("locked_forecasts", root)
    if not lf.empty and target in set(lf["target"]):
        from common_layer.forecast_evaluation import metrics
        from common_layer.forecast_evaluation.dm import diebold_mariano
        x = lf[lf["target"] == target].copy()
        x["local_date"] = [d.toordinal() for d in pd.to_datetime(x["ts_utc"], utc=True).dt.tz_convert("Europe/Madrid").dt.date]
        x = x[~x["local_date"].isin({pd.Timestamp(d).toordinal() for d in lk["synthetic_fill"]["days"]})]
        a = metrics.add_error_columns(x.rename(columns={"champion": "y_pred"})[["y_true", "y_pred", "local_date"]].dropna())
        b = metrics.add_error_columns(x.rename(columns={"reference": "y_pred"})[["y_true", "y_pred", "local_date"]].dropna())
        out["p_value"] = float(diebold_mariano(a, b, lk.get("loss", "absolute"), 7, "hln")["p_value"])
    return out


def process_table(root: Path = REPO_ROOT) -> pd.DataFrame:
    """One row per price target: how much evidence its data gives and which steps of the process it has been through (from the decision and locked-test files)."""
    r = config("rollout", root).get("targets", {})
    dec, lk = decisions(root), locked_results(root)
    names = {"standard": "Full", "limited_evidence": "Limited", "very_limited_evidence": "Very limited", "deferred": "Deferred"}
    rows = []
    for t in TARGETS:
        d = dec.get(t)
        done = bool(d)                                              # a decision exists only after the tuned evaluation and the algorithm comparison
        rows.append({"Target": label(t), "Evidence": names.get(r.get(t, {}).get("treatment"), "–"),
                     "Training": "✓" if done else "–", "Tuning": "✓" if done else "–",
                     "Validation": f"✓ {int(d['n_days']):,} days" if done and d.get("n_days") else "–",
                     "Champion": "✓" if d and d.get("champion") else "–", "Locked test": "✓" if t in lk else "–"})
    return pd.DataFrame(rows)


def results_table(root: Path = REPO_ROOT) -> pd.DataFrame:
    """One row per target: treatment, development days, champion, development MAE, locked MAE (from the decision and locked files)."""
    rollout = config("rollout", root)
    d, lk = decisions(root), locked_results(root)
    rows = []
    for t in TARGETS:
        spec = rollout.get("targets", {}).get(t, {})
        dec, loc = d.get(t), lk.get(t)
        real = locked_real_only(t, root) if loc else None            # a target with artificial outcome days is shown on its real days only
        ref = None
        if dec and dec.get("ranking"):
            ref = next((r["mean_loss"] for r in dec["ranking"] if r["model"] == dec["policy"]["reference"]), None)
        rows.append({"target": t, "label": label(t), "treatment": spec.get("treatment", "standard"),
                     "dev_days": dec.get("n_days") if dec else None, "champion": champion_text(dec.get("champion")) if dec and dec.get("champion") else None,
                     "champion_short": champion_short(dec.get("champion")) if dec and dec.get("champion") else "–",
                     "dev_mae": dec.get("champion_dev_loss") if dec else None, "reference_dev_mae": ref,
                     "locked_mae": (real["mae"] if real else loc["locked_scores"]["mae"]) if loc else None,
                     "locked_reference_mae": (real["reference_mae"] if real else (loc.get("reference_scores") or {}).get("mae")) if loc else None,
                     "locked_p_vs_reference": (real["p_value"] if real else (loc.get("dm_vs_reference") or {}).get("p_value")) if loc else None,
                     "locked_vs_dev": loc["loss_ratio_locked_over_dev"] if loc else None, "unit": UNITS[t],
                     "reason": spec.get("reason") if spec.get("treatment") == "deferred" else None})
    df = pd.DataFrame(rows)
    bm = benchmarks(root)
    for col, stage in (("gain_vs_naive", "development"), ("locked_gain_vs_naive", "locked")):
        df[col] = [strongest_naive_gain(bm.get(t), stage) for t in df["target"]]
    df["gain_vs_reference"] = 1 - df["dev_mae"] / df["reference_dev_mae"]
    df["locked_gain_vs_reference"] = 1 - df["locked_mae"] / df["locked_reference_mae"]      # level-independent: same year, same hours
    return df


# ------------------------------------------------------------------------------------------------- comparison and evidence
def comparison_tables(target: str, root: Path = REPO_ROOT) -> Dict[str, pd.DataFrame]:
    """The tables of the algorithm comparison of a target: scoreboard, pairwise, versus_best, by_hour, by_month, by_regime."""
    if target == "da_price":
        base, prefix = md(root) / "tables", "algorithm_"
    else:
        base, prefix = md(root) / "rollout" / target / "tables", f"{target}_"
    return {k: read_csv(base / f"{prefix}{k}.csv") for k in ("scoreboard", "pairwise_dm", "versus_best", "by_hour", "by_month", "by_regime")}


def evidence_table(name: str, root: Path = REPO_ROOT) -> pd.DataFrame:
    """A table of the day-ahead tuning evidence (convergence, importance_random_search, stability, inner_vs_outer_tpe, pruning_tpe, effects_<model>)."""
    return read_csv(md(root) / "tables" / f"{name}.csv")


def figure_path(name: str, root: Path = REPO_ROOT) -> Optional[Path]:
    p = md(root) / "figures" / name
    return p if p.exists() else None


# -------------------------------------------------------------------------------------------------------------- stages
def stage_status(root: Path = REPO_ROOT) -> List[dict]:
    """The ten stages with a state (done, partial, not run) and a one-line fact, from what exists."""
    d, lk = decisions(root), locked_results(root)
    snap = pack_json("registry_snapshot", root) or {}
    trials = pack("trials", root)
    live = config("live_models", root).get("targets", {})
    rollout = config("rollout", root).get("targets", {})
    runs = snap.get("runs", [])
    champs = snap.get("champions", {})
    n_dec = len(d)
    comparisons = sum(1 for t in TARGETS if not comparison_tables(t, root)["scoreboard"].empty)
    active = sum(1 for t, vs in champs.items() if any(v.get("status") == "active" for v in vs))
    switched = [t for t, v in live.items() if v.get("mode") != "legacy"]
    ready = [t for t, v in rollout.items() if v.get("treatment") != "deferred"]

    def st(n_done: int, n_total: int) -> str:
        return "done" if n_done >= n_total else ("partial" if n_done else "not run")
    return [
        {"stage": STAGES[0], "state": st(len(runs), 1), "fact": f"{len(runs)} experiments recorded"},
        {"stage": STAGES[1], "state": st(len(trials), 1), "fact": f"{len(trials):,} tuning trials logged"},
        {"stage": STAGES[2], "state": "done" if (md(root) / "tuning_evidence.md").exists() else "not run", "fact": "day-ahead evidence tables and figures"},
        {"stage": STAGES[3], "state": st(comparisons, len(ready)), "fact": f"{comparisons} of {len(ready)} targets compared"},
        {"stage": STAGES[4], "state": st(n_dec, len(ready)), "fact": f"{n_dec} of {len(ready)} champion decisions"},
        {"stage": STAGES[5], "state": st(len(lk), len(ready)), "fact": f"{len(lk)} of {len(ready)} targets scored on the locked year"},
        {"stage": STAGES[6], "state": st(active, len(ready)), "fact": f"{active} champions registered"},
        {"stage": STAGES[7], "state": "partial", "fact": "built; needs registered champions to monitor"},
        {"stage": STAGES[8], "state": st(len(ready), len(rollout) or 1), "fact": f"{len(ready)} of {len(rollout)} targets rolled out" + ("" if len(ready) == len(rollout) else "; the rest deferred")},
        {"stage": STAGES[9], "state": "done" if switched else "partial", "fact": f"{len(switched)} of {len(live)} live forecasts switched to the champion"},
    ]


# ------------------------------------------------------------------------------------------------- evaluation design
def folds_for(first_data_day, test_start, test_end, schedule: dict) -> pd.DataFrame:
    """The rolling-origin folds of a schedule as a table (fold, train_start, train_end, test_start, test_end), computed by the framework's own code."""
    from common_layer.forecast_evaluation.folds import make_folds
    folds = make_folds(test_start, test_end, first_data_day, schedule)
    return pd.DataFrame([{"fold": f.fold_id, "train_start": f.train_start, "train_end": f.train_end, "test_start": f.test_days[0],
                          "test_end": f.test_days[-1], "train_days": (f.train_end - f.train_start).days + 1, "test_days": len(f.test_days)} for f in folds])


def dm_example(daily: pd.DataFrame, target: str, a: str = "champion", b: str = "reference", loss: str = "mae") -> Optional[dict]:
    """The Diebold-Mariano test of model `a` against `b` on the daily losses of the development window (the framework's own test function).

    `daily` is the daily_loss table of the data pack. Returns the statistic, p-value, mean difference and the daily differences, or None."""
    from common_layer.forecast_evaluation.dm import diebold_mariano
    d = daily[daily["target"] == target]
    x, y = d[d["model"] == a].set_index("date"), d[d["model"] == b].set_index("date")
    common = x.index.intersection(y.index)
    if len(common) < 20:
        return None
    ordinal = {day: pd.Timestamp(day).toordinal() for day in common}

    def table(frame):
        return pd.DataFrame({"local_date": [ordinal[day] for day in common], "err": frame.loc[common, loss].to_numpy()})
    cfg = config("evaluation").get("comparison", {})
    r = diebold_mariano(table(x), table(y), "absolute", int(cfg.get("hac_lags", 7)), cfg.get("small_sample", "hln"))
    diff = (x.loc[common, loss] - y.loc[common, loss]).rename("difference")
    return {"statistic": r["statistic"], "p_value": r["p_value"], "mean_diff": r["mean_diff"], "n_days": int(r["n_days"]), "differences": diff.reset_index()}


# ------------------------------------------------------------------------------------------------------------ tuning
def prepared_trials(trials: pd.DataFrame, target: str, search: Optional[str] = None) -> pd.DataFrame:
    """The trials of a target with the settings parsed (`params_d`) and `rel` = trial score / score of the defaults in the same round."""
    import ast
    t = trials[trials["target"] == target].copy()
    if search:
        t = t[t["search"] == search]
    if t.empty:
        return t

    def parse(s):
        try:
            return ast.literal_eval(str(s)) if str(s).strip() not in ("", "{}", "nan") else {}
        except (ValueError, SyntaxError):
            return {}
    t["params_d"] = t["params"].map(parse)
    default = t[t["is_default"]].set_index(["search", "model", "tuned_at"])["score"]
    t["default_score"] = [default.get((a, m, r), float("nan")) for a, m, r in zip(t["search"], t["model"], t["tuned_at"])]
    t["rel"] = t["score"] / t["default_score"]
    return t


def best_so_far(trials: pd.DataFrame) -> pd.DataFrame:
    """Convergence of a search: the best relative score after k trials, averaged over the tuning rounds (the framework's own function)."""
    from common_layer.model_development.evidence import best_so_far as bsf
    return bsf(trials)


# ---------------------------------------------------------------------------------------------------------- monitoring
def monitoring_replay(locked: pd.DataFrame, daily: pd.DataFrame, target: str, window: int = 28, quantile: float = 0.95,
                      shock_day: Optional[str] = None, factor: float = 1.0) -> Optional[dict]:
    """Replay of the monitoring rule over the locked year (a genuine out-of-sample year).

    The threshold is the `quantile` of the rolling `window`-day mean absolute error of the champion on the DEVELOPMENT window (what normal looked
    like before going live). The rolling error of the locked year is then compared with it. With `shock_day` and `factor` the errors from that
    day on are multiplied by `factor`: a SIMULATION of a market change, to show the flag firing."""
    dev = daily[(daily["target"] == target) & (daily["model"] == "champion")].sort_values("date")
    lk = locked[locked["target"] == target].copy()
    if len(dev) < 2 * window or lk.empty:
        return None
    thr = dev["mae"].rolling(window).mean().dropna()
    level, threshold = float(thr.mean()), float(thr.quantile(quantile))
    lk["date"] = pd.to_datetime(lk["ts_utc"], utc=True).dt.tz_convert("Europe/Madrid").dt.tz_localize(None).dt.normalize()
    lk["err"] = lk["champion"] - lk["y_true"]
    if shock_day is not None and factor != 1.0:
        lk.loc[lk["date"] >= pd.Timestamp(shock_day), "err"] *= factor
    d = lk.groupby("date")["err"].agg(mae=lambda e: float(e.abs().mean()), bias="mean").reset_index()
    d["rolling_mae"] = d["mae"].rolling(window, min_periods=window).mean()
    d["flag"] = d["rolling_mae"] > threshold
    return {"daily": d, "level": level, "threshold": threshold, "window": window, "quantile": quantile,
            "flagged_days": int(d["flag"].sum()), "first_flag": (d.loc[d["flag"], "date"].min() if d["flag"].any() else None)}
