"""
evidence.py — what the tuning did, from the logged trials (stage 3 of docs/Model_Development_Plan.pdf).

Inputs are the files a tuned evaluation run writes (tuning_trials.csv, tuning_log.csv, predictions_*.csv). Everything here
is a table; the figures are in evidence_figures.py. Scores are compared as `relative score` = trial score / the score of
the defaults in the same tuning round, so rounds with different price levels can be pooled (below 1 is better than the
defaults).

    best_so_far           convergence of the search: best relative score after k trials
    effects               mean and best relative score for every value of every searched setting
    importance            share of the variance of the relative score explained by each setting (eta squared)
    selected_settings     the setting chosen in each round, and how often each value is the most frequent one
    inner_vs_outer        the score that chose a setting against the error it then had on the days that followed

`importance` and `effects` are marginal summaries. They are unbiased only for uniformly drawn trials (the random-search
control); under TPE the trials concentrate where the score is good, so the same tables describe where the sampler looked.
"""
from __future__ import annotations

import ast
import datetime as dt
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def parse_params(s) -> dict:
    if isinstance(s, dict):
        return s
    if s is None or (isinstance(s, float) and s != s) or str(s).strip() in ("", "{}"):
        return {}
    return ast.literal_eval(str(s))


def _ordinal(x) -> int:
    return pd.Timestamp(x).date().toordinal()


def load_trials(path: str) -> pd.DataFrame:
    """tuning_trials.csv of a run folder with the settings parsed (column `params_d`) and `rel` = score / default score
    of the same model and round (NaN for pruned trials)."""
    df = pd.read_csv(os.path.join(path, "tuning_trials.csv"))
    df["params_d"] = df["params"].map(parse_params)
    default = df[df["is_default"]].set_index(["model", "tuned_at"])["score"]
    df["default_score"] = [default.get((m, t), np.nan) for m, t in zip(df["model"], df["tuned_at"])]
    df["rel"] = df["score"] / df["default_score"]
    return df


def best_so_far(trials: pd.DataFrame) -> pd.DataFrame:
    """For each model and trial number k: the mean over rounds of the best relative score among trials 0..k (pruned
    trials count as not better; trial 0, the defaults, is 1.0)."""
    rows = []
    for (m, t), g in trials.groupby(["model", "tuned_at"]):
        g = g.sort_values("trial")
        run_min = g["rel"].fillna(np.inf).cummin()
        for k, v in zip(g["trial"], run_min):
            rows.append({"model": m, "tuned_at": t, "trial": int(k), "best_rel": float(v)})
    d = pd.DataFrame(rows)
    return d.groupby(["model", "trial"], as_index=False).agg(best_rel=("best_rel", "mean"), rounds=("tuned_at", "nunique"))


def _searched(trials: pd.DataFrame, model: str) -> pd.DataFrame:
    g = trials[(trials["model"] == model) & (~trials["is_default"]) & trials["rel"].notna()].copy()
    return g


def effects(trials: pd.DataFrame, model: str) -> pd.DataFrame:
    """One row per setting and value: number of trials, mean and best relative score."""
    g = _searched(trials, model)
    rows = []
    for p in sorted({k for d in g["params_d"] for k in d}):
        vals = g["params_d"].map(lambda d: d.get(p, np.nan))
        for v in sorted(vals.dropna().unique(), key=str):
            r = g.loc[vals.map(lambda x: x == v), "rel"]
            rows.append({"model": model, "setting": p, "value": v, "n": int(len(r)), "mean_rel": float(r.mean()),
                         "best_rel": float(r.min())})
    return pd.DataFrame(rows)


def importance(trials: pd.DataFrame, model: str) -> pd.DataFrame:
    """Eta squared of each setting: between-value sum of squares / total sum of squares of the relative score."""
    g = _searched(trials, model)
    if len(g) < 3:
        return pd.DataFrame(columns=["model", "setting", "eta2"])
    y = g["rel"].to_numpy()
    total = float(((y - y.mean()) ** 2).sum())
    rows = []
    for p in sorted({k for d in g["params_d"] for k in d}):
        vals = g["params_d"].map(lambda d: str(d.get(p, "")))
        between = sum(len(grp) * (grp.mean() - y.mean()) ** 2 for _, grp in pd.Series(y, index=g.index).groupby(vals))
        rows.append({"model": model, "setting": p, "eta2": float(between / total) if total > 0 else 0.0})
    return pd.DataFrame(rows).sort_values("eta2", ascending=False).reset_index(drop=True)


def selected_settings(log: pd.DataFrame) -> pd.DataFrame:
    """The setting chosen in each round: one row per model, round and setting."""
    rows = []
    for _, r in log.iterrows():
        d = parse_params(r["params"])
        for k, v in (d.items() if d else [("(defaults)", "")]):
            rows.append({"model": r["model"], "tuned_at": r["tuned_at"], "setting": k, "value": v})
    return pd.DataFrame(rows)


def stability(sel: pd.DataFrame) -> pd.DataFrame:
    """Per model and setting: rounds in which it was chosen at a non-default value, and the share of those rounds that
    chose the most frequent value (1.0 = the same value every time)."""
    rows = []
    for (m, p), g in sel.groupby(["model", "setting"]):
        vc = g["value"].astype(str).value_counts()
        rows.append({"model": m, "setting": p, "rounds": int(len(g)), "most_frequent": vc.index[0],
                     "share_most_frequent": float(vc.iloc[0] / len(g))})
    return pd.DataFrame(rows)


def inner_vs_outer(log: pd.DataFrame, tuned_pred: Dict[str, pd.DataFrame], default_pred: Optional[Dict[str, pd.DataFrame]] = None,
                   gap_days: int = 1) -> pd.DataFrame:
    """For each model and round: the inner score of the chosen setting and of the defaults, and the outer MAE of the tuned
    (and untuned) model on the days that used the setting: the first day after the gap that follows the round's training
    end, up to the same day of the next round (a round's test days start `gap_days` + 1 days after its training end). `gap` = outer tuned MAE - inner score."""
    rows = []
    for m, g in log.groupby("model"):
        g = g.sort_values("tuned_at")
        starts = [_ordinal(t) for t in g["tuned_at"]]
        ends = starts[1:] + [10 ** 9]
        for (_, r), a, b in zip(g.iterrows(), starts, ends):
            def mae(pred):
                if pred is None:
                    return np.nan
                d = pred[(pred["local_date"] >= a + gap_days + 1) & (pred["local_date"] <= b + gap_days)].dropna(subset=["y_true", "y_pred"])
                return float((d["y_pred"] - d["y_true"]).abs().mean()) if len(d) else np.nan
            rows.append({"model": m, "tuned_at": r["tuned_at"], "inner_default": r["default_score"], "inner_chosen": r["score"],
                         "inner_gain": 1 - r["score"] / r["default_score"], "outer_tuned": mae(tuned_pred.get(m)),
                         "outer_default": mae((default_pred or {}).get(m))})
    d = pd.DataFrame(rows)
    d["outer_gain"] = 1 - d["outer_tuned"] / d["outer_default"]
    d["gap"] = d["outer_tuned"] - d["inner_chosen"]
    return d


def pruning_summary(trials: pd.DataFrame) -> pd.DataFrame:
    g = trials.groupby("model").agg(trials=("trial", "size"), pruned=("pruned", "sum"), duplicates=("duplicate", "sum"))
    g["share_pruned"] = g["pruned"] / g["trials"]
    return g.reset_index()


def read_predictions(path: str, suffix: str = "") -> Dict[str, pd.DataFrame]:
    """predictions_<model><suffix>.csv of a run folder, keyed by the model name without the suffix."""
    out = {}
    for f in os.listdir(path):
        if f.startswith("predictions_") and f.endswith(suffix + ".csv") and not f.startswith("predictions_ensemble_"):
            name = f[len("predictions_"):-len(suffix + ".csv")] if suffix else f[len("predictions_"):-4]
            if suffix == "" and name.endswith("_tuned"):
                continue
            out[name] = pd.read_csv(os.path.join(path, f), index_col=0, parse_dates=True)
    return out
