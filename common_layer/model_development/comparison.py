"""
comparison.py — which algorithm is better, and where (stage 4 of docs/Model_Development_Plan.pdf).

The algorithms are compared by their out-of-sample forecast errors on the SAME hours (never by their settings):

    scoreboard      SSE, MAE, RMSE, and how often each algorithm has the smallest error of the day
    pairwise        Diebold-Mariano test for every pair (one loss difference per delivery day, Newey-West variance,
                    Harvey-Leybourne-Newbold correction), with the Holm correction for the number of pairs
    versus_best     the algorithm with the smallest mean loss, and which others cannot be told apart from it
                    (Holm over the comparisons with the best only)
    win_table       where each algorithm wins: by hour of day, calendar month, or market regime

Reading a difference: a negative DM statistic means the first algorithm has the smaller loss. "Not distinguishable from
the best" means the test cannot separate them with the data at hand, not that they are equal.
"""
from __future__ import annotations

import datetime as dt
import itertools
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from common_layer.forecast_evaluation import metrics
from common_layer.forecast_evaluation.dm import daily_losses, diebold_mariano

ALPHA = 0.05


def prepare(preds: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Keep only the hours every algorithm predicted, and add the error columns."""
    idx = None
    for p in preds.values():
        ok = p.dropna(subset=["y_true", "y_pred"]).index
        idx = ok if idx is None else idx.intersection(ok)
    return {m: metrics.add_error_columns(p.loc[idx].copy()) for m, p in preds.items()}


def holm(p: List[float]) -> List[float]:
    """Holm step-down adjusted p-values (NaN stays NaN and is not counted)."""
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    order = ok[np.argsort(p[ok])]
    m, running = len(order), 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        out[i] = min(1.0, running)
    return list(out)


def scoreboard(prep: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per algorithm, best MAE first."""
    daily = pd.DataFrame({m: daily_losses(p, "absolute") for m, p in prep.items()})
    best = daily.idxmin(axis=1).value_counts(normalize=True)
    rows = []
    for m, p in prep.items():
        s = metrics.score(p)
        rows.append({"model": m, "n": s["n"], "sse": s["sse"], "mae": s["mae"], "rmse": s["rmse"],
                     "days_best": float(best.get(m, 0.0)), "mean_daily_rank": float(daily.rank(axis=1)[m].mean())})
    return pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)


def pairwise(prep: Dict[str, pd.DataFrame], loss: str = "absolute", hac_lags: int = 7, small_sample: str = "hln") -> pd.DataFrame:
    """Diebold-Mariano for every pair (a, b) with a before b in the input order, plus Holm-adjusted p-values."""
    rows = []
    for a, b in itertools.combinations(prep, 2):
        r = diebold_mariano(prep[a], prep[b], loss, hac_lags, small_sample)
        rows.append({"a": a, "b": b, **r})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p_value"].tolist())
    df["better"] = np.where(df["mean_diff"] < 0, df["a"], df["b"])
    df["significant"] = df["p_holm"] < ALPHA
    return df


def matrix(pairs: pd.DataFrame, models: List[str], value: str = "statistic") -> pd.DataFrame:
    """Square table: entry (row, column) is `value` of the row algorithm against the column algorithm (negative statistic = row better)."""
    out = pd.DataFrame(np.nan, index=models, columns=models)
    for _, r in pairs.iterrows():
        v = r[value]
        out.loc[r["a"], r["b"]] = v
        out.loc[r["b"], r["a"]] = -v if value in ("statistic", "mean_diff") else v
    return out


def versus_best(prep: Dict[str, pd.DataFrame], loss: str = "absolute", hac_lags: int = 7, small_sample: str = "hln") -> pd.DataFrame:
    """The algorithm with the smallest mean loss, and each other algorithm tested against it (Holm over n - 1 tests)."""
    mean = {m: float(daily_losses(p, loss).mean()) for m, p in prep.items()}
    best = min(mean, key=mean.get)
    rows = []
    for m in prep:
        if m == best:
            continue
        r = diebold_mariano(prep[m], prep[best], loss, hac_lags, small_sample)
        rows.append({"model": m, "best": best, "mean_loss": mean[m], "best_loss": mean[best], "mean_diff": r["mean_diff"],
                     "statistic": r["statistic"], "p_value": r["p_value"]})
    df = pd.DataFrame(rows)
    df["p_holm"] = holm(df["p_value"].tolist())
    df["distinguishable_from_best"] = df["p_holm"] < ALPHA
    return df


def top_set(vb: pd.DataFrame) -> List[str]:
    """The best algorithm and every algorithm that cannot be told apart from it."""
    best = vb["best"].iloc[0]
    return [best] + vb.loc[~vb["distinguishable_from_best"], "model"].tolist()


def _regime_of(local_date: int, periods: List[dict]) -> str:
    d = dt.date.fromordinal(int(local_date))
    for p in periods:
        if p["start"] <= d <= p["end"]:
            return p["id"]
    return "other"


def win_table(prep: Dict[str, pd.DataFrame], by: str, periods: Optional[List[dict]] = None) -> pd.DataFrame:
    """Per group: each algorithm's MAE, the one with the lowest MAE, which one wins most units, and its share of the units.

    A unit is a delivery day (for month and regime) or one hour of a day (for hour); the winner of a unit is the algorithm
    with the smallest mean absolute error in it. `lowest_mae` and `winner` can differ: an algorithm can win most days and
    still have the larger average error because of a few bad days.
    """
    first = next(iter(prep.values()))
    key = pd.DataFrame({"local_date": first["local_date"].to_numpy(), "hour": first["local_hour"].to_numpy()}, index=first.index)
    if by == "hour":
        unit = ["local_date", "hour"]
        group = key["hour"]
    elif by == "month":
        unit = ["local_date"]
        group = pd.Series([dt.date.fromordinal(int(d)).month for d in key["local_date"]], index=key.index)
    elif by == "regime":
        if not periods:
            raise ValueError("regime grouping needs the period definitions")
        unit = ["local_date"]
        group = pd.Series([_regime_of(d, periods) for d in key["local_date"]], index=key.index)
    else:
        raise ValueError("by must be hour, month or regime")
    losses = pd.DataFrame({m: p["abs_err"] for m, p in prep.items()})
    u = pd.concat([key.loc[:, unit], group.rename("group"), losses], axis=1)
    per_unit = u.groupby(unit + ["group"], as_index=False)[list(prep)].mean()
    per_unit["winner"] = per_unit[list(prep)].idxmin(axis=1)
    rows = []
    for g, d in per_unit.groupby("group"):
        wins = d["winner"].value_counts(normalize=True)
        top = wins.index[0]
        means = {m: float(d[m].mean()) for m in prep}
        rows.append({"group": g, "units": int(len(d)), **means, "lowest_mae": min(means, key=means.get), "winner": top,
                     "winner_share": float(wins.iloc[0])})
    return pd.DataFrame(rows)
