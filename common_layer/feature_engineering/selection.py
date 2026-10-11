"""
selection.py — which features matter, and whether that is stable across periods.

    table = training_table(view, builder, days, transform)         # features + modelled target for each hour
    imp   = permutation_importance(table, learner, train_fraction, repeats, seed, transform)
    rank  = stability(importances_by_period, min_share, min_period_fraction)

Permutation importance of a feature = increase of the mean absolute error (in price units, on days not used for
training) when that feature's values are shuffled. The days are split in time order, so the importance is measured
out of sample. A feature is called stable when its share of the total positive importance reaches `min_share` in
at least `min_period_fraction` of the periods; the average pairwise Spearman correlation of the importance
rankings across periods summarises how much the picture changes from one period to the next.
"""
from __future__ import annotations

import datetime as dt
import itertools
import warnings
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from common_layer.feature_engineering.builder import FeatureBuilder
from common_layer.forecast_evaluation.asof import AsOfView

# learners are fitted and applied on plain arrays in the same column order; the feature-name notice is noise
warnings.filterwarnings("ignore", message="X does not have valid feature names")


def training_table(view: AsOfView, builder: FeatureBuilder, days: List[dt.date], transform, history_days: int = 400) -> pd.DataFrame:
    """Feature matrix with columns `y_raw` (price units), `y` (transformed) and `local_date` for the given days."""
    frames = []
    for D in days:
        X = builder.build(view.at(D, history_days), D)
        t = view.truth_day(D)
        y_raw = view.modelled(t).reindex(X.index)
        if y_raw.notna().sum() == 0:
            continue
        X = X.assign(y_raw=y_raw.to_numpy(), local_date=D.toordinal())
        frames.append(X[y_raw.notna().to_numpy()])
    out = pd.concat(frames)
    out["y"] = transform.forward(out["y_raw"].to_numpy())
    return out


def permutation_importance(table: pd.DataFrame, make_learner: Callable[[], object], features: List[str],
                           train_fraction: float, repeats: int, seed: int, transform) -> pd.DataFrame:
    """Out-of-sample permutation importance in price units (MAE increase), days split in time order."""
    days = np.sort(table["local_date"].unique())
    cut = days[int(len(days) * train_fraction)]
    tr, te = table[table["local_date"] < cut], table[table["local_date"] >= cut]
    cols = [c for c in features if tr[c].notna().any()]
    fill = tr[cols].mean()
    A_tr = tr[cols].fillna(fill).to_numpy()
    A_te = te[cols].fillna(fill).to_numpy()
    model = make_learner().fit(A_tr, tr["y"].to_numpy())

    def mae(A) -> float:
        return float(np.mean(np.abs(transform.inverse(model.predict(A)) - te["y_raw"].to_numpy())))

    base = mae(A_te)
    rng = np.random.default_rng(seed)
    rows = []
    for j, c in enumerate(cols):
        incr = []
        for _ in range(repeats):
            B = A_te.copy()
            B[:, j] = rng.permutation(B[:, j])
            incr.append(mae(B) - base)
        rows.append({"feature": c, "importance": float(np.mean(incr))})
    out = pd.DataFrame(rows)
    out.attrs["base_mae"], out.attrs["train_days"], out.attrs["test_days"] = base, int((days < cut).sum()), int((days >= cut).sum())
    return out


def stability(importances: Dict[str, pd.DataFrame], min_share: float, min_period_fraction: float) -> pd.DataFrame:
    """Per feature: share of positive importance in each period, how often it passes, and the selection."""
    shares = {}
    for p, df in importances.items():
        pos = df.set_index("feature")["importance"].clip(lower=0)
        total = pos.sum()
        shares[p] = pos / total if total > 0 else pos * 0.0
    S = pd.DataFrame(shares).fillna(0.0)
    out = pd.DataFrame({"mean_share": S.mean(axis=1), "periods_passed": (S >= min_share).sum(axis=1)})
    out["period_fraction"] = out["periods_passed"] / S.shape[1]
    out["selected"] = out["period_fraction"] >= min_period_fraction
    out = out.join(S.add_prefix("share_"))
    out = out.sort_values(["selected", "mean_share"], ascending=[False, False])
    out.index.name = "feature"
    ranks = S.rank(ascending=False)
    pairs = [stats.spearmanr(ranks[a], ranks[b])[0] for a, b in itertools.combinations(S.columns, 2)] if S.shape[1] > 1 else []
    out.attrs["mean_rank_correlation"] = float(np.nanmean(pairs)) if pairs else float("nan")
    return out
