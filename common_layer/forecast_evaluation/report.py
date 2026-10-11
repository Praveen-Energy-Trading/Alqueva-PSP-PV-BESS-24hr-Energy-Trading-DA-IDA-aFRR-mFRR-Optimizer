"""
report.py — the report of one evaluation, and the digests that make it reproducible.

report.md contains no clock time: the same code, configuration and data give a byte-identical file. The digests
say what was scored: `config_digest` (evaluation settings and the target's formulation), `data_digest` (the
outcomes over the scored window) and the code revision. run.json holds the time and the file paths.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def data_digest(values: pd.Series) -> str:
    """Digest of the outcome series (rounded to 6 decimals, NaN as 'nan') in index order."""
    h = hashlib.sha256()
    for ts, v in zip(values.index.astype("int64"), values.to_numpy(dtype=float)):
        h.update(f"{ts}:{'nan' if np.isnan(v) else f'{v:.6f}'};".encode())
    return h.hexdigest()


def config_digest(eval_cfg: dict, formulation_target: dict) -> str:
    return sha256_text(json.dumps({"evaluation": eval_cfg, "target": formulation_target}, sort_keys=True, default=str))


def git_revision() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_REPO, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _md(df: pd.DataFrame, floats: str = "{:.3f}") -> str:
    if df is None or df.empty:
        return "_none_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(floats.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(lines) + "\n"


def render_report(meta: dict, overall: pd.DataFrame, by_group: Dict[str, Dict[str, pd.DataFrame]],
                  comparisons: pd.DataFrame, folds: pd.DataFrame) -> str:
    out = [f"# Evaluation: {meta['target']} ({meta['kind']})", "",
           f"Stage **{meta['stage']}**, delivery days {meta['test_start']} to {meta['test_end']}, "
           f"{meta['n_folds']} folds (refit every {meta['retrain_every_days']} days, gap {meta['gap_days']} day(s), "
           f"training window {meta['train_window_days'] or 'expanding'} days).", "",
           "Metrics are in price units (EUR/MWh; EUR/MW/h for the reserve band): SSE (sum of squared errors), MAE and RMSE. "
           "Compare SSE only between forecasters scored on the same hours.", "",
           f"config digest `{meta['config_digest'][:16]}`  data digest `{meta['data_digest'][:16]}`  code `{meta['code']}`"
           + (f"  locked data: **{meta['lock_status']}**" if meta.get("lock_status") else ""), "",
           f"Feature set: {meta.get('feature_set', 'n/a')}.", "",
           "## Overall", "", _md(overall)]
    if meta.get("note_common"):
        out += [f"_{meta['note_common']}_", ""]
    for grp, tables in by_group.items():
        out += [f"## By {grp.replace('_', ' ')}", ""]
        for name, t in tables.items():
            out += [f"**{name}**", "", _md(t), ""]
    out += ["## Comparison with the reference (Diebold-Mariano on daily absolute-error means)", "",
            f"Reference model: **{meta['reference']}** (a row for a combination of models is compared with the best single model instead, "
            f"named in its `reference` column). A negative statistic means the forecaster has the smaller loss.", "",
            _md(comparisons, "{:.4f}"), "", "## Folds", "", _md(folds.head(60), "{:.3f}"), ""]
    return "\n".join(out)


def write_outputs(out_dir: str, report_md: str, overall: pd.DataFrame, predictions: Dict[str, pd.DataFrame],
                  run_info: dict) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {"report": os.path.join(out_dir, "report.md"), "overall": os.path.join(out_dir, "metrics_overall.csv"),
             "run": os.path.join(out_dir, "run.json")}
    with open(paths["report"], "w", encoding="utf-8", newline="\n") as fh:
        fh.write(report_md)
    overall.to_csv(paths["overall"], index=False, float_format="%.6f", lineterminator="\n")
    for name, p in predictions.items():
        keep = p[["y_true", "y_pred", "fold", "local_date", "local_hour"]]
        keep.to_csv(os.path.join(out_dir, f"predictions_{name}.csv"), float_format="%.6f", lineterminator="\n")
    with open(paths["run"], "w", encoding="utf-8") as fh:
        json.dump(run_info, fh, indent=2, default=str)
    return paths
