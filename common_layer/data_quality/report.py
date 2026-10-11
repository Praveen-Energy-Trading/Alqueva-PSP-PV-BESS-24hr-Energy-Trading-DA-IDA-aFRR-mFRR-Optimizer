"""report.py — Markdown data-quality report from a QualityRun."""
from __future__ import annotations

import datetime as dt
from typing import List

import pandas as pd


def _ranges(days: List[str]) -> List[str]:
    """['2024-01-01','2024-01-02','2024-01-05'] -> ['2024-01-01..2024-01-02', '2024-01-05']"""
    out, run = [], []
    for d in sorted(days):
        x = dt.date.fromisoformat(d)
        if run and (x - run[-1]).days == 1:
            run.append(x)
        else:
            if run:
                out.append(run)
            run = [x]
    if run:
        out.append(run)
    return [f"{r[0]}" if len(r) == 1 else f"{r[0]}..{r[-1]}" for r in out]


def _table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    if df.empty:
        return "_none_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def render_markdown(run) -> str:
    g, issues, daily = run.gate, run.issues, run.daily
    out = [f"# Data quality report\n",
           f"Generated {run.generated_utc}. Checked delivery days {run.start} to {run.end}.\n",
           f"## Gate: {'GREEN' if g.passed else 'RED'}\n"]
    if g.reasons:
        out += [f"- {r}" for r in g.reasons] + [""]
    out += ["## Datasets\n", _table(g.table.assign(
        missing_share=g.table["missing_share"].map("{:.1%}".format),
        mean_score=g.table["mean_score"].map("{:.1f}".format),
        share_below_70=g.table["share_below_70"].map("{:.1%}".format)) if len(g.table) else g.table)]

    warn = issues[issues["severity"] == "warning"]
    out.append("## Warnings by check\n")
    if len(warn):
        pivot = warn.groupby(["dataset", "check"]).agg(days=("delivery_date", "nunique")).reset_index()
        out.append(_table(pivot))
    else:
        out.append("_no warnings_\n")

    gaps = warn[warn["check"] == "missing_day"]
    out.append("## Missing days (with the last answer of the source)\n")
    if len(gaps):
        rows = []
        for (ds, sess, cause), d in gaps.groupby(["dataset", "session", "detail"]):
            rows.append({"dataset": ds + (f" s{sess}" if sess else ""), "cause": cause, "days": len(d),
                         "ranges": ", ".join(_ranges(list(d["delivery_date"]))[:6]) +
                                   (" ..." if len(_ranges(list(d["delivery_date"]))) > 6 else "")})
        out.append(_table(pd.DataFrame(rows)))
    else:
        out.append("_none_\n")

    out.append("## Lowest-scoring days (present days below 100)\n")
    present = daily[daily["present"] & (daily["score"] < 100)].sort_values(["score", "dataset", "delivery_date"]).head(15)
    out.append(_table(present[["dataset", "session", "delivery_date", "score", "n_warnings"]]))

    info = issues[issues["severity"] == "info"]
    out.append("## Information (counted, not penalised)\n")
    if len(info):
        t = info.assign(year=info["delivery_date"].str[:4]).groupby(["dataset", "check", "year"])["value"].sum().reset_index()
        t = t.pivot_table(index=["dataset", "check"], columns="year", values="value", fill_value=0).reset_index()
        t.columns = [str(c) for c in t.columns]
        out.append(_table(t, "{:.0f}"))
    else:
        out.append("_none_\n")
    return "\n".join(out)
