"""
evidence_figures.py — figures of the tuning evidence (stage 3). Tables come from evidence.py. Each function writes one PNG
and returns its path. The Agg backend is used, so no display is needed.
"""
from __future__ import annotations

import os
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"tpe": "#1f4e79", "random": "#c0504d", "grid": "#7f7f7f"}


def _save(fig, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def convergence(curves: Dict[str, pd.DataFrame], out_dir: str, name: str = "tuning_convergence.png") -> str:
    """Best relative score (1 = defaults) after k trials, one panel per model, one line per search method."""
    models = sorted({m for c in curves.values() for m in c["model"].unique()})
    fig, axes = plt.subplots(1, len(models), figsize=(3.3 * len(models), 3.2), sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        for label, c in curves.items():
            d = c[c["model"] == m].sort_values("trial")
            ax.plot(d["trial"], d["best_rel"], label=label, color=COLORS.get(label), lw=1.8)
        ax.set_title(m.replace("_", " "), fontsize=10)
        ax.set_xlabel("trials used")
        ax.axhline(1.0, color="0.7", lw=0.8)
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("best score / defaults\n(mean over tuning rounds)")
    axes[0][-1].legend(frameon=False, fontsize=9)
    return _save(fig, out_dir, name)


def setting_effects(eff: pd.DataFrame, model: str, out_dir: str) -> str:
    """Mean relative score by value of every searched setting of one model (bars; the line marks the defaults)."""
    sets = list(dict.fromkeys(eff["setting"]))
    cols = min(4, len(sets))
    rows = int(np.ceil(len(sets) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 2.6 * rows), squeeze=False)
    lo, hi = float(eff["mean_rel"].min()), float(eff["mean_rel"].max())
    for ax, s in zip(axes.ravel(), sets):
        d = eff[eff["setting"] == s]
        ax.bar([str(v) for v in d["value"]], d["mean_rel"], color="#1f4e79", alpha=0.85)
        ax.plot([str(v) for v in d["value"]], d["best_rel"], "o", color="#c0504d", ms=4)
        ax.axhline(1.0, color="0.5", lw=0.8)
        ax.set_ylim(max(0.0, lo - 0.05), min(hi + 0.05, 1.6))
        ax.set_title(s, fontsize=9)
        ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    for ax in axes.ravel()[len(sets):]:
        ax.axis("off")
    fig.suptitle(f"{model.replace('_', ' ')}: score by value (bars = mean, dots = best; 1 = defaults)", fontsize=10)
    return _save(fig, out_dir, f"tuning_effects_{model}.png")


def importance_bars(imp: pd.DataFrame, out_dir: str, name: str = "tuning_importance.png") -> str:
    models = list(dict.fromkeys(imp["model"]))
    fig, axes = plt.subplots(1, len(models), figsize=(3.4 * len(models), 3.0), squeeze=False)
    for ax, m in zip(axes[0], models):
        d = imp[imp["model"] == m].sort_values("eta2")
        ax.barh(d["setting"], d["eta2"], color="#1f4e79")
        ax.set_title(m.replace("_", " "), fontsize=10)
        ax.set_xlabel("variance explained")
        ax.set_xlim(0, max(0.05, float(imp["eta2"].max()) * 1.1))
    return _save(fig, out_dir, name)


def inner_outer(df: pd.DataFrame, out_dir: str, name: str = "tuning_inner_vs_outer.png") -> str:
    """Gain over the defaults seen when choosing (inner) and on the days that followed (outer), per model: one dot per round."""
    models = list(dict.fromkeys(df["model"]))
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for i, m in enumerate(models):
        d = df[df["model"] == m]
        ax.scatter(np.full(len(d), i - 0.12), 100 * d["inner_gain"], color="#1f4e79", s=22, label="inner (when choosing)" if i == 0 else None)
        ax.scatter(np.full(len(d), i + 0.12), 100 * d["outer_gain"], color="#c0504d", s=22, label="outer (days that followed)" if i == 0 else None)
        ax.plot([i - 0.12, i + 0.12], [100 * d["inner_gain"].mean(), 100 * d["outer_gain"].mean()], color="0.4", lw=1.2)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=9)
    ax.set_ylabel("MAE gain over defaults (%)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    return _save(fig, out_dir, name)
