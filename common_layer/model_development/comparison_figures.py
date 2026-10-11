"""
comparison_figures.py — figures of the algorithm comparison (stage 4). Tables come from comparison.py.
"""
from __future__ import annotations

import os
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PALETTE = ["#1f4e79", "#c0504d", "#9bbb59", "#8064a2", "#f79646", "#4bacc6"]


def _save(fig, out_dir: str, name: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def dm_heatmap(stat: pd.DataFrame, p_holm: pd.DataFrame, out_dir: str, name: str = "algorithm_dm_matrix.png") -> str:
    """Signed DM statistic of the row algorithm against the column algorithm (blue = row has the smaller loss);
    a star marks a difference that survives the Holm correction."""
    models = list(stat.index)
    v = stat.to_numpy(dtype=float)
    lim = max(3.0, float(np.nanmax(np.abs(v)))) if np.isfinite(v).any() else 3.0
    fig, ax = plt.subplots(figsize=(1.2 * len(models) + 2.2, 1.0 * len(models) + 1.6))
    im = ax.imshow(v, cmap="RdBu", vmin=-lim, vmax=lim)
    for i in range(len(models)):
        for j in range(len(models)):
            if i == j or not np.isfinite(v[i, j]):
                continue
            star = "*" if p_holm.iloc[i, j] < 0.05 else ""
            ax.text(j, i, f"{v[i, j]:.1f}{star}", ha="center", va="center", fontsize=9)
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels([m.replace("_", "\n") for m in models], fontsize=8)
    ax.set_yticklabels([m.replace("_", " ") for m in models], fontsize=8)
    ax.set_title("Diebold-Mariano statistic, row against column\n(negative = row better; * = significant after Holm)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    return _save(fig, out_dir, name)


def mae_by_hour(tab: pd.DataFrame, models: List[str], out_dir: str, name: str = "algorithm_mae_by_hour.png") -> str:
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for c, m in zip(PALETTE, models):
        ax.plot(tab["group"], tab[m], label=m.replace("_", " "), color=c, lw=1.6)
    ax.set_xlabel("hour of day (local)")
    ax.set_ylabel("MAE")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    return _save(fig, out_dir, name)


def mae_by_group(tab: pd.DataFrame, models: List[str], out_dir: str, name: str, xlabel: str) -> str:
    """Grouped bars of MAE per group (month or market regime); the algorithm with the lowest MAE of each group is named above it."""
    fig, ax = plt.subplots(figsize=(max(6.5, 1.1 * len(tab)), 3.8))
    w = 0.8 / len(models)
    x = np.arange(len(tab))
    for k, (c, m) in enumerate(zip(PALETTE, models)):
        ax.bar(x + k * w - 0.4 + w / 2, tab[m], width=w, label=m.replace("_", " "), color=c)
    for i, win in enumerate(tab["lowest_mae"]):
        ax.text(i, float(tab[models].iloc[i].max()) * 1.02, win.split("_")[0][:4], ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(g) for g in tab["group"]], rotation=20, fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("MAE")
    ax.legend(frameon=False, fontsize=8, ncol=len(models))
    ax.grid(alpha=0.25, axis="y")
    return _save(fig, out_dir, name)
