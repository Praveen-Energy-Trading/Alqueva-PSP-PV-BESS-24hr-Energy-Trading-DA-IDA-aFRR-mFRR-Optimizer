"""_style.py - the look shared by every figure: palette, type, canvas furniture (title, subtitle, footer) and the SVG writer.

Figures are written as SVG with the text turned into paths, so they look the same on every machine and stay sharp at any size. One idea per figure;
the title says what the figure is, the subtitle says what to look at."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / "output"

# ---------------------------------------------------------------------------- palette
INK = "#0f172a"        # titles and numbers
TEXT = "#334155"       # labels
MUTED = "#64748b"      # secondary text and ticks
FAINT = "#94a3b8"      # footers
GRID = "#e2e8f0"       # gridlines
PANEL = "#f8fafc"      # soft panel behind a chart
LINE = "#cbd5e1"       # axis lines

BLUE = "#2563eb"       # generation, day-ahead energy
ORANGE = "#f97316"     # pumping, mFRR
TEAL = "#0d9488"       # PV, aFRR
VIOLET = "#7c3aed"     # prices
AMBER = "#f59e0b"      # battery, intraday
ROSE = "#e11d48"       # down, losses, risk
GREEN = "#16a34a"      # gains, passed checks
SKY = "#38bdf8"        # lower reservoir, water

_FONT = next((n for n in ("Segoe UI", "Calibri", "Arial", "DejaVu Sans") if any(f.name == n for f in fm.fontManager.ttflist)), "DejaVu Sans")

W, H = 12.0, 6.75      # inches; the SVG viewBox is 1200 x 675 units at 100 units per inch


def setup() -> None:
    plt.rcParams.update({
        "font.family": _FONT, "font.size": 10, "text.color": TEXT, "axes.labelcolor": MUTED, "axes.edgecolor": LINE,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "svg.fonttype": "path", "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.9,
        "axes.grid": False, "axes.axisbelow": True, "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "none",
        "savefig.facecolor": "white", "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })


def canvas(title: str, subtitle: str, source: str, accent: str = BLUE, height: float = H):
    """A blank figure with the title block and the footer."""
    setup()
    fig = plt.figure(figsize=(W, height))
    fig.patch.set_facecolor("white")
    # a short accent bar, then the title and the subtitle
    fig.add_artist(FancyBboxPatch((0.032, 1 - 0.074 * H / height), 0.006, 0.05 * H / height, boxstyle="round,pad=0,rounding_size=0.003",
                                  transform=fig.transFigure, facecolor=accent, edgecolor="none", mutation_aspect=H / W))
    fig.text(0.046, 1 - 0.042 * H / height, title, fontsize=19, fontweight="bold", color=INK, va="center", ha="left")
    fig.text(0.046, 1 - 0.083 * H / height, subtitle, fontsize=10.5, color=MUTED, va="center", ha="left")
    fig.text(0.032, 0.022 * H / height, source, fontsize=8, color=FAINT, va="center", ha="left")
    fig.text(0.968, 0.022 * H / height, "Alqueva  PSP + PV + BESS", fontsize=8, color=FAINT, va="center", ha="right", fontweight="bold")
    return fig


def clean(ax, grid: str = "y", ticks: bool = True) -> None:
    """The quiet look of a chart: no frame on top and right, light gridlines."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(LINE)
    ax.spines["bottom"].set_color(LINE)
    if "y" in grid:
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    if "x" in grid:
        ax.grid(axis="x", color=GRID, linewidth=0.8)
    if not ticks:
        ax.tick_params(length=0)
    else:
        ax.tick_params(length=3, width=0.8, color=LINE)


def label(ax, text: str, x: float = 0.0, y: float = 1.04, size: float = 10.5, color: str = INK, weight: str = "bold") -> None:
    """A heading above a panel, in axes coordinates."""
    ax.text(x, y, text, transform=ax.transAxes, fontsize=size, fontweight=weight, color=color, ha="left", va="bottom")


def clock_axis(ax, n: int = 96, step_h: int = 4) -> None:
    """A time-of-day axis for quarter-hour data: ISP 1 starts at 00:00, ISP n ends at 24:00."""
    ax.set_xlim(0, n)
    ticks = list(range(0, n + 1, n * step_h // 24))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t * 24 // n:02d}:00" for t in ticks])


def save(fig, name: str, preview: Path | None = None) -> Path:
    OUT.mkdir(exist_ok=True)
    path = OUT / f"{name}.svg"
    fig.savefig(path, format="svg")
    if preview is not None:
        preview.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview / f"{name}.png", dpi=110)
    plt.close(fig)
    return path
