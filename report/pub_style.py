"""Shared plotting style for manuscript figure scripts."""

from __future__ import annotations

import os

import matplotlib as mpl
import matplotlib.pyplot as plt

OKABE_ITO = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "black": "#000000",
}

GREY = "#7A7A7A"
HIGHLIGHT = OKABE_ITO["vermillion"]
SEQ_CMAP = "viridis"
DIV_CMAP = "RdBu_r"

C = {
    "mature_OC": OKABE_ITO["vermillion"],
    "macrophage": OKABE_ITO["blue"],
    "OCP_mono": OKABE_ITO["bluish_green"],
    "myeloid": OKABE_ITO["blue"],
    "tumour": OKABE_ITO["reddish_purple"],
    "GCTB": OKABE_ITO["orange"],
    "adult": OKABE_ITO["sky_blue"],
    "canonical": "#4D4D4D",
}


def _apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def clean(ax: plt.Axes) -> plt.Axes:
    """Use a compact, publication-style axis."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=0.5, pad=1.5)
    return ax


def panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="right",
    )


def umap_axes(ax: plt.Axes) -> plt.Axes:
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return ax


def save_fig(fig: plt.Figure, stem: str, dpi: int = 300) -> None:
    """Save a figure as PDF and PNG using a path stem without extension."""

    _apply_style()
    outdir = os.path.dirname(stem)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


_apply_style()
