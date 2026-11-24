"""Amodo house style for matplotlib.

Drop this file next to any plotting script and import it. Typical use:

    import amodo_style as st
    from amodo_style import apply_amodo_style, style_axis, save_plot

    apply_amodo_style()                              # white background (default)
    # apply_amodo_style(background=st.AMODO_PAPER)  # brand cream instead

    fig, ax = plt.subplots()
    ax.plot(x, y, color=st.AMODO_DEEP_TEAL)
    style_axis(ax, "", "x label", "y label")
    save_plot(fig, Path("figure.png"), dpi=200)

Use ``st.AMODO_BG`` wherever you need "the background colour" — bar edges,
marker outlines, text on dark fills — so it tracks whichever background you
chose. Palette constants (AMODO_INK, AMODO_ACCENT, ...) never change.

Optional assets, auto-detected next to this file or passed to apply_amodo_style:
  - full logo.png       -> dropped top-right of every saved figure
  - assets/fonts/*.ttf  -> IBM Plex Sans/Serif registered with matplotlib
Both silently no-op if absent.

Palette and helpers are lifted from the LVHEM inference-verification analysis
script so figures match the amododesign.com notes charts.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
AMODO_PAPER = "#f5f3e7"  # brand cream; pass as background= to use it
AMODO_SURFACE = "#fcfbf4"
AMODO_INK = "#3e4e4d"
AMODO_MUTED = "#5f5f5f"
AMODO_RULE = "#c9c5b8"
AMODO_GRID = "#ded9cc"
AMODO_ACCENT = "#ca4f3e"
AMODO_TEAL = "#79bcba"
AMODO_BLUE = "#2f6fa3"
AMODO_GREEN = "#4f7e6b"
AMODO_AMBER = "#95713a"
AMODO_DEEP_TEAL = "#147f7b"
AMODO_TINT = "#f3f2ee"  # light stripe tint that reads on a white background
AMODO_SERIES = [
    AMODO_ACCENT,
    AMODO_DEEP_TEAL,
    AMODO_INK,
    AMODO_AMBER,
    AMODO_GREEN,
    AMODO_BLUE,
    "#7a4f93",
    "#b05f2d",
]
AMODO_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "*"]
AMODO_TITLE_FONT_FAMILY = ["IBM Plex Serif", "Georgia", "Times New Roman", "serif"]
AMODO_BODY_FONT_FAMILY = ["IBM Plex Sans", "Arial", "Helvetica", "sans-serif"]
AMODO_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "amodo", [AMODO_PAPER, "#e7c779", AMODO_ACCENT, AMODO_DEEP_TEAL, AMODO_INK]
)

# ---------------------------------------------------------------------------
# Runtime configuration (set by apply_amodo_style)
# ---------------------------------------------------------------------------
AMODO_BG = "#ffffff"  # figure/axes background and the contrast colour on dark fills
LOGO_PATH: Path = Path(__file__).parents[1] / "assets" / "full logo.png"
FONT_DIR: Path = Path(__file__).parents[1] / "assets" / "fonts"
LOGO_TOP = 0.985
LOGO_RIGHT = 0.985
LOGO_HEIGHT_FRACTION = 0.04
LOGO_CONTENT_TOP = 0.935
_LOGO_IMAGE: np.ndarray | None = None


def register_bundled_fonts() -> None:
    """Register IBM Plex TTFs from FONT_DIR. No-ops if the directory is absent."""
    if not FONT_DIR.is_dir():
        return
    for font_path in sorted(FONT_DIR.glob("*.ttf")):
        with contextlib.suppress(Exception):  # a font matplotlib cannot read is left out
            font_manager.fontManager.addfont(str(font_path))


def apply_amodo_style(
    background: str = "#ffffff",
    logo_path: str | Path | None = None,
    font_dir: str | Path | None = None,
) -> None:
    """Install the house style into matplotlib rcParams. Call once, before plotting."""
    global AMODO_BG, LOGO_PATH, FONT_DIR, _LOGO_IMAGE
    AMODO_BG = background
    if logo_path is not None:
        LOGO_PATH = Path(logo_path)
        _LOGO_IMAGE = None
    if font_dir is not None:
        FONT_DIR = Path(font_dir)
    register_bundled_fonts()
    plt.rcParams.update(
        {
            "figure.facecolor": AMODO_BG,
            "axes.facecolor": AMODO_BG,
            "savefig.facecolor": AMODO_BG,
            "savefig.edgecolor": AMODO_BG,
            "font.family": AMODO_BODY_FONT_FAMILY,
            "font.sans-serif": AMODO_BODY_FONT_FAMILY,
            "font.serif": AMODO_TITLE_FONT_FAMILY,
            "font.size": 13,
            "axes.edgecolor": AMODO_INK,
            "axes.labelcolor": AMODO_INK,
            "axes.labelsize": 13,
            "axes.linewidth": 0.8,
            "axes.titlecolor": AMODO_INK,
            "axes.titlelocation": "left",
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "xtick.color": AMODO_MUTED,
            "xtick.labelsize": 12,
            "ytick.color": AMODO_MUTED,
            "ytick.labelsize": 12,
            "grid.color": AMODO_GRID,
            "grid.linewidth": 0.55,
            "legend.frameon": False,
            "legend.labelcolor": AMODO_INK,
            "patch.edgecolor": AMODO_INK,
            "text.color": AMODO_INK,
        }
    )


def style_axis(ax: plt.Axes, title: str, xlabel: str, ylabel: str, *, grid_axis: str = "both") -> None:
    """House axis: uppercase left serif title, hidden top/right spines, faint grid."""
    ax.set_title(title.upper(), loc="left", pad=14, fontfamily=AMODO_TITLE_FONT_FAMILY)
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel(ylabel, labelpad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(AMODO_RULE)
    ax.spines["bottom"].set_color(AMODO_RULE)
    ax.tick_params(axis="both", length=3, width=0.7, color=AMODO_RULE)
    ax.grid(True, axis=grid_axis, alpha=1.0)
    ax.set_axisbelow(True)


def register_bbox_extra_artist(fig: plt.Figure, artist: Any) -> None:
    """Track artists outside the axes (legends, suptitles) so tight saving keeps them."""
    extra = getattr(fig, "_amodo_bbox_extra_artists", None)
    if extra is None:
        extra = []
        fig._amodo_bbox_extra_artists = extra
    extra.append(artist)


def style_figure_title(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    artist = fig.suptitle(
        title.upper(),
        x=0.01,
        y=0.985,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        fontfamily=AMODO_TITLE_FONT_FAMILY,
    )
    register_bbox_extra_artist(fig, artist)
    if subtitle:
        sub = fig.text(0.01, 0.952, subtitle, ha="left", va="top", fontsize=9, color=AMODO_MUTED)
        register_bbox_extra_artist(fig, sub)


def load_logo_image() -> np.ndarray | None:
    global _LOGO_IMAGE
    if _LOGO_IMAGE is not None:
        return _LOGO_IMAGE
    if not LOGO_PATH.exists():
        return None
    try:
        _LOGO_IMAGE = plt.imread(LOGO_PATH)
    except (OSError, ValueError):
        return None
    return _LOGO_IMAGE


def add_logo(fig: plt.Figure) -> None:
    """Drop LOGO_PATH into the top-right band. No-ops if the asset is missing."""
    logo = load_logo_image()
    if logo is None:
        return
    image_height, image_width = logo.shape[:2]
    if image_height <= 0 or image_width <= 0:
        return
    figure_width, figure_height = fig.get_size_inches()
    logo_aspect = image_width / image_height
    width_fraction = min(LOGO_HEIGHT_FRACTION * logo_aspect * (figure_height / figure_width), 0.14)
    height_fraction = width_fraction / logo_aspect * (figure_width / figure_height)
    logo_ax = fig.add_axes(
        [LOGO_RIGHT - width_fraction, LOGO_TOP - height_fraction, width_fraction, height_fraction],
        zorder=1000,
    )
    logo_ax.imshow(logo)
    logo_ax.axis("off")
    logo_ax.set_in_layout(False)


def save_plot(fig: plt.Figure, path: Path, dpi: int = 200) -> Path:
    """Lay out, add the logo, save with a tight bbox, and close the figure."""
    content_top = getattr(fig, "_amodo_content_top", LOGO_CONTENT_TOP)
    fig.tight_layout(rect=(0, 0, 1, content_top))
    add_logo(fig)
    extra = getattr(fig, "_amodo_bbox_extra_artists", [])
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.18, bbox_extra_artists=extra or None)
    plt.close(fig)
    return path


def annotate_series(ax: plt.Axes, items: list[dict[str, Any]]) -> None:
    """Direct, coloured, legend-free labels near each line (the notes-chart convention).

    Each item: {"x", "y", "text", "color"} plus optional "dx", "dy" (offset points),
    "ha", "va", "fontsize".
    """
    for it in items:
        ax.annotate(
            it["text"],
            xy=(it["x"], it["y"]),
            xycoords="data",
            xytext=(it.get("dx", 6), it.get("dy", 0)),
            textcoords="offset points",
            color=it["color"],
            fontsize=it.get("fontsize", 13),
            fontweight="normal",
            ha=it.get("ha", "left"),
            va=it.get("va", "center"),
            multialignment="center",
            zorder=6,
        )


def style_colorbar(cbar: Any) -> None:
    cbar.ax.yaxis.label.set_color(AMODO_INK)
    cbar.ax.tick_params(color=AMODO_RULE, labelcolor=AMODO_MUTED)
    cbar.outline.set_edgecolor(AMODO_RULE)


def style_table(table: Any) -> None:
    """Ink header with contrast text, alternating tinted rows, rule-coloured cell edges."""
    for (row, _col), cell in table.get_celld().items():
        cell.set_edgecolor(AMODO_RULE)
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor(AMODO_INK)
            cell.set_text_props(color=AMODO_BG, fontweight="bold", fontfamily=AMODO_TITLE_FONT_FAMILY)
        else:
            cell.set_facecolor(AMODO_TINT if row % 2 == 0 else AMODO_BG)
            cell.set_text_props(color=AMODO_INK)
