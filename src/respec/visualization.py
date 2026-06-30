from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

CHANNEL_COLORS = ["#0B6E4F", "#FF9F1C", "#3A86FF"]
METHOD_COLORS = {
    "Cold": "#2E6F95",
    "Param": "#7B2CBF",
    "State": "#D0006F",
    "Combined": "#1B9C73",
    "Greedy": "#F77F00",
    "Local Search": "#6C757D",
    "Offline DP": "#111111",
}
REGIME_COLORS = {
    "stationary": "#577590",
    "gradual": "#43AA8B",
    "sudden": "#F94144",
    "continuous_sudden": "#F3722C",
}


def apply_style() -> ListedColormap:
    plt.rcParams.update(
        {
            "figure.facecolor": "#F7F5EF",
            "axes.facecolor": "#FCFBF8",
            "axes.edgecolor": "#252525",
            "axes.labelcolor": "#252525",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "grid.color": "#D9D3C7",
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "savefig.facecolor": "#F7F5EF",
            "savefig.bbox": "tight",
        }
    )
    return ListedColormap(CHANNEL_COLORS, name="channel_map")


def polish_axes(ax) -> None:
    ax.grid(True, alpha=0.55)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240)
    plt.close(fig)
