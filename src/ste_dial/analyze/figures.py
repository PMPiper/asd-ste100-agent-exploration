"""Figures. Every figure uses the same colors, fonts, and level order."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from ste_dial.analyze import stats  # noqa: E402

# One blue ramp, light to dark, for the ordered STE levels.
LEVEL_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")
MARKERS = ("o", "s", "^", "D", "v", "P")
ACCENT = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
HEATMAP = LinearSegmentedColormap.from_list("ste_blue", ("#eef4fd", "#0d366b"))

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": SECONDARY_INK,
        "axes.edgecolor": AXIS,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": SECONDARY_INK,
        "ytick.labelcolor": SECONDARY_INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.frameon": False,
        "svg.fonttype": "none",
    }
)


class Style:
    """Maps levels to colors and subsystems to markers, the same everywhere."""

    def __init__(self, levels: list[int], subsystems: dict[str, str]) -> None:
        """Initializes the style.

        Args:
            levels: The STE levels in ascending order.
            subsystems: Display name per subsystem id, in config order.
        """
        steps = np.linspace(0, len(LEVEL_RAMP) - 1, len(levels)).round().astype(int)
        self.levels = levels
        self.subsystems = subsystems
        self.color = {level: LEVEL_RAMP[step] for level, step in zip(levels, steps)}
        self.marker = dict(zip(subsystems, MARKERS))

    def level_handles(self) -> list[Line2D]:
        return [
            Line2D(
                [], [], marker="o", ls="", color=self.color[level], label=f"{level}%"
            )
            for level in self.levels
        ]

    def subsystem_handles(self) -> list[Line2D]:
        return [
            Line2D([], [], marker=self.marker[key], ls="", color=MUTED, label=name)
            for key, name in self.subsystems.items()
        ]


def save(figure: Figure, directory: Path, name: str) -> None:
    """Writes the figure as PNG (200 dpi) and SVG, then closes it."""
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{name}.png", dpi=200, bbox_inches="tight")
    figure.savefig(directory / f"{name}.svg", bbox_inches="tight")
    plt.close(figure)


def _facets(style: Style, title: str, ylabel: str) -> tuple[Figure, list]:
    count = len(style.subsystems)
    figure, axes = plt.subplots(
        1, count, figsize=(3.1 * count, 3.4), sharey=True, squeeze=False
    )
    axes = list(axes[0])
    for axis, name in zip(axes, style.subsystems.values()):
        axis.set_title(name)
        axis.set_xticks(range(len(style.levels)))
        axis.set_xticklabels([f"{level}%" for level in style.levels])
        axis.set_xlabel("STE level")
        axis.set_xlim(-0.6, len(style.levels) - 0.4)
    axes[0].set_ylabel(ylabel)
    figure.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")
    return figure, axes


def sentence_length(sentences: pd.DataFrame, style: Style) -> Figure:
    """Box plot of words per sentence by level, one panel per subsystem."""
    figure, axes = _facets(style, "Sentence length by STE level", "Words per sentence")
    top = (
        max(40.0, float(sentences["n_words"].quantile(0.995))) if len(sentences) else 40
    )
    for axis, key in zip(axes, style.subsystems):
        for position, level in enumerate(style.levels):
            mask = (sentences["subsystem"] == key) & (sentences["level"] == level)
            values = sentences.loc[mask, "n_words"].to_numpy()
            if len(values) == 0:
                continue
            axis.boxplot(
                values,
                positions=[position],
                widths=0.6,
                patch_artist=True,
                boxprops=dict(facecolor=style.color[level], edgecolor=SURFACE),
                medianprops=dict(color=SURFACE, linewidth=2),
                whiskerprops=dict(color=MUTED),
                capprops=dict(color=MUTED),
                flierprops=dict(
                    marker=".",
                    markersize=3,
                    markerfacecolor=MUTED,
                    markeredgecolor="none",
                    alpha=0.5,
                ),
            )
        for limit, label in (
            (20, "20: procedure limit"),
            (25, "25: description limit"),
        ):
            axis.axhline(limit, color=ACCENT, linestyle=(0, (4, 3)), linewidth=1)
            if axis is axes[-1]:
                axis.annotate(
                    label,
                    (1.0, limit),
                    xycoords=("axes fraction", "data"),
                    xytext=(4, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=7,
                    color=SECONDARY_INK,
                )
        axis.set_xticks(range(len(style.levels)))
        axis.set_xticklabels([f"{level}%" for level in style.levels])
        axis.set_ylim(0, top)
    figure.tight_layout()
    return figure


def metric_by_level(
    runs: pd.DataFrame,
    column: str,
    style: Style,
    title: str,
    ylabel: str,
    rng: np.random.Generator,
    band: tuple[float, float, str] | None = None,
) -> Figure:
    """Per-run values by level with the mean and its bootstrap 95% CI.

    Args:
        runs: One row per run with `subsystem`, `level`, and the column.
        column: The metric to plot.
        style: The shared style.
        title: Figure title.
        ylabel: Label of the y axis.
        rng: Seeded generator for the bootstrap.
        band: Optional (low, high, label) reference band drawn in every panel.

    Returns:
        The figure, one panel per subsystem.
    """
    figure, axes = _facets(style, title, ylabel)
    for axis, key in zip(axes, style.subsystems):
        if band is not None:
            axis.axhspan(band[0], band[1], color=GRID, alpha=0.7, linewidth=0)
        means = []
        for position, level in enumerate(style.levels):
            mask = (runs["subsystem"] == key) & (runs["level"] == level)
            values = runs.loc[mask, column].dropna().to_numpy()
            if len(values) == 0:
                means.append(np.nan)
                continue
            jitter = np.linspace(-0.12, 0.12, len(values)) if len(values) > 1 else [0]
            axis.scatter(
                position + np.asarray(jitter),
                values,
                s=22,
                color=style.color[level],
                alpha=0.55,
                linewidths=0,
                zorder=2,
            )
            mean = float(values.mean())
            means.append(mean)
            low, high = stats.bootstrap_mean_ci(values, rng)
            if not np.isnan(low):
                axis.plot(
                    [position, position],
                    [low, high],
                    color=INK,
                    linewidth=1.5,
                    zorder=3,
                )
            axis.scatter(
                [position],
                [mean],
                s=64,
                marker=style.marker[key],
                color=style.color[level],
                edgecolors=SURFACE,
                linewidths=2,
                zorder=4,
            )
        axis.plot(range(len(style.levels)), means, color=MUTED, linewidth=1, zorder=1)
    handles = [
        Line2D([], [], marker="o", ls="", color=MUTED, markersize=8, label="mean"),
        Line2D([], [], color=INK, linewidth=1.5, label="95% CI of the mean"),
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=MUTED,
            alpha=0.55,
            markersize=4,
            label="one run",
        ),
    ]
    if band is not None:
        handles.append(Line2D([], [], color=GRID, linewidth=6, label=band[2]))
    figure.legend(handles=handles, loc="lower center", ncols=len(handles), fontsize=8)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    return figure


def consistency(cells: pd.DataFrame, style: Style) -> Figure:
    """Mean pairwise similarity of the repeats in each cell, with its CI."""
    figure, axes = _facets(
        style,
        "Consistency of repeats by STE level",
        "Mean pairwise cosine similarity",
    )
    for axis, key in zip(axes, style.subsystems):
        rows = cells[cells["subsystem"] == key].set_index("level")
        means = [rows["mean_similarity"].get(level, np.nan) for level in style.levels]
        axis.plot(range(len(style.levels)), means, color=MUTED, linewidth=1, zorder=1)
        for position, level in enumerate(style.levels):
            if level not in rows.index or np.isnan(rows.loc[level, "mean_similarity"]):
                continue
            row = rows.loc[level]
            if not np.isnan(row["ci_low"]):
                axis.plot(
                    [position, position],
                    [row["ci_low"], row["ci_high"]],
                    color=INK,
                    linewidth=1.5,
                    zorder=2,
                )
            axis.scatter(
                [position],
                [row["mean_similarity"]],
                s=64,
                marker=style.marker[key],
                color=style.color[level],
                edgecolors=SURFACE,
                linewidths=2,
                zorder=3,
            )
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            color=MUTED,
            markersize=8,
            label="mean over pairs of runs",
        ),
        Line2D([], [], color=INK, linewidth=1.5, label="95% CI of the mean"),
    ]
    figure.legend(handles=handles, loc="lower center", ncols=2, fontsize=8)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    return figure


def projection(
    points: np.ndarray, runs: pd.DataFrame, style: Style, title: str, note: str = ""
) -> Figure:
    """2D projection of the runs: color is the level, shape is the subsystem."""
    figure, axis = plt.subplots(figsize=(6.4, 5.0))
    for key in style.subsystems:
        for level in style.levels:
            mask = ((runs["subsystem"] == key) & (runs["level"] == level)).to_numpy()
            axis.scatter(
                points[mask, 0],
                points[mask, 1],
                s=70,
                marker=style.marker[key],
                color=style.color[level],
                edgecolors=SURFACE,
                linewidths=1.5,
            )
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)
    axis.set_xlabel(note)
    first = axis.legend(
        handles=style.level_handles(),
        title="STE level",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    axis.add_artist(first)
    axis.legend(
        handles=style.subsystem_handles(),
        title="Subsystem",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.55),
    )
    figure.tight_layout()
    return figure


def cell_heatmap(
    cells: list[tuple[str, int]], matrix: np.ndarray, style: Style
) -> Figure:
    """Heatmap of the mean cosine similarity between cells."""
    labels = [f"{style.subsystems[key]} {level}%" for key, level in cells]
    figure, axis = plt.subplots(figsize=(8.2, 7.0))
    image = axis.imshow(matrix, cmap=HEATMAP)
    axis.set_xticks(range(len(cells)))
    axis.set_yticks(range(len(cells)))
    axis.set_xticklabels(labels, rotation=90, fontsize=7)
    axis.set_yticklabels(labels, fontsize=7)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    finite = matrix[np.isfinite(matrix)]
    middle = (finite.min() + finite.max()) / 2 if len(finite) else 0.0
    for (i, j), value in np.ndenumerate(matrix):
        if np.isfinite(value):
            axis.text(
                j,
                i,
                f"{value:.2f}".lstrip("0"),
                ha="center",
                va="center",
                fontsize=5.5,
                color=SURFACE if value > middle else INK,
            )
    # White gaps between the cells, and wider ones between subsystems.
    axis.set_xticks(np.arange(len(cells) + 1) - 0.5, minor=True)
    axis.set_yticks(np.arange(len(cells) + 1) - 0.5, minor=True)
    axis.grid(which="minor", color=SURFACE, linewidth=1)
    axis.tick_params(which="both", length=0)
    boundaries = [i for i in range(1, len(cells)) if cells[i][0] != cells[i - 1][0]]
    for boundary in boundaries:
        axis.axhline(boundary - 0.5, color=SURFACE, linewidth=4)
        axis.axvline(boundary - 0.5, color=SURFACE, linewidth=4)
    axis.set_title("Mean cosine similarity between cells")
    bar = figure.colorbar(image, ax=axis, fraction=0.04, pad=0.02)
    bar.outline.set_visible(False)
    bar.set_label("Cosine similarity")
    figure.tight_layout()
    return figure


def sentence_projection(
    points: np.ndarray, sentences: pd.DataFrame, style: Style
) -> Figure:
    """UMAP of the sentences: color is the level, jargon sentences are ringed."""
    figure, axis = plt.subplots(figsize=(7.2, 5.6))
    for level in style.levels:
        mask = (sentences["level"] == level).to_numpy()
        axis.scatter(
            points[mask, 0],
            points[mask, 1],
            s=7,
            color=style.color[level],
            alpha=0.6,
            linewidths=0,
        )
    jargon = sentences["has_jargon"].to_numpy()
    axis.scatter(
        points[jargon, 0],
        points[jargon, 1],
        s=46,
        facecolors="none",
        edgecolors=ACCENT,
        linewidths=1.4,
    )
    axis.set_title("Sentences, with jargon sentences marked")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(False)
    axis.set_xlabel(
        f"UMAP of {len(sentences)} prose sentences; {int(jargon.sum())} have a jargon term"
    )
    handles = style.level_handles() + [
        Line2D(
            [],
            [],
            marker="o",
            ls="",
            markerfacecolor="none",
            markeredgecolor=ACCENT,
            markeredgewidth=1.4,
            markersize=8,
            label="has a jargon term",
        )
    ]
    axis.legend(
        handles=handles, title="STE level", loc="upper left", bbox_to_anchor=(1.01, 1.0)
    )
    figure.tight_layout()
    return figure
