"""Builds results/summary.md: tables that can be pasted into a post."""

import math

import numpy as np
import pandas as pd

# Rows of the level comparison table: label, metric column, statistic, format.
LEVEL_TABLE = (
    ("Words (median)", "words", "median", "{:.0f}"),
    ("Words per sentence (mean)", "words_per_sentence_mean", "mean", "{:.1f}"),
    ("Sentences over 25 words (mean)", "share_sentences_over_25", "mean", "{:.1%}"),
    ("Passive voice (mean)", "passive_rate", "mean", "{:.1%}"),
    ("Jargon terms per 1k words (mean)", "jargon_per_1k", "mean", "{:.2f}"),
    ("Flesch-Kincaid grade (mean)", "flesch_kincaid_grade", "mean", "{:.1f}"),
    ("Files read by the agent (median)", "files_read", "median", "{:.0f}"),
)


DENSITY_TABLE = (
    ("Stacked noun phrases (mean)", "stacked_noun_phrase_share", "mean", "{:.1%}"),
    ("Hyphenated compounds per 1k words (mean)", "compounds_per_1k", "mean", "{:.1f}"),
    ("Nominalizations per 1k words (mean)", "nominalizations_per_1k", "mean", "{:.1f}"),
    ("Nouns and adjectives per verb (mean)", "noun_verb_ratio", "mean", "{:.2f}"),
    ("Rare-word share (mean)", "rare_word_share", "mean", "{:.1%}"),
    ("Flesch-Kincaid grade (mean)", "flesch_kincaid_grade", "mean", "{:.1f}"),
)


def fmt(value: float, pattern: str = "{:.3f}") -> str:
    """Formats a number, or "n/a" for a missing value."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return pattern.format(value)


def markdown_table(header: list[str], rows: list[list[str]]) -> str:
    """Renders a Markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def level_table(by_level: pd.DataFrame, levels: list[int]) -> str:
    """The level comparison table, with the number of runs per level."""
    return _table_by_level(by_level, levels, LEVEL_TABLE)


def density_table(by_level: pd.DataFrame, levels: list[int]) -> str:
    """The dense-prose measures by level, with the number of runs per level."""
    return _table_by_level(by_level, levels, DENSITY_TABLE)


def _table_by_level(by_level: pd.DataFrame, levels: list[int], spec: tuple) -> str:
    indexed = by_level.set_index(["metric", "level"])

    def value(metric: str, level: int, statistic: str) -> float:
        if (metric, level) not in indexed.index:
            return float("nan")
        return float(indexed.loc[(metric, level), statistic])

    rows = [
        [label] + [fmt(value(metric, level, statistic), pattern) for level in levels]
        for label, metric, statistic, pattern in spec
    ]
    counts = [
        (
            str(int(value("words", level, "n")))
            if ("words", level) in indexed.index
            else "0"
        )
        for level in levels
    ]
    rows.append(["Runs (n)"] + counts)
    return markdown_table(["Metric"] + [f"{level}%" for level in levels], rows)


def cell_count_table(
    runs: pd.DataFrame, names: dict[str, str], levels: list[int]
) -> str:
    """The number of runs in every subsystem and level."""
    counts = runs.groupby(["subsystem", "level"]).size()
    rows = [
        [name] + [str(int(counts.get((key, level), 0))) for level in levels]
        for key, name in names.items()
    ]
    return markdown_table(["Subsystem"] + [f"{level}%" for level in levels], rows)


def detail_table(
    by_cell: pd.DataFrame,
    metric: str,
    names: dict[str, str],
    levels: list[int],
    pattern: str,
) -> str:
    """Median [Q1, Q3], mean, and 95% CI of one metric in every cell."""
    indexed = by_cell[by_cell["metric"] == metric].set_index(["subsystem", "level"])
    rows = []
    for key, name in names.items():
        for level in levels:
            if (key, level) not in indexed.index:
                continue
            r = indexed.loc[(key, level)]
            rows.append(
                [
                    name,
                    f"{level}%",
                    str(int(r["n"])),
                    f"{fmt(r['median'], pattern)} [{fmt(r['q1'], pattern)}, {fmt(r['q3'], pattern)}]",
                    fmt(r["mean"], pattern),
                    f"[{fmt(r['ci_low'], pattern)}, {fmt(r['ci_high'], pattern)}]",
                ]
            )
    return markdown_table(
        ["Subsystem", "Level", "n", "Median [Q1, Q3]", "Mean", "95% CI of mean"], rows
    )


def trend_table(trends: pd.DataFrame) -> str:
    """Spearman and Kruskal-Wallis results for every metric."""
    rows = [
        [
            r["metric"],
            str(int(r["n"])),
            fmt(r["spearman_rho"], "{:+.2f}"),
            f"[{fmt(r['spearman_ci_low'], '{:+.2f}')}, {fmt(r['spearman_ci_high'], '{:+.2f}')}]",
            fmt(r["spearman_p"], "{:.3g}"),
            fmt(r["kruskal_h"], "{:.2f}"),
            fmt(r["kruskal_p"], "{:.3g}"),
        ]
        for _, r in trends.iterrows()
    ]
    return markdown_table(
        ["Metric", "n", "Spearman rho", "95% CI", "p", "Kruskal-Wallis H", "p"], rows
    )


def silhouette_table(scores: pd.DataFrame, names: dict[str, str]) -> str:
    """The silhouette scores by grouping and scope."""
    rows = []
    for _, r in scores.iterrows():
        scope = r["scope"]
        for key, name in names.items():
            scope = scope.replace(f"within {key}", f"within {name}")
        label = "By subsystem" if r["grouping"] == "subsystem" else "By STE level"
        rows.append([f"{label}, {scope}", str(int(r["n"])), fmt(r["silhouette"])])
    return markdown_table(["Grouping", "n", "Silhouette score"], rows)


def jargon_table(term_counts: pd.DataFrame, levels: list[int], top: int = 10) -> str:
    """The most frequent jargon terms at the lowest level, counted by level.

    Args:
        term_counts: Rows of term, level, and count.
        levels: The STE levels in ascending order.
        top: How many terms to list.
    """
    if term_counts.empty:
        return "No jargon term from the list matched in any run."
    pivot = term_counts.pivot_table(
        index="term", columns="level", values="count", aggfunc="sum", fill_value=0
    ).reindex(columns=levels, fill_value=0)
    pivot = pivot.assign(total=pivot.sum(axis=1)).reset_index()
    ordered = pivot.sort_values(
        [levels[0], "total", "term"], ascending=[False, False, True]
    ).set_index("term")
    rows = [
        [term] + [str(int(row[level])) for level in levels]
        for term, row in ordered.head(top).iterrows()
    ]
    return markdown_table(["Term"] + [f"{level}%" for level in levels], rows)


def noise_floor_text(pairs: np.ndarray) -> str:
    """One sentence that describes the similarity among 0% runs."""
    if len(pairs) == 0:
        return "Noise floor: not defined, because no subsystem has two 0% runs."
    q1, median, q3 = np.percentile(pairs, [25, 50, 75])
    return (
        f"Noise floor (cosine similarity between pairs of 0% runs of the same "
        f"subsystem, {len(pairs)} pairs): median {median:.3f}, "
        f"IQR [{q1:.3f}, {q3:.3f}], minimum {pairs.min():.3f}."
    )


def consistency_table(cells: pd.DataFrame, names: dict[str, str]) -> str:
    """Mean pairwise similarity within every cell."""
    rows = [
        [
            name,
            f"{int(r['level'])}%",
            str(int(r["n"])),
            str(int(r["pairs"])),
            fmt(r["mean_similarity"]),
            f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]",
        ]
        for key, name in names.items()
        for _, r in cells[cells["subsystem"] == key].iterrows()
    ]
    return markdown_table(
        ["Subsystem", "Level", "n", "Pairs", "Mean pairwise similarity", "95% CI"], rows
    )
