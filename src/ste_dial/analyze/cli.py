"""Entry point `ste-analyze`: metrics, figures, and summary of the runs."""

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ste_dial import run as batch
from ste_dial.analyze import density, figures, geometry, stats, summary, surface
from ste_dial.analyze.embed import CachedEmbedder, Embedder
from ste_dial.analyze.preprocess import Document, parse_document
from ste_dial.config import Config, load_config

logger = logging.getLogger(__name__)

METRICS = [
    "words",
    "sentences",
    "words_per_sentence_mean",
    "words_per_sentence_median",
    "share_sentences_over_20",
    "share_sentences_over_25",
    "passive_rate",
    "flesch_kincaid_grade",
    "flesch_reading_ease",
    "jargon_per_1k",
    "rare_word_share",
    "hedges_per_1k",
    "compounds_per_1k",
    "nominalizations_per_1k",
    "stacked_noun_phrase_share",
    "noun_verb_ratio",
    "identifier_retention",
    "files_read",
    "turns",
    "cost_usd",
    "duration_s",
]
EXCERPT_SENTENCES = 4
BASELINE_LEVEL = 0
_HEADING = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)
_PROBLEM_HEADING = re.compile(r"problem|solve|purpose|why", re.IGNORECASE)


def problem_section(markdown: str) -> str:
    """Returns the section that says what problem the subsystem solves.

    The prompt asks for this part first, so most answers have it. Returns the
    whole text if no heading names it.
    """
    headings = list(_HEADING.finditer(markdown))
    for index, heading in enumerate(headings):
        if _PROBLEM_HEADING.search(heading.group(1)):
            following = headings[index + 1 :]
            end = following[0].start() if following else len(markdown)
            return markdown[heading.end() : end]
    return markdown


@dataclass(frozen=True)
class Run:
    """One usable run: its metadata and parsed answer."""

    meta: dict
    document: Document


def load_runs(config: Config) -> tuple[list[Run], list[dict]]:
    """Reads the runs that are valid and used the configured agent.

    Returns:
        The usable runs in a stable order, and the metadata of every saved
        run, usable or not.
    """
    order = {subsystem.id: i for i, subsystem in enumerate(config.subsystems)}
    raw = config.results_dir / "raw"
    runs = []
    metas = []
    for path in sorted(raw.glob("*/*/*/meta.json")):
        meta = json.loads(path.read_text())
        metas.append(meta)
        if batch.is_usable(meta, config) and meta["subsystem"]["id"] in order:
            answer = (path.parent / "answer.md").read_text()
            runs.append(Run(meta, parse_document(answer)))
    runs.sort(
        key=lambda r: (
            order[r.meta["subsystem"]["id"]],
            r.meta["level"],
            r.meta["repeat"],
            r.meta["run_id"],
        )
    )
    return runs, metas


def build_run_table(runs: list[Run], terms: list[surface.JargonTerm]) -> pd.DataFrame:
    """Computes one row of metrics per run."""
    baseline_runs: dict[str, list[Run]] = {}
    for r in runs:
        if r.meta["level"] == BASELINE_LEVEL:
            baseline_runs.setdefault(r.meta["subsystem"]["id"], []).append(r)

    def retention(run: Run) -> tuple[float, float]:
        """Returns the retention score and the mean size of the baseline sets.

        Every score uses baseline sets built from all but one baseline run.
        A baseline run is scored against the set that leaves it out. Any
        other run gets the mean over all such sets. The sets then have the
        same size rule for every run, and no run is scored against itself.
        """
        pool = baseline_runs.get(run.meta["subsystem"]["id"], [])
        if run in pool:
            subsets = [[r for r in pool if r is not run]]
        elif len(pool) > 1:
            subsets = [[r for r in pool if r is not left_out] for left_out in pool]
        else:
            subsets = [pool]
        sets = [
            surface.baseline_identifiers([set(r.document.identifiers) for r in subset])
            for subset in subsets
            if subset
        ]
        if not sets:
            return float("nan"), float("nan")
        scores = [surface.identifier_retention(run.document.raw, s) for s in sets]
        return float(np.mean(scores)), float(np.mean([len(s) for s in sets]))

    rows = []
    for r in runs:
        meta = r.meta
        subsystem = meta["subsystem"]["id"]
        metrics = surface.surface_metrics(r.document, terms)
        jargon_terms = metrics.pop("jargon_terms")
        rows.append(
            {
                "run_id": meta["run_id"],
                "subsystem": subsystem,
                "level": meta["level"],
                "repeat": meta["repeat"],
                **metrics,
                **density.density_metrics(r.document.plain_prose),
                "identifier_retention": retention(r)[0],
                "baseline_identifiers": retention(r)[1],
                "files_read": len(meta["files_read"]),
                "turns": meta["turns"],
                "cost_usd": meta["cost_usd"],
                "duration_s": meta["duration_s"],
                "started_at": meta["started_at"],
                "jargon_terms": json.dumps(jargon_terms, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def build_sentence_table(
    runs: list[Run], terms: list[surface.JargonTerm]
) -> pd.DataFrame:
    """Returns one row per prose sentence, tagged with its run."""
    rows = [
        {
            "run_id": r.meta["run_id"],
            "subsystem": r.meta["subsystem"]["id"],
            "level": r.meta["level"],
            "text": sentence.text,
            "n_words": sentence.n_words,
            "has_jargon": bool(surface.jargon_hits(sentence, terms)),
        }
        for r in runs
        for sentence in r.document.sentences
    ]
    columns = ["run_id", "subsystem", "level", "text", "n_words", "has_jargon"]
    return pd.DataFrame(rows, columns=columns)


def _term_counts(table: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"term": term, "level": level, "count": count}
        for level, terms in zip(table["level"], table["jargon_terms"])
        for term, count in json.loads(terms).items()
    ]
    return pd.DataFrame(rows, columns=["term", "level", "count"])


def _excerpts(
    runs: list[Run],
    names: dict[str, str],
    levels: list[int],
    chosen: pd.Series | None,
) -> str:
    by_id = {r.meta["run_id"]: r for r in runs}
    parts = []
    for key, name in names.items():
        parts.append(f"#### {name}")
        for level in levels:
            cell = [
                r
                for r in runs
                if r.meta["subsystem"]["id"] == key and r.meta["level"] == level
            ]
            if not cell:
                continue
            if chosen is not None:
                picked = by_id[chosen[(key, level)]]
            else:
                picked = cell[0]
            section = parse_document(problem_section(picked.document.raw))
            sentences = section.sentences[:EXCERPT_SENTENCES]
            text = " ".join(sentence.text for sentence in sentences)
            parts.append(f"**{level}%** (run `{key}/{level}/{picked.meta['run_id']}`)")
            parts.append(f"> {text}")
    return "\n\n".join(parts)


def analyze(config: Config, embedder: Embedder | None) -> None:
    """Writes metrics.csv, the figures, and summary.md.

    Args:
        config: The experiment configuration.
        embedder: The embedding model, or None to compute only the surface
            metrics.

    Raises:
        SystemExit: If there is no usable run.
    """
    rng = np.random.default_rng(config.seed)
    out = config.results_dir
    figure_dir = out / "figures"
    names = {subsystem.id: subsystem.name for subsystem in config.subsystems}
    levels = list(config.levels)
    style = figures.Style(levels, names)
    terms = surface.load_jargon(config.jargon_file)

    runs, metas = load_runs(config)
    if not runs:
        raise SystemExit("No usable run found under results/raw.")
    table = build_run_table(runs, terms)
    sentences = build_sentence_table(runs, terms)
    logger.info("%d usable runs, %d prose sentences", len(runs), len(sentences))

    sections = ["# Summary", f"Generated by `ste-analyze` from {len(runs)} runs."]
    by_level = stats.aggregate(table, METRICS, ["level"], rng)
    by_cell = stats.aggregate(table, METRICS, ["subsystem", "level"], rng)
    trends = stats.trend_tests(table, METRICS, rng)
    by_level.to_csv(out / "aggregates_by_level.csv", index=False)
    by_cell.to_csv(out / "aggregates_by_cell.csv", index=False)
    trends.to_csv(out / "trend_tests.csv", index=False)

    sections += [
        "## Runs per cell",
        summary.cell_count_table(table, names, levels),
        "## Level comparison",
        summary.level_table(by_level, levels),
        "All subsystems pooled. Sentence length counts each inline code "
        "identifier as one word.",
        "## Words per sentence by cell",
        summary.detail_table(
            by_cell, "words_per_sentence_mean", names, levels, "{:.1f}"
        ),
        "## Trend with STE level",
        "Spearman correlation of each metric with the level over all runs, "
        "with a bootstrap 95% CI, and a Kruskal-Wallis test across levels.",
        summary.trend_table(trends),
        "## Dense prose",
        "Measures of stacked, abstract wording, all subsystems pooled.",
        summary.density_table(by_level, levels),
        "## Stacked noun phrases by cell",
        summary.detail_table(
            by_cell, "stacked_noun_phrase_share", names, levels, "{:.1%}"
        ),
        "## Jargon",
        f"Top terms at {levels[0]}%, with the number of occurrences by level "
        f"({len(terms)} terms in the list).",
        summary.jargon_table(_term_counts(table), levels),
        "## Identifier retention by cell",
        "The fraction of the identifiers that at least half of a subsystem's "
        f"{levels[0]}% runs mention. Each baseline set leaves one {levels[0]}% "
        f"run out. A {levels[0]}% run is scored against the set without it, "
        "and every other run gets the mean over all the sets.",
        summary.detail_table(by_cell, "identifier_retention", names, levels, "{:.2f}"),
    ]

    figures.save(
        figures.sentence_length(sentences, style),
        figure_dir,
        "sentence_length_by_level",
    )
    figures.save(
        figures.metric_by_level(
            table,
            "jargon_per_1k",
            style,
            "Jargon by STE level",
            "Jargon terms per 1k words",
            rng,
        ),
        figure_dir,
        "jargon_by_level",
    )
    figures.save(
        figures.metric_by_level(
            table,
            "identifier_retention",
            style,
            "Identifier retention by STE level",
            f"{levels[0]}% identifiers mentioned (share)",
            rng,
        ),
        figure_dir,
        "identifier_retention",
    )

    figures.save(
        figures.metric_by_level(
            table,
            "stacked_noun_phrase_share",
            style,
            "Stacked noun phrases by STE level",
            "Stacked noun phrases (share)",
            rng,
        ),
        figure_dir,
        "stacked_noun_phrases_by_level",
    )
    figures.save(
        figures.metric_by_level(
            table,
            "compounds_per_1k",
            style,
            "Hyphenated compounds by STE level",
            "Compounds per 1,000 words",
            rng,
        ),
        figure_dir,
        "compounds_by_level",
    )

    chosen = None
    truncated = None
    if embedder is not None:
        vectors = embedder.embed([r.document.text_without_code_blocks for r in runs])
        table["similarity_to_baseline"] = geometry.drift_from_baseline(vectors, table)
        if config.anchor_dense is not None and config.anchor_plain is not None:
            dense, plain = embedder.embed(
                [config.anchor_dense.read_text(), config.anchor_plain.read_text()]
            )
            table["density_axis"] = density.density_axis_scores(vectors, dense, plain)
            axis = stats.aggregate(table, ["density_axis"], ["subsystem", "level"], rng)
            axis.to_csv(out / "density_axis.csv", index=False)
            figures.save(
                figures.metric_by_level(
                    table,
                    "density_axis",
                    style,
                    "Position between the plain and the dense anchor text",
                    "Dense minus plain similarity",
                    rng,
                ),
                figure_dir,
                "density_axis_by_level",
            )
            axis_section = [
                "## Density axis",
                "Cosine similarity of each run to the dense anchor text minus "
                "its similarity to the plain anchor text. Higher is denser.",
                summary.detail_table(axis, "density_axis", names, levels, "{:+.3f}"),
            ]
        else:
            axis_section = []
        noise = geometry.baseline_pairwise(vectors, table)
        scores = geometry.silhouette_table(vectors, table)
        cells = geometry.cell_consistency(vectors, table, rng)
        cell_keys, matrix = geometry.cell_similarity(vectors, table)
        # Show the cells in the subsystem order of the config.
        ranks = sorted(
            range(len(cell_keys)),
            key=lambda i: (list(names).index(cell_keys[i][0]), cell_keys[i][1]),
        )
        cell_keys = [cell_keys[i] for i in ranks]
        matrix = matrix[np.ix_(ranks, ranks)]
        chosen = geometry.closest_to_centroid(vectors, table)
        drift = stats.aggregate(
            table, ["similarity_to_baseline"], ["subsystem", "level"], rng
        )
        scores.to_csv(out / "silhouette.csv", index=False)
        cells.to_csv(out / "consistency.csv", index=False)
        drift.to_csv(out / "drift.csv", index=False)

        pca_points, explained = geometry.project_pca(vectors)
        note = f"PC1 {explained[0]:.1%}, PC2 {explained[1]:.1%} of the variance"
        for name, points, title, label in (
            ("doc_pca", pca_points, "PCA of the explanations", note),
            (
                "doc_tsne",
                geometry.project_tsne(vectors, config.seed),
                "t-SNE of the explanations",
                "",
            ),
            (
                "doc_umap",
                geometry.project_umap(vectors, config.seed),
                "UMAP of the explanations",
                "",
            ),
        ):
            figures.save(
                figures.projection(points, table, style, title, label), figure_dir, name
            )
        band = None
        if len(noise):
            q1, q3 = np.percentile(noise, [25, 75])
            band = (
                float(q1),
                float(q3),
                f"IQR of similarity between {levels[0]}% runs",
            )
        figures.save(
            figures.metric_by_level(
                table,
                "similarity_to_baseline",
                style,
                f"Similarity to the {levels[0]}% baseline by STE level",
                f"Similarity to the {levels[0]}% centroid",
                rng,
                band,
            ),
            figure_dir,
            "drift_vs_level",
        )
        figures.save(
            figures.consistency(cells, style), figure_dir, "consistency_by_level"
        )
        figures.save(
            figures.cell_heatmap(cell_keys, matrix, style),
            figure_dir,
            "cell_similarity_heatmap",
        )
        sentence_vectors = embedder.embed(sentences["text"].tolist())
        figures.save(
            figures.sentence_projection(
                geometry.project_umap(sentence_vectors, config.seed), sentences, style
            ),
            figure_dir,
            "sentence_umap_jargon",
        )
        truncated = getattr(embedder, "truncated", None)
        sections += [
            "## Silhouette scores",
            "Cosine distance on the full-dimension document embeddings.",
            summary.silhouette_table(scores, names),
            f"PCA explained variance: {note}.",
            "## Drift from the baseline",
            f"Cosine similarity of each run to the centroid of its subsystem's "
            f"{levels[0]}% runs. A {levels[0]}% run is compared with the "
            f"centroid of the other {levels[0]}% runs.",
            summary.detail_table(
                drift, "similarity_to_baseline", names, levels, "{:.3f}"
            ),
            summary.noise_floor_text(noise),
            "## Consistency of repeats",
            summary.consistency_table(cells, names),
        ]
        sections += axis_section
    else:
        sections += ["## Embeddings", "Skipped (`--skip-embeddings`)."]

    table.to_csv(out / "metrics.csv", index=False)
    sentences.to_csv(out / "sentences.csv", index=False)

    how = (
        "the run closest to the centroid of its cell"
        if chosen is not None
        else "the first run of its cell, because embeddings were skipped"
    )
    usable = [r.meta for r in runs]
    invalid = [m for m in metas if not m.get("valid")]
    other = [m for m in metas if m.get("valid") and not batch.is_usable(m, config)]
    started = sorted(m["started_at"] for m in usable)
    reproducibility = [
        f"- Claude Code version: {', '.join(sorted({m['claude_code_version'] for m in usable}))}",
        f"- Model: `{config.claude.model}`, effort `{config.claude.effort}`",
        f"- Beam commit: `{config.beam_sha}`",
        f"- Embedding model: {embedder.name if embedder is not None else 'not used'}"
        + (
            f" ({truncated} texts truncated; texts read from the cache are not counted)"
            if truncated is not None
            else ""
        ),
        f"- Seed: {config.seed}",
        f"- Run dates (UTC): {started[0]} to {started[-1]}",
        f"- Runs analyzed: {len(usable)}",
        f"- Invalid runs (kept on disk, not analyzed): {len(invalid)}",
        f"- Valid runs of another model or effort (not analyzed): {len(other)}",
        f"- Cost reported by Claude Code, analyzed runs: "
        f"${sum(m['cost_usd'] or 0 for m in usable):.2f}",
        f"- Cost reported by Claude Code, all saved runs: "
        f"${sum(m.get('cost_usd') or 0 for m in metas):.2f}",
    ]
    sections += [
        "## Paired excerpts",
        f"The first {EXCERPT_SENTENCES} prose sentences of the section on the "
        "problem that the subsystem solves, or of the whole answer if no "
        "heading names that section. "
        f"Each excerpt is from {how}.",
        _excerpts(runs, names, levels, chosen),
        "## Reproducibility",
        "\n".join(reproducibility),
    ]
    (out / "summary.md").write_text("\n\n".join(sections) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Runs the analysis. Returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="compute only the surface metrics",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    config = load_config(args.config)
    embedder = None
    if not args.skip_embeddings:
        from ste_dial.analyze.embed import SentenceTransformerEmbedder

        model = SentenceTransformerEmbedder(
            config.embedding_model, config.embedding_max_tokens
        )
        embedder = CachedEmbedder(model, config.results_dir / "cache" / "embeddings")
    analyze(config, embedder)
    print(f"Wrote {config.results_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
