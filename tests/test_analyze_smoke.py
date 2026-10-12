"""Smoke tests of ste-analyze on three fake runs."""

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from ste_dial.analyze import cli
from ste_dial.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG = load_config(REPO_ROOT / "config.yaml")

ANSWERS = {
    0: "## Coders\n\nUnder the hood, a `Coder` is wired up by the registry, which "
    "might surprise you. The bytes are written by `CoderImpl`.\n\n"
    "```python\ncoder.encode(value)\n```\n",
    80: "## Coders\n\nA `Coder` changes an element into bytes. `CoderImpl` writes "
    "the bytes. The registry selects the coder.\n",
    100: "## Coders\n\nA `Coder` changes an element into bytes. The registry "
    "selects the coder. Use one coder for each collection.\n",
}


class FakeEmbedder:
    """Deterministic bag-of-words vectors. No model is loaded."""

    name = "fake-embedder"

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            vector = np.full(16, 1e-3)
            for word in text.lower().split():
                vector[hashlib.sha256(word.encode()).digest()[0] % 16] += 1.0
            rows.append(vector / np.linalg.norm(vector))
        return np.array(rows)


def _write_run(results_dir: Path, level: int) -> None:
    run_dir = results_dir / "raw" / "coders" / str(level) / "1-aaaaaaaa"
    run_dir.mkdir(parents=True)
    (run_dir / "answer.md").write_text(ANSWERS[level])
    meta = {
        "run_id": "1-aaaaaaaa",
        "subsystem": {"id": "coders", "name": "Coders", "path": "coders"},
        "level": level,
        "repeat": 1,
        "valid": True,
        "model": CONFIG.claude.model,
        "effort": CONFIG.claude.effort,
        "files_read": ["sdks/python/apache_beam/coders/coders.py"],
        "turns": 5,
        "cost_usd": 0.1,
        "duration_s": 30.0,
        "started_at": "2026-10-11T00:00:00+00:00",
        "claude_code_version": "2.1.296 (Claude Code)",
    }
    (run_dir / "meta.json").write_text(json.dumps(meta))


def _three_runs(results_dir: Path):
    _write_run(results_dir, 0)
    _write_run(results_dir, 80)
    _write_run(results_dir, 100)
    return dataclasses.replace(CONFIG, results_dir=results_dir)


def should_write_one_metrics_row_per_run_without_embeddings(tmp_path):
    config = _three_runs(tmp_path)

    cli.analyze(config, None)

    rows = (tmp_path / "metrics.csv").read_text().splitlines()
    assert len(rows) == 1 + 3
    assert (tmp_path / "figures" / "sentence_length_by_level.png").exists()
    assert "Skipped (`--skip-embeddings`)" in (tmp_path / "summary.md").read_text()


def should_report_jargon_only_for_the_run_that_uses_it(tmp_path):
    config = _three_runs(tmp_path)

    cli.analyze(config, None)

    summary = (tmp_path / "summary.md").read_text()
    assert "| under the hood | 1 | 0 | 0 | 0 | 0 |" in summary


def should_write_every_figure_the_post_refers_to_with_embeddings(tmp_path):
    config = _three_runs(tmp_path)

    cli.analyze(config, FakeEmbedder())

    written = sorted(path.name for path in (tmp_path / "figures").glob("*.png"))
    assert written == [
        "cell_similarity_heatmap.png",
        "compounds_by_level.png",
        "consistency_by_level.png",
        "density_axis_by_level.png",
        "doc_pca.png",
        "doc_tsne.png",
        "doc_umap.png",
        "drift_vs_level.png",
        "identifier_retention.png",
        "jargon_by_level.png",
        "sentence_length_by_level.png",
        "sentence_umap_jargon.png",
        "stacked_noun_phrases_by_level.png",
    ]
    assert (tmp_path / "figures" / "doc_tsne.svg").exists()


def should_run_from_the_command_line_with_skip_embeddings(tmp_path):
    _three_runs(tmp_path / "results")
    raw = yaml.safe_load((REPO_ROOT / "config.yaml").read_text())
    raw["results_dir"] = str(tmp_path / "results")
    raw["jargon_file"] = str(REPO_ROOT / "jargon.txt")
    raw["prompts"] = {
        "template": str(REPO_ROOT / "prompts" / "explain.txt"),
        "levels": str(REPO_ROOT / "prompts" / "levels.yaml"),
    }
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(raw))

    exit_code = cli.main(
        ["--config", str(tmp_path / "config.yaml"), "--skip-embeddings"]
    )

    assert exit_code == 0
    assert "## Reproducibility" in (tmp_path / "results" / "summary.md").read_text()


def should_take_the_excerpt_from_the_problem_section():
    answer = "Intro line.\n\n## The problem it solves\n\nBytes must move.\n\n## Classes\n\nMany.\n"

    section = cli.problem_section(answer)

    assert section.strip() == "Bytes must move."


def should_fall_back_to_the_whole_answer_without_a_problem_heading():
    answer = "## Classes\n\nMany.\n"

    assert cli.problem_section(answer) == answer
