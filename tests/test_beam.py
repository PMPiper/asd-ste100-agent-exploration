"""Tests for the Beam checkout preparation and the agent-file scan."""

import subprocess
from pathlib import Path

import pytest

from ste_dial import beam


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _make_upstream(path: Path, files: dict[str, str]) -> str:
    """Creates a repository with one commit and returns the commit SHA."""
    path.mkdir()
    _git(path, "init", "--quiet")
    for name, text in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_text(text)
    _git(path, "add", "-A")
    _git(path, "commit", "--quiet", "-m", "initial")
    return _git(path, "rev-parse", "HEAD")


def should_flag_skill_file_inside_agent_folder():
    assert beam.is_agent_instruction_file(".agent/skills/beam-concepts/SKILL.md")


def should_flag_claude_md_in_any_folder_ignoring_case():
    assert beam.is_agent_instruction_file("sdks/python/Claude.MD")


def should_flag_cursor_rule_files_by_extension():
    assert beam.is_agent_instruction_file("rules/python.mdc")


def should_flag_github_prompts_folder():
    assert beam.is_agent_instruction_file(".github/prompts/review.prompt.md")


def should_flag_windsurf_rules_file():
    assert beam.is_agent_instruction_file(".windsurfrules")


def should_not_flag_ordinary_source_file():
    assert not beam.is_agent_instruction_file(
        "sdks/python/apache_beam/coders/coders.py"
    )


def should_not_flag_file_whose_name_only_contains_agent():
    assert not beam.is_agent_instruction_file(
        "sdks/python/apache_beam/ml/inference/agent_development_kit.py"
    )


def should_return_only_the_agent_files_from_a_listing():
    hits = beam.find_agent_instruction_files(["a.py", "AGENTS.md", "docs/llms.txt"])

    assert hits == ["AGENTS.md", "docs/llms.txt"]


def should_remove_agent_folder_with_one_commit_on_top_of_pinned_sha(tmp_path):
    sha = _make_upstream(
        tmp_path / "upstream",
        {"src/a.py": "x = 1\n", ".agent/skills/concepts/README.md": "notes\n"},
    )
    workdir = tmp_path / "work"

    head = beam.prepare(str(tmp_path / "upstream"), sha, workdir)

    assert beam.tracked_files(workdir) == ["src/a.py"]
    assert _git(workdir, "rev-parse", "HEAD~1") == sha
    assert head == _git(workdir, "rev-parse", "HEAD")


def should_return_the_same_scrub_commit_when_prepared_twice(tmp_path):
    sha = _make_upstream(
        tmp_path / "upstream", {"a.py": "x = 1\n", ".agent/README.md": "notes\n"}
    )
    first = beam.prepare(str(tmp_path / "upstream"), sha, tmp_path / "work")

    second = beam.prepare(str(tmp_path / "upstream"), sha, tmp_path / "work")

    assert second == first


def should_fail_when_another_agent_file_is_still_tracked(tmp_path):
    sha = _make_upstream(
        tmp_path / "upstream",
        {"a.py": "x = 1\n", ".agent/README.md": "notes\n", "AGENTS.md": "rules\n"},
    )

    with pytest.raises(beam.BeamCheckoutError, match="AGENTS.md"):
        beam.prepare(str(tmp_path / "upstream"), sha, tmp_path / "work")


def should_report_untracked_and_ignored_files_as_dirty(tmp_path):
    sha = _make_upstream(
        tmp_path / "upstream",
        {"a.py": "x = 1\n", ".gitignore": "*.log\n", ".agent/README.md": "notes\n"},
    )
    workdir = tmp_path / "work"
    beam.prepare(str(tmp_path / "upstream"), sha, workdir)
    (workdir / "new.txt").write_text("new\n")
    (workdir / "debug.log").write_text("log\n")

    dirty = beam.dirty_paths(workdir)

    assert dirty == ["?? new.txt", "!! debug.log"]


def should_restore_a_clean_checkout_on_reset(tmp_path):
    sha = _make_upstream(
        tmp_path / "upstream", {"a.py": "x = 1\n", ".agent/README.md": "notes\n"}
    )
    workdir = tmp_path / "work"
    beam.prepare(str(tmp_path / "upstream"), sha, workdir)
    (workdir / "a.py").write_text("x = 2\n")
    (workdir / "new.txt").write_text("new\n")

    beam.reset(workdir)

    assert beam.dirty_paths(workdir) == []
    assert (workdir / "a.py").read_text() == "x = 1\n"
