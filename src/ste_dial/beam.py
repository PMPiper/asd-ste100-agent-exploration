"""Prepares and guards the Apache Beam checkout that the agent explores."""

import os
import subprocess
from pathlib import Path, PurePosixPath

BRANCH = "ste-dial"
SCRUB_MESSAGE = "Remove .agent/ before agent runs"

# The scrub commit uses a fixed identity and date so that its SHA is the same
# on every machine.
_SCRUB_ENV = {
    "GIT_AUTHOR_NAME": "ste-dial",
    "GIT_AUTHOR_EMAIL": "ste-dial@localhost",
    "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+00:00",
    "GIT_COMMITTER_NAME": "ste-dial",
    "GIT_COMMITTER_EMAIL": "ste-dial@localhost",
    "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+00:00",
}

_FILE_NAMES = frozenset(
    {
        "claude.md",
        "agents.md",
        "gemini.md",
        ".cursorrules",
        "copilot-instructions.md",
        "llms.txt",
        "skill.md",
    }
)
_DIR_NAMES = frozenset({".claude", ".agent", ".agents", ".cursor"})
_NAME_PREFIXES = (".windsurf", ".codex")
_PATH_FRAGMENTS = (".github/instructions/", ".github/prompts/")


class BeamCheckoutError(RuntimeError):
    """The Beam checkout is not in the state the experiment needs."""


def is_agent_instruction_file(path: str) -> bool:
    """Tells whether a repository path is an agent-instruction file.

    Args:
        path: A path relative to the repository root, with "/" separators.

    Returns:
        True if the path matches a known agent-instruction file or folder,
        ignoring case.
    """
    lowered = path.lower()
    parts = PurePosixPath(lowered).parts
    name = parts[-1]
    return (
        name in _FILE_NAMES
        or name.endswith(".mdc")
        or any(part in _DIR_NAMES for part in parts[:-1])
        or any(part.startswith(_NAME_PREFIXES) for part in parts)
        or any(fragment in lowered for fragment in _PATH_FRAGMENTS)
    )


def find_agent_instruction_files(paths: list[str]) -> list[str]:
    """Returns the paths that are agent-instruction files."""
    return [path for path in paths if is_agent_instruction_file(path)]


def _git(workdir: Path, *args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(workdir), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    return result.stdout


def tracked_files(workdir: Path) -> list[str]:
    """Returns the files tracked at HEAD of the checkout."""
    return _git(workdir, "ls-files").splitlines()


def head_sha(workdir: Path) -> str:
    """Returns the commit SHA at HEAD of the checkout."""
    return _git(workdir, "rev-parse", "HEAD").strip()


def prepare(repo_url: str, sha: str, workdir: Path) -> str:
    """Creates the scrubbed Beam checkout if needed and verifies it.

    Fetches only the pinned commit, checks it out on a local branch, removes
    .agent/ with one commit, and fails if any agent-instruction file is still
    tracked.

    Args:
        repo_url: URL of the Beam repository.
        sha: The pinned commit.
        workdir: Where the checkout lives.

    Returns:
        The SHA of the scrub commit that the runs use.

    Raises:
        BeamCheckoutError: If the checkout is not based on the pinned commit
            or still tracks agent-instruction files.
    """
    if not (workdir / ".git").exists():
        workdir.mkdir(parents=True, exist_ok=True)
        _git(workdir, "init", "--quiet")
        _git(workdir, "remote", "add", "origin", repo_url)
        _git(workdir, "fetch", "--quiet", "--depth", "1", "origin", sha)
        _git(workdir, "checkout", "--quiet", "-B", BRANCH, sha)
    if (workdir / ".agent").exists() or any(
        path.startswith(".agent/") for path in tracked_files(workdir)
    ):
        _git(workdir, "rm", "-r", "--quiet", ".agent")
        _git(
            workdir,
            "commit",
            "--quiet",
            "--no-gpg-sign",
            "-m",
            SCRUB_MESSAGE,
            env={**os.environ, **_SCRUB_ENV},
        )
    parent = _git(workdir, "rev-parse", "HEAD~1").strip()
    if parent != sha:
        raise BeamCheckoutError(
            f"{workdir} is not the scrub commit on top of {sha} (parent is {parent})"
        )
    changed = _git(workdir, "diff", "--name-only", "HEAD~1", "HEAD").splitlines()
    outside = [path for path in changed if not path.startswith(".agent/")]
    if outside:
        raise BeamCheckoutError(
            f"Scrub commit changes files outside .agent/: {outside}"
        )
    remaining = find_agent_instruction_files(tracked_files(workdir))
    if remaining:
        raise BeamCheckoutError(
            "Agent-instruction files are still tracked:\n" + "\n".join(remaining)
        )
    return head_sha(workdir)


def reset(workdir: Path) -> None:
    """Restores the checkout to HEAD and removes every untracked file."""
    _git(workdir, "reset", "--quiet", "--hard")
    _git(workdir, "clean", "--quiet", "-fdx")


def dirty_paths(workdir: Path) -> list[str]:
    """Returns the status lines of changed, untracked, and ignored files."""
    return _git(workdir, "status", "--porcelain", "--ignored").splitlines()
