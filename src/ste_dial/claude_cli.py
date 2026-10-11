"""Builds the headless `claude` command and reads its stream-json output."""

import glob
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ste_dial.config import ClaudeConfig

# Environment variables that start with CLAUDE are set by a parent Claude Code
# session. They are removed so a run behaves the same from any terminal. These
# two only select the credentials and are kept.
_KEPT_ENV = frozenset({"CLAUDE_CONFIG_DIR", "CLAUDE_CODE_OAUTH_TOKEN"})
# Shell commands that print file contents, and commands that search files.
_READ_COMMANDS = frozenset({"cat", "head", "tail", "sed", "nl", "awk", "less"})
_SEARCH_COMMANDS = frozenset({"grep", "egrep", "fgrep", "rg"})
_OPERATOR_CHARS = frozenset("();<>|&")
_GLOB_CHARS = frozenset("*?[")
_AGENT_DIR = re.compile(r"\.agent\b")


@dataclass
class StreamSummary:
    """What one run's stream-json transcript contains."""

    init: dict | None = None
    result: dict | None = None
    files_read: list[str] = field(default_factory=list)
    paths_searched: list[str] = field(default_factory=list)
    bash_commands: list[str] = field(default_factory=list)
    tool_counts: dict[str, int] = field(default_factory=dict)
    touched_agent_dir: bool = False
    rate_limit: dict | None = None
    unparsed_lines: int = 0


def build_command(claude: ClaudeConfig, prompt: str) -> list[str]:
    """Returns the argv of one isolated, read-only, headless run."""
    return [
        "claude",
        "--safe-mode",
        "-p",
        prompt,
        "--model",
        claude.model,
        "--effort",
        claude.effort,
        "--tools",
        ",".join(claude.tools),
        "--allowedTools",
        *claude.allowed_tools,
        "--disallowedTools",
        *claude.disallowed_tools,
        "--permission-prompts",
        "none",
        "--strict-mcp-config",
        "--setting-sources",
        "",
        "--no-session-persistence",
        "--output-format",
        "stream-json",
        "--verbose",
        "--max-turns",
        str(claude.max_turns),
        "--max-budget-usd",
        str(claude.max_budget_usd),
    ]


def child_env(env: dict[str, str]) -> dict[str, str]:
    """Returns the environment for the agent process.

    Args:
        env: The environment of the harness.

    Returns:
        The same environment without the variables that a parent Claude Code
        session sets.
    """
    return {
        key: value
        for key, value in env.items()
        if not key.startswith("CLAUDE") or key in _KEPT_ENV
    }


def claude_version() -> str:
    """Returns the output of `claude --version`."""
    result = subprocess.run(
        ["claude", "--version"], check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def run_claude(
    command: list[str], cwd: Path, transcript: Path, stderr: Path, timeout_s: int
) -> tuple[int | None, bool]:
    """Runs the agent and writes its raw output to disk.

    Args:
        command: The argv from build_command.
        cwd: The Beam checkout.
        transcript: File that receives the raw stream-json output.
        stderr: File that receives the standard error output.
        timeout_s: Wall-clock limit in seconds.

    Returns:
        The exit code (None if the process was killed) and whether the run
        timed out.
    """
    with transcript.open("wb") as out, stderr.open("wb") as err:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            env=child_env(dict(os.environ)),
        )
        try:
            return process.wait(timeout=timeout_s), False
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            return None, True


def _relative(path: str, workdir: str) -> str:
    normalized = os.path.normpath(os.path.join(workdir, path))
    prefix = workdir.rstrip("/") + "/"
    return normalized[len(prefix) :] if normalized.startswith(prefix) else normalized


@dataclass
class BashTrace:
    """The files that one Bash tool call touched."""

    read: list[str] = field(default_factory=list)
    searched: list[str] = field(default_factory=list)
    cwd: str = ""


def _expand(token: str, cwd: str) -> list[str]:
    path = os.path.normpath(os.path.join(cwd, token))
    if _GLOB_CHARS & set(token):
        return sorted(glob.glob(path))
    return [path]


def trace_bash_command(command: str, cwd: str, workdir: str) -> BashTrace:
    """Finds the files that a shell command prints or searches.

    This is an approximation. It follows `cd`, expands globs, and keeps only
    the arguments that are existing paths. It does not see files that reach a
    command through `xargs`, `find -exec`, or a shell variable.

    Args:
        command: The command line of one Bash tool call.
        cwd: The shell's directory when the command starts.
        workdir: Absolute path of the Beam checkout.

    Returns:
        The files printed by cat, head, tail, sed, nl, awk, or less, the paths
        given to grep or rg, and the shell's directory after the command.
        Paths inside the checkout are relative to it.
    """
    trace = BashTrace(cwd=cwd)
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return trace
    program = None
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
        elif set(token) <= _OPERATOR_CHARS:
            if "<" in token or ">" in token:
                skip_next = True
            else:
                program = None
        elif program is None:
            program = token
        elif program == "cd":
            target = os.path.normpath(os.path.join(trace.cwd, token))
            if os.path.isdir(target):
                trace.cwd = target
        elif program in _READ_COMMANDS or program in _SEARCH_COMMANDS:
            if token.startswith("-"):
                continue
            for path in _expand(token, trace.cwd):
                if os.path.isfile(path) and program in _READ_COMMANDS:
                    trace.read.append(_relative(path, workdir))
                elif os.path.exists(path) and program in _SEARCH_COMMANDS:
                    trace.searched.append(_relative(path, workdir))
    return trace


def parse_stream(lines: list[str], workdir: str) -> StreamSummary:
    """Summarizes a stream-json transcript.

    Args:
        lines: The lines of the transcript.
        workdir: Absolute path of the Beam checkout. Paths inside it are
            reported relative to it.

    Returns:
        The init and result events, and what the tool calls touched.
        `files_read` holds the files whose contents were shown: the Read tool
        and the shell commands that print files. `paths_searched` holds the
        paths given to the Grep and Glob tools and to grep or rg.
    """
    summary = StreamSummary()
    files: set[str] = set()
    searched: set[str] = set()
    cwd = workdir
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            summary.unparsed_lines += 1
            continue
        kind = event.get("type")
        if kind == "system" and event.get("subtype") == "init":
            summary.init = event
        elif kind == "result":
            summary.result = event
        elif kind == "rate_limit_event":
            summary.rate_limit = event.get("rate_limit_info")
        elif kind == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") != "tool_use":
                    continue
                name = block.get("name", "")
                tool_input = block.get("input", {})
                summary.tool_counts[name] = summary.tool_counts.get(name, 0) + 1
                if _AGENT_DIR.search(json.dumps(tool_input)):
                    summary.touched_agent_dir = True
                if name == "Read" and "file_path" in tool_input:
                    files.add(_relative(tool_input["file_path"], workdir))
                elif name in ("Grep", "Glob"):
                    searched.add(_relative(tool_input.get("path") or ".", workdir))
                elif name == "Bash":
                    command = tool_input.get("command", "")
                    summary.bash_commands.append(command)
                    trace = trace_bash_command(command, cwd, workdir)
                    files.update(trace.read)
                    searched.update(trace.searched)
                    # The shell keeps its directory between calls, unless the
                    # command left the checkout.
                    inside = os.path.commonpath([workdir, trace.cwd]) == workdir
                    cwd = trace.cwd if inside else workdir
    summary.files_read = sorted(files)
    summary.paths_searched = sorted(searched)
    return summary


def init_fields(init: dict) -> dict:
    """Returns the init-event fields that describe the agent's setup."""
    return {
        "tools": init.get("tools"),
        "mcp_servers": init.get("mcp_servers"),
        "slash_commands": init.get("slash_commands"),
        "skills": init.get("skills"),
        "plugins": init.get("plugins"),
        "agents": init.get("agents"),
        "output_style": init.get("output_style"),
        "model": init.get("model"),
        "permission_mode": init.get("permissionMode"),
        "api_key_source": init.get("apiKeySource"),
        "claude_code_version": init.get("claude_code_version"),
    }


def _difference(label: str, actual: list[str], expected: tuple[str, ...]) -> list[str]:
    extra = sorted(set(actual) - set(expected))
    missing = sorted(set(expected) - set(actual))
    problems = []
    if extra:
        problems.append(f"unexpected {label}: {extra}")
    if missing:
        problems.append(f"missing {label}: {missing}")
    return problems


def check_init(init: dict | None, claude: ClaudeConfig) -> list[str]:
    """Lists every way the init event differs from an isolated setup.

    Args:
        init: The system/init event, or None if the stream had none.
        claude: The configured agent parameters.

    Returns:
        One message per problem. Empty if the run was isolated.
    """
    if init is None:
        return ["no init event in the stream"]
    expected = claude.expected_init
    problems = []
    if init.get("mcp_servers"):
        problems.append(f"MCP servers present: {init['mcp_servers']}")
    if init.get("model") != claude.model:
        problems.append(f"model is {init.get('model')}, expected {claude.model}")
    if init.get("permissionMode") != expected.permission_mode:
        problems.append(f"permission mode is {init.get('permissionMode')}")
    if init.get("output_style") != expected.output_style:
        problems.append(f"output style is {init.get('output_style')}")
    plugins = init.get("plugins") or []
    not_builtin = [p.get("name") for p in plugins if p.get("path") != "builtin"]
    if not_builtin:
        problems.append(f"plugins that are not built in: {not_builtin}")
    problems += _difference("tools", init.get("tools") or [], claude.tools)
    problems += _difference(
        "plugins", [p.get("name") for p in plugins], expected.plugins
    )
    problems += _difference("skills", init.get("skills") or [], expected.skills)
    problems += _difference("agents", init.get("agents") or [], expected.agents)
    problems += _difference(
        "slash commands", init.get("slash_commands") or [], expected.slash_commands
    )
    return problems
