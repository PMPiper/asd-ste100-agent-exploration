"""Tests for the command builder, stream parser, and isolation checks."""

import json
from pathlib import Path

from ste_dial import claude_cli
from ste_dial.config import load_config

CLAUDE = load_config(Path(__file__).resolve().parent.parent / "config.yaml").claude
WORKDIR = "/work/beam"


def _isolated_init() -> dict:
    expected = CLAUDE.expected_init
    return {
        "type": "system",
        "subtype": "init",
        "tools": list(CLAUDE.tools),
        "mcp_servers": [],
        "model": CLAUDE.model,
        "permissionMode": "default",
        "output_style": "default",
        "slash_commands": list(expected.slash_commands),
        "skills": list(expected.skills),
        "agents": list(expected.agents),
        "plugins": [{"name": name, "path": "builtin"} for name in expected.plugins],
    }


def _tool_use(name: str, tool_input: dict) -> str:
    block = {"type": "tool_use", "name": name, "input": tool_input}
    return json.dumps({"type": "assistant", "message": {"content": [block]}})


def should_pass_the_prompt_as_the_argument_after_print_flag():
    command = claude_cli.build_command(CLAUDE, "Explain Coders.")

    assert command[:4] == ["claude", "--safe-mode", "-p", "Explain Coders."]


def should_ignore_all_settings_files_and_mcp_configs():
    command = claude_cli.build_command(CLAUDE, "Explain Coders.")

    assert command[command.index("--setting-sources") + 1] == ""
    assert "--strict-mcp-config" in command


def should_not_allow_git_commands_that_could_show_the_removed_folder():
    command = claude_cli.build_command(CLAUDE, "Explain Coders.")

    assert "Bash(git show *)" not in command
    assert "Bash(git log *)" not in command


def should_drop_parent_session_variables_from_the_child_environment():
    env = claude_cli.child_env(
        {"PATH": "/bin", "CLAUDE_EFFORT": "low", "CLAUDECODE": "1"}
    )

    assert env == {"PATH": "/bin"}


def should_keep_the_credential_variables_in_the_child_environment():
    env = claude_cli.child_env({"CLAUDE_CONFIG_DIR": "/cfg", "CLAUDE_PID": "7"})

    assert env == {"CLAUDE_CONFIG_DIR": "/cfg"}


def _checkout(tmp_path: Path) -> str:
    """Creates a small checkout: pkg/a.py, pkg/b.py, and pkg/c.txt."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("a = 1\n")
    (tmp_path / "pkg" / "b.py").write_text("b = 2\n")
    (tmp_path / "pkg" / "c.txt").write_text("c\n")
    return str(tmp_path)


def should_list_files_printed_by_cat_and_not_the_piped_head(tmp_path):
    workdir = _checkout(tmp_path)

    trace = claude_cli.trace_bash_command(
        "cat pkg/a.py pkg/b.py | head -n 20", workdir, workdir
    )

    assert trace.read == ["pkg/a.py", "pkg/b.py"]


def should_list_file_printed_by_sed_and_not_its_line_range(tmp_path):
    workdir = _checkout(tmp_path)

    trace = claude_cli.trace_bash_command("sed -n 1,5p pkg/a.py", workdir, workdir)

    assert trace.read == ["pkg/a.py"]


def should_resolve_files_against_the_directory_set_by_cd(tmp_path):
    workdir = _checkout(tmp_path)

    trace = claude_cli.trace_bash_command("cd pkg && tail -3 b.py", workdir, workdir)

    assert trace.read == ["pkg/b.py"]
    assert trace.cwd == str(tmp_path / "pkg")


def should_expand_globs_to_the_files_that_exist(tmp_path):
    workdir = _checkout(tmp_path)

    trace = claude_cli.trace_bash_command("cat pkg/*.py", workdir, workdir)

    assert trace.read == ["pkg/a.py", "pkg/b.py"]


def should_not_list_redirect_targets_as_files(tmp_path):
    workdir = _checkout(tmp_path)

    trace = claude_cli.trace_bash_command(
        "cat pkg/a.py 2>/dev/null; ls pkg", workdir, workdir
    )

    assert trace.read == ["pkg/a.py"]


def should_list_grep_targets_as_searched_and_not_as_read(tmp_path):
    workdir = _checkout(tmp_path)

    trace = claude_cli.trace_bash_command(
        'grep -n "^class " pkg/a.py | head -5', workdir, workdir
    )

    assert trace.searched == ["pkg/a.py"]
    assert trace.read == []


def should_not_list_files_that_wc_only_counts(tmp_path):
    workdir = _checkout(tmp_path)

    trace = claude_cli.trace_bash_command("wc -l pkg/a.py", workdir, workdir)

    assert trace.read == []


def should_keep_the_shell_directory_between_bash_calls(tmp_path):
    workdir = _checkout(tmp_path)
    lines = [
        _tool_use("Bash", {"command": "cd pkg && ls"}),
        _tool_use("Bash", {"command": "sed -n 1,2p a.py"}),
    ]

    summary = claude_cli.parse_stream(lines, workdir)

    assert summary.files_read == ["pkg/a.py"]


def should_report_read_paths_relative_to_the_checkout():
    lines = [_tool_use("Read", {"file_path": "/work/beam/sdks/python/setup.py"})]

    summary = claude_cli.parse_stream(lines, WORKDIR)

    assert summary.files_read == ["sdks/python/setup.py"]


def should_keep_absolute_path_for_reads_outside_the_checkout():
    lines = [_tool_use("Read", {"file_path": "/etc/hosts"})]

    summary = claude_cli.parse_stream(lines, WORKDIR)

    assert summary.files_read == ["/etc/hosts"]


def should_count_a_file_once_when_read_by_two_tools(tmp_path):
    workdir = _checkout(tmp_path)
    lines = [
        _tool_use("Read", {"file_path": f"{workdir}/pkg/a.py"}),
        _tool_use("Bash", {"command": "cat pkg/a.py"}),
    ]

    summary = claude_cli.parse_stream(lines, workdir)

    assert summary.files_read == ["pkg/a.py"]
    assert summary.tool_counts == {"Read": 1, "Bash": 1}


def should_record_search_paths_separately_from_files_read():
    lines = [_tool_use("Grep", {"pattern": "Coder", "path": "/work/beam/sdks"})]

    summary = claude_cli.parse_stream(lines, WORKDIR)

    assert summary.paths_searched == ["sdks"]
    assert summary.files_read == []


def should_flag_a_tool_call_that_refers_to_the_agent_folder():
    lines = [_tool_use("Bash", {"command": "ls .agent/skills"})]

    summary = claude_cli.parse_stream(lines, WORKDIR)

    assert summary.touched_agent_dir


def should_not_flag_a_source_file_with_agent_in_its_name():
    lines = [_tool_use("Read", {"file_path": "/work/beam/agent_development_kit.py"})]

    summary = claude_cli.parse_stream(lines, WORKDIR)

    assert not summary.touched_agent_dir


def should_capture_init_and_result_events():
    lines = [
        json.dumps(_isolated_init()),
        json.dumps({"type": "result", "subtype": "success", "result": "Done."}),
    ]

    summary = claude_cli.parse_stream(lines, WORKDIR)

    assert summary.init["model"] == CLAUDE.model
    assert summary.result["result"] == "Done."


def should_count_lines_that_are_not_json():
    summary = claude_cli.parse_stream(["not json"], WORKDIR)

    assert summary.unparsed_lines == 1


def should_accept_an_init_event_with_only_built_in_parts():
    assert claude_cli.check_init(_isolated_init(), CLAUDE) == []


def should_reject_a_missing_init_event():
    assert claude_cli.check_init(None, CLAUDE) == ["no init event in the stream"]


def should_reject_an_mcp_server():
    init = _isolated_init() | {"mcp_servers": [{"name": "drive"}]}

    problems = claude_cli.check_init(init, CLAUDE)

    assert problems == ["MCP servers present: [{'name': 'drive'}]"]


def should_reject_a_different_model():
    init = _isolated_init() | {"model": "claude-haiku-5-5"}

    problems = claude_cli.check_init(init, CLAUDE)

    assert problems == [f"model is claude-haiku-5-5, expected {CLAUDE.model}"]


def should_reject_a_personal_skill():
    init = _isolated_init()
    init["skills"].append("my-style")

    problems = claude_cli.check_init(init, CLAUDE)

    assert problems == ["unexpected skills: ['my-style']"]


def should_reject_a_plugin_that_is_not_built_in():
    init = _isolated_init()
    init["plugins"][0]["path"] = "/home/me/.claude/plugins/x"

    problems = claude_cli.check_init(init, CLAUDE)

    assert problems == ["plugins that are not built in: ['cc-plugin-agents-md']"]


def should_reject_a_custom_output_style():
    init = _isolated_init() | {"output_style": "terse"}

    problems = claude_cli.check_init(init, CLAUDE)

    assert problems == ["output style is terse"]


def should_reject_an_extra_tool():
    init = _isolated_init()
    init["tools"].append("Edit")

    problems = claude_cli.check_init(init, CLAUDE)

    assert problems == ["unexpected tools: ['Edit']"]
