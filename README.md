# ste-dial

Measures how asking a coding agent to write "X% of the way to ASD-STE100
Simplified Technical English" changes its explanations of an unfamiliar
codebase.

The agent under test is Claude Code, run headless. The codebase is the Python
SDK of Apache Beam at a pinned commit. Each run asks for an explanation of one
subsystem. The only text that changes between conditions is one style sentence
at the end of the prompt.

| | |
|---|---|
| Subsystems | RunInference, Coders, FnApiRunner |
| STE levels | 0, 25, 50, 80, 100 |
| Repeats | 5 per subsystem and level (75 runs) |

All parameters are in `config.yaml`. The prompt template is in
`prompts/explain.txt`, and the style sentence for each level is in
`prompts/levels.yaml`.

## Setup

Requirements: Linux or WSL, `git`, [`uv`](https://docs.astral.sh/uv/), and
Claude Code, logged in.

```bash
uv sync
uv run pytest
```

The results in this repository came from Claude Code 2.1.296. Another version
can report different built-in skills or plugins. The harness then stops, and
you must update `claude.expected_init` in `config.yaml`.

## Reproduce the runs

```bash
# Print every rendered prompt and command. Runs nothing.
uv run ste-run --dry-run

# One repeat of every subsystem and level (15 runs).
uv run ste-run --pilot

# The full grid. Shows a cost projection from the saved runs and asks first.
uv run ste-run
```

`--limit N` runs only the first N jobs of the shuffled grid.

The first run fetches the pinned Beam commit into `beam.workdir`, which is
outside this repository. It then removes `.agent/` with one local commit, and
fails if any other agent-instruction file is tracked.

The batch is resumable. A job that already has a valid run is skipped, so the
pilot runs count as repeat 1 of the full batch. A failed run is tried again up
to two times. Each attempt keeps its own folder and failure reasons. If the
account's usage limit rejects a run, the batch stops; run the same command
again later.

The costs that the harness prints are the API-equivalent figures that Claude
Code reports. On a subscription plan, the runs count against the usage limits
instead.

### What each run saves

`results/raw/<subsystem>/<level>/<repeat>-<id>/` contains:

- `transcript.jsonl`: the raw `stream-json` output
- `answer.md`: the final answer
- `meta.json`: the rendered prompt, command, timestamps, turns, cost, model,
  Claude Code version, Beam SHA, init-event fields, validity flags, and the
  files that the agent read
- `stderr.txt`: the standard error output of the process

`files_read` in `meta.json` lists the files whose contents the agent saw:
the Read tool, and `cat`, `head`, `tail`, `sed`, `nl`, `awk`, and `less` in
shell commands. `paths_searched` lists the paths given to Grep, Glob, `grep`,
and `rg`. The shell part is an approximation: it follows `cd` and expands
globs, but it does not see files that reach a command through `xargs` or
`find -exec`.

`results/raw/` is in `.gitignore`.

## How a run is isolated

Each run is a new `claude` process. Its working directory is the Beam checkout.

```
claude --safe-mode -p "<rendered prompt>" \
  --model <model> --effort <effort> \
  --tools "Read,Grep,Glob,Bash" \
  --allowedTools "Bash(ls *)" "Bash(find *)" "Bash(rg *)" "Bash(grep *)" \
                 "Bash(cat *)" "Bash(head *)" "Bash(wc *)" \
  --disallowedTools "Edit" "Write" "mcp__*" \
  --permission-prompts none \
  --strict-mcp-config \
  --setting-sources "" \
  --no-session-persistence \
  --output-format stream-json --verbose \
  --max-turns <n> --max-budget-usd <usd>
```

| Flag | Effect |
|---|---|
| `--safe-mode` | Disables CLAUDE.md, personal skills, installed plugins, hooks, MCP servers, custom commands and agents, output styles, and auto memory. The skills, plugins, and agents that ship with Claude Code stay. |
| `-p` | Runs without a terminal session and exits after the answer. |
| `--model`, `--effort` | Pin the model by its full ID, and the effort level. |
| `--tools` | Limits the built-in tools to these four. No edit, write, web, or subagent tools. |
| `--allowedTools` | Shell commands that run without a permission prompt. Claude Code also runs other commands that it classifies as read-only, such as `sed -n`, `tail`, and `echo`, without a prompt. |
| `--disallowedTools` | Denies the edit and write tools and every MCP tool, even if another setting allows them. |
| `--permission-prompts none` | Denies everything that would otherwise prompt. Nobody is there to answer. |
| `--strict-mcp-config` | Ignores every MCP configuration. None is passed, so there are no MCP servers. |
| `--setting-sources ""` | Loads no user, project, or local settings file. Without it, the allow rules in a personal `settings.json` still apply. |
| `--no-session-persistence` | Does not save the session to disk. |
| `--output-format stream-json --verbose` | Emits the full transcript, including every tool call. |
| `--max-turns`, `--max-budget-usd` | Stop a run that does not finish. |

The allowlist has no `git` commands. The removed `.agent/` folder is still in
the parent of the scrub commit, and `git show` could read it.

The harness also removes the environment variables that start with `CLAUDE`
from the agent process, apart from `CLAUDE_CONFIG_DIR` and
`CLAUDE_CODE_OAUTH_TOKEN`. A run then behaves the same whether you start the
batch from a terminal or from inside a Claude Code session.

### Checks on every run

A run is marked invalid, and its record is kept, if one of these does not hold:

- The init event reports no MCP servers, the configured model, the default
  permission mode and output style, and exactly the tools, skills, plugins,
  agents, and slash commands in `claude.expected_init`. Every plugin must have
  the path `builtin`. If this check fails, the batch stops.
- The run ends with a successful result and a non-empty answer.
- After the run, `git status --porcelain --ignored` is empty and `HEAD` did not
  move. The checkout is reset with `git reset --hard` and `git clean -fdx`
  before every run.
- No tool call refers to `.agent`.

In a test with these flags, `mkdir` and `python3` were denied, and a run with
`--max-turns 1` stopped with `error_max_turns`.

Known limit: `find` can delete files or run a command through `-exec`. The
status check after the run catches a changed checkout, but not a change
outside it.
