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
| Repeats | 3 per subsystem and level (45 runs) |

All parameters are in `config.yaml`. The prompt template is in
`prompts/explain.txt`, and the style sentence for each level is in
`prompts/levels.yaml`.

## Status

| Experiment | Config | Runs done | Model |
|---|---|---|---|
| Explanation | `config.yaml` | 45 of 45 | Sonnet 5.5, medium effort |
| Design document | `config-design-doc.yaml` | 15 of 45 (one run per cell) | Sonnet 5.5, high effort |
| Handoff document | `config-handoff.yaml` | 15 of 45 (one run per cell) | Sonnet 5.5, high effort |

The design-document and handoff results are a first look. With one run per cell they
have no noise floor and no meaningful confidence intervals.

## Setup

Requirements: Linux or WSL, `git`, [`uv`](https://docs.astral.sh/uv/), and
Claude Code, logged in.

```bash
uv sync --extra embed
uv run pytest
```

The `embed` extra installs PyTorch from the CUDA 12.8 wheel index, which an
RTX 50-series (Blackwell) GPU needs, and `sentence-transformers`. Without the
extra, `uv sync` is enough for `ste-run` and for
`ste-analyze --skip-embeddings`.

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

`--limit N` runs only the first N jobs of the shuffled grid. `--levels 0 100`
runs only those STE levels.
`--by-repeat` finishes repeat 1 of every cell before repeat 2, so a batch
that stops early stays balanced. `--stop-at-usage 0.9` stops the batch when
the five-hour usage window of a subscription is 90% full.

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

## Follow-up: document tasks

The explanation task above produces plain prose even at 0%. Two more configs
ask for the kinds of text where dense, abstract prose shows up: a design
document and a handoff document. They use Sonnet 5.5 at high effort and write
to their own results folders.

| Config | Prompt | Results |
|---|---|---|
| `config-design-doc.yaml` | `prompts/design_doc.txt` | `results-design-doc/` |
| `config-handoff.yaml` | `prompts/handoff.txt` | `results-handoff/` |

Check first that the baseline is dense at all. This runs only the 0% level,
one run per subsystem (3 runs), and reports the dense-prose measures:

```bash
uv run ste-run --config config-design-doc.yaml --pilot --levels 0
uv run ste-analyze --config config-design-doc.yaml --skip-embeddings
```

Compare the "Dense prose" table in `results-design-doc/summary.md` with the
anchor text in `anchors/dense.txt`, which has 33% stacked noun phrases and 96
hyphenated compounds per 1,000 words. If the baseline is not dense, change
`claude.model` or `claude.effort` in the config and run the check again. Runs
of another model stay on disk and are not analyzed.

Then run the batch and the analysis:

```bash
uv run ste-run --config config-design-doc.yaml --by-repeat --stop-at-usage 0.9
uv run ste-analyze --config config-design-doc.yaml
```

The same commands work with `config-handoff.yaml`.

## Analyze the runs

```bash
# Surface metrics only. No GPU or model download.
uv run ste-analyze --skip-embeddings

# Everything, including the embedding analyses.
uv run ste-analyze
```

The analysis reads every valid run that used the configured model and effort
level, and writes to `results/`:

- `metrics.csv`: one row per run
- `aggregates_by_level.csv`, `aggregates_by_cell.csv`: n, median, quartiles,
  mean, and a bootstrap 95% CI of the mean (10,000 resamples, seeded)
- `trend_tests.csv`: Spearman correlation with the level and its bootstrap CI,
  and a Kruskal-Wallis test across levels
- `silhouette.csv`, `drift.csv`, `consistency.csv`: the embedding analyses
- `figures/`: each figure as PNG (200 dpi) and SVG
- `summary.md`: tables for the post, paired excerpts, and a reproducibility
  block
- `sentences.csv`: every prose sentence (ignored by git, like the transcripts)

How the text is prepared:

- Fenced code blocks, headings, table rows, and horizontal rules are not prose.
- Each remaining line is split into sentences with spaCy (`en_core_web_sm`).
- An inline code span counts as one word in the sentence length. It is left
  out of the readability, rare-word, hedge, and jargon metrics.
- Document embeddings use the answer without fenced code blocks. Inline code
  stays in.

Identifier retention compares a run with the inline code identifiers of the
0% runs of its subsystem. Each baseline set is built from all 0% runs but
one, and holds the identifiers that at least half of them mention. A 0% run
is scored against the set that leaves it out. Every other run gets the mean
over all the sets.

The dense-prose measures look for stacked, abstract wording:

- stacked noun phrases: the share of noun phrases with three or more nouns
  and adjectives ("durable intent ledger")
- hyphenated compounds per 1,000 words ("leader-fenced")
- nominalizations per 1,000 words: long nouns that end in -tion, -ment, -ity,
  and similar
- nouns and adjectives per verb
- the density axis: the similarity of a run's embedding to `anchors/dense.txt`
  minus its similarity to `anchors/plain.txt`. The two anchors say the same
  thing in a dense and in a plain style. The anchors are about another topic
  than the runs, so this measure is weak at the document level.

The jargon list is `jargon.txt`, one term per line. A term matches inflected
forms and ignores case. `[verb]`, `[noun]`, or `[adj]` after a term limits it
to that part of speech.

Embeddings come from the model in `embedding.model`, loaded with
`sentence-transformers` in bfloat16 on CUDA, or in float32 on the CPU with a
warning. Texts are embedded as documents, without a query instruction, and
L2-normalized. Vectors are cached in `results/cache/`, keyed by a hash of the
model name and the text.

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
