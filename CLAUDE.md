# ste-dial

Experiment harness: runs Claude Code headless over a grid of STE levels and
Beam subsystems, then analyzes the explanations. See `README.md`.

## Commands
- `uv sync` — install from the lockfile
- `uv run pytest` — run the tests
- `uv run ste-run --dry-run` — print prompts and commands, run nothing
- `uv run ste-run --pilot` / `uv run ste-run` — live runs; these use quota

## Structure
- `config.yaml` — every experiment parameter
- `prompts/` — prompt template and the style sentence per level
- `src/ste_dial/beam.py` — Beam checkout: fetch, scrub, scan, reset
- `src/ste_dial/claude_cli.py` — command builder, stream parser, init checks
- `src/ste_dial/run.py` — `ste-run` batch runner
- `src/ste_dial/analyze/` — `ste-analyze`

## Conventions
- Never change anything in the Beam checkout except the `.agent/` removal
  commit that `beam.prepare` makes.
- `results/raw/` is ignored by git. Do not commit transcripts.
- Do not weaken an isolation check to make a run pass. Stop and ask.
- Tests are named `should_<behavior>`; pytest collects that prefix.
