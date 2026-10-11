"""Loads the experiment configuration and renders the prompts."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Subsystem:
    """A part of the target codebase that the agent is asked to explain."""

    id: str
    name: str
    path: str


@dataclass(frozen=True)
class ExpectedInit:
    """What the init event of an isolated run must report."""

    permission_mode: str
    output_style: str
    plugins: tuple[str, ...]
    skills: tuple[str, ...]
    agents: tuple[str, ...]
    slash_commands: tuple[str, ...]


@dataclass(frozen=True)
class ClaudeConfig:
    """Parameters of the agent under test."""

    model: str
    effort: str
    max_turns: int
    max_budget_usd: float
    timeout_s: int
    tools: tuple[str, ...]
    allowed_tools: tuple[str, ...]
    disallowed_tools: tuple[str, ...]
    expected_init: ExpectedInit


@dataclass(frozen=True)
class Config:
    """All experiment parameters, with paths resolved to absolute paths."""

    root: Path
    beam_repo_url: str
    beam_sha: str
    beam_workdir: Path
    subsystems: tuple[Subsystem, ...]
    levels: tuple[int, ...]
    repeats: int
    seed: int
    claude: ClaudeConfig
    embedding_model: str
    embedding_max_tokens: int
    results_dir: Path
    prompt_template: str
    style_sentences: dict[int, str]
    jargon_file: Path


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def load_config(path: Path) -> Config:
    """Reads config.yaml and the prompt files it points to.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        The parsed configuration.

    Raises:
        ValueError: If a level has no style sentence.
    """
    path = path.resolve()
    root = path.parent
    raw = yaml.safe_load(path.read_text())
    claude = raw["claude"]
    expected = claude["expected_init"]
    template = _resolve(root, raw["prompts"]["template"]).read_text()
    sentences = yaml.safe_load(_resolve(root, raw["prompts"]["levels"]).read_text())
    levels = tuple(int(level) for level in raw["levels"])
    missing = [level for level in levels if level not in sentences]
    if missing:
        raise ValueError(f"No style sentence for levels {missing}")
    return Config(
        root=root,
        beam_repo_url=raw["beam"]["repo_url"],
        beam_sha=raw["beam"]["sha"],
        beam_workdir=_resolve(root, raw["beam"]["workdir"]),
        subsystems=tuple(Subsystem(**item) for item in raw["subsystems"]),
        levels=levels,
        repeats=int(raw["repeats"]),
        seed=int(raw["seed"]),
        claude=ClaudeConfig(
            model=claude["model"],
            effort=claude["effort"],
            max_turns=int(claude["max_turns"]),
            max_budget_usd=float(claude["max_budget_usd"]),
            timeout_s=int(claude["timeout_s"]),
            tools=tuple(claude["tools"]),
            allowed_tools=tuple(claude["allowed_tools"]),
            disallowed_tools=tuple(claude["disallowed_tools"]),
            expected_init=ExpectedInit(
                permission_mode=expected["permission_mode"],
                output_style=expected["output_style"],
                plugins=tuple(expected["plugins"]),
                skills=tuple(expected["skills"]),
                agents=tuple(expected["agents"]),
                slash_commands=tuple(expected["slash_commands"]),
            ),
        ),
        embedding_model=raw["embedding"]["model"],
        embedding_max_tokens=int(raw["embedding"]["max_tokens"]),
        results_dir=_resolve(root, raw["results_dir"]),
        prompt_template=template,
        style_sentences={int(k): str(v) for k, v in sentences.items()},
        jargon_file=_resolve(root, raw["jargon_file"]),
    )


def render_prompt(
    template: str, style_sentences: dict[int, str], subsystem: Subsystem, level: int
) -> str:
    """Renders the prompt for one condition.

    The style sentence is the only text that differs between levels. Trailing
    whitespace is removed, so level 0 is exactly the base prompt.

    Args:
        template: Template with {name}, {path} and {style_sentence}.
        style_sentences: Style sentence per level; may contain {level}.
        subsystem: The subsystem to explain.
        level: The STE level, from 0 to 100.

    Returns:
        The fully rendered prompt.
    """
    sentence = style_sentences[level].format(level=level)
    rendered = template.format(
        name=subsystem.name, path=subsystem.path, style_sentence=sentence
    )
    return rendered.rstrip()
