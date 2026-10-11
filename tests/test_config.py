"""Tests for configuration loading and prompt rendering."""

from pathlib import Path

from ste_dial.config import Subsystem, load_config, render_prompt

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = "Explain {name} in apache_beam/{path}.\n{style_sentence}\n"
SENTENCES = {0: "", 50: "Write about {level}% STE.", 100: "Write in STE."}
CODERS = Subsystem(id="coders", name="Coders", path="coders")


def should_render_base_prompt_without_style_sentence_at_level_zero():
    prompt = render_prompt(TEMPLATE, SENTENCES, CODERS, 0)

    assert prompt == "Explain Coders in apache_beam/coders."


def should_append_style_sentence_with_level_number():
    prompt = render_prompt(TEMPLATE, SENTENCES, CODERS, 50)

    assert prompt == "Explain Coders in apache_beam/coders.\nWrite about 50% STE."


def should_differ_from_level_zero_only_by_the_style_sentence():
    base = render_prompt(TEMPLATE, SENTENCES, CODERS, 0)

    styled = render_prompt(TEMPLATE, SENTENCES, CODERS, 100)

    assert styled == base + "\nWrite in STE."


def should_load_the_shipped_config_with_default_grid():
    config = load_config(REPO_ROOT / "config.yaml")

    assert [s.id for s in config.subsystems] == [
        "runinference",
        "coders",
        "fn_api_runner",
    ]
    assert config.levels == (0, 25, 50, 80, 100)
    assert config.style_sentences[0] == ""


def should_resolve_relative_paths_against_the_config_file():
    config = load_config(REPO_ROOT / "config.yaml")

    assert config.results_dir == REPO_ROOT / "results"
