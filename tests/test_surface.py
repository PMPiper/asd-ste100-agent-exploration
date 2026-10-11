"""Tests for the surface metrics."""

import pytest

from ste_dial.analyze import surface
from ste_dial.analyze.preprocess import parse_document

TERMS = surface.parse_jargon(
    ["# comment", "", "wire up", "under the hood", "surface [verb]", "load-bearing"]
)


def _hits(text: str) -> list[str]:
    return surface.jargon_hits(parse_document(text).sentences[0], TERMS)


def should_skip_comments_and_blank_lines_in_the_jargon_list():
    assert [term.label for term in TERMS] == [
        "wire up",
        "under the hood",
        "surface [verb]",
        "load-bearing",
    ]


def should_match_an_inflected_two_word_phrase():
    assert _hits("We wired up the retry path.") == ["wire up"]


def should_match_a_phrase_regardless_of_case():
    assert _hits("Under The Hood, it calls the runner.") == ["under the hood"]


def should_match_a_hyphenated_term():
    assert _hits("This check is load-bearing.") == ["load-bearing"]


def should_match_a_verb_only_term_used_as_a_verb():
    assert _hits("The errors surface in the logs.") == ["surface [verb]"]


def should_not_match_a_verb_only_term_used_as_a_noun():
    assert _hits("The surface of the table is flat.") == []


def should_not_match_a_phrase_whose_words_are_not_adjacent():
    assert _hits("Pull the wire slowly up the wall.") == []


def should_count_words_and_sentences():
    metrics = surface.surface_metrics(
        parse_document("The runner writes bytes. The coder reads them back."), TERMS
    )

    assert metrics["sentences"] == 2
    assert metrics["words"] == 9
    assert metrics["words_per_sentence_mean"] == 4.5


def should_report_share_of_sentences_over_20_words():
    long_sentence = "The runner " + "really " * 19 + "writes bytes."
    document = parse_document(f"{long_sentence}\n\nThe coder reads them.")

    metrics = surface.surface_metrics(document, TERMS)

    assert metrics["share_sentences_over_20"] == 0.5
    assert metrics["share_sentences_over_25"] == 0.0


def should_report_passive_rate_as_share_of_sentences():
    document = parse_document(
        "The bytes are written by the runner. The coder reads them."
    )

    metrics = surface.surface_metrics(document, TERMS)

    assert metrics["passive_rate"] == 0.5


def should_report_jargon_per_thousand_words_and_matched_terms():
    document = parse_document("We wired up the retry path today.")

    metrics = surface.surface_metrics(document, TERMS)

    assert metrics["jargon_terms"] == {"wire up": 1}
    assert metrics["jargon_per_1k"] == pytest.approx(1000 / 7)


def should_report_hedges_per_thousand_words():
    document = parse_document("This might fail and could retry.")

    metrics = surface.surface_metrics(document, TERMS)

    assert metrics["hedges_per_1k"] == pytest.approx(2000 / 6)


def should_count_a_rare_word_in_the_rare_word_share():
    document = parse_document("The cat ate the idempotent fish.")

    metrics = surface.surface_metrics(document, TERMS)

    assert metrics["rare_word_share"] == pytest.approx(1 / 6)


def should_not_count_code_identifiers_as_rare_words():
    document = parse_document("The cat ate the `xqzvkw` fish.")

    metrics = surface.surface_metrics(document, TERMS)

    assert metrics["rare_word_share"] == 0.0


def should_keep_identifiers_that_half_of_the_baseline_runs_mention():
    baseline = surface.baseline_identifiers(
        [{"Coder", "CoderImpl"}, {"Coder", "FastCoder"}, {"Coder", "CoderImpl"}]
    )

    assert baseline == {"Coder", "CoderImpl"}


def should_report_fraction_of_baseline_identifiers_mentioned():
    retention = surface.identifier_retention(
        "A `Coder` makes bytes.", {"Coder", "CoderImpl"}
    )

    assert retention == 0.5


def should_not_count_an_identifier_inside_a_longer_name():
    retention = surface.identifier_retention("Use `FastCoderImpl`.", {"Coder"})

    assert retention == 0.0


def should_count_an_identifier_used_as_an_attribute():
    retention = surface.identifier_retention("Call `Coder.encode`.", {"Coder"})

    assert retention == 1.0
