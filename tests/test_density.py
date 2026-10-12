"""Tests for the measures of dense, abstract prose."""

import numpy as np

from ste_dial.analyze import density

DENSE = (
    "A leader-fenced reconciler performs level-triggered convergence, hydrating "
    "each intent into a hermetic, owner-scoped execution unit."
)
PLAIN = "A controller picks up the request and runs it as a pod."


def should_count_hyphenated_compounds_per_thousand_words():
    metrics = density.density_metrics("The leader-fenced reconciler runs it now.")

    assert metrics["compounds_per_1k"] == 1000 / 7


def should_count_no_compounds_in_plain_prose():
    assert density.density_metrics(PLAIN)["compounds_per_1k"] == 0.0


def should_count_a_long_noun_with_a_nominal_suffix():
    metrics = density.density_metrics("The convergence of the system is slow.")

    assert metrics["nominalizations_per_1k"] == 1000 / 7


def should_not_count_a_short_word_with_a_nominal_suffix():
    metrics = density.density_metrics("The city is big.")

    assert metrics["nominalizations_per_1k"] == 0.0


def should_report_a_stacked_noun_phrase():
    metrics = density.density_metrics("The durable intent ledger stores it.")

    assert metrics["stacked_noun_phrase_share"] == 0.5


def should_report_no_stacked_noun_phrases_in_plain_prose():
    assert density.density_metrics(PLAIN)["stacked_noun_phrase_share"] == 0.0


def should_score_dense_prose_above_plain_prose_on_noun_verb_ratio():
    dense = density.density_metrics(DENSE)["noun_verb_ratio"]
    plain = density.density_metrics(PLAIN)["noun_verb_ratio"]

    assert dense > plain


def should_score_a_text_at_the_dense_anchor_above_one_at_the_plain_anchor():
    dense = np.array([1.0, 0.0])
    plain = np.array([0.0, 1.0])

    scores = density.density_axis_scores(np.array([dense, plain]), dense, plain)

    assert scores.tolist() == [1.0, -1.0]
