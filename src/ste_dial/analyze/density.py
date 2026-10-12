"""Measures of dense, abstract prose: stacked noun phrases and compounds."""

import re

import numpy as np

from ste_dial.analyze.preprocess import load_nlp

_COMPOUND = re.compile(r"\b[A-Za-z]+(?:-[A-Za-z]+)+\b")
_NOMINAL_SUFFIX = re.compile(r"(tion|sion|ment|ity|ence|ance|ness)s?$")
_CONTENT_TAGS = frozenset({"NOUN", "PROPN", "ADJ"})
MIN_NOMINALIZATION_LENGTH = 7
STACK_SIZE = 3


def density_metrics(prose: str) -> dict:
    """Computes the density measures of a prose text.

    Args:
        prose: Prose without code. Code identifiers must already be removed.

    Returns:
        compounds_per_1k: Hyphenated compounds ("leader-fenced") per 1,000
            words.
        nominalizations_per_1k: Nouns of seven letters or more that end in
            -tion, -sion, -ment, -ity, -ence, -ance, or -ness, per 1,000
            words.
        stacked_noun_phrase_share: The share of noun phrases with three or
            more nouns and adjectives ("durable intent ledger").
        noun_verb_ratio: Nouns and adjectives divided by verbs.
    """
    parsed = load_nlp()(prose)
    words = [token for token in parsed if token.is_alpha]
    n_words = len(words)
    nan = float("nan")
    if n_words == 0:
        return dict(
            compounds_per_1k=nan,
            nominalizations_per_1k=nan,
            stacked_noun_phrase_share=nan,
            noun_verb_ratio=nan,
        )
    nominalizations = [
        token
        for token in words
        if token.pos_ == "NOUN"
        and len(token.text) >= MIN_NOMINALIZATION_LENGTH
        and _NOMINAL_SUFFIX.search(token.text.lower())
    ]
    chunk_sizes = [
        sum(token.pos_ in _CONTENT_TAGS for token in chunk)
        for chunk in parsed.noun_chunks
    ]
    stacked = sum(size >= STACK_SIZE for size in chunk_sizes)
    content = sum(token.pos_ in _CONTENT_TAGS for token in words)
    verbs = sum(token.pos_ == "VERB" for token in words)
    return dict(
        compounds_per_1k=1000.0 * len(_COMPOUND.findall(prose)) / n_words,
        nominalizations_per_1k=1000.0 * len(nominalizations) / n_words,
        stacked_noun_phrase_share=stacked / len(chunk_sizes) if chunk_sizes else nan,
        noun_verb_ratio=content / verbs if verbs else nan,
    )


def density_axis_scores(
    vectors: np.ndarray, dense: np.ndarray, plain: np.ndarray
) -> np.ndarray:
    """Places texts between a plain and a dense text that say the same thing.

    Args:
        vectors: One unit vector per text.
        dense: Unit vector of the dense anchor text.
        plain: Unit vector of the plain anchor text.

    Returns:
        One score per text: its cosine similarity to the dense anchor minus
        its similarity to the plain anchor. Higher means closer to the dense
        style.
    """
    return vectors @ dense - vectors @ plain
