"""Surface metrics of one explanation: length, voice, jargon, readability."""

import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import textstat
from wordfreq import zipf_frequency

from ste_dial.analyze.preprocess import Document, Sentence, load_nlp

HEDGES = frozenset(
    {"might", "may", "could", "should", "likely", "generally", "typically"}
)
RARE_ZIPF = 3.0
_POS_TAG = re.compile(r"^(?P<term>.*?)\s*\[(?P<pos>verb|noun|adj)\]$", re.IGNORECASE)
_POS_NAMES = {"verb": ("VERB", "AUX"), "noun": ("NOUN", "PROPN"), "adj": ("ADJ",)}


@dataclass(frozen=True)
class JargonTerm:
    """One entry of the jargon list.

    Attributes:
        label: The entry as written in the file.
        tokens: The lower-case words of the term.
        pos: Allowed part-of-speech tags of the first word, or empty for any.
    """

    label: str
    tokens: tuple[str, ...]
    pos: tuple[str, ...] = ()


def parse_jargon(lines: list[str]) -> list[JargonTerm]:
    """Parses jargon list lines.

    Blank lines and lines that start with "#" are skipped. A term can end
    with [verb], [noun], or [adj] to match only that part of speech.
    """
    tokenizer = load_nlp().tokenizer
    terms = []
    for line in lines:
        label = line.strip()
        if not label or label.startswith("#"):
            continue
        tagged = _POS_TAG.match(label)
        text = tagged["term"] if tagged else label
        pos = _POS_NAMES[tagged["pos"].lower()] if tagged else ()
        tokens = tuple(
            t.text.lower() for t in tokenizer(text) if not t.is_punct and not t.is_space
        )
        terms.append(JargonTerm(label, tokens, pos))
    return terms


def load_jargon(path: Path) -> list[JargonTerm]:
    """Reads the jargon list from a file."""
    return parse_jargon(path.read_text().splitlines())


def jargon_hits(sentence: Sentence, terms: list[JargonTerm]) -> list[str]:
    """Returns the label of every jargon term occurrence in a sentence.

    A term matches a run of consecutive words when each word equals the term
    word in its written form or its lemma, ignoring case.
    """
    words = sentence.words
    hits = []
    for term in terms:
        size = len(term.tokens)
        for start in range(len(words) - size + 1):
            window = words[start : start + size]
            if term.pos and window[0].pos not in term.pos:
                continue
            if all(
                token in (word.text.lower(), word.lemma)
                for token, word in zip(term.tokens, window)
            ):
                hits.append(term.label)
    return hits


def _per_thousand(count: int, n_words: int) -> float:
    return 1000.0 * count / n_words if n_words else float("nan")


def surface_metrics(document: Document, terms: list[JargonTerm]) -> dict:
    """Computes the surface metrics of one explanation.

    Args:
        document: The parsed explanation.
        terms: The jargon list.

    Returns:
        The metric values by column name. `jargon_terms` maps each matched
        term to its count.
    """
    sentences = document.sentences
    lengths = [s.n_words for s in sentences]
    n_sentences = len(sentences)
    n_words = sum(lengths)
    words = [word for s in sentences for word in s.words]
    alphabetic = [w.text.lower() for w in words if w.text.isalpha()]
    rare = [w for w in alphabetic if zipf_frequency(w, "en") < RARE_ZIPF]
    hedges = [w for w in alphabetic if w in HEDGES]
    jargon = Counter(hit for s in sentences for hit in jargon_hits(s, terms))

    def share(count: int, total: int) -> float:
        return count / total if total else float("nan")

    return {
        "words": n_words,
        "sentences": n_sentences,
        "words_per_sentence_mean": (
            statistics.fmean(lengths) if lengths else float("nan")
        ),
        "words_per_sentence_median": (
            statistics.median(lengths) if lengths else float("nan")
        ),
        "share_sentences_over_20": share(sum(n > 20 for n in lengths), n_sentences),
        "share_sentences_over_25": share(sum(n > 25 for n in lengths), n_sentences),
        "passive_rate": share(sum(s.is_passive for s in sentences), n_sentences),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(document.plain_prose),
        "flesch_reading_ease": textstat.flesch_reading_ease(document.plain_prose),
        "jargon_per_1k": _per_thousand(sum(jargon.values()), n_words),
        "jargon_terms": dict(jargon),
        "rare_word_share": share(len(rare), len(alphabetic)),
        "hedges_per_1k": _per_thousand(len(hedges), n_words),
        "identifiers": len(set(document.identifiers)),
    }


def baseline_identifiers(identifier_sets: list[set[str]]) -> set[str]:
    """Returns the identifiers that at least half of the given runs mention."""
    counts = Counter(name for names in identifier_sets for name in names)
    return {name for name, n in counts.items() if 2 * n >= len(identifier_sets)}


def identifier_retention(text: str, baseline: set[str]) -> float:
    """Returns the fraction of baseline identifiers that a text mentions.

    An identifier is mentioned if it appears anywhere in the text, code
    blocks included, and is not part of a longer name.
    """
    if not baseline:
        return float("nan")
    found = [
        name
        for name in baseline
        if re.search(rf"(?<![\w]){re.escape(name)}(?![\w])", text)
    ]
    return len(found) / len(baseline)
