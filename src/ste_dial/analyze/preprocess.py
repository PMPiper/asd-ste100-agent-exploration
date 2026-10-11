"""Separates the prose of a Markdown explanation from its code."""

import functools
import re
from dataclasses import dataclass

import spacy
from spacy.language import Language

# Stands in for one inline code span while spaCy parses the sentence, so that
# an identifier counts as one word and its punctuation does not split it.
PLACEHOLDER = "CODEID"

_FENCE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.MULTILINE | re.DOTALL)
_UNCLOSED_FENCE = re.compile(r"^(```|~~~).*", re.MULTILINE | re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s")
_TABLE_ROW = re.compile(r"^\s*\|")
_RULE = re.compile(r"^\s*([-*_]\s*){3,}$")
_LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_QUOTE_MARKER = re.compile(r"^\s*>+\s?")
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_EMPHASIS = re.compile(r"(\*\*|__|\*|(?<!\w)_|_(?!\w))")
_LINE_REFERENCE = re.compile(r":\d+(-\d+)?$")
_CALL_SUFFIX = re.compile(r"\(\)$")


@dataclass(frozen=True)
class Word:
    """One word of a prose sentence."""

    text: str
    lemma: str
    pos: str


@dataclass(frozen=True)
class Sentence:
    """One prose sentence.

    Attributes:
        text: The sentence with its inline code restored.
        words: The words, without punctuation and without code identifiers.
        n_words: Number of words, counting each code identifier as one word.
        is_passive: Whether the sentence has a passive subject or auxiliary.
    """

    text: str
    words: tuple[Word, ...]
    n_words: int
    is_passive: bool


@dataclass(frozen=True)
class Document:
    """The parts of one explanation.

    Attributes:
        sentences: The prose sentences, in order.
        identifiers: The inline code spans, normalized, in order.
        plain_prose: The prose without code identifiers, for readability
            formulas.
        text_without_code_blocks: The Markdown without fenced code blocks.
        raw: The original Markdown.
    """

    sentences: tuple[Sentence, ...]
    identifiers: tuple[str, ...]
    plain_prose: str
    text_without_code_blocks: str
    raw: str


@functools.cache
def load_nlp() -> Language:
    """Returns the spaCy pipeline used for sentence splitting and tagging."""
    return spacy.load("en_core_web_sm")


def strip_code_blocks(markdown: str) -> str:
    """Removes fenced code blocks, including one that is never closed."""
    return _UNCLOSED_FENCE.sub("", _FENCE.sub("", markdown))


def normalize_identifier(span: str) -> str:
    """Returns an inline code span without a line reference or call suffix.

    `coders.py:141` becomes `coders.py`, and `encode()` becomes `encode`.
    """
    return _CALL_SUFFIX.sub("", _LINE_REFERENCE.sub("", span.strip()))


def extract_identifiers(markdown: str) -> list[str]:
    """Returns the normalized inline code spans outside fenced code blocks."""
    spans = _INLINE_CODE.findall(strip_code_blocks(markdown))
    return [normalize_identifier(span) for span in spans]


def prose_lines(markdown: str) -> list[str]:
    """Returns the prose lines with Markdown markup removed.

    Fenced code blocks, headings, table rows, and horizontal rules are left
    out. List and quote markers, emphasis marks, and link targets are removed.
    Inline code stays, in its backticks.
    """
    lines = []
    for line in strip_code_blocks(markdown).splitlines():
        if (
            not line.strip()
            or _HEADING.match(line)
            or _TABLE_ROW.match(line)
            or _RULE.match(line)
        ):
            continue
        line = _LIST_MARKER.sub("", _QUOTE_MARKER.sub("", line))
        parts = _INLINE_CODE.split(line)
        # Odd parts are inline code. Only the text between them is cleaned.
        parts[0::2] = [_EMPHASIS.sub("", _LINK.sub(r"\1", p)) for p in parts[0::2]]
        parts[1::2] = [f"`{code}`" for code in parts[1::2]]
        lines.append("".join(parts).strip())
    return [line for line in lines if line]


def parse_document(markdown: str) -> Document:
    """Splits an explanation into prose sentences and code identifiers.

    Each prose line is parsed on its own, so a list item without a full stop
    does not merge with the next item.

    Args:
        markdown: The final answer of one run.

    Returns:
        The parsed document.
    """
    nlp = load_nlp()
    sentences: list[Sentence] = []
    plain_parts: list[str] = []
    for line in prose_lines(markdown):
        spans = iter(_INLINE_CODE.findall(line))
        parsed = nlp(_INLINE_CODE.sub(PLACEHOLDER, line))
        for sent in parsed.sents:
            tokens = [t for t in sent if not t.is_punct and not t.is_space]
            if not tokens:
                continue
            text = sent.text.strip()
            for _ in range(text.count(PLACEHOLDER)):
                text = text.replace(PLACEHOLDER, f"`{next(spans, '')}`", 1)
            words = tuple(
                Word(t.text, t.lemma_.lower(), t.pos_)
                for t in tokens
                if PLACEHOLDER not in t.text
            )
            sentences.append(
                Sentence(
                    text=text,
                    words=words,
                    n_words=len(tokens),
                    is_passive=any(t.dep_ in ("nsubjpass", "auxpass") for t in sent),
                )
            )
            plain = re.sub(r"\s+", " ", sent.text.replace(PLACEHOLDER, "")).strip()
            plain_parts.append(plain)
    return Document(
        sentences=tuple(sentences),
        identifiers=tuple(extract_identifiers(markdown)),
        plain_prose=" ".join(plain_parts),
        text_without_code_blocks=strip_code_blocks(markdown).strip(),
        raw=markdown,
    )
