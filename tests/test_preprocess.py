"""Tests for separating prose from code."""

from ste_dial.analyze import preprocess

ANSWER = """# Coders

A `Coder` turns elements into bytes. The bytes are written by the runner.

```python
class Coder:
    def encode(self, value): ...
```

- First item without a full stop
- See `coders.py:141` and call `encode()`.

| Class | Role |
|---|---|
| `CoderImpl` | does the work |
"""


def should_remove_fenced_code_blocks():
    text = preprocess.strip_code_blocks("Before.\n```python\nx = 1\n```\nAfter.")

    assert text == "Before.\n\nAfter."


def should_remove_a_code_block_that_is_never_closed():
    text = preprocess.strip_code_blocks("Before.\n```python\nx = 1\n")

    assert text == "Before.\n"


def should_extract_inline_code_spans_as_identifiers():
    identifiers = preprocess.extract_identifiers("Use `Coder` and `CoderImpl`.")

    assert identifiers == ["Coder", "CoderImpl"]


def should_not_extract_identifiers_from_fenced_code_blocks():
    identifiers = preprocess.extract_identifiers("```\n`hidden`\n```\nUse `shown`.")

    assert identifiers == ["shown"]


def should_drop_line_reference_and_call_suffix_from_identifiers():
    identifiers = preprocess.extract_identifiers("See `coders.py:141` and `encode()`.")

    assert identifiers == ["coders.py", "encode"]


def should_leave_headings_and_table_rows_out_of_the_prose():
    lines = preprocess.prose_lines(ANSWER)

    assert lines == [
        "A `Coder` turns elements into bytes. The bytes are written by the runner.",
        "First item without a full stop",
        "See `coders.py:141` and call `encode()`.",
    ]


def should_remove_emphasis_marks_and_link_targets_from_prose():
    lines = preprocess.prose_lines(
        "The **hot path** uses [Cython](https://cython.org)."
    )

    assert lines == ["The hot path uses Cython."]


def should_keep_underscores_inside_inline_code():
    lines = preprocess.prose_lines("Call `__init__` and `_private_name` first.")

    assert lines == ["Call `__init__` and `_private_name` first."]


def should_split_two_sentences_on_one_line():
    document = preprocess.parse_document(
        "A coder turns elements into bytes. The runner reads them."
    )

    assert [s.text for s in document.sentences] == [
        "A coder turns elements into bytes.",
        "The runner reads them.",
    ]


def should_treat_each_list_item_as_its_own_sentence():
    document = preprocess.parse_document("- First item here\n- Second item here\n")

    assert [s.text for s in document.sentences] == [
        "First item here",
        "Second item here",
    ]


def should_count_an_inline_code_span_as_one_word():
    document = preprocess.parse_document("Call `registry.get_coder(int, str)` first.")

    assert document.sentences[0].n_words == 3


def should_restore_inline_code_in_the_sentence_text():
    document = preprocess.parse_document("A `Coder` makes a `CoderImpl`.")

    assert document.sentences[0].text == "A `Coder` makes a `CoderImpl`."


def should_leave_identifiers_out_of_the_plain_prose():
    document = preprocess.parse_document("A `Coder` makes bytes.")

    assert document.plain_prose == "A makes bytes."


def should_mark_a_passive_sentence():
    document = preprocess.parse_document("The bytes are written by the runner.")

    assert document.sentences[0].is_passive


def should_not_mark_an_active_sentence():
    document = preprocess.parse_document("The runner writes the bytes.")

    assert not document.sentences[0].is_passive
