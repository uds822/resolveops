import pytest

from resolveops.ingestion.identifiers import make_stable_id


def test_same_inputs_produce_same_id():
    first = make_stable_id(
        "doc",
        "markdown",
        "data/raw/guide.md",
    )

    second = make_stable_id(
        "doc",
        "markdown",
        "data/raw/guide.md",
    )

    assert first == second
    assert first.startswith("doc_")


def test_different_inputs_produce_different_ids():
    first = make_stable_id(
        "doc",
        "markdown",
        "data/raw/guide.md",
    )

    second = make_stable_id(
        "doc",
        "markdown",
        "data/raw/another-guide.md",
    )

    assert first != second


def test_input_boundaries_are_preserved():
    first = make_stable_id("block", "ab", "c")
    second = make_stable_id("block", "a", "bc")

    assert first != second


def test_empty_namespace_is_rejected():
    with pytest.raises(
        ValueError,
        match="namespace cannot be empty",
    ):
        make_stable_id("", "guide.md")


def test_missing_identity_parts_are_rejected():
    with pytest.raises(
        ValueError,
        match="at least one identity part is required",
    ):
        make_stable_id("doc")