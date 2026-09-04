"""Tests for the parsing / binning helpers of `llm_judge` (no model involved)."""

import pandas as pd
import pytest

pytest.importorskip("torch", reason="torch not installed")
pytest.importorskip("transformers", reason="transformers not installed")

from llm_judge import (  # noqa: E402
    build_prompt,
    extract_json,
    scores_to_categories,
    strip_markdown_fence,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a": 1}', '{"a": 1}'),
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('```\n{"a": 1}\n```', '{"a": 1}'),
        ('  {"a": 1}  ', '{"a": 1}'),
    ],
)
def test_strip_markdown_fence(raw, expected):
    assert strip_markdown_fence(raw) == expected


def test_extract_json_from_fenced_and_surrounded_text():
    assert extract_json('```json\n{"a": 3}\n```') == {"a": 3}
    assert extract_json('Here you go: {"a": 3, "b": 4} — hope it helps') == {"a": 3, "b": 4}


def test_extract_json_returns_none_on_garbage():
    assert extract_json("no json here") is None
    assert extract_json('{"a": }') is None


def test_build_prompt_lists_every_marker():
    prompt = build_prompt([{"marker": "inflation"}, {"marker": "gdp"}])

    assert '"inflation"' in prompt and '"gdp"' in prompt
    assert "1 (simplest) to 10 (most complex)" in prompt


def test_scores_to_categories_splits_into_thirds():
    scores = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    categories = scores_to_categories(scores)

    assert list(categories) == [
        "simple", "simple", "intermediate", "intermediate", "complex", "complex",
    ]


def test_scores_to_categories_breaks_ties_by_order():
    """Equal scores must still yield balanced thirds (rank method='first')."""
    categories = scores_to_categories(pd.Series([5.0] * 6))

    assert categories.value_counts().to_dict() == {
        "simple": 2, "intermediate": 2, "complex": 2,
    }


def test_scores_to_categories_keeps_missing_scores_unlabelled():
    scores = pd.Series([1.0, None, 3.0])
    categories = scores_to_categories(scores)

    assert categories.isna()[1]
    assert categories[0] == "simple" and categories[2] == "complex"


def test_scores_to_categories_all_missing():
    assert scores_to_categories(pd.Series([None, None], dtype=float)).isna().all()
