"""Shared fixtures: a tiny occurrence table with hand-computable statistics."""

import sys
from pathlib import Path

import polars as pl
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# Articles a1..a5, markers m1/m2/m3 (+ an unrelated one), with a marker repeated
# inside an article and an article citing none of m1/m2/m3:
#
#   a1: m1, m1, m2   a2: m1, m2   a3: m1, m3   a4: m2   a5: other
#
# => N = 5, n(m1) = n(m2) = 3, n(m3) = 1, n(m1,m2) = 2, n(m1,m3) = 1, n(m2,m3) = 0
OCCURRENCES = [
    ("a1", "m1", "pub_a", "economie"),
    ("a1", "m1", "pub_a", "economie"),
    ("a1", "m2", "pub_a", "economie"),
    ("a2", "m1", "pub_a", "economie"),
    ("a2", "m2", "pub_b", "economie"),
    ("a3", "m1", "pub_b", "sport"),
    ("a3", "m3", "pub_b", "sport"),
    ("a4", "m2", "pub_b", "sante"),
    ("a5", "other", "pub_a", "economie"),
]

MARKERS = ["m1", "m2", "m3"]


@pytest.fixture
def occurrences() -> pl.DataFrame:
    return pl.DataFrame(
        OCCURRENCES,
        schema=["id", "marker", "publisher_label", "journal_theme"],
        orient="row",
    )


@pytest.fixture
def markers() -> list:
    return list(MARKERS)


@pytest.fixture
def conv() -> dict:
    return {m: i for i, m in enumerate(MARKERS)}
