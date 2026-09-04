"""`peter_clark_scm` recomputes complexities on its own; check it against the pipeline."""

import numpy as np
import polars as pl
import pytest

from complexity_clusters import (
    compute_cocitation_probability_matrix,
    compute_lift_matrix,
    get_complexity_fast,
)

causallearn = pytest.importorskip("causallearn", reason="causal-learn not installed")
from peter_clark_scm import compute_local_complexities  # noqa: E402


@pytest.fixture
def cluster_df() -> pl.DataFrame:
    """Every article cites at least two distinct markers and no duplicate rows —
    the conditions under which both implementations must agree (compute_local_complexities
    skips single-marker articles and does not deduplicate markers within an article)."""
    rows = [
        ("a1", "m1"), ("a1", "m2"),
        ("a2", "m1"), ("a2", "m2"), ("a2", "m3"),
        ("a3", "m1"), ("a3", "m3"),
        ("a4", "m2"), ("a4", "m3"),
    ]
    return pl.DataFrame(rows, schema=["id", "marker"], orient="row")


def test_local_complexities_match_get_complexity_fast(cluster_df):
    markers = ["m1", "m2", "m3"]
    conv = {m: i for i, m in enumerate(markers)}
    lift = compute_lift_matrix(
        compute_cocitation_probability_matrix(np.array(markers), cluster_df, conv)
    )
    expected = {m: get_complexity_fast(lift, conv, m) for m in markers}

    local = compute_local_complexities(cluster_df, markers)

    assert set(local) == set(expected)
    for marker in markers:
        assert local[marker] == pytest.approx(expected[marker])


def test_local_complexities_ignore_articles_with_a_single_marker(cluster_df):
    """Documented behaviour: `if len(idxs) > 1` drops such articles from the counts,
    so adding one only changes the result through the article count N."""
    padded = pl.concat([cluster_df, pl.DataFrame({"id": ["a5"], "marker": ["m1"]})])

    base = compute_local_complexities(cluster_df, ["m1", "m2", "m3"])
    with_single = compute_local_complexities(padded, ["m1", "m2", "m3"])

    # N goes from 4 to 5 and every count is unchanged: lifts scale by 5/4
    for marker in base:
        assert with_single[marker] == pytest.approx(base[marker] * 5 / 4)
