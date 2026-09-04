"""Tests for the per-publisher decomposition."""

import numpy as np
import polars as pl

from complexity_clusters import compute_sub_lift_matrix
from complexity_clusters_publisher import (
    compute_publisher_lift_matrices,
    top_publishers_for_cluster,
)


def test_top_publishers_ranked_by_occurrence_count(occurrences):
    # rows citing m1/m2/m3: pub_a has 4 (a1 x3, a2 x1), pub_b has 4 (a2, a3 x2, a4)
    publishers = top_publishers_for_cluster(np.array(["m1", "m2", "m3"]), occurrences)
    assert sorted(publishers) == ["pub_a", "pub_b"]

    # restricted to m3, only pub_b cites it
    assert top_publishers_for_cluster(np.array(["m3"]), occurrences) == ["pub_b"]


def test_top_publishers_honours_top_n(occurrences):
    assert len(top_publishers_for_cluster(np.array(["m1", "m2"]), occurrences, top_n=1)) == 1


def test_publisher_lift_matrices_are_computed_on_the_publisher_subset(occurrences):
    markers = np.array(["m1", "m2"])
    matrices, convs = compute_publisher_lift_matrices(markers, occurrences, ["pub_a", "pub_b"])

    assert set(matrices) == {"pub_a", "pub_b"}
    for publisher in ("pub_a", "pub_b"):
        expected, expected_conv = compute_sub_lift_matrix(
            markers, occurrences.filter(pl.col("publisher_label") == publisher)
        )
        np.testing.assert_allclose(matrices[publisher], expected)
        assert convs[publisher] == expected_conv


def test_publisher_lift_uses_only_that_publisher_articles(occurrences):
    """pub_a cites m1 and m2 together in a1 and a2 out of its 3 articles."""
    matrices, convs = compute_publisher_lift_matrices(
        np.array(["m1", "m2"]), occurrences, ["pub_a"]
    )
    lift = matrices["pub_a"]

    # pub_a rows: a1 (m1,m2), a2 (m1), a5 (other) => N = 3, n(m1) = 2, n(m2) = 1
    assert lift[convs["pub_a"]["m1"], convs["pub_a"]["m1"]] ** -1 == 2 / 3
    assert lift[convs["pub_a"]["m2"], convs["pub_a"]["m2"]] ** -1 == 1 / 3
