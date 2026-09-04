"""Tests for the synthetic benchmark data of `marker_clustering`."""

import numpy as np
import pytest

from marker_clustering import generate_cluster_markers, simulate_markers


def test_generate_cluster_markers_is_upper_triangular():
    C, labels = generate_cluster_markers(
        n_clusters=3, n_markers=12, p_intra=0.8, w_min=0.1, w_max=0.2, p_inter=0.1
    )

    assert C.shape == (12, 12)
    np.testing.assert_array_equal(np.tril(C), np.zeros_like(C))
    assert np.all(C[C > 0] >= 0.1) and np.all(C[C > 0] <= 0.2)


def test_generate_cluster_markers_balances_cluster_sizes():
    _, labels = generate_cluster_markers(3, 10, 0.5, 0.1, 0.2, 0.1)

    counts = np.bincount(labels)
    assert list(counts) == [4, 3, 3]  # 10 markers over 3 clusters
    assert np.all(np.diff(labels) >= 0), "labels are contiguous blocks"


def test_generate_cluster_markers_edges_follow_the_block_structure():
    """p_intra=1 / p_inter=0 gives exactly the within-cluster upper-triangular edges."""
    C, labels = generate_cluster_markers(2, 6, p_intra=1.0, w_min=0.1, w_max=0.2, p_inter=0.0)

    same_cluster = labels[:, None] == labels[None, :]
    expected = np.triu(same_cluster, k=1)
    np.testing.assert_array_equal(C > 0, expected)


def test_simulate_markers_shape_and_binary():
    np.random.seed(0)
    C, _ = generate_cluster_markers(2, 6, 0.5, 0.1, 0.2, 0.05)
    docs = simulate_markers(C, u=np.full(6, 0.2), n_docs=100)

    assert docs.shape == (100, 6)
    assert set(np.unique(docs)) <= {0, 1}


def test_simulate_markers_respects_unary_probabilities():
    np.random.seed(0)
    docs = simulate_markers(np.zeros((2, 2)), u=np.array([0.1, 0.9]), n_docs=5000)

    np.testing.assert_allclose(docs.mean(axis=0), [0.1, 0.9], atol=0.02)


def test_simulate_markers_dependency_increases_cooccurrence():
    """A strong C[1,0] makes marker 1 appear mostly together with marker 0."""
    np.random.seed(0)
    C = np.zeros((2, 2))
    C[1, 0] = 0.9
    docs = simulate_markers(C, u=np.array([0.5, 0.0]), n_docs=2000)

    p_1_given_0 = docs[docs[:, 0] == 1, 1].mean()
    assert p_1_given_0 == pytest.approx(0.9, abs=0.05)
    assert docs[docs[:, 0] == 0, 1].sum() == 0
