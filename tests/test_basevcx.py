"""Tests for the synthetic generators and the simulation of `basevcx`."""

import networkx as nx
import numpy as np
import pytest

import basevcx as bv


N = 12

# name -> (generator, is_dag). Cliques and dense random matrices are cyclic by design;
# note that some generators return C and others C.T, so orientation is not asserted.
GENERATORS = {
    "chain": (lambda: bv.gen_C_chain(N), True),
    "tree": (lambda: bv.gen_C_tree(N), True),
    "cliques": (lambda: bv.gen_C_cliques(N, [4, 8]), False),
    "hierarchical": (lambda: bv.gen_C_hierarchical(N, levels=3), True),
    "random": (lambda: bv.gen_C_random(N), False),
    "dag": (lambda: bv.random_dag(N), True),
    "fractal": (lambda: bv.gen_C_fractal(N), True),
    "funnel": (lambda: bv.gen_C_funnel(N), True),
    "skip_hierarchical": (lambda: bv.gen_C_skip_hierarchical(N), True),
    "dense_progressive": (lambda: bv.gen_C_dense_progressive(N), True),
    "mostly_full": (lambda: bv.gen_C_mostly_full(N), True),
    "croissante_rang": (lambda: bv.gen_C_croissante_rang(N, bv.max_strength), True),
}


@pytest.mark.parametrize("name", list(GENERATORS))
def test_generators_shape_and_no_self_dependency(name):
    np.random.seed(0)
    C = GENERATORS[name][0]()

    assert C.shape == (N, N)
    assert np.all(np.diag(C) == 0), "a marker must not depend on itself"
    assert np.all(C >= 0)


@pytest.mark.parametrize("name", [n for n, (_, dag) in GENERATORS.items() if dag])
def test_generators_are_acyclic(name):
    """No feedback loop: the dependency graph must stay a DAG."""
    np.random.seed(0)
    C = GENERATORS[name][0]()

    graph = nx.DiGraph((j, i) for i, j in zip(*np.nonzero(C)))
    assert nx.is_directed_acyclic_graph(graph)


def test_gen_C_chain_links_consecutive_markers():
    C = bv.gen_C_chain(4, strength=0.3)
    np.testing.assert_allclose(np.diag(C, k=-1), [0.3, 0.3, 0.3])
    assert C.sum() == pytest.approx(0.9)


def test_gen_C_cliques_is_block_structured():
    C = bv.gen_C_cliques(6, [3, 3], internal_strength=0.25, cross_strength=0.0)
    for block in (slice(0, 3), slice(3, 6)):
        block_values = C[block, block]
        assert np.all(block_values[~np.eye(3, dtype=bool)] == 0.25)
    assert C[0, 3] == 0.0 and C[3, 0] == 0.0


def test_gen_C_random_density():
    np.random.seed(0)
    C = bv.gen_C_random(20, density=0.1, strength_range=(0.1, 0.3))

    assert int(np.count_nonzero(C)) <= int(0.1 * 20 * 20)  # diagonal draws are dropped
    assert np.all(C[C > 0] >= 0.1) and np.all(C[C > 0] <= 0.3)


def test_max_strength_decreases_with_rank():
    values = [bv.max_strength(r) for r in range(5)]
    assert values[0] == pytest.approx(2.0)
    assert all(a > b for a, b in zip(values, values[1:]))


def test_compute_depth_in_dag_on_a_chain():
    C = bv.gen_C_chain(5)
    np.testing.assert_array_equal(bv.compute_depth_in_dag(C), [0, 1, 2, 3, 4])


def test_compute_depth_in_dag_takes_the_shortest_path():
    """NB: the docstring says 'longest path' but the implementation uses np.min."""
    C = np.zeros((4, 4))
    C[1, 0] = 0.5  # 0 -> 1
    C[2, 1] = 0.5  # 1 -> 2
    C[3, 0] = 0.5  # 0 -> 3  (depth 1)
    C[3, 2] = 0.5  # 2 -> 3  (depth 3 through the long branch)

    np.testing.assert_array_equal(bv.compute_depth_in_dag(C), [0, 1, 2, 1])


def test_simulate_markers_shape_and_binary():
    np.random.seed(0)
    C = bv.gen_C_chain(4)
    markers = bv.simulate_markers(C, u=np.full(4, 0.3), n_docs=50)

    assert markers.shape == (50, 4)
    assert set(np.unique(markers)) <= {0, 1}


def test_simulate_markers_respects_unary_probabilities():
    """Rank-0 markers (no dependency) appear at frequency u."""
    np.random.seed(0)
    C = np.zeros((2, 2))
    markers = bv.simulate_markers(C, u=np.array([0.2, 0.8]), n_docs=5000)

    np.testing.assert_allclose(markers.mean(axis=0), [0.2, 0.8], atol=0.02)


def test_simulate_markers_propagates_certain_dependency():
    """With u=0 and C[1,0]=1, marker 1 appears exactly when marker 0 does."""
    np.random.seed(0)
    C = np.zeros((2, 2))
    C[1, 0] = 1.0
    markers = bv.simulate_markers(C, u=np.array([0.5, 0.0]), n_docs=200)

    np.testing.assert_array_equal(markers[:, 1], markers[:, 0])
    assert 0 < markers[:, 0].sum() < 200
