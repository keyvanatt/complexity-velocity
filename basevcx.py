import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from tqdm import tqdm
from itertools import product


# ── Dependency matrix generators ──────────────────────────────────────────────

def gen_C_chain(N, strength=0.2):
    """Linear chain: M_i depends only on M_{i-1}"""
    C = np.zeros((N, N))
    for i in range(1, N):
        C[i, i-1] = strength
    return C


def gen_C_tree(N, branching=2, strength=0.15):
    """Tree structure: each parent influences children"""
    C = np.zeros((N, N))
    for i in range(N):
        for b in range(1, branching + 1):
            child = i * branching + b
            if child < N:
                C[child, i] = strength
    return C


def gen_C_cliques(N, clique_sizes, internal_strength=0.25, cross_strength=0.05):
    """Multiple cliques with internal and cross-dependencies"""
    C = np.zeros((N, N))
    idx = 0
    for size in clique_sizes:
        for i in range(size):
            row = idx + i
            if row < N:
                for j in range(size):
                    col = idx + j
                    if col < N and row != col:
                        C[row, col] = internal_strength
                next_start = idx + size
                if next_start < N:
                    C[row, next_start] = cross_strength
        idx += size
    return C


def gen_C_hierarchical(N, levels=3, base_strength=0.3):
    """Hierarchical: higher ranks depend on multiple lower ranks"""
    C = np.zeros((N, N))
    markers_per_level = N // levels
    for k in range(1, levels):
        start = k * markers_per_level
        end = min((k + 1) * markers_per_level, N)
        for i in range(start, end):
            prev_start = (k - 1) * markers_per_level
            prev_end = k * markers_per_level
            for j in range(prev_start, min(prev_end, N)):
                C[i, j] = base_strength / (prev_end - prev_start)
    return C


def gen_C_random(N, density=0.1, strength_range=(0.1, 0.3)):
    """Random dependencies with given density"""
    C = np.zeros((N, N))
    n_edges = int(density * N * N)
    positions = np.random.choice(N * N, n_edges, replace=False)
    for pos in positions:
        i, j = divmod(pos, N)
        if i != j:
            C[i, j] = np.random.uniform(*strength_range)
    return C


def diag_to_zero(C):
    for i in range(C.shape[0]):
        C[i, i] = 0
    return C


def random_dag(N, edge_prob=0.2, weight_range=(0.1, 0.3)):
    """Generate a random DAG adjacency matrix"""
    C = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if np.random.rand() < edge_prob:
                C[j, i] = np.random.uniform(*weight_range)
    return C


def gen_C_fractal(N, strength_base=0.4):
    C = np.zeros((N, N))

    def fill_fractal(start, end, level):
        if end - start <= 1:
            return
        mid = (start + end) // 2
        s = strength_base / (level + 1)
        for i in range(mid, end):
            for j in range(start, mid):
                if np.random.rand() > 0.3:
                    C[i, j] = s
        fill_fractal(start, mid, level + 1)
        fill_fractal(mid, end, level + 1)

    fill_fractal(0, N, 0)
    return C.T


def gen_C_funnel(N, strength_base=0.3):
    C = np.zeros((N, N))
    mid_point = N // 2
    for i in range(1, mid_point):
        parents = np.random.choice(i, size=min(i, 2), replace=False)
        C[i, parents] = strength_base
    for i in range(mid_point, N):
        n_parents = int(i * 0.8)
        parents = np.random.choice(i, size=n_parents, replace=False)
        C[i, parents] = strength_base / n_parents
    return C.T


def gen_C_skip_hierarchical(N, strength_base=0.2):
    C = np.zeros((N, N))
    for i in range(1, N):
        C[i, i-1] = strength_base * 0.5
        if i > N // 2:
            n_skips = np.random.randint(1, N // 4)
            sources = np.random.choice(range(N // 4), size=n_skips, replace=False)
            C[i, sources] = strength_base / n_skips
    return C.T


def gen_C_dense_progressive(N, max_lookback=5, base_strength=0.05):
    C = np.zeros((N, N))
    for i in range(1, N):
        C[i, i-1] = base_strength * (1 + i / N)
        current_lookback = min(i, int(max_lookback * (i / N)) + 1)
        if current_lookback > 1:
            potential_parents = np.arange(max(0, i - current_lookback), i - 1)
            for p in potential_parents:
                C[i, p] = base_strength * (1.5 * i / N)
    return C.T


def gen_C_mostly_full(N, density=0.8, strength_range=(0.02, 0.08)):
    C = np.zeros((N, N))
    strengths = np.linspace(strength_range[0], strength_range[1], N)
    for i in range(1, N):
        for j in range(i):
            if np.random.rand() < density:
                C[i, j] = 1 / strengths[i] * np.random.uniform(0.9, 1.1)
    return C.T


def gen_C_croissante_rang(N, max_strength, proba_app=0.2):
    """Generate a dependency matrix where ranks increase with marker index"""
    C = np.zeros((N, N))
    matrix_rank = np.zeros(N)
    for i in range(N):
        if i > 0:
            matrix_rank[i] = matrix_rank[i-1]
        for j in range(i):
            if j < matrix_rank[i-1]:
                C[i, j] = np.random.uniform(0, max_strength(matrix_rank[i]))
            elif np.random.rand() < proba_app:
                C[i, j] = np.random.uniform(0, max_strength(matrix_rank[i] + 1))
                matrix_rank[i] += 1
    return C


# ── Graph utilities ────────────────────────────────────────────────────────────

def compute_depth_in_dag(C):
    """
    Compute the depth (longest path length) of each node in a DAG.
    C: (N, N) adjacency matrix where C[i,j] > 0 means j -> i
    Returns: (N,) array of depths
    """
    N = C.shape[0]
    depth = np.zeros(N, dtype=int)
    for i in range(N):
        predecessors = np.where(C[i, :] > 0)[0]
        if len(predecessors) > 0:
            depth[i] = 1 + np.min(depth[predecessors])
    return depth


def show_graph(C):
    """
    Visualize the dependency graph defined by adjacency matrix C.
    C: (N, N) adjacency matrix where C[i,j] > 0 means j -> i
    """
    G = nx.DiGraph()
    G.add_nodes_from(range(C.shape[0]))
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            if C[i, j] > 0:
                G.add_edge(j, i, weight=C[i, j])
    depths = compute_depth_in_dag(C)
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42, k=2)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
    plt.title('Dependency Graph')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# ── Simulation ─────────────────────────────────────────────────────────────────

def simulate_markers(C, u, n_docs=1):
    """
    C: (N, N) dependency matrix
    u: (N,) unary probabilities for markers
    n_docs: number of documents to generate
    """
    N = len(u)
    ranks = np.sum(C != 0, axis=1)
    markers = np.zeros((n_docs, N), dtype=int)
    for doc in tqdm(range(n_docs)):
        present = np.zeros(N, dtype=bool)
        for r in range(ranks.max() + 1):
            idx = np.where(ranks == r)[0]
            for i in idx:
                if r == 0:
                    p = u[i]
                else:
                    p = u[i] + np.sum(C[i, present])
                if np.random.rand() < np.clip(p, 0, 1):
                    present[i] = 1
        markers[doc] = present.astype(int)
    return markers


# ── Helper ─────────────────────────────────────────────────────────────────────

def max_strength(rang):
    return 2 / (rang + 1) ** 2


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Sanity-check all generators
    N = 8
    for name, C_gen in [
        ("Chain",        lambda: gen_C_chain(N)),
        ("Tree",         lambda: gen_C_tree(N, branching=2)),
        ("Cliques",      lambda: gen_C_cliques(N, [3, 3, 2])),
        ("Hierarchical", lambda: gen_C_hierarchical(N, levels=3)),
        ("Random",       lambda: gen_C_random(N, density=0.15)),
        ("Random DAG",   lambda: random_dag(N, edge_prob=0.3)),
    ]:
        C = C_gen()
        ranks = np.sum(C != 0, axis=1)
        print(f"{name}: ranks = {ranks}")
        u = np.ones(N) * 0.1
        sums = u + np.sum(C, axis=1)
        print(f"  Property holds: {(sums > 0).all()} and {(sums <= 1).all()}\n")

    # ── Build matrix and unary probs ──────────────────────────────────────────
    N = 150
    C = gen_C_mostly_full(N, density=0.8, strength_range=(0.02, 0.08))
    min_c = np.min(C[C > 0])
    c_rank = np.sum(C > min_c / 2, axis=0)

    u = np.random.uniform(0.00, 0.001, size=N)
    u.sort()
    u = u[::-1]

    for m in np.where(c_rank == 0)[0]:
        u[m] = 0.3

    # ── Visualise C ───────────────────────────────────────────────────────────
    sns.heatmap(C, cmap='viridis')
    plt.title('Dependency Matrix C')
    plt.xlabel('Markers')
    plt.ylabel('Markers')
    plt.show()

    sns.barplot(c_rank)
    plt.title('C Rank per Marker')
    plt.xlabel('Marker')
    plt.ylabel('C rank')
    plt.show()

    sns.barplot(np.sum(C, axis=0))
    plt.title('Sum Dependency Strength per Marker')
    plt.xlabel('Marker')
    plt.ylabel('Sum Strength')
    plt.show()

    # ── Simulate and compute lifts ────────────────────────────────────────────
    n_sims = 20_000
    markers = simulate_markers(C, u, n_docs=n_sims)
    p_marge = np.mean(markers, axis=0)
    if p_marge.min() == 0:
        p_marge += 1e-5

    p_joint = (markers.T @ markers) / n_sims
    p_product = p_marge[:, None] * p_marge[None, :]
    lifts = p_joint / p_product
    np.fill_diagonal(lifts, 0)

    # ── Heatmaps: C vs lifts ──────────────────────────────────────────────────
    eps = 1e-8
    C_plot = C.astype(float).copy()
    lifts_plot = lifts.astype(float).copy()
    C_plot[C_plot == 0] = np.nan
    lifts_plot[lifts_plot == 0] = np.nan

    cmap = sns.color_palette("magma", as_cmap=True)
    cmap.set_bad("white")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sns.heatmap(
        C_plot + eps, cmap=cmap, ax=axes[0],
        norm=mcolors.LogNorm(vmin=np.nanmin(C_plot + eps), vmax=np.nanmax(C_plot + eps)),
    )
    axes[0].set_title('Dependency Matrix (C) - log scale')
    sns.heatmap(
        lifts_plot + eps, cmap=cmap, ax=axes[1],
        norm=mcolors.LogNorm(vmin=np.nanmin(lifts_plot + eps), vmax=np.nanmax(lifts_plot + eps)),
    )
    axes[1].set_title('Lift Matrix - log scale')
    plt.tight_layout()
    plt.show()

    # ── Derived metrics ───────────────────────────────────────────────────────
    complexity = np.sum(lifts, axis=1)
    min_c = np.min(C[C > 0])
    c_rank = np.sum(C > min_c / 2, axis=0)
    depth = compute_depth_in_dag(C)

    # Scatter: p_marge vs c_rank
    sns.scatterplot(y=p_marge, x=c_rank)
    plt.ylabel('Marker Marginal Probability (p_marge)')
    plt.xlabel('C Rank')
    plt.title('C Rank vs Marker Marginal Probability')
    plt.show()

    # Bar plots
    f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(figsize=(10, 6), nrows=5)
    sns.barplot(complexity, ax=ax1)
    ax1.set_xlabel('Marker')
    ax1.set_ylabel('Complexity')
    sns.barplot(c_rank, ax=ax2)
    ax2.set_xlabel('Marker')
    ax2.set_ylabel('C rank')
    sns.barplot(depth, ax=ax3)
    ax3.set_xlabel('Marker')
    ax3.set_ylabel('Depth in DAG')
    sns.barplot(p_marge, ax=ax4)
    ax4.set_xlabel('Marker')
    ax4.set_ylabel('p_marge')
    sns.barplot(u, ax=ax5)
    ax5.set_xlabel('Marker')
    ax5.set_ylabel('Unary prob. u_i')
    plt.show()

    # Scatter: velocity / c_rank / depth vs complexity
    f, (ax1, ax2, ax3) = plt.subplots(figsize=(8, 4), ncols=3, sharey=True)
    ax1.plot(1 / p_marge, complexity, 'o')
    ax1.grid(True)
    ax1.set_xlabel('Velocity')
    ax1.set_ylabel('Complexity')
    ax2.plot(c_rank, complexity, 'o')
    ax2.grid(True)
    ax2.set_xlabel('C rank')
    ax3.plot(depth, complexity, 'o')
    ax3.grid(True)
    ax3.set_xlabel('Depth in DAG')
    plt.show()
