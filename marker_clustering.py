from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from itertools import combinations
from matplotlib.colors import LogNorm
import umap
import hdbscan
import matplotlib.pyplot as plt
import numpy as np


def generate_cluster_markers(n_clusters, n_markers, p_intra, w_min, w_max, p_inter):
    """
    Generate an upper-triangular dependency matrix C with cluster structure.

    Markers are split as evenly as possible into n_clusters groups.
    C[i,j] != 0 only for i < j (upper triangular).
    Within-cluster edges : drawn with prob p_intra, weight ~ Uniform(w_min, w_max).
    Cross-cluster edges  : drawn with prob p_inter, weight ~ Uniform(w_min, w_max).

    Parameters
    ----------
    n_clusters : int
    n_markers  : int
    p_intra    : float  probability of an edge within the same cluster
    w_min      : float  minimum edge weight
    w_max      : float  maximum edge weight
    p_inter    : float  probability of an edge between different clusters

    Returns
    -------
    C      : (n_markers, n_markers) upper-triangular weight matrix
    labels : (n_markers,) integer cluster labels
    """
    C = np.zeros((n_markers, n_markers))

    # distribute markers as evenly as possible across clusters
    sizes = np.full(n_clusters, n_markers // n_clusters)
    sizes[:n_markers % n_clusters] += 1
    labels = np.repeat(np.arange(n_clusters), sizes)

    rng = np.random.default_rng()
    for i in range(n_markers):
        for j in range(i + 1, n_markers):
            p = p_intra if labels[i] == labels[j] else p_inter
            if rng.random() < p:
                C[i, j] = rng.uniform(w_min, w_max)

    return C, labels


def simulate_markers(C, u, n_docs=1):
    """
    C: (N, N) dependency matrix, C[i,j] = probability M_i given M_j
    u: (N,) unary probabilities for markers
    n_docs: number of documents to generate
    """
    N = len(u)
    # Compute rank from C: count non-zero dependencies per row
    ranks = np.sum(C != 0, axis=1)
    markers = np.zeros((n_docs, N), dtype=int)

    for doc in range(n_docs):
        present = np.zeros(N, dtype=bool)
        # Draw in order of increasing rank
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


def plot_dependency_matrix(C, title="Dependency Matrix C", save_path=None):
    """
    Plot the dependency matrix C using a logarithmic colour scale.
    Zero entries are masked (shown in white).
    """
    C_plot = np.where(C > 0, C, np.nan)
    vmin = np.nanmin(C_plot)
    vmax = np.nanmax(C_plot)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        C_plot,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        cmap="viridis",
        aspect="auto",
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="Weight (log scale)")
    ax.set_title(title)
    ax.set_xlabel("Marker j")
    ax.set_ylabel("Marker i")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def assess_kmeans(C, true_labels, k_max=None, n_repeats=20, save_path=None):
    """
    Assess k-means clustering on StandardScaler([C, C^T]).

    The optimal k is inferred by maximising  μ − σ  where μ and σ are the
    mean and standard deviation of pairwise ARI across n_repeats independent
    k-means runs (stability criterion).

    Parameters
    ----------
    C           : (n, n) dependency matrix
    true_labels : (n,) ground-truth cluster labels
    k_max       : int or None — upper bound on k to explore (default: min(12, n-1))
    n_repeats   : int — number of k-means runs per k value

    Returns
    -------
    best_k  : int
    pred    : (n,) predicted labels from the final k-means run
    ari     : float — ARI of pred vs. true_labels
    """
    n = C.shape[0]
    X = StandardScaler().fit_transform(np.hstack([C, C.T]))
    k_max = k_max or min(12, n - 1)

    stability = {}
    for k in range(2, k_max + 1):
        runs = [
            KMeans(n_clusters=k, n_init=1, random_state=seed).fit_predict(X)
            for seed in range(n_repeats)
        ]
        aris = [
            adjusted_rand_score(runs[a], runs[b])
            for a, b in combinations(range(n_repeats), 2)
        ]
        stability[k] = np.mean(aris) - np.std(aris)

    best_k = max(stability, key=stability.get)
    pred = KMeans(n_clusters=best_k, n_init=20, random_state=0).fit_predict(X)
    ari = adjusted_rand_score(true_labels, pred)

    # stability curve
    ks = list(stability.keys())
    vals = [stability[k] for k in ks]
    plt.figure(figsize=(7, 4))
    plt.plot(ks, vals, marker="o")
    plt.axvline(best_k, color="r", linestyle="--", label=f"best k = {best_k}")
    plt.xlabel("k")
    plt.ylabel("μ - σ  (pairwise ARI stability)")
    plt.title(f"K-Means stability  |  best k = {best_k}  |  ARI = {ari:.3f}")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    return best_k, pred, ari


def assess_umap_hdbscan(C, true_labels, n_sim=5000, min_cluster_size=None, u=None, save_path=None):
    """
    Assess two UMAP + HDBSCAN pipelines on the same dependency matrix C.

    Approach A — euclidean
        UMAP(metric='euclidean') on StandardScaler([C, C^T]) → HDBSCAN

    Approach B — lift dissimilarity
        Simulate n_sim marker vectors from C, compute empirical pairwise lifts,
        build a precomputed dissimilarity matrix
            D[i,j] = log(1 + 1 / lift[i,j]),  lift[i,j] = P(i,j) / (P(i)·P(j))
        then UMAP(metric='precomputed') on D → HDBSCAN.

    Parameters
    ----------
    C                : (n, n) upper-triangular dependency matrix
    true_labels      : (n,) ground-truth cluster labels
    n_sim            : int — documents simulated for lift estimation (approach B)
    min_cluster_size : int or None — HDBSCAN parameter (default: n // (n_true_clusters * 3))
    u                : (n,) unary probabilities; defaults to 0.1 for all markers

    Returns
    -------
    (pred_A, ari_A), (pred_B, ari_B), D
    """
    n = C.shape[0]
    n_true = len(np.unique(true_labels))
    min_cs = min_cluster_size or max(2, n // (n_true * 3))

    # ── Approach A : euclidean on [C, C^T] ───────────────────────────────────
    X = StandardScaler().fit_transform(np.hstack([C, C.T]))
    emb_A = umap.UMAP(n_components=2, metric="euclidean", random_state=42).fit_transform(X)
    pred_A = hdbscan.HDBSCAN(min_cluster_size=min_cs).fit_predict(emb_A)
    ari_A = adjusted_rand_score(true_labels, pred_A)

    # ── Approach B : precomputed lift dissimilarity ───────────────────────────
    if u is None:
        u = np.full(n, 0.1)

    sim = simulate_markers(C, u, n_docs=n_sim).astype(float)  # (n_sim, n)

    p_i  = sim.mean(axis=0)               # marginal probabilities, shape (n,)
    p_ij = (sim.T @ sim) / n_sim          # joint probabilities,    shape (n, n)
    denom = np.outer(p_i, p_i)
    denom = np.where(denom > 0, denom, 1e-10)   # guard against zero marginals
    lift  = p_ij / denom
    lift  = np.where(lift > 0, lift, 1e-10)     # guard against zero joint probs

    D = np.log1p(1.0 / lift)              # dissimilarity matrix
    np.fill_diagonal(D, 0.0)              # zero self-distance
    D = (D + D.T) / 2                    # symmetrise finite-sample estimates

    emb_B  = umap.UMAP(n_components=2, metric="precomputed", random_state=42).fit_transform(D)
    pred_B = hdbscan.HDBSCAN(min_cluster_size=min_cs).fit_predict(emb_B)
    ari_B  = adjusted_rand_score(true_labels, pred_B)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, emb, pred, ari, title in zip(
        axes,
        [emb_A, emb_B],
        [pred_A, pred_B],
        [ari_A, ari_B],
        ["A: Euclidean on [C, C\u1d40]", "B: Lift dissimilarity (precomputed)"],
    ):
        sc = ax.scatter(
            emb[:, 0], emb[:, 1],
            c=true_labels, cmap="tab10",
            s=60, alpha=0.8, edgecolors="none",
        )
        ax.set_title(f"{title}\nHDBSCAN  |  ARI = {ari:.3f}")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return (pred_A, ari_A), (pred_B, ari_B), D


def plot_dissimilarity_matrix(D, pred_labels, title="Dissimilarity matrix", save_path=None):
    """
    Plot the pairwise dissimilarity matrix D, with markers reordered by
    their assigned cluster and colour-coded accordingly.

    Parameters
    ----------
    D           : (n, n) symmetric dissimilarity matrix
    pred_labels : (n,) cluster labels assigned by the algorithm
                  (use -1 for noise, as returned by HDBSCAN)
    title       : str
    """
    n = D.shape[0]
    unique_labels = sorted(set(pred_labels))

    # build colour map: noise (-1) → grey, clusters → tab10
    palette = plt.cm.tab10.colors
    color_map = {
        lab: (palette[i % 10] if lab >= 0 else (0.65, 0.65, 0.65))
        for i, lab in enumerate(unique_labels)
    }
    marker_colors = np.array([color_map[l] for l in pred_labels])  # (n, 3)

    # reorder markers so that clusters appear as contiguous blocks
    order = np.argsort(pred_labels)
    D_ord = D[np.ix_(order, order)]
    colors_ord = marker_colors[order]                               # (n, 3)

    # layout: small colour strip on top + left, main heatmap in centre
    fig = plt.figure(figsize=(9, 8))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[0.04, 1],
        height_ratios=[0.04, 1],
        hspace=0.02, wspace=0.02,
    )
    ax_top  = fig.add_subplot(gs[0, 1])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_main = fig.add_subplot(gs[1, 1])

    # ── main heatmap ─────────────────────────────────────────────────────────
    im = ax_main.imshow(
        D_ord, cmap="magma_r", aspect="auto", interpolation="nearest"
    )
    plt.colorbar(im, ax=ax_main, label="Dissimilarity  log(1 + 1/lift)")
    ax_main.set_xlabel("Marker j  (sorted by cluster)")
    ax_main.set_ylabel("Marker i  (sorted by cluster)")
    ax_main.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    # ── cluster boundary lines ────────────────────────────────────────────────
    boundaries = np.where(np.diff(pred_labels[order]) != 0)[0] + 0.5
    for b in boundaries:
        ax_main.axvline(b, color="white", linewidth=1.0, linestyle="--", alpha=0.7)
        ax_main.axhline(b, color="white", linewidth=1.0, linestyle="--", alpha=0.7)

    # ── colour strips ─────────────────────────────────────────────────────────
    ax_top.imshow(colors_ord[np.newaxis, :],  aspect="auto", interpolation="nearest")
    ax_top.set_xticks([]); ax_top.set_yticks([])

    ax_left.imshow(colors_ord[:, np.newaxis], aspect="auto", interpolation="nearest")
    ax_left.set_xticks([]); ax_left.set_yticks([])

    # ── legend ────────────────────────────────────────────────────────────────
    handles = [
        plt.Rectangle(
            (0, 0), 1, 1,
            color=color_map[lab],
            label=f"Cluster {lab}" if lab >= 0 else "Noise",
        )
        for lab in unique_labels
    ]
    ax_main.legend(
        handles=handles,
        loc="upper right",
        fontsize=8,
        framealpha=0.8,
        ncol=max(1, len(unique_labels) // 4),
    )

    fig.suptitle(title, y=0.99)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_cluster_dissimilarity(D, true_labels, pred_km, pred_A, pred_B, save_path=None):
    """
    Plot the upper-triangular dissimilarity matrix D three times, once per
    clustering result (k-means, UMAP-euclidean, UMAP-lift).

    Each cell (i, j) with i < j is coloured by cluster membership:
      - same cluster (non-noise) → that cluster's colour
      - different clusters        → light grey
      - noise pair (-1)           → dark grey
    Markers are sorted by predicted cluster so same-cluster cells form
    contiguous blocks on the diagonal. Cluster boundaries are marked with
    thin black lines.

    Parameters
    ----------
    D           : (n, n) symmetric dissimilarity matrix (used for ordering)
    true_labels : (n,) ground-truth labels (used to compute ARI in titles)
    pred_km     : (n,) k-means predicted labels
    pred_A      : (n,) UMAP-euclidean HDBSCAN predicted labels
    pred_B      : (n,) UMAP-lift-dissimilarity HDBSCAN predicted labels
    """
    n = D.shape[0]
    palette = plt.cm.tab10.colors
    all_preds  = [pred_km,    pred_A,              pred_B]
    subtitles  = ['K-Means', 'UMAP (euclidean)', 'UMAP (lift dissimilarity)']

    # upper-triangle index pairs (same for every subplot)
    rows, cols = np.triu_indices(n, k=1)   # i < j

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))

    for ax, pred, subtitle in zip(axes, all_preds, subtitles):
        ari = adjusted_rand_score(true_labels, pred)

        # sort markers by predicted cluster (stable so ties keep original order)
        order    = np.argsort(pred, kind='stable')
        pred_ord = pred[order]

        # cluster index for each cell endpoint after reordering
        ci = pred_ord[rows]   # cluster of row marker
        cj = pred_ord[cols]   # cluster of col marker
        same = ci == cj

        # start with a white canvas (lower triangle + diagonal stay white)
        img = np.ones((n, n, 4))   # RGBA

        # different clusters → light grey
        img[rows[~same], cols[~same]] = [0.85, 0.85, 0.85, 1.0]

        # same cluster noise (-1) → mid grey
        noise_same = same & (ci < 0)
        img[rows[noise_same], cols[noise_same]] = [0.5, 0.5, 0.5, 1.0]

        # same cluster (non-noise) → cluster colour
        for k in sorted(set(pred_ord[pred_ord >= 0])):
            mask = same & (ci == k)
            img[rows[mask], cols[mask], :3] = palette[k % 10]
            img[rows[mask], cols[mask],  3] = 1.0

        ax.imshow(img, aspect='auto', interpolation='nearest')

        # cluster boundary lines
        bounds = np.where(np.diff(pred_ord) != 0)[0] + 0.5
        for b in bounds:
            ax.axvline(b, color='black', linewidth=0.7, alpha=0.8)
            ax.axhline(b, color='black', linewidth=0.7, alpha=0.8)

        ax.set_title(f'{subtitle}\nARI = {ari:.3f}')
        ax.set_xlabel('Marker j  (sorted by cluster)')
        ax.set_ylabel('Marker i  (sorted by cluster)')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        # per-subplot legend
        unique_labels = sorted(set(pred))
        handles = [
            plt.Rectangle(
                (0, 0), 1, 1,
                color=palette[l % 10] if l >= 0 else (0.5, 0.5, 0.5),
                label=f'Cluster {l}' if l >= 0 else 'Noise',
            )
            for l in unique_labels
        ]
        ax.legend(
            handles=handles, loc='lower right',
            fontsize=7, framealpha=0.9,
            ncol=max(1, len(unique_labels) // 5),
        )

    fig.suptitle(
        'Predicted cluster assignment — upper-triangular dissimilarity matrix',
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    from pathlib import Path
    Path("plots").mkdir(exist_ok=True)

    C_clust, labels_clust = generate_cluster_markers(
    n_clusters=5, n_markers=100,
    p_intra=0.99, w_min=0.8, w_max=1.0,
    p_inter=0.01,
    )

    plot_dependency_matrix(C_clust, title="Cluster-structured C (5 clusters, 100 markers)", save_path="plots/cluster_structured_c.png")

    best_k, pred_km, ari_km = assess_kmeans(C_clust, labels_clust, k_max=10, n_repeats=20, save_path="plots/kmeans_comparison.png")

    (pred_A, ari_A), (pred_B, ari_B), D_lift = assess_umap_hdbscan(
        C_clust, labels_clust, n_sim=5000, min_cluster_size=None, u=None, save_path="plots/umap_hdbscan_comparison.png"
    )

    print(f"K-Means       ARI = {ari_km:.3f}  (inferred k = {best_k})")
    print(f"UMAP eucl.    ARI = {ari_A:.3f}")
    print(f"UMAP lift-dis ARI = {ari_B:.3f}")

    plot_cluster_dissimilarity(D_lift, labels_clust, pred_km, pred_A, pred_B, save_path="plots/cluster_dissimilarity_comparison.png")