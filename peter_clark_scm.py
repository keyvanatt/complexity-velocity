"""Within-cluster causal structure discovery with the PC algorithm.

Runs the hierarchical pipeline of the paper on the CausalityLink corpus:

  1. global lift-based UMAP + DBSCAN clustering of the most frequent markers,
  2. per-cluster local complexity scores,
  3. PC (causal-learn, chi-squared independence test) on the binary
     marker-presence matrix of the selected cluster,
  4. a figure per cluster: the recovered causal graph next to its adjacency
     matrix, with markers ordered by increasing complexity.

Reproduces the "Intra-cluster dependency structures recovered by the PC
algorithm" figure of the paper.

Usage:
    python peter_clark_scm.py --clusters all
    python peter_clark_scm.py --clusters 5 11
    python peter_clark_scm.py                 # interactive cluster selection
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from umap import UMAP

try:
    from causallearn.search.ConstraintBased.PC import pc
except ImportError:  # pragma: no cover - dependency guard
    print("ERROR: causal-learn is required. Install it with: pip install causal-learn")
    sys.exit(1)

from causalityTable import CausalityTable
from lift_dissimilarity import lift_dissimilarity

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path("data/causalitylink_sample")
PLOTS_DIR = Path("plots")

# Number of most frequent markers retained for the global clustering step.
N_GLOBAL_MARKERS = 2000

# PC scales combinatorially with the number of variables, so large clusters are
# reduced to their complexity extremes before the search: the N_LOWEST least
# complex and the N_HIGHEST most complex markers of the cluster.
MAX_CLUSTER_SIZE = 25
N_LOWEST = 10
N_HIGHEST = 8

# Cap on the number of articles fed to PC (rows of the binary design matrix).
MAX_ARTICLES = 120_000

PC_ALPHA = 0.0005


# =============================================================================
# PART 1: PRE-PROCESSING AND GLOBAL CLUSTERING
# =============================================================================

def get_clusters(root_path: Path):
    """Load the corpus, keep the most frequent non-country markers and cluster them.

    Returns ``(filtered_df, selected_markers, dbscan_labels)``.
    """
    markerTable = CausalityTable(root_path / "Markers")
    markerTable.load_one_mounth(year=2025, month=1)
    treeTable = CausalityTable(root_path / "Tree")
    treeTable.load_data(date_parsing=False)

    markers_filter = treeTable.df.filter(pl.col("country").is_null())["marker"].to_list()
    df_filtered = markerTable.df.filter(pl.col("marker").is_in(markers_filter))

    counts = df_filtered.group_by("marker").len().sort("len", descending=True)
    sel_markers = counts.head(N_GLOBAL_MARKERS)["marker"].to_numpy()
    conv = {m: i for i, m in enumerate(sel_markers)}

    n = len(sel_markers)
    cm_counts = np.zeros((n, n), dtype=np.int64)
    df_grouped = df_filtered.filter(pl.col("marker").is_in(sel_markers)).group_by("id").agg(pl.col("marker"))

    for row in tqdm(df_grouped.iter_rows(), desc="Global clustering", total=len(df_grouped)):
        idxs = [conv[m] for m in row[1] if m in conv]
        if len(idxs) > 1:
            ix = np.array(idxs)
            cm_counts[np.ix_(ix, ix)] += 1

    cm_prob = cm_counts.astype(float) / float(max(1, df_filtered["id"].n_unique()))
    diag_p = np.diag(cm_prob).astype(float)
    denom = diag_p[:, None] * diag_p[None, :]
    lift = np.divide(cm_prob, denom, out=np.zeros_like(cm_prob), where=denom != 0)
    lift = 0.5 * (lift + lift.T)

    dist = lift_dissimilarity(lift)
    X_latent = UMAP(metric="precomputed", n_neighbors=15, random_state=42).fit_transform(dist)
    labels = DBSCAN(eps=0.2, min_samples=10).fit(X_latent).labels_

    return df_filtered, sel_markers, labels


# =============================================================================
# PART 2: LOCAL (WITHIN-CLUSTER) COMPLEXITY
# =============================================================================

def compute_local_complexities(df_cluster: pl.DataFrame, cluster_markers) -> dict:
    """Mean pairwise lift of each marker against the rest of its own cluster."""
    n = len(cluster_markers)
    conv = {m: i for i, m in enumerate(cluster_markers)}
    cm_counts = np.zeros((n, n), dtype=np.int64)
    df_grouped = df_cluster.group_by("id").agg(pl.col("marker"))
    for row in df_grouped.iter_rows():
        idxs = [conv[m] for m in row[1] if m in conv]
        if len(idxs) > 1:
            ix = np.array(idxs)
            cm_counts[np.ix_(ix, ix)] += 1
    cm_prob = cm_counts.astype(float) / float(max(1, df_cluster["id"].n_unique()))
    diag_p = np.diag(cm_prob).astype(float)
    denom = diag_p[:, None] * diag_p[None, :]
    lift_local = np.divide(cm_prob, denom, out=np.zeros_like(cm_prob), where=denom != 0)
    return {
        m: (np.sum(lift_local[conv[m], :]) - lift_local[conv[m], conv[m]]) / (n - 1)
        for m in cluster_markers
    }


# =============================================================================
# PART 3: VISUALISATION
# =============================================================================

def build_and_save_final_results(cg, markers, complexities, filename, cluster_id):
    """Save the recovered causal graph next to its complexity-sorted adjacency matrix."""
    G = nx.DiGraph()
    G.add_nodes_from(markers)
    adj = cg.G.graph

    undirected_edges = []
    directed_edges = []

    for i in range(len(markers)):
        for j in range(i + 1, len(markers)):
            m_i, m_j = markers[i], markers[j]
            if adj[i, j] == -1 and adj[j, i] == 1:
                G.add_edge(m_i, m_j)
                directed_edges.append((m_i, m_j))
            elif adj[i, j] == 1 and adj[j, i] == -1:
                G.add_edge(m_j, m_i)
                directed_edges.append((m_j, m_i))
            elif adj[i, j] == -1 and adj[j, i] == -1:
                undirected_edges.append((m_i, m_j))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 12))

    # --- causal graph ---
    pos = nx.kamada_kawai_layout(G)
    node_size = 2200
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=node_size, node_color="#FFDAB9", edgecolors="black")
    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=8, font_weight="bold")
    nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=directed_edges, edge_color="black",
                           style="solid", width=2, arrowsize=25, arrowstyle="-|>",
                           node_size=node_size, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=undirected_edges, edge_color="orange",
                           style="solid", width=2, arrows=False, alpha=0.6)

    ax1.set_title(f"PC causal graph — cluster {cluster_id}", fontsize=14)
    ax1.axis("off")

    # --- adjacency matrix, markers sorted by increasing complexity ---
    ordered_nodes = sorted(markers, key=lambda n: complexities.get(n, 0))
    labels_with_comp = [f"{n} (LC:{complexities.get(n, 0):.2f})" for n in ordered_nodes]

    size = len(ordered_nodes)
    mat = np.zeros((size, size))
    node_to_idx = {n: i for i, n in enumerate(ordered_nodes)}

    # 1 for a directed edge, 0.5 for an undirected one (symmetric)
    for u, v in directed_edges:
        if u in node_to_idx and v in node_to_idx:
            mat[node_to_idx[u], node_to_idx[v]] = 1

    for u, v in undirected_edges:
        if u in node_to_idx and v in node_to_idx:
            mat[node_to_idx[u], node_to_idx[v]] = 0.5
            mat[node_to_idx[v], node_to_idx[u]] = 0.5

    ax2.imshow(mat, cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
    ax2.set_xticks(range(size))
    ax2.set_yticks(range(size))
    ax2.set_xticklabels(labels_with_comp, rotation=90, fontsize=9)
    ax2.set_yticklabels(labels_with_comp, fontsize=9)
    ax2.set_title("Adjacency matrix (sorted by complexity)", fontsize=14)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def select_clusters(eligible_clusters, all_markers, cluster_labels, requested):
    """Resolve the ``--clusters`` argument, falling back to an interactive prompt."""
    if requested is None:
        print("\n" + "=" * 50)
        print(f"{'ID':<6} | {'Size':<8} | {'Sample markers'}")
        print("-" * 50)
        for lab in eligible_clusters:
            c_markers = [all_markers[i] for i, l in enumerate(cluster_labels) if l == lab]
            preview = ", ".join(c_markers[:5]) + ("..." if len(c_markers) > 5 else "")
            print(f"{lab:<6} | {len(c_markers):<8} | {preview}")
        print("=" * 50 + "\n")
        requested = input("Cluster IDs to process (comma-separated) or 'all': ").split(",")

    requested = [str(x).strip() for x in requested if str(x).strip()]
    if any(x.lower() == "all" for x in requested):
        return list(eligible_clusters)
    try:
        selected_ids = [int(x) for x in requested]
    except ValueError:
        print("Invalid selection. Aborting.")
        sys.exit(1)
    return [c for c in selected_ids if c in eligible_clusters]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Directory holding the Markers/ and Tree/ AVRO folders (default: %(default)s)")
    parser.add_argument("--clusters", nargs="+", default=None,
                        help="Cluster IDs to process, or 'all'. Omit for an interactive prompt.")
    parser.add_argument("--alpha", type=float, default=PC_ALPHA,
                        help="Significance level of the PC independence test (default: %(default)s)")
    parser.add_argument("--out-dir", type=Path, default=PLOTS_DIR,
                        help="Directory for the generated figures (default: %(default)s)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. global clustering
    df_filtered, all_markers, cluster_labels = get_clusters(args.root)

    # 2. every cluster except the DBSCAN noise label
    eligible_clusters = [int(lab) for lab in np.unique(cluster_labels) if lab != -1]

    if not eligible_clusters:
        logger.info("No cluster matched the criteria.")
        sys.exit(0)

    clusters_to_process = select_clusters(eligible_clusters, all_markers, cluster_labels, args.clusters)

    if not clusters_to_process:
        print("No valid cluster selected. Exiting.")
        sys.exit(0)

    # 3. run PC on each selected cluster
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for target_cluster in tqdm(clusters_to_process, desc="Running PC"):
        cluster_markers = [all_markers[i] for i, l in enumerate(cluster_labels) if l == target_cluster]
        df_cluster_full = df_filtered.filter(pl.col("marker").is_in(cluster_markers))

        # complexity over the whole cluster, used to pick the extremes below
        local_complexities = compute_local_complexities(df_cluster_full, cluster_markers)

        if len(cluster_markers) > MAX_CLUSTER_SIZE:
            logger.info(
                "Cluster %d is too large (%d markers). Keeping the %d least and %d most complex.",
                target_cluster, len(cluster_markers), N_LOWEST, N_HIGHEST,
            )
            sorted_by_comp = sorted(cluster_markers, key=lambda m: local_complexities[m])
            cluster_markers = sorted_by_comp[:N_LOWEST] + sorted_by_comp[-N_HIGHEST:]
            df_cluster = df_filtered.filter(pl.col("marker").is_in(cluster_markers))
            local_complexities = {m: local_complexities[m] for m in cluster_markers}
        else:
            df_cluster = df_cluster_full

        # --- binary article x marker design matrix ---
        X_pivot = df_cluster.pivot(index="id", on="marker", values="marker", aggregate_function="len").fill_null(0)
        X_bin = (X_pivot.to_pandas().set_index("id") > 0).astype(int)

        if len(X_bin) > MAX_ARTICLES:
            X_bin = X_bin.sample(n=MAX_ARTICLES, random_state=30)

        output_filename = args.out_dir / f"causal_cluster{target_cluster}_{timestamp}.png"
        cg = pc(X_bin.to_numpy(), alpha=args.alpha, indep_test="chisq", show_progress=True)

        build_and_save_final_results(
            cg, X_bin.columns.tolist(), local_complexities, output_filename, target_cluster
        )
        logger.info("Saved %s", output_filename)


if __name__ == "__main__":
    main()
