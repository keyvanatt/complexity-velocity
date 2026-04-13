import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from causalityTable import CausalityTable
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from umap import UMAP  # pip install umap-learn


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def compute_cocitation_probability_matrix(
    markers: np.ndarray,
    df: pl.DataFrame,
    conv: Dict[str, int],
) -> np.ndarray:
    """Compute cocitation probability matrix for a given set of markers.

    Args:
        markers: array-like of selected marker ids (order defines indices).
        df: Polars DataFrame containing at least columns `id` and `marker`.
        conv: mapping marker -> index in `markers`.

    Returns:
        square numpy array (n_markers x n_markers) of cocitation probabilities (counts / n_articles).
    """
    n_markers = len(markers)
    cm_counts = np.zeros((n_markers, n_markers), dtype=np.int64)


    # filter rows to selected markers and group markers per article
    df_filtered = df.filter(pl.col("marker").is_in(markers))
    df_grouped = df_filtered.group_by("id").agg(pl.col("marker").unique().alias("markers"))

    # total unique articles
    n_articles = int(df_filtered["id"].n_unique())
    logger.info("Computing cocitation counts for %d entries, %d markers, %d articles", 
                len(df_filtered), df_filtered["marker"].n_unique(), df_filtered["id"].n_unique())

    conv_local = conv
    for marker_list in tqdm(df_grouped["markers"].to_list(), desc="computing cocitation counts"):
        if not marker_list:
            continue
        idxs = np.array([conv_local[m] for m in marker_list if m in conv_local], dtype=np.int64)
        if idxs.size == 0:
            continue
        cm_counts[np.ix_(idxs, idxs)] += 1

    # convert to probabilities
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_prob = cm_counts.astype(float) / float(max(1, n_articles))

    return cm_prob


def compute_lift_matrix(cocitation_prob_matrix: np.ndarray) -> np.ndarray:
    """Compute lift matrix from cocitation probabilities.

    lift[i,j] = p_ij / (p_i * p_j) with p_i = p_ii.
    Entries with zero denominator are set to 0. The result is symmetrized.
    """
    cm = np.asarray(cocitation_prob_matrix, dtype=float)
    p = np.diag(cm).astype(float)
    denom = p[:, None] * p[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = cm / denom
    lift[~np.isfinite(lift)] = 0.0
    lift = 0.5 * (lift + lift.T)
    return lift


def get_complexity_fast(lift_matrix: np.ndarray, conv: Dict[str, int], marker: str) -> float:
    """Return a simple complexity score for `marker` as mean lift to all others.

    Complexity = mean_j lift[marker, j] over j != marker.
    """
    ind = conv[marker]
    n = lift_matrix.shape[0]
    if n <= 1:
        return 0.0
    acc = float(np.sum(lift_matrix[ind, :]))-float(lift_matrix[ind, ind])
    return acc / float(n - 1)


def prepare_filtered_marker_table(path: Path, list_themes: Optional[List[str]] = None) -> pl.DataFrame:
    """Load tables and prepare the filtered marker DataFrame enriched with publisher info.

    Expects files under `path` and CSVs `CausalityLinkPublishers.csv`, `journaux_themes.csv` in working dir.
    Returns a Polars DataFrame ready for further analysis.
    """
    markerTable = CausalityTable(path / "Markers")
    markerTable.load_one_mounth(year=2025, month=1)

    treeTable = CausalityTable(path / "Tree")
    treeTable.load_data(date_parsing=False)

    publishers = pl.read_csv("CausalityLinkPublishers.csv")
    journaux_themes = pd.read_csv("journaux_themes.csv", index_col=0).to_dict()["theme"]

    logger.info("Initial marker table: %d entries, %d markers, %d articles", 
                len(markerTable.df), markerTable.df["marker"].n_unique(), markerTable.df["id"].n_unique())

    # filter markers with no country information in treeTable
    markers_filter = treeTable.df.filter(pl.col("country").is_null())
    markers_filter = markers_filter["marker"].to_list()
    filtered_marker_df = markerTable.df.filter(pl.col("marker").is_in(markers_filter))
    
    logger.info("After tree & country filter: %d entries, %d markers, %d articles", 
                len(filtered_marker_df), filtered_marker_df["marker"].n_unique(), filtered_marker_df["id"].n_unique())

    # add publisher information to marker table
    filtered_marker_df = filtered_marker_df.with_columns(
        pl.col("id").str.split("_").list.get(0).alias("publisher_id")
    )
    filtered_marker_df = filtered_marker_df.join(
        publishers.select("publisher", "label"), left_on="publisher_id", right_on="publisher", how="left"
    )
    filtered_marker_df = filtered_marker_df.filter(pl.col("label").is_not_null()).rename({"label": "publisher_label"})
    
    logger.info("After publisher join: %d entries, %d markers, %d articles", 
                len(filtered_marker_df), filtered_marker_df["marker"].n_unique(), filtered_marker_df["id"].n_unique())
    
    filtered_marker_df = filtered_marker_df.with_columns(
        pl.col("publisher_label").replace(journaux_themes).alias("journal_theme")
    )

    if list_themes is not None:
        filtered_marker_df = filtered_marker_df.filter(pl.col("journal_theme").is_in(list_themes))
        logger.info("After theme filter: %d entries, %d markers, %d articles", 
                    len(filtered_marker_df), filtered_marker_df["marker"].n_unique(), filtered_marker_df["id"].n_unique())

    return filtered_marker_df


def select_markers_by_theme(filtered_marker_df: pl.DataFrame, themes: Optional[List[str]] = None, fraction: float = 1 / 3, top : bool = True):
    """Select markers appearing in given journal themes and return markers array and conv mapping.

    Args:
        filtered_marker_df: Polars DataFrame with `journal_theme` and `marker` columns.
        themes: list of themes to keep.
        fraction: fraction of markers by count to keep.
        top: if True, select top markers by count, else random sample.
    """
    if themes is None:
        themes = filtered_marker_df["journal_theme"].unique().to_list()
    
    selected_markers_df = (
        filtered_marker_df
        .filter(pl.col("journal_theme").is_in(themes))["marker", "publisher_label"]
        .group_by("marker")
        .agg(pl.col("publisher_label").unique().alias("publishers_label"), pl.col("marker").count().alias("marker_count"))
    )
    keep_n = max(1, int(len(selected_markers_df) * fraction))
    if top:
        selected_markers_df = selected_markers_df.sort("marker_count", descending=True).head(keep_n)
    else:
        selected_markers_df = selected_markers_df.sample(keep_n, shuffle=True)
    markers_journals = np.array(selected_markers_df["publishers_label"].to_list(), dtype=object)
    selected_markers = np.array(selected_markers_df["marker"].to_list())
    conv = {selected_markers[k]: int(k) for k in range(len(selected_markers))}
    return selected_markers, conv, markers_journals


def plot_complexity_vs_velocity(
    lift_matrix: np.ndarray,
    conv: Dict[str, int],
    selected_markers: np.ndarray,
    out_prefix: str = "complexity_vs_velocity",
    n_bins: int = 15,
):
    """Produce and save scatter and boxplot comparing complexity vs velocity.
    Files saved: '{out_prefix}.png' and '{out_prefix}_categories.png'.
    """
    complexities = {marker: get_complexity_fast(lift_matrix, conv, marker) for marker in selected_markers}
    velocities = [lift_matrix[i, i] ** (-1) if lift_matrix[i, i] > 0 else np.nan for i in range(len(lift_matrix))]

    complexities_values = list(complexities.values())

    plt.figure()
    plt.scatter(complexities_values, velocities)
    plt.xlabel("Complexity")
    plt.ylabel("Velocity")
    plt.title("Marker Complexity vs Velocity (Log-Log Scale)")
    plt.xscale("log")
    plt.yscale("log")
    plt.tight_layout()
    if out_prefix != "SHOW":
        plt.savefig(f"{out_prefix}.png")
    else:
        plt.show()
    plt.close()

    categories = pd.qcut(pd.Series(complexities_values).dropna(), n_bins, duplicates="drop")
    plt.figure()
    boxplot = sns.boxplot(x=categories, y=np.log(velocities))
    boxplot.set_xlabel("Complexity")
    boxplot.set_ylabel("Log Velocity")
    boxplot.tick_params(axis="x", rotation=45)
    boxplot.set_title("Complexity vs Velocity Categories of Markers")
    plt.tight_layout()
    if out_prefix != "SHOW":
        plt.savefig(f"{out_prefix}_categories.png")
    else:
        plt.show()
    plt.close()

    return complexities



def compute_latent_and_cluster(lift_matrix: np.ndarray, selected_markers: np.ndarray, markers_journals: np.ndarray, out_prefix: str = "projection_2d",
                               eps_dbscan: float = 0.10, min_samples_dbscan: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Compute latent 2D embedding (UMAP) from lift matrix-derived distances and run DBSCAN clustering.

    Saves projection and projection with DBSCAN into PNG files with given prefix.
    Returns embedding and cluster labels.
    """
    epsilon = 1e-4
    velocities = np.array([lift_matrix[i, i] ** (-1) if lift_matrix[i, i] > 0 else np.nan for i in range(len(lift_matrix))])
    # safeguard and produce a symmetric non-negative distance matrix
    distance_matrix = np.log1p((lift_matrix + epsilon) ** (-1))
    distance_matrix[np.diag_indices_from(distance_matrix)] = 0
    distance_matrix = (distance_matrix + distance_matrix.T) / 2.0
    distance_matrix -= distance_matrix.min()

    umap = UMAP(n_components=2, metric="precomputed", min_dist=0.10, random_state=42)
    X_latent = umap.fit_transform(distance_matrix)

    plt.figure(figsize=(10, 8))
    plt.scatter(X_latent[:, 0], X_latent[:, 1], s=3, alpha=0.2)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title("Projection 2D")
    plt.tight_layout()
    if out_prefix != "SHOW":
        plt.savefig(f"{out_prefix}.png")
    else:
        plt.show()
    plt.close()

    dbscan = DBSCAN(metric="euclidean", eps=eps_dbscan, min_samples=min_samples_dbscan)
    dbscan.fit(X_latent)

    plt.figure(figsize=(10, 8))
    labels = dbscan.labels_
    norm = np.max(labels) if np.max(labels) > 0 else 1
    plt.scatter(X_latent[:, 0], X_latent[:, 1], c=labels / norm, cmap="gist_ncar", s=3, alpha=0.4)

    # annotate representative point per cluster (skip noise -1)
    for lab in np.unique(labels):
        if lab == -1:
            continue
        idxs = np.where(labels == lab)[0]
        if idxs.size == 0:
            continue
        centroid = X_latent[idxs].mean(axis=0)
        rel_idx = idxs[int(np.argmin(np.linalg.norm(X_latent[idxs] - centroid, axis=1)))]
        plt.annotate(f"{int(lab)} : {selected_markers[rel_idx]}", xy=(X_latent[rel_idx, 0], X_latent[rel_idx, 1]), xytext=(4, 4), textcoords="offset points", fontsize=6)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title(f"Projection 2D avec Clustering DBSCAN (n_clusters = {n_clusters})")
    plt.tight_layout()
    if out_prefix != "SHOW":
        plt.savefig(f"{out_prefix}_dbscan.png")
    else:
        plt.show()
    plt.close()

    return X_latent, labels

def markers_from_cluster(dbscan_labels, cluster_label, selected_markers):
    indices = np.where(dbscan_labels == cluster_label)[0]
    return np.array([selected_markers[i] for i in indices])

def compute_sub_lift_matrix(sub_selected_markers: np.ndarray, filtered_marker_df: pl.DataFrame) -> Tuple[np.ndarray, Dict[str, int]]:
    """Compute lift matrix for a subset of markers."""
    sub_conv = {marker: idx for idx, marker in enumerate(sub_selected_markers)}

    sub_cocitation_matrix = compute_cocitation_probability_matrix(sub_selected_markers, filtered_marker_df, sub_conv)
    sub_lift_matrix = compute_lift_matrix(sub_cocitation_matrix)
    return sub_lift_matrix, sub_conv

def compute_complexity_df(complexities: Dict[str, float]) -> pd.DataFrame:
    complexities_df = pd.DataFrame.from_dict(complexities, orient='index', columns=['complexity']).sort_values(by='complexity')
    complexities_df.index.name = 'marker'
    return complexities_df

def top_lifters(marker: str, lift_matrix: np.ndarray, conv: Dict[str, int], complexity_df: pd.DataFrame, top_n: int = 15) -> List[Tuple[str, float]]:
    """Return the top N markers with highest lift with respect to the given marker."""
    if marker not in conv:
        return []
    ind = conv[marker]
    lifts = lift_matrix[ind, :]
    top_indices = np.argsort(lifts)[-top_n-1:][::-1]
    if ind in top_indices:
        top_indices = top_indices[top_indices != ind]
    else:
        top_indices = top_indices[:-1]
    top_markers = [(list(conv.keys())[i], lifts[i]) for i in top_indices ]
    return top_markers

def run_all(root: Path = Path("/Data/rc/data/causalitylink_sample")) -> None:
    """Main pipeline: load data, select markers, compute matrices, plot and cluster.

    This function mirrors the previous top-level script but organizes steps and saves outputs.
    """
    logger.info("Preparing filtered marker table...")
    list_themes = ["sante", "economie", "sport", "politique", "transport", "information"]
    filtered_marker_df = prepare_filtered_marker_table(root,None)

    selected_markers, conv, markers_journals = select_markers_by_theme(filtered_marker_df, list_themes, fraction= 1 / 3)

    logger.info("Computing cocitation probabilities for %d markers", len(selected_markers))
    cocitation_matrix = compute_cocitation_probability_matrix(selected_markers, filtered_marker_df, conv)

    logger.info("Computing lift matrix")
    lift_matrix = compute_lift_matrix(cocitation_matrix)

    logger.info("Plotting complexity vs velocity")
    complexities = plot_complexity_vs_velocity(lift_matrix, conv, selected_markers, out_prefix="complexity_vs_velocity")

    logger.info("Computing latent embedding and clustering")
    X_latent, labels =compute_latent_and_cluster(lift_matrix, selected_markers,markers_journals, out_prefix="projection_2d")

    logger.info("All done.")
    return filtered_marker_df, selected_markers, conv, markers_journals,lift_matrix,complexities,labels


if __name__ == "__main__":
    filtered_marker_df, selected_markers, conv, markers_journals,lift_matrix,complexities = run_all()
