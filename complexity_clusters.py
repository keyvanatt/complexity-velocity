import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from causalityTable import CausalityTable
from scipy import stats
from scipy.sparse import csr_matrix
from sklearn.cluster import DBSCAN
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

    # filter rows to selected markers only
    df_filtered = df.filter(pl.col("marker").is_in(markers))
    n_articles = int(df_filtered["id"].n_unique())
    logger.info(
        "Computing cocitation counts for %d entries, %d markers, %d articles",
        len(df_filtered), df_filtered["marker"].n_unique(), n_articles,
    )

    # map article ids -> contiguous integers
    article_ids = df_filtered["id"].unique()
    article_conv = {aid: i for i, aid in enumerate(article_ids.to_list())}
    n_art = len(article_conv)

    # build flat arrays for sparse matrix construction (no Python list-of-lists)
    marker_col = df_filtered["marker"].to_list()
    id_col = df_filtered["id"].to_list()
    row_indices = np.fromiter((article_conv[a] for a in id_col), dtype=np.int32, count=len(id_col))
    col_indices = np.fromiter((conv[m] for m in marker_col), dtype=np.int32, count=len(marker_col))

    # article-marker indicator matrix A  (n_art x n_markers)
    data = np.ones(len(row_indices), dtype=np.float32)
    A = csr_matrix((data, (row_indices, col_indices)), shape=(n_art, n_markers))

    # co-citation counts = A^T @ A  (sparse -> dense, shape n_markers x n_markers)
    cm_counts = (A.T @ A).toarray()

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
    logger.info("Loading Markers AVRO files from %s...", path / "Markers")
    markerTable = CausalityTable(path / "Markers")
    markerTable.load_one_mounth(year=2025, month=1)
    logger.info("Markers loaded: %d rows", len(markerTable.df))

    logger.info("Loading Tree AVRO files from %s...", path / "Tree")
    treeTable = CausalityTable(path / "Tree")
    treeTable.load_data(date_parsing=False)
    logger.info("Tree loaded: %d rows", len(treeTable.df))

    logger.info("Loading publishers and journal themes CSV...")
    publishers = pl.read_csv("data/CausalityLinkPublishers.csv")
    journaux_themes = pd.read_csv("data/journaux_themes.csv", index_col=0).to_dict()["theme"]
    logger.info("Publishers: %d entries, journal themes: %d entries", len(publishers), len(journaux_themes))

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

    logger.info("Selecting markers for themes: %s", themes)
    selected_markers_df = (
        filtered_marker_df
        .filter(pl.col("journal_theme").is_in(themes))["marker", "publisher_label"]
        .group_by("marker")
        .agg(pl.col("publisher_label").unique().alias("publishers_label"), pl.col("marker").count().alias("marker_count"))
    )
    keep_n = max(1, int(len(selected_markers_df) * fraction))
    logger.info("Total distinct markers in themes: %d — keeping top %d (fraction=%.2f)", len(selected_markers_df), keep_n, fraction)
    if top:
        selected_markers_df = selected_markers_df.sort("marker_count", descending=True).head(keep_n)
    else:
        selected_markers_df = selected_markers_df.sample(keep_n, shuffle=True)
    markers_journals = np.array(selected_markers_df["publishers_label"].to_list(), dtype=object)
    selected_markers = np.array(selected_markers_df["marker"].to_list())
    conv = {selected_markers[k]: int(k) for k in range(len(selected_markers))}
    logger.info("Selected %d markers", len(selected_markers))
    return selected_markers, conv, markers_journals


def fit_loglog_regression(
    complexities_values: np.ndarray,
    velocities: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """Fit a log-log OLS regression (velocity ~ complexity^beta1) and return statistics.

    Returns a dict with keys:
        beta0, beta1        : intercept and slope in log-log space
        beta1_ci            : (low, high) confidence interval for beta1
        r2                  : coefficient of determination
        pearson_r, pearson_p: Pearson correlation on log-log values
        kendall_tau, kendall_p: Kendall's tau on original values
    """
    c = np.asarray(complexities_values, dtype=float)
    v = np.asarray(velocities, dtype=float)
    valid = np.isfinite(c) & np.isfinite(v) & (c > 0) & (v > 0)
    log_c = np.log(c[valid])
    log_v = np.log(v[valid])
    n = valid.sum()

    # OLS in log-log space
    result = stats.linregress(log_c, log_v)
    beta1 = result.slope
    beta0 = result.intercept
    r2 = result.rvalue ** 2

    # Confidence interval for beta1 (t-distribution, two-tailed)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 2)
    beta1_ci = (beta1 - t_crit * result.stderr, beta1 + t_crit * result.stderr)

    # Pearson on log-log
    pearson_r, pearson_p = stats.pearsonr(log_c, log_v)

    # Kendall's tau on original values
    kendall_tau, kendall_p = stats.kendalltau(c[valid], v[valid])

    return dict(
        beta0=beta0,
        beta1=beta1,
        beta1_ci=beta1_ci,
        r2=r2,
        pearson_r=pearson_r,
        pearson_p=pearson_p,
        kendall_tau=kendall_tau,
        kendall_p=kendall_p,
        n=int(n),
    )


def plot_complexity_vs_velocity(
    lift_matrix: np.ndarray,
    conv: Dict[str, int],
    selected_markers: np.ndarray,
    out_prefix: str = "complexity_vs_velocity",
    n_bins: int = 15,
):
    """Produce and save scatter and boxplot comparing complexity vs velocity.

    Fits a log-log OLS regression and annotates the plot with beta1, its 95% CI,
    Pearson r and Kendall tau. Files saved: '{out_prefix}.png' and
    '{out_prefix}_categories.png'.
    """
    complexities = {marker: get_complexity_fast(lift_matrix, conv, marker) for marker in selected_markers}
    velocities = np.array([lift_matrix[i, i] ** (-1) if lift_matrix[i, i] > 0 else np.nan for i in range(len(lift_matrix))])

    complexities_values = np.array(list(complexities.values()))

    # --- regression & stats ---
    reg = fit_loglog_regression(complexities_values, velocities)
    logger.info(
        "Log-log regression: beta1=%.4f  95%%CI=[%.4f, %.4f]  R²=%.4f  "
        "Pearson r=%.4f (p=%.2e)  Kendall tau=%.4f (p=%.2e)  n=%d",
        reg["beta1"], reg["beta1_ci"][0], reg["beta1_ci"][1], reg["r2"],
        reg["pearson_r"], reg["pearson_p"],
        reg["kendall_tau"], reg["kendall_p"], reg["n"],
    )

    # regression line: 2 endpoints are enough (straight line in log-log)
    valid = np.isfinite(complexities_values) & np.isfinite(velocities) & (complexities_values > 0) & (velocities > 0)
    c_fit = np.array([complexities_values[valid].min(), complexities_values[valid].max()])
    v_fit = np.exp(reg["beta0"]) * c_fit ** reg["beta1"]

    annotation = (
        f"β₁ = {reg['beta1']:.3f}  95% CI [{reg['beta1_ci'][0]:.3f}, {reg['beta1_ci'][1]:.3f}]\n"
        f"R² = {reg['r2']:.3f}   Pearson r = {reg['pearson_r']:.3f}\n"
        f"Kendall τ = {reg['kendall_tau']:.3f} (p={reg['kendall_p']:.2e})   n = {reg['n']}"
    )

    fig, ax = plt.subplots()
    ax.scatter(complexities_values, velocities, s=4, alpha=0.4, label="markers")
    ax.plot(c_fit, v_fit, color="red", linewidth=1.5, label=f"fit  β₁={reg['beta1']:.3f}")
    ax.set_xlabel("Complexity")
    ax.set_ylabel("Velocity")
    ax.set_title("Marker Complexity vs Velocity (Log-Log Scale)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.text(0.02, 0.02, annotation, transform=ax.transAxes, fontsize=7,
            verticalalignment="bottom", bbox=dict(boxstyle="round,pad=0.3", alpha=0.15))
    fig.tight_layout()
    if out_prefix != "SHOW":
        fig.savefig(f"{out_prefix}.png")
    else:
        plt.show()
    plt.close(fig)

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

    return complexities, reg



def compute_latent_and_cluster(lift_matrix: np.ndarray, selected_markers: np.ndarray, markers_journals: np.ndarray, out_prefix: str = "projection_2d",
                               eps_dbscan: float = 0.10, min_samples_dbscan: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Compute latent 2D embedding (UMAP) from lift matrix-derived distances and run DBSCAN clustering.

    Saves projection and projection with DBSCAN into PNG files with given prefix.
    Returns embedding and cluster labels.
    """
    epsilon = 1e-4
    velocities = np.array([lift_matrix[i, i] ** (-1) if lift_matrix[i, i] > 0 else np.nan for i in range(len(lift_matrix))])
    # safeguard and produce a symmetric non-negative distance matrix
    logger.info("Building distance matrix (%dx%d)...", lift_matrix.shape[0], lift_matrix.shape[1])
    distance_matrix = np.log1p((lift_matrix + epsilon) ** (-1))
    distance_matrix[np.diag_indices_from(distance_matrix)] = 0
    distance_matrix = (distance_matrix + distance_matrix.T) / 2.0
    distance_matrix -= distance_matrix.min()

    logger.info("Running UMAP (precomputed, %d markers)...", len(selected_markers))
    umap = UMAP(n_components=2, metric="precomputed", min_dist=0.10, random_state=42)
    X_latent = umap.fit_transform(distance_matrix)
    logger.info("UMAP done.")

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

    logger.info("Running DBSCAN (eps=%.3f, min_samples=%d)...", eps_dbscan, min_samples_dbscan)
    dbscan = DBSCAN(metric="euclidean", eps=eps_dbscan, min_samples=min_samples_dbscan)
    dbscan.fit(X_latent)
    labels = dbscan.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    logger.info("DBSCAN done: %d clusters, %d noise points", n_clusters, n_noise)

    plt.figure(figsize=(10, 8))
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

def run_all(root: Path = Path("data/causalitylink_sample"), cluster_id: Optional[int] = None) -> None:
    """Main pipeline: load data, select markers, compute matrices, plot and cluster.

    This function mirrors the previous top-level script but organizes steps and saves outputs.

    Args:
        root: Path to data root directory.
        cluster_id: If provided, after clustering compute a sub-lift matrix for markers
                    belonging to that DBSCAN cluster and plot complexity vs velocity within it.
    """
    Path("plots").mkdir(exist_ok=True)

    logger.info("=== Step 1/5 : Loading and filtering data ===")
    list_themes = ["sante", "economie", "sport", "politique", "transport", "information"]
    filtered_marker_df = prepare_filtered_marker_table(root, None)

    logger.info("=== Step 2/5 : Selecting markers ===")
    selected_markers, conv, markers_journals = select_markers_by_theme(filtered_marker_df, list_themes, fraction=1 / 3)

    logger.info("=== Step 3/5 : Computing cocitation matrix (%d markers) — this is the slow step ===", len(selected_markers))
    cocitation_matrix = compute_cocitation_probability_matrix(selected_markers, filtered_marker_df, conv)

    logger.info("=== Step 4/5 : Computing lift matrix and stats ===")
    lift_matrix = compute_lift_matrix(cocitation_matrix)
    logger.info("Lift matrix computed (%dx%d)", lift_matrix.shape[0], lift_matrix.shape[1])

    logger.info("Plotting complexity vs velocity...")
    complexities, reg = plot_complexity_vs_velocity(lift_matrix, conv, selected_markers, out_prefix="plots/complexity_vs_velocity")

    logger.info("=== Step 5/5 : UMAP + DBSCAN clustering ===")
    X_latent, labels = compute_latent_and_cluster(lift_matrix, selected_markers, markers_journals, out_prefix="plots/projection_2d", eps_dbscan=0.25, min_samples_dbscan=50)

    if cluster_id is not None:
        logger.info("=== Sub-complexity analysis for cluster %d ===", cluster_id)
        cluster_markers = markers_from_cluster(labels, cluster_id, selected_markers)
        if len(cluster_markers) < 10:
            logger.warning("Cluster %d has fewer than 10 markers (%d), skipping sub-analysis.", cluster_id, len(cluster_markers))
        else:
            logger.info("Cluster %d: %d markers — computing sub-lift matrix...", cluster_id, len(cluster_markers))
            sub_lift_matrix, sub_conv = compute_sub_lift_matrix(cluster_markers, filtered_marker_df)
            logger.info("Sub-lift matrix computed (%dx%d)", sub_lift_matrix.shape[0], sub_lift_matrix.shape[1])
            out_prefix = f"plots/cluster_{cluster_id}_complexity_vs_velocity"
            logger.info("Plotting cluster %d complexity vs velocity -> %s.png", cluster_id, out_prefix)
            sub_complexities, sub_reg = plot_complexity_vs_velocity(
                sub_lift_matrix, sub_conv, cluster_markers, out_prefix=out_prefix
            )
            logger.info("Cluster %d sub-analysis done.", cluster_id)

    logger.info("=== All done. Plots saved in plots/ ===")
    return filtered_marker_df, selected_markers, conv, markers_journals, lift_matrix, complexities, reg, labels


if __name__ == "__main__":

    root = Path("data/causalitylink_sample")
    cluster_id = 0
   
    filtered_marker_df, selected_markers, conv, markers_journals, lift_matrix, complexities, reg, labels = run_all(
        root=root,
        cluster_id=cluster_id,  # set to None to skip sub-analysis
    )
