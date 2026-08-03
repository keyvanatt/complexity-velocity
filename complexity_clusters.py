"""Main pipeline: lift matrix, complexity/velocity, UMAP + DBSCAN clustering.

Run ``python complexity_clusters.py --help`` for the command-line interface.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from causalityTable import CausalityTable
from lift_dissimilarity import lift_dissimilarity
from scipy import stats
from sklearn.cluster import DBSCAN
from tqdm import tqdm
from umap import UMAP  # pip install umap-learn


logger = logging.getLogger(__name__)

# Default locations, all relative to the repository root so the pipeline runs
# out of the box from a fresh clone.
DEFAULT_DATA_DIR = Path("data")
DEFAULT_ROOT = DEFAULT_DATA_DIR / "causalitylink_sample"
PLOTS_DIR = Path("plots")
CLUSTERS_DIR = Path("clusters")

# Journal themes retained for marker selection (labels as they appear in
# data/journaux_themes.csv).
DEFAULT_THEMES = ["sante", "economie", "sport", "politique", "transport", "information"]


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

    # total unique articles
    n_articles = int(df["id"].n_unique())

    # filter rows to selected markers and group markers per article
    df_filtered = df.filter(pl.col("marker").is_in(markers))
    df_grouped = df_filtered.group_by("id").agg(pl.col("marker").unique().alias("markers"))
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


def prepare_filtered_marker_table(
    path: Path,
    list_themes: Optional[List[str]] = None,
    data_dir: Path = DEFAULT_DATA_DIR,
    year: int = 2025,
    month: int = 1,
) -> pl.DataFrame:
    """Load tables and prepare the filtered marker DataFrame enriched with publisher info.

    Args:
        path: directory holding the ``Markers/`` and ``Tree/`` AVRO folders.
        list_themes: optional journal themes to keep (``None`` keeps all).
        data_dir: directory holding ``CausalityLinkPublishers.csv`` and
            ``journaux_themes.csv``.
        year, month: the monthly snapshot to load from ``Markers/``.

    Returns:
        A Polars DataFrame ready for further analysis.
    """
    logger.info("Loading Markers AVRO files from %s...", path / "Markers")
    markerTable = CausalityTable(path / "Markers")
    markerTable.load_one_mounth(year=year, month=month)
    logger.info("Markers loaded: %d rows", len(markerTable.df))

    logger.info("Loading Tree AVRO files from %s...", path / "Tree")
    treeTable = CausalityTable(path / "Tree")
    treeTable.load_data(date_parsing=False)
    logger.info("Tree loaded: %d rows", len(treeTable.df))

    data_dir = Path(data_dir)
    logger.info("Loading publishers and journal themes CSV from %s...", data_dir)
    publishers = pl.read_csv(data_dir / "CausalityLinkPublishers.csv")
    journaux_themes = pd.read_csv(data_dir / "journaux_themes.csv", index_col=0).to_dict()["theme"]
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


def select_markers_by_theme(filtered_marker_df: pl.DataFrame, themes: Optional[List[str]] = None, fraction: float = 1 / 3, top: bool = True, seed: int = 42):
    """Select markers appearing in given journal themes and return markers array and conv mapping.

    Args:
        filtered_marker_df: Polars DataFrame with `journal_theme` and `marker` columns.
        themes: list of themes to keep.
        fraction: fraction of markers by count to keep.
        top: if True, select top markers by count, else random sample.
    """
    if themes is None:
        themes = filtered_marker_df["journal_theme"].unique().to_list()

    logger.info("Selecting markers for themes: %s", themes, )
    selected_markers_df = (
        filtered_marker_df
        .filter(pl.col("journal_theme").is_in(themes))["marker", "publisher_label"]
        .group_by("marker")
        .agg(pl.col("publisher_label").unique().alias("publishers_label"), pl.col("marker").count().alias("marker_count"))
    )
    logger.info("Selected markers from %d publishers", selected_markers_df["publishers_label"].explode().n_unique())
    keep_n = max(1, int(len(selected_markers_df) * fraction))
    logger.info("Total distinct markers in themes: %d - keeping top %d (fraction=%.2f)", len(selected_markers_df), keep_n, fraction)
    if top:
        selected_markers_df = selected_markers_df.sort(["marker_count", "marker"], descending=[True, False]).head(keep_n)
    else:
        selected_markers_df = selected_markers_df.sample(keep_n, shuffle=True, seed=seed)
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
        "Log-log regression: beta1=%.4f  95%%CI=[%.4f, %.4f]  R2=%.4f  "
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



def compute_latent_and_cluster(lift_matrix: np.ndarray, selected_markers: np.ndarray, out_prefix: str = "projection_2d",
                               eps_dbscan: float = 0.10, min_samples_dbscan: int = 20, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Compute latent 2D embedding (UMAP) from lift matrix-derived distances and run DBSCAN clustering.

    The dissimilarity fed to UMAP is ``lift_dissimilarity.lift_dissimilarity``,
    shared with the synthetic benchmarks so the two cannot drift apart.

    Saves projection and projection with DBSCAN into PNG files with given prefix.
    Returns embedding and cluster labels.
    """
    logger.info("Building distance matrix (%dx%d)...", lift_matrix.shape[0], lift_matrix.shape[1])
    distance_matrix = lift_dissimilarity(lift_matrix)

    logger.info("Running UMAP (precomputed, %d markers)...", len(selected_markers))
    umap = UMAP(n_components=2, metric="precomputed", min_dist=0.10, random_state=seed)
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

    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.title(f"2D projection with DBSCAN clustering (n_clusters = {n_clusters})")
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

def top_lifters(marker: str, lift_matrix: np.ndarray, conv: Dict[str, int], top_n: int = 15) -> List[Tuple[str, float]]:
    """Return the top N markers with highest lift with respect to the given marker."""
    if marker not in conv:
        return []
    ind = conv[marker]
    lifts = lift_matrix[ind, :]
    index_to_marker = list(conv.keys())
    top_indices = np.argsort(lifts)[-top_n - 1:][::-1]
    if ind in top_indices:
        top_indices = top_indices[top_indices != ind]
    else:
        top_indices = top_indices[:-1]
    return [(index_to_marker[i], float(lifts[i])) for i in top_indices]

def save_top_bottom_csv(complexities: Dict[str, float], out_path, top_n: int = 10) -> pd.DataFrame:
    """Save a CSV with the top_n lowest and top_n highest markers by complexity.

    Returns the combined DataFrame.
    """
    df = (
        pd.DataFrame.from_dict(complexities, orient="index", columns=["complexity"])
        .rename_axis("marker")
        .sort_values("complexity")
    )
    bottom = df.head(top_n).assign(rank="bottom")
    top = df.tail(top_n).assign(rank="top")
    result = pd.concat([bottom, top])
    result.to_csv(out_path)
    logger.info("Saved top/bottom %d markers to %s", top_n, out_path)
    return result


def plot_cluster_distributions(
    filtered_marker_df: pl.DataFrame,
    selected_markers: np.ndarray,
    lift_matrix: np.ndarray,
    conv: Dict[str, int],
    labels: np.ndarray,
) -> None:
    """Dot plots (vertical layout) of beta1, kendall tau, pearson r per cluster.

    Also saves ``clusters/all_clusters_stats.csv`` and, for each cluster,
    ``clusters/cluster_<id>_all_markers.csv``.
    """
    CLUSTERS_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)
    rows: List[Dict] = []

    ids_to_run = sorted(int(l) for l in np.unique(labels) if l != -1)
    for cluster_id in ids_to_run:
        cluster_markers = markers_from_cluster(labels, cluster_id, selected_markers)
        n = len(cluster_markers)
        if n < 10:
            logger.warning("Cluster %d has fewer than 10 markers, skipping.", cluster_id)
            continue

        sub_lift_matrix, sub_conv = compute_sub_lift_matrix(cluster_markers, filtered_marker_df)
        c_vals = np.array([get_complexity_fast(sub_lift_matrix, sub_conv, m) for m in cluster_markers])
        v_vals = np.array([
            sub_lift_matrix[i, i] ** (-1) if sub_lift_matrix[i, i] > 0 else np.nan
            for i in range(len(sub_lift_matrix))
        ])

        reg = fit_loglog_regression(c_vals, v_vals)
        valid = np.isfinite(c_vals) & (c_vals > 0)

        # n_articles: unique articles mentioning at least one cluster marker
        n_articles = int(
            filtered_marker_df.filter(pl.col("marker").is_in(cluster_markers.tolist()))["id"].n_unique()
        )

        # intra-cluster lift: mean pairwise lift (off-diagonal) from global matrix
        cluster_idx = np.array([conv[m] for m in cluster_markers if m in conv], dtype=int)
        sub_intra = lift_matrix[np.ix_(cluster_idx, cluster_idx)]
        n_c = len(cluster_idx)
        if n_c > 1:
            mean_intra_lift = float((sub_intra.sum() - np.trace(sub_intra)) / (n_c * (n_c - 1)))
        else:
            mean_intra_lift = np.nan

        # external lift: mean lift from markers outside the cluster onto cluster members
        all_idx = np.arange(lift_matrix.shape[0])
        external_mask = np.ones(lift_matrix.shape[0], dtype=bool)
        external_mask[cluster_idx] = False
        external_idx = all_idx[external_mask]
        if len(external_idx) > 0 and len(cluster_idx) > 0:
            mean_external_lift = float(lift_matrix[np.ix_(cluster_idx, external_idx)].mean())
        else:
            mean_external_lift = np.nan

        # Save full per-cluster marker data for LLM judge
        pd.DataFrame({
            "marker": cluster_markers,
            "complexity": c_vals,
            "velocity": v_vals,
        }).to_csv(CLUSTERS_DIR / f"cluster_{cluster_id}_all_markers.csv", index=False)

        rows.append({
            "cluster_id": cluster_id,
            "n_kpi": n,
            "n_articles": n_articles,
            "n_valid": reg["n"],
            "mean_intra_lift": mean_intra_lift,
            "mean_external_lift": mean_external_lift,
            "complexity_mean": float(c_vals[valid].mean()) if valid.sum() > 0 else np.nan,
            "complexity_median": float(np.median(c_vals[valid])) if valid.sum() > 0 else np.nan,
            "beta0": reg["beta0"],
            "beta1": reg["beta1"],
            "beta1_ci_low": reg["beta1_ci"][0],
            "beta1_ci_high": reg["beta1_ci"][1],
            "r2": reg["r2"],
            "kendall_tau": reg["kendall_tau"],
            "kendall_p": reg["kendall_p"],
            "pearson_r": reg["pearson_r"],
            "pearson_p": reg["pearson_p"],
            "complexity_min": float(c_vals[valid].min()) if valid.sum() > 0 else np.nan,
            "complexity_max": float(c_vals[valid].max()) if valid.sum() > 0 else np.nan,
            "complexity_span": float(c_vals[valid].max() - c_vals[valid].min()) if valid.sum() > 0 else np.nan,
        })

    # save CSV
    csv_path = CLUSTERS_DIR / "all_clusters_stats.csv"
    df_stats = pd.DataFrame(rows).set_index("cluster_id")
    df_stats.to_csv(csv_path)
    logger.info("Saved cluster stats CSV to %s (%d clusters)", csv_path, len(df_stats))

    # dot plots — 3 rows × 1 col
    metrics = [
        ("beta1",       "β₁",           "steelblue"),
        ("kendall_tau", "Kendall's τ",   "darkorange"),
        ("pearson_r",   "Pearson r",     "green"),
    ]

    cluster_ids = [r["cluster_id"] for r in rows]
    y_pos = np.arange(len(cluster_ids))
    y_labels = [str(c) for c in cluster_ids]

    fig, axes = plt.subplots(3, 1, figsize=(5, 8), sharex=False)

    for ax, (col, xlabel, color) in zip(axes, metrics):
        values = np.array([r[col] for r in rows])
        ax.scatter(values, y_pos, color=color, s=30, zorder=3)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_ylabel("Cluster ID", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_title(f"{xlabel} per cluster", fontsize=10)
        ax.grid(axis="x", alpha=0.3)

    fig.tight_layout(pad=2.0)
    dot_plot_path = PLOTS_DIR / "distributions_beta1_kendall_pearson.png"
    fig.savefig(dot_plot_path, dpi=150)
    logger.info("Saved dot plot to %s", dot_plot_path)
    plt.close(fig)


def plot_all_clusters_grid(
    filtered_marker_df: pl.DataFrame,
    selected_markers: np.ndarray,
    labels: np.ndarray,
    n_cols: int = 4,
    out_path: Optional[Path] = None,
) -> None:
    """Grid of complexity vs velocity scatter + fit line for every cluster.

    Layout: n_cols columns, ceil(n_clusters / n_cols) rows.
    No stats annotations — scatter and fit only.
    """
    out_path = Path(out_path) if out_path is not None else PLOTS_DIR / "all_clusters_complexity_velocity.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_ids = sorted(int(l) for l in np.unique(labels) if l != -1)
    n_clusters = len(cluster_ids)
    n_rows = (n_clusters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.0))
    axes_flat = np.array(axes).flatten()

    for ax_idx, cluster_id in enumerate(cluster_ids):
        ax = axes_flat[ax_idx]
        cluster_markers = markers_from_cluster(labels, cluster_id, selected_markers)

        if len(cluster_markers) < 5:
            ax.set_visible(False)
            continue

        sub_lift_matrix, sub_conv = compute_sub_lift_matrix(cluster_markers, filtered_marker_df)
        c_vals = np.array([get_complexity_fast(sub_lift_matrix, sub_conv, m) for m in cluster_markers])
        v_vals = np.array([
            sub_lift_matrix[i, i] ** (-1) if sub_lift_matrix[i, i] > 0 else np.nan
            for i in range(len(sub_lift_matrix))
        ])

        valid = np.isfinite(c_vals) & np.isfinite(v_vals) & (c_vals > 0) & (v_vals > 0)
        ax.scatter(c_vals[valid], v_vals[valid], s=3, alpha=0.4)

        if valid.sum() >= 3:
            reg = fit_loglog_regression(c_vals, v_vals)
            c_fit = np.array([c_vals[valid].min(), c_vals[valid].max()])
            v_fit = np.exp(reg["beta0"]) * c_fit ** reg["beta1"]
            ax.plot(c_fit, v_fit, color="red", linewidth=1.2)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"Cluster {cluster_id}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("Complexity", fontsize=7)
        ax.set_ylabel("Velocity", fontsize=7)

    for ax_idx in range(len(cluster_ids), len(axes_flat)):
        axes_flat[ax_idx].set_visible(False)

    # single shared legend via proxy artists
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="steelblue", markersize=5, alpha=0.6, label="markers"),
        Line2D([0], [0], color="red", linewidth=1.5, label="log-log fit"),
    ]
    fig.legend(handles=legend_handles, loc="lower right", fontsize=8, framealpha=0.8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    logger.info("Saved all-clusters grid to %s", out_path)
    plt.close(fig)


def run_all(
    root: Path = DEFAULT_ROOT,
    data_dir: Path = DEFAULT_DATA_DIR,
    cluster_ids: Optional[List[int]] = None,
    all_clusters: bool = False,
    list_themes: Optional[List[str]] = None,
    marker_fraction: float = 1 / 3,
    eps_dbscan: float = 0.25,
    min_samples_dbscan: int = 60,
    seed: int = 42,
) -> Tuple:
    """Main pipeline: load data, select markers, compute matrices, plot and cluster.

    Args:
        root: path to the directory holding the ``Markers/`` and ``Tree/`` AVRO folders.
        data_dir: path to the directory holding the publisher/theme CSVs.
        cluster_ids: explicit list of DBSCAN cluster IDs to analyse.
        all_clusters: if True, run the sub-analysis on every cluster found by
            DBSCAN (noise -1 excluded). Overrides ``cluster_ids``.
        list_themes: journal themes used to select markers (default: DEFAULT_THEMES).
        marker_fraction: fraction of the most frequent markers to keep.
        eps_dbscan, min_samples_dbscan: DBSCAN parameters on the UMAP embedding.
        seed: seed for UMAP and for the marker sampling.

    Returns:
        ``(filtered_marker_df, selected_markers, conv, markers_journals,
        lift_matrix, complexities, reg, labels)``
    """
    PLOTS_DIR.mkdir(exist_ok=True)
    CLUSTERS_DIR.mkdir(exist_ok=True)
    list_themes = list_themes if list_themes is not None else DEFAULT_THEMES

    logger.info("=== Step 1/5 : Loading and filtering data ===")
    filtered_marker_df = prepare_filtered_marker_table(root, None, data_dir=data_dir)

    logger.info("=== Step 2/5 : Selecting markers ===")
    selected_markers, conv, markers_journals = select_markers_by_theme(filtered_marker_df, list_themes, fraction=marker_fraction, seed=seed)

    logger.info("=== Step 3/5 : Computing cocitation matrix (%d markers) - this is the slow step ===", len(selected_markers))
    cocitation_matrix = compute_cocitation_probability_matrix(selected_markers, filtered_marker_df, conv)

    logger.info("=== Step 4/5 : Computing lift matrix and stats ===")
    lift_matrix = compute_lift_matrix(cocitation_matrix)
    logger.info("Lift matrix computed (%dx%d)", lift_matrix.shape[0], lift_matrix.shape[1])

    logger.info("Plotting global complexity vs velocity...")
    complexities, reg = plot_complexity_vs_velocity(lift_matrix, conv, selected_markers, out_prefix=str(PLOTS_DIR / "complexity_vs_velocity"))

    logger.info("=== Step 5/5 : UMAP + DBSCAN clustering ===")
    _, labels = compute_latent_and_cluster(
        lift_matrix, selected_markers, out_prefix=str(PLOTS_DIR / "projection_2d"),
        eps_dbscan=eps_dbscan, min_samples_dbscan=min_samples_dbscan, seed=seed,
    )

    ids_to_run: List[int] = sorted(int(l) for l in np.unique(labels) if l != -1) if all_clusters else (cluster_ids or [])
    if ids_to_run:
        logger.info("=== Sub-cluster analysis for clusters: %s ===", ids_to_run)
        for cluster_id in ids_to_run:
            logger.info("--- Cluster %d ---", cluster_id)
            cluster_markers = markers_from_cluster(labels, cluster_id, selected_markers)
            if len(cluster_markers) < 10:
                logger.warning("Cluster %d has fewer than 10 markers (%d), skipping.", cluster_id, len(cluster_markers))
                continue
            logger.info("Cluster %d: %d markers - computing sub-lift matrix...", cluster_id, len(cluster_markers))
            sub_lift_matrix, sub_conv = compute_sub_lift_matrix(cluster_markers, filtered_marker_df)
            logger.info("Sub-lift matrix computed (%dx%d)", sub_lift_matrix.shape[0], sub_lift_matrix.shape[1])

            plot_prefix = CLUSTERS_DIR / f"cluster_{cluster_id}_complexity_vs_velocity"
            sub_complexities, _ = plot_complexity_vs_velocity(
                sub_lift_matrix, sub_conv, cluster_markers, out_prefix=str(plot_prefix)
            )

            save_top_bottom_csv(sub_complexities, CLUSTERS_DIR / f"cluster_{cluster_id}_top_bottom.csv")
            logger.info("Cluster %d done.", cluster_id)

    if all_clusters or ids_to_run:
        logger.info("=== Plotting cluster distributions ===")
        plot_cluster_distributions(filtered_marker_df, selected_markers, lift_matrix, conv, labels)
        logger.info("=== Plotting all-clusters grid ===")
        plot_all_clusters_grid(filtered_marker_df, selected_markers, labels)

    logger.info("=== All done. Figures in %s/, per-cluster tables in %s/ ===", PLOTS_DIR, CLUSTERS_DIR)

    return filtered_marker_df, selected_markers, conv, markers_journals, lift_matrix, complexities, reg, labels


def configure_logging(level: int = logging.INFO) -> None:
    """Send this package's progress logs to stderr.

    Called from the ``__main__`` blocks rather than at import time, so importing
    the module from a notebook or another script leaves logging untouched.
    """
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="Directory holding the Markers/ and Tree/ AVRO folders (default: %(default)s)")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="Directory holding CausalityLinkPublishers.csv and journaux_themes.csv (default: %(default)s)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all-clusters", action="store_true",
                       help="Run the per-cluster sub-analysis on every DBSCAN cluster")
    group.add_argument("--cluster-ids", nargs="+", type=int,
                       help="Run the sub-analysis only on these DBSCAN cluster IDs")
    parser.add_argument("--marker-fraction", type=float, default=1 / 3,
                        help="Fraction of the most frequent markers to keep (default: %(default)s)")
    parser.add_argument("--eps-dbscan", type=float, default=0.25, help="DBSCAN eps (default: %(default)s)")
    parser.add_argument("--min-samples-dbscan", type=int, default=60,
                        help="DBSCAN min_samples (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: %(default)s)")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    configure_logging()
    np.random.seed(args.seed)
    run_all(
        root=args.root,
        data_dir=args.data_dir,
        cluster_ids=args.cluster_ids,
        all_clusters=args.all_clusters,
        marker_fraction=args.marker_fraction,
        eps_dbscan=args.eps_dbscan,
        min_samples_dbscan=args.min_samples_dbscan,
        seed=args.seed,
    )
