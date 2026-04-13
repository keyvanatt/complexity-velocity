"""Per-publisher complexity vs velocity analysis for a given DBSCAN cluster.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl

from complexity_clusters import (
    compute_cocitation_probability_matrix,
    compute_latent_and_cluster,
    compute_lift_matrix,
    compute_sub_lift_matrix,
    get_complexity_fast,
    markers_from_cluster,
    prepare_filtered_marker_table,
    select_markers_by_theme,
)


def compute_publisher_lift_matrices(
    sub_selected_markers: np.ndarray,
    filtered_marker_df: pl.DataFrame,
    publishers: List[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, int]]]:
    """Compute per-publisher lift matrices for a subset of markers.

    Args:
        sub_selected_markers: markers belonging to the cluster of interest.
        filtered_marker_df: full filtered marker DataFrame.
        publishers: list of publisher labels to process.

    Returns:
        Tuple of (lift_matrices, conv_dicts) keyed by publisher label.
    """
    lift_matrices: Dict[str, np.ndarray] = {}
    conv_dicts: Dict[str, Dict[str, int]] = {}
    for publisher in publishers:
        publisher_df = filtered_marker_df.filter(pl.col("publisher_label") == publisher)
        lift_matrices[publisher], conv_dicts[publisher] = compute_sub_lift_matrix(
            sub_selected_markers, publisher_df
        )
    return lift_matrices, conv_dicts


def top_publishers_for_cluster(
    sub_selected_markers: np.ndarray,
    filtered_marker_df: pl.DataFrame,
    top_n: int = 10,
) -> List[str]:
    """Return the top N publishers by article count for a set of cluster markers."""
    return (
        filtered_marker_df.filter(pl.col("marker").is_in(sub_selected_markers))
        .group_by("publisher_label")
        .len()
        .sort("len", descending=True)
        .head(top_n)["publisher_label"]
        .to_list()
    )


def plot_complexity_vs_velocity_publishers(
    lift_matrices: Dict[str, np.ndarray],
    conv_dicts: Dict[str, Dict[str, int]],
    sub_selected_markers: np.ndarray,
    publishers: List[str],
    journaux_themes: Dict[str, str],
    out_prefix: str = "complexity_vs_velocity_publishers",
    center_percentile: Tuple[float, float] = (20.0, 80.0),
) -> None:
    """Scatter plot of complexity vs velocity with a power-law fit per publisher.

    For each publisher a log-log linear regression is fitted on the central
    ``center_percentile`` range of complexity values and overlaid as a dashed
    curve.

    Args:
        lift_matrices: per-publisher lift matrices.
        conv_dicts: per-publisher marker -> index mappings.
        sub_selected_markers: markers of the cluster.
        publishers: publishers to include in the plot.
        journaux_themes: mapping publisher_label -> theme string (used in legend).
        out_prefix: file prefix for saved PNG, or ``"SHOW"`` to display inline.
        center_percentile: (low, high) percentile bounds used to exclude outliers
            before fitting the power law.
    """
    fig, ax = plt.subplots()
    eps = 1e-10
    low_p, high_p = center_percentile

    for publisher in publishers:
        lift_matrix = lift_matrices[publisher]
        conv = conv_dicts[publisher]

        complexities = np.array(
            [get_complexity_fast(lift_matrix, conv, m) for m in sub_selected_markers],
            dtype=float,
        )
        velocities = np.array(
            [
                lift_matrix[i, i] ** (-1) if lift_matrix[i, i] > 0 else np.nan
                for i in range(len(lift_matrix))
            ],
            dtype=float,
        )

        theme = journaux_themes.get(publisher, "?")
        ax.scatter(complexities, velocities, label=f"{publisher} ({theme})", alpha=1, s=3)

        log_c = np.log(complexities + eps)
        log_v = np.log(velocities + eps)

        valid = (
            np.isfinite(log_c)
            & np.isfinite(log_v)
            & (complexities > 0)
            & (velocities > 0)
        )
        if valid.sum() < 2:
            continue

        c_low = np.percentile(complexities[valid], low_p)
        c_high = np.percentile(complexities[valid], high_p)
        center = valid & (complexities >= c_low) & (complexities <= c_high)

        if center.sum() < 2 or np.var(log_c[center]) == 0:
            continue

        a = np.cov(log_c[center], log_v[center])[0, 1] / np.var(log_c[center])
        b = np.mean(log_v[center]) - a * np.mean(log_c[center])

        x_fit = np.linspace(c_low, c_high, 100)
        y_fit = np.exp(a * np.log(x_fit + eps) + b) - eps
        ax.plot(x_fit, y_fit, linestyle="--", linewidth=2, label=f"{publisher} fit (α={a:.2f})")

    ax.set_xlabel("Complexity")
    ax.set_ylabel("Velocity")
    ax.set_title("Marker Complexity vs Velocity — per publisher (log-log)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    fig.tight_layout()

    if out_prefix == "SHOW":
        plt.show()
    else:
        fig.savefig(f"{out_prefix}.png")
    plt.close(fig)


def run_publisher_analysis(
    root: Path = Path("data/causalitylink_sample"),
    list_themes: Optional[List[str]] = None,
    marker_fraction: float = 1 / 3,
    cluster_id: int = 12,
    top_n_publishers: int = 10,
    publishers_to_plot: Optional[List[str]] = None,
    eps_dbscan: float = 0.23,
    min_samples_dbscan: int = 30,
    out_prefix: str = "plots/complexity_vs_velocity_publishers",
) -> None:
    """End-to-end pipeline: load data, cluster markers, plot per-publisher complexity.

    Args:
        root: path to the CausalityLink sample directory.
        list_themes: journal themes used to filter markers (default: 6 broad themes).
        marker_fraction: fraction of top markers to keep for the global analysis.
        cluster_id: DBSCAN cluster label to analyse per publisher.
        top_n_publishers: number of top publishers to use if ``publishers_to_plot`` is None.
        publishers_to_plot: explicit list of publishers to plot (overrides ``top_n_publishers``).
        eps_dbscan: epsilon parameter for DBSCAN.
        min_samples_dbscan: min_samples parameter for DBSCAN.
        out_prefix: prefix for output PNG files, or ``"SHOW"`` to display inline.
    """
    if list_themes is None:
        list_themes = ["sante", "economie", "sport", "politique", "transport", "information"]

    Path("plots").mkdir(exist_ok=True)
    filtered_marker_df = prepare_filtered_marker_table(root, None)

    journaux_themes: Dict[str, str] = (
        pd.read_csv("data/journaux_themes.csv", index_col=0).squeeze().to_dict()
    )

    selected_markers, conv, markers_journals = select_markers_by_theme(
        filtered_marker_df, list_themes, fraction=marker_fraction, top=True
    )

    cocitation_matrix = compute_cocitation_probability_matrix(
        selected_markers, filtered_marker_df, conv
    )
    lift_matrix = compute_lift_matrix(cocitation_matrix)

    _, labels = compute_latent_and_cluster(
        lift_matrix,
        selected_markers,
        markers_journals,
        out_prefix=out_prefix + "_projection" if out_prefix != "SHOW" else "SHOW",
        eps_dbscan=eps_dbscan,
        min_samples_dbscan=min_samples_dbscan,
    )

    cluster_markers = markers_from_cluster(labels, cluster_id, selected_markers)

    if publishers_to_plot is None:
        publishers_to_plot = top_publishers_for_cluster(
            cluster_markers, filtered_marker_df, top_n=top_n_publishers
        )

    lift_matrices, conv_dicts = compute_publisher_lift_matrices(
        cluster_markers, filtered_marker_df, publishers_to_plot
    )

    plot_complexity_vs_velocity_publishers(
        lift_matrices,
        conv_dicts,
        cluster_markers,
        publishers_to_plot,
        journaux_themes,
        out_prefix=out_prefix,
    )


if __name__ == "__main__":
    np.random.seed(42)
    run_publisher_analysis(
        root=Path("data/causalitylink_sample"),
        cluster_id=12,           # cars cluster
        top_n_publishers=10,
        out_prefix="plots/complexity_vs_velocity_publishers",
    )
