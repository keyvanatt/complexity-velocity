"""Combined scatter + log-log regression plot for clusters 5, 9, 11, 13
and a comprehensive CSV with stats for all clusters.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from complexity_clusters import (
    compute_cocitation_probability_matrix,
    compute_latent_and_cluster,
    compute_lift_matrix,
    compute_sub_lift_matrix,
    fit_loglog_regression,
    get_complexity_fast,
    markers_from_cluster,
    prepare_filtered_marker_table,
    select_markers_by_theme,
)

CLUSTERS_TO_PLOT = [5, 9, 11, 13]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
MARKERS_STYLE = ["o", "s", "^", "D"]


def plot_selected_clusters(cluster_data: dict, out_path: str = "plots/clusters_5_9_11_13.png") -> None:
    """Scatter + log-log regression for clusters 5, 9, 11, 13 on a single figure."""
    fig, ax = plt.subplots(figsize=(10, 7))
    eps = 1e-10

    for i, cluster_id in enumerate(CLUSTERS_TO_PLOT):
        if cluster_id not in cluster_data:
            continue
        d = cluster_data[cluster_id]
        c_vals = d["complexities"]
        v_vals = d["velocities"]
        reg = d["reg"]

        valid = np.isfinite(c_vals) & np.isfinite(v_vals) & (c_vals > 0) & (v_vals > 0)

        ax.scatter(
            c_vals[valid], v_vals[valid],
            s=6, alpha=0.35,
            color=COLORS[i], marker=MARKERS_STYLE[i],
            label=f"Cluster {cluster_id}  (n={valid.sum()})",
        )

        # log-log regression line over valid range
        c_fit = np.array([c_vals[valid].min(), c_vals[valid].max()])
        v_fit = np.exp(reg["beta0"]) * c_fit ** reg["beta1"]
        ax.plot(
            c_fit, v_fit,
            color=COLORS[i], linewidth=2, linestyle="--",
            label=(
                f"Cluster {cluster_id} fit  "
                f"β₁={reg['beta1']:.3f}  "
                f"R²={reg['r2']:.3f}"
            ),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Complexity", fontsize=12)
    ax.set_ylabel("Velocity", fontsize=12)
    ax.set_title("Complexity vs Velocity — Clusters 5, 9, 11, 13 (Log-Log)", fontsize=13)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {out_path}")
    plt.close(fig)



def main():
    np.random.seed(42)
    root = Path("data/causalitylink_sample")
    Path("plots").mkdir(exist_ok=True)

    print("=== Step 1: Loading and filtering data ===")
    filtered_marker_df = prepare_filtered_marker_table(root, None)

    print("=== Step 2: Selecting markers ===")
    list_themes = ["sante", "economie", "sport", "politique", "transport", "information"]
    selected_markers, conv, markers_journals = select_markers_by_theme(
        filtered_marker_df, list_themes, fraction=1 / 3, seed=42
    )

    print("=== Step 3: Computing cocitation matrix ===")
    cocitation_matrix = compute_cocitation_probability_matrix(selected_markers, filtered_marker_df, conv)

    print("=== Step 4: Computing lift matrix ===")
    lift_matrix = compute_lift_matrix(cocitation_matrix)

    print("=== Step 5: UMAP + DBSCAN clustering ===")
    _, labels = compute_latent_and_cluster(
        lift_matrix, selected_markers, markers_journals,
        out_prefix="plots/projection_2d",
        eps_dbscan=0.25, min_samples_dbscan=60, seed=42,
    )

    unique_cluster_ids = sorted(int(l) for l in np.unique(labels) if l != -1)
    print(f"Found {len(unique_cluster_ids)} clusters: {unique_cluster_ids}")

    print("=== Step 6: Per-cluster sub-analysis ===")
    cluster_data = {}
    for cluster_id in unique_cluster_ids:
        cluster_markers = markers_from_cluster(labels, cluster_id, selected_markers)
        n = len(cluster_markers)
        if n < 5:
            print(f"  Cluster {cluster_id}: only {n} markers, skipping.")
            continue
        print(f"  Cluster {cluster_id}: {n} markers")

        sub_lift, sub_conv = compute_sub_lift_matrix(cluster_markers, filtered_marker_df)
        c_vals = np.array([get_complexity_fast(sub_lift, sub_conv, m) for m in cluster_markers])
        v_vals = np.array([
            sub_lift[i, i] ** (-1) if sub_lift[i, i] > 0 else np.nan
            for i in range(len(sub_lift))
        ])
        reg = fit_loglog_regression(c_vals, v_vals)

        cluster_data[cluster_id] = {
            "n_elements": n,
            "complexities": c_vals,
            "velocities": v_vals,
            "reg": reg,
        }

    print("=== Step 7: Generating combined plot for clusters 5, 9, 11, 13 ===")
    plot_selected_clusters(cluster_data, out_path="plots/clusters_5_9_11_13.png")



if __name__ == "__main__":
    main()
