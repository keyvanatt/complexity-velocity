"""
Difficulty sweep for the cluster-recovery experiment.

Same three methods as cluster_recovery_experiment.py, but now the
inter-cluster connection probability p_inter is swept upward so the block
structure of the raw dependency matrix C progressively degrades. This is the
regime where the raw matrix stops being trivially separable, and where the
lift/complexity dissimilarity is expected to earn its keep.

For each p_inter level we run N_TRIALS_PER seeded matrices and record ARI for:
    - kmeans            : k-means on StandardScaler([C, C^T])
    - umap_euclidean    : UMAP(euclidean) + HDBSCAN            (no complexity metric)
    - umap_complexity   : UMAP(precomputed lift dissimilarity) + HDBSCAN

Run:  python cluster_recovery_sweep.py
Outputs: results/cluster_recovery_sweep.csv,
         results/cluster_recovery_sweep_summary.csv,
         plots/cluster_recovery_sweep.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

from marker_clustering import (
    generate_cluster_markers,
    run_kmeans,
    run_umap_hdbscan,
)
from cluster_recovery_experiment import bootstrap_ci, t_ci  # reuse stat helpers

# ── Sweep configuration ───────────────────────────────────────────────────────
P_INTER_GRID  = [0.01, 0.05, 0.10, 0.20, 0.30]
N_TRIALS_PER  = 20
N_CLUSTERS    = 5
N_MARKERS     = 100
P_INTRA       = 0.99
W_MIN         = 0.8
W_MAX         = 1.0
N_SIM         = 5000
BASE_SEED     = 1000     # offset so seeds differ from the single-regime run

METHODS = ["kmeans", "umap_euclidean", "umap_complexity"]
METHOD_LABELS = {
    "kmeans":          "K-Means",
    "umap_euclidean":  "UMAP (euclidean)",
    "umap_complexity": "UMAP + complexity",
}
METHOD_COLORS = {
    "kmeans":          plt.cm.tab10.colors[0],
    "umap_euclidean":  plt.cm.tab10.colors[1],
    "umap_complexity": plt.cm.tab10.colors[2],
}


def run_sweep():
    """Run all p_inter levels x trials; return per-trial DataFrame."""
    rows = []
    seed = BASE_SEED
    for p_inter in P_INTER_GRID:
        for t in tqdm(range(N_TRIALS_PER), desc=f"p_inter={p_inter}"):
            rng = np.random.default_rng(seed)
            seed += 1
            C, labels = generate_cluster_markers(
                N_CLUSTERS, N_MARKERS, P_INTRA, W_MIN, W_MAX, p_inter, rng=rng
            )
            _, _, ari_km, _ = run_kmeans(C, labels)
            (_, ari_A), (_, ari_B), _, _ = run_umap_hdbscan(
                C, labels, n_sim=N_SIM, rng=rng
            )
            rows.append({
                "p_inter":         p_inter,
                "trial":           t,
                "kmeans":          ari_km,
                "umap_euclidean":  ari_A,
                "umap_complexity": ari_B,
            })
    return pd.DataFrame(rows)


def summarize(df):
    """Per (p_inter, method): mean, std, bootstrap CI, t CI."""
    rows = []
    for p_inter in P_INTER_GRID:
        sub = df[df["p_inter"] == p_inter]
        for m in METHODS:
            x = sub[m].to_numpy()
            b_lo, b_hi = bootstrap_ci(x)
            t_lo, t_hi = t_ci(x)
            rows.append({
                "p_inter":       p_inter,
                "method":        m,
                "label":         METHOD_LABELS[m],
                "mean":          x.mean(),
                "std":           x.std(ddof=1),
                "boot_ci_low":   b_lo,
                "boot_ci_high":  b_hi,
                "t_ci_low":      t_lo,
                "t_ci_high":     t_hi,
            })
    return pd.DataFrame(rows)


def plot_sweep(summary, save_path):
    """Mean ARI vs p_inter, one line per method with 95% bootstrap CI band."""
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.array(P_INTER_GRID, dtype=float)
    for m in METHODS:
        sub = summary[summary["method"] == m].set_index("p_inter").loc[P_INTER_GRID]
        means = sub["mean"].to_numpy()
        lo = sub["boot_ci_low"].to_numpy()
        hi = sub["boot_ci_high"].to_numpy()
        c = METHOD_COLORS[m]
        ax.plot(x, means, marker="o", color=c, lw=2, label=METHOD_LABELS[m])
        ax.fill_between(x, lo, hi, color=c, alpha=0.18)

    ax.set_xlabel("p_inter  (inter-cluster connection probability)")
    ax.set_ylabel("Mean ARI")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(x)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_title(
        f"Cluster recovery vs difficulty  "
        f"(p_intra={P_INTRA}, {N_CLUSTERS} clusters, {N_MARKERS} markers, "
        f"{N_TRIALS_PER} trials/level)\nbands = 95% bootstrap CI"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    Path("results").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)

    df = run_sweep()
    df.to_csv("results/cluster_recovery_sweep.csv", index=False)

    summary = summarize(df)
    summary.to_csv("results/cluster_recovery_sweep_summary.csv", index=False)

    plot_sweep(summary, "plots/cluster_recovery_sweep.png")

    # ── console report ────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"Cluster-recovery difficulty sweep  |  {N_TRIALS_PER} trials/level")
    print(f"regime: {N_CLUSTERS} clusters, {N_MARKERS} markers, "
          f"p_intra={P_INTRA}, n_sim={N_SIM}")
    print("=" * 78)
    header = f"{'p_inter':>8}"
    for m in METHODS:
        header += f"   {METHOD_LABELS[m]:>20}"
    print(header)
    print("-" * 78)
    for p_inter in P_INTER_GRID:
        line = f"{p_inter:>8.2f}"
        for m in METHODS:
            r = summary[(summary["p_inter"] == p_inter) & (summary["method"] == m)].iloc[0]
            line += f"   {r['mean']:>6.3f} [{r['boot_ci_low']:.2f},{r['boot_ci_high']:.2f}]"
        print(line)
    print("-" * 78)
    # paired complexity - euclidean and complexity - kmeans per level (Wilcoxon)
    print("Paired Wilcoxon p-values (complexity vs baseline) per level:")
    for p_inter in P_INTER_GRID:
        sub = df[df["p_inter"] == p_inter]
        dk = sub["umap_complexity"].to_numpy() - sub["kmeans"].to_numpy()
        de = sub["umap_complexity"].to_numpy() - sub["umap_euclidean"].to_numpy()
        pk = 1.0 if np.allclose(dk, 0) else stats.wilcoxon(dk).pvalue
        pe = 1.0 if np.allclose(de, 0) else stats.wilcoxon(de).pvalue
        print(f"  p_inter={p_inter:.2f}: "
              f"vs kmeans  Δ={dk.mean():+.3f} (p={pk:.2e}) | "
              f"vs umap_eucl  Δ={de.mean():+.3f} (p={pe:.2e})")
    print("=" * 78)
    print("Wrote: results/cluster_recovery_sweep.csv, "
          "results/cluster_recovery_sweep_summary.csv, "
          "plots/cluster_recovery_sweep.png")


if __name__ == "__main__":
    main()
