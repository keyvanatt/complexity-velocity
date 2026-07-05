"""
FAIR cluster-recovery experiment (difficulty sweep).

Unlike cluster_recovery_experiment.py / cluster_recovery_sweep.py, here **no
method is allowed to read the generating matrix C**. Instead, every trial:

  1. generates C + ground-truth labels (low p_inter, high p_intra),
  2. simulates n_sim binary documents from C  (the ONLY thing methods observe),
  3. estimates the pairwise lift matrix from those documents, and
  4. clusters markers using representations derived *only* from that estimate.

This mirrors the real pipeline (complexity_clusters.py), where there is no
ground-truth C — lift computed from observed co-occurrence is the only signal.
All three methods share the same C-blind estimated lift; the only differences
are the algorithm and whether the "complexity" log-dissimilarity transform is
used:

    - kmeans           : k-means on StandardScaler([L, L^T])          (L = est. lift)
    - umap_euclidean   : UMAP(euclidean) on StandardScaler([L, L^T]) + HDBSCAN
    - umap_complexity  : UMAP(precomputed D=log(1+1/L)) + HDBSCAN     (complexity metric)

Run:  python cluster_recovery_fair.py
Outputs: results/cluster_recovery_fair.csv,
         results/cluster_recovery_fair_summary.csv,
         plots/cluster_recovery_fair.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
import umap
import hdbscan

from marker_clustering import generate_cluster_markers, simulate_markers, run_kmeans
from cluster_recovery_experiment import bootstrap_ci, t_ci

# ── Sweep configuration (matches cluster_recovery_sweep.py) ────────────────────
P_INTER_GRID  = [0.01, 0.05, 0.10, 0.20, 0.30]
N_TRIALS_PER  = 20
N_CLUSTERS    = 5
N_MARKERS     = 100
P_INTRA       = 0.99
W_MIN         = 0.8
W_MAX         = 1.0
N_SIM         = 5000
BASE_SEED     = 2000     # fresh, reproducible seeds

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


def estimate_lift_and_dissimilarity(sim, n_sim):
    """Empirical pairwise lift + complexity dissimilarity from documents.

    Identical construction to marker_clustering.run_umap_hdbscan (approach B):
        lift[i,j] = P(i,j) / (P(i)*P(j)),   D = log(1 + 1/lift)
    """
    p_i  = sim.mean(axis=0)
    p_ij = (sim.T @ sim) / n_sim
    denom = np.outer(p_i, p_i)
    denom = np.where(denom > 0, denom, 1e-10)
    lift  = p_ij / denom
    lift  = np.where(lift > 0, lift, 1e-10)

    D = np.log1p(1.0 / lift)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    return lift, D


def run_fair(C, true_labels, n_sim=N_SIM, min_cluster_size=None, u=None, rng=None):
    """Cluster with all three methods from a C-blind estimated lift matrix.

    Returns (ari_kmeans, ari_umap_euclidean, ari_umap_complexity).
    """
    n = C.shape[0]
    n_true = len(np.unique(true_labels))
    min_cs = min_cluster_size or max(2, n // (n_true * 3))
    if u is None:
        u = np.full(n, 0.1)

    # ── the only thing any method observes: simulated co-occurrence ───────────
    sim = simulate_markers(C, u, n_docs=n_sim, rng=rng).astype(float)  # (n_sim, n)
    lift, D = estimate_lift_and_dissimilarity(sim, n_sim)

    # ── K-Means on standardized [L, L^T] (stability-selected k) ───────────────
    # run_kmeans internally does StandardScaler(hstack([lift, lift.T])); since
    # lift is symmetric this is the fair euclidean analogue of the old [C, C^T].
    _, _, ari_km, _ = run_kmeans(lift, true_labels)

    # ── UMAP (euclidean) on the same estimated-lift features + HDBSCAN ────────
    X = StandardScaler().fit_transform(np.hstack([lift, lift.T]))
    emb_E = umap.UMAP(n_components=2, metric="euclidean", random_state=42).fit_transform(X)
    ari_ue = adjusted_rand_score(true_labels,
                                 hdbscan.HDBSCAN(min_cluster_size=min_cs).fit_predict(emb_E))

    # ── UMAP + complexity: precomputed log-dissimilarity + HDBSCAN ────────────
    emb_C = umap.UMAP(n_components=2, metric="precomputed", random_state=42).fit_transform(D)
    ari_uc = adjusted_rand_score(true_labels,
                                 hdbscan.HDBSCAN(min_cluster_size=min_cs).fit_predict(emb_C))

    return ari_km, ari_ue, ari_uc


def run_sweep():
    rows = []
    seed = BASE_SEED
    for p_inter in P_INTER_GRID:
        for t in tqdm(range(N_TRIALS_PER), desc=f"p_inter={p_inter}"):
            rng = np.random.default_rng(seed)
            seed += 1
            C, labels = generate_cluster_markers(
                N_CLUSTERS, N_MARKERS, P_INTRA, W_MIN, W_MAX, p_inter, rng=rng
            )
            ari_km, ari_ue, ari_uc = run_fair(C, labels, n_sim=N_SIM, rng=rng)
            rows.append({
                "p_inter":         p_inter,
                "trial":           t,
                "kmeans":          ari_km,
                "umap_euclidean":  ari_ue,
                "umap_complexity": ari_uc,
            })
    return pd.DataFrame(rows)


def summarize(df):
    rows = []
    for p_inter in P_INTER_GRID:
        sub = df[df["p_inter"] == p_inter]
        for m in METHODS:
            x = sub[m].to_numpy()
            b_lo, b_hi = bootstrap_ci(x)
            t_lo, t_hi = t_ci(x)
            rows.append({
                "p_inter":      p_inter, "method": m, "label": METHOD_LABELS[m],
                "mean":         x.mean(), "std": x.std(ddof=1),
                "boot_ci_low":  b_lo, "boot_ci_high": b_hi,
                "t_ci_low":     t_lo, "t_ci_high": t_hi,
            })
    return pd.DataFrame(rows)


def plot_sweep(summary, save_path):
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
        f"FAIR cluster recovery vs difficulty — all methods C-blind\n"
        f"(p_intra={P_INTRA}, {N_CLUSTERS} clusters, {N_MARKERS} markers, "
        f"{N_TRIALS_PER} trials/level; bands = 95% bootstrap CI)"
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    Path("results").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)

    df = run_sweep()
    df.to_csv("results/cluster_recovery_fair.csv", index=False)

    summary = summarize(df)
    summary.to_csv("results/cluster_recovery_fair_summary.csv", index=False)

    plot_sweep(summary, "plots/cluster_recovery_fair.png")

    print("\n" + "=" * 78)
    print(f"FAIR cluster-recovery sweep (all methods C-blind) | {N_TRIALS_PER} trials/level")
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
    print("Wrote: results/cluster_recovery_fair.csv, "
          "results/cluster_recovery_fair_summary.csv, "
          "plots/cluster_recovery_fair.png")


if __name__ == "__main__":
    main()
