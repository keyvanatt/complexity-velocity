"""
Cluster-recovery experiment: k-means vs UMAP + HDBSCAN (complexity metric).

Generates many cluster-structured dependency matrices with known ground-truth
labels (low inter-cluster connection probability, high intra-cluster), then
runs three clustering pipelines on each and measures how well each recovers the
true clusters via the Adjusted Rand Index (ARI):

    - kmeans            : k-means on StandardScaler([C, C^T]), stability-selected k
    - umap_euclidean    : UMAP(euclidean) + HDBSCAN            (approach A, no complexity metric)
    - umap_complexity   : UMAP(precomputed lift dissimilarity) + HDBSCAN  (approach B)

The lift dissimilarity D = log(1 + 1/lift) is the "complexity" metric: it is
derived from empirical pairwise lifts, lift[i,j] = P(i,j) / (P(i)*P(j)).

Reports, per method, mean ARI with 95% confidence intervals (bootstrap +
t-interval), and a paired superiority test (Wilcoxon signed-rank + bootstrap CI
on the paired difference) showing whether UMAP + complexity beats k-means.

Run:  python cluster_recovery_experiment.py
Outputs: results/cluster_recovery_results.csv,
         results/cluster_recovery_summary.csv,
         plots/cluster_recovery_summary.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: never block on plt.show()
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

from marker_clustering import (
    generate_cluster_markers,
    run_kmeans,
    run_umap_hdbscan,
)

# ── Experiment configuration ──────────────────────────────────────────────────
N_TRIALS   = 30
N_CLUSTERS = 5
N_MARKERS  = 100
P_INTRA    = 0.99      # high intra-cluster connection probability
P_INTER    = 0.01      # low inter-cluster connection probability
W_MIN      = 0.8
W_MAX      = 1.0
N_SIM      = 5000      # documents simulated for lift estimation (approach B)
BASE_SEED  = 0

N_BOOT     = 10_000    # bootstrap resamples for CIs
BOOT_SEED  = 12345

METHODS = ["kmeans", "umap_euclidean", "umap_complexity"]
METHOD_LABELS = {
    "kmeans":          "K-Means",
    "umap_euclidean":  "UMAP (euclidean)",
    "umap_complexity": "UMAP + complexity",
}


def run_trials():
    """Run all trials; return a DataFrame with one row per trial."""
    rows = []
    for t in tqdm(range(N_TRIALS), desc="trials"):
        rng = np.random.default_rng(BASE_SEED + t)
        C, labels = generate_cluster_markers(
            N_CLUSTERS, N_MARKERS, P_INTRA, W_MIN, W_MAX, P_INTER, rng=rng
        )

        _, _, ari_km, _ = run_kmeans(C, labels)
        (_, ari_A), (_, ari_B), _, _ = run_umap_hdbscan(
            C, labels, n_sim=N_SIM, rng=rng
        )

        rows.append({
            "trial":            t,
            "kmeans":           ari_km,
            "umap_euclidean":   ari_A,
            "umap_complexity":  ari_B,
        })
    return pd.DataFrame(rows)


def bootstrap_ci(x, statistic=np.mean, n_boot=N_BOOT, seed=BOOT_SEED, alpha=0.05):
    """Percentile bootstrap CI for a 1-D sample statistic."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boot = statistic(x[idx], axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi


def t_ci(x, alpha=0.05):
    """Student-t confidence interval for the mean."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    h = se * stats.t.ppf(1 - alpha / 2, df=n - 1)
    return mean - h, mean + h


def paired_comparison(a, b, name_a, name_b):
    """Paired difference (a - b) stats: mean diff, bootstrap CI, Wilcoxon p."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    lo, hi = bootstrap_ci(d)
    # Wilcoxon is undefined when all differences are zero.
    if np.allclose(d, 0.0):
        w_p = 1.0
    else:
        w_p = stats.wilcoxon(d, zero_method="wilcox").pvalue
    return {
        "comparison":     f"{name_a} - {name_b}",
        "mean_diff":      d.mean(),
        "ci_low":         lo,
        "ci_high":        hi,
        "wilcoxon_p":     w_p,
        "n_wins":         int(np.sum(d > 0)),
        "n_losses":       int(np.sum(d < 0)),
        "n_ties":         int(np.sum(d == 0)),
    }


def summarize(df):
    """Per-method summary rows (mean, std, bootstrap CI, t CI)."""
    rows = []
    for m in METHODS:
        x = df[m].to_numpy()
        b_lo, b_hi = bootstrap_ci(x)
        t_lo, t_hi = t_ci(x)
        rows.append({
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


def plot_summary(df, summary, save_path):
    """Bar chart (mean ARI ± 95% bootstrap CI) + boxplot of per-trial ARI."""
    fig, (ax_bar, ax_box) = plt.subplots(1, 2, figsize=(13, 5))
    palette = plt.cm.tab10.colors
    labels = [METHOD_LABELS[m] for m in METHODS]

    # ── left: mean ARI with asymmetric 95% bootstrap CI error bars ────────────
    means = summary.set_index("method").loc[METHODS, "mean"].to_numpy()
    lo = summary.set_index("method").loc[METHODS, "boot_ci_low"].to_numpy()
    hi = summary.set_index("method").loc[METHODS, "boot_ci_high"].to_numpy()
    yerr = np.vstack([means - lo, hi - means])
    x = np.arange(len(METHODS))
    ax_bar.bar(x, means, color=[palette[i] for i in range(len(METHODS))],
               alpha=0.85, edgecolor="black", linewidth=0.6)
    ax_bar.errorbar(x, means, yerr=yerr, fmt="none", ecolor="black",
                    capsize=6, capthick=1.2, lw=1.2)
    for xi, m in zip(x, means):
        ax_bar.text(xi, m + 0.02, f"{m:.3f}", ha="center", va="bottom", fontsize=10)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=10)
    ax_bar.set_ylabel("Mean ARI")
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_title(f"Mean cluster-recovery ARI  (N={N_TRIALS} matrices)\n"
                     "error bars = 95% bootstrap CI")

    # ── right: distribution of per-trial ARI ──────────────────────────────────
    data = [df[m].to_numpy() for m in METHODS]
    bp = ax_box.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55,
                        showmeans=True, meanprops=dict(marker="D", markerfacecolor="white",
                                                       markeredgecolor="black", markersize=6))
    for patch, i in zip(bp["boxes"], range(len(METHODS))):
        patch.set_facecolor(palette[i]); patch.set_alpha(0.6)
    # jittered points
    jitter_rng = np.random.default_rng(0)
    for i, y in enumerate(data):
        xj = (i + 1) + jitter_rng.uniform(-0.12, 0.12, size=len(y))
        ax_box.scatter(xj, y, s=18, color="black", alpha=0.45, zorder=3)
    ax_box.set_ylabel("ARI per trial")
    ax_box.set_ylim(-0.05, 1.05)
    ax_box.set_xticklabels(labels, rotation=10)
    ax_box.set_title("Per-trial ARI distribution")

    fig.suptitle(
        f"Cluster recovery: k-means vs UMAP+HDBSCAN  "
        f"(p_intra={P_INTRA}, p_inter={P_INTER}, {N_CLUSTERS} clusters, {N_MARKERS} markers)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    Path("results").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)

    df = run_trials()
    df.to_csv("results/cluster_recovery_results.csv", index=False)

    summary = summarize(df)

    # paired superiority tests (paired: same matrices per trial)
    paired = [
        paired_comparison(df["umap_complexity"], df["kmeans"],
                          "umap_complexity", "kmeans"),
        paired_comparison(df["umap_complexity"], df["umap_euclidean"],
                          "umap_complexity", "umap_euclidean"),
    ]
    paired_df = pd.DataFrame(paired)

    # write a combined summary CSV: per-method rows then paired-difference rows
    with open("results/cluster_recovery_summary.csv", "w") as f:
        f.write("# per-method summary\n")
        summary.to_csv(f, index=False)
        f.write("\n# paired differences\n")
        paired_df.to_csv(f, index=False)

    plot_summary(df, summary, "plots/cluster_recovery_summary.png")

    # ── console report ────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"Cluster-recovery experiment  |  N = {N_TRIALS} matrices")
    print(f"regime: {N_CLUSTERS} clusters, {N_MARKERS} markers, "
          f"p_intra={P_INTRA}, p_inter={P_INTER}, n_sim={N_SIM}")
    print("=" * 68)
    print(f"{'method':<20}{'mean ARI':>10}   {'95% CI (bootstrap)':>22}")
    print("-" * 68)
    for _, r in summary.iterrows():
        print(f"{r['label']:<20}{r['mean']:>10.3f}   "
              f"[{r['boot_ci_low']:.3f}, {r['boot_ci_high']:.3f}]")
    print("-" * 68)
    print("Paired superiority tests (positive diff favours the first method):")
    for _, r in paired_df.iterrows():
        verdict = "SIGNIFICANT" if r["wilcoxon_p"] < 0.05 else "not significant"
        print(f"  {r['comparison']:<34} "
              f"Δ={r['mean_diff']:+.3f}  "
              f"95% CI [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  "
              f"Wilcoxon p={r['wilcoxon_p']:.2e}  ({verdict})")
        print(f"    wins/losses/ties = "
              f"{r['n_wins']}/{r['n_losses']}/{r['n_ties']}")
    print("=" * 68)
    print("Wrote: results/cluster_recovery_results.csv, "
          "results/cluster_recovery_summary.csv, "
          "plots/cluster_recovery_summary.png")


if __name__ == "__main__":
    main()
