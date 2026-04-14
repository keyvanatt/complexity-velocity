# complexity-velocity

Analysis of the complexity and velocity of markers from the CausalityLink database.

## Context

A **marker** is a concept extracted from news articles. This project measures two properties of each marker based on their co-occurrences across articles:

- **Complexity**: degree of interconnection of a marker with others — computed as the sum of lifts towards all other markers.
- **Velocity**: frequency of isolated appearance of a marker — computed as the inverse of its marginal probability (1/P(marker)).

The lift between two markers i and j is defined as:

```
lift(i,j) = P(i,j) / (P(i) * P(j))
```

## Project structure

```
complexity-velocity/
├── causalityTable.py               # Loading AVRO files (markers, tree)
├── complexity_clusters.py          # Main pipeline: lift, complexity, UMAP, DBSCAN
├── complexity_clusters_publisher.py # Complexity/velocity analysis per publisher
├── marker_clustering.py            # Simulation and benchmarking of clustering methods
├── peter_clark_scm.py              # Causal structure discovery (PC algorithm)
├── kb_visualisation.py             # Interactive KB tree explorer
├── basevcx.py                      # Synthetic simulation: C matrix generators, lifts, visualisations
├── data/
│   ├── causalitylink_sample/       # CausalityLink data (AVRO)
│   ├── CausalityLinkPublishers.csv # publisher_id → label mapping
│   └── journaux_themes.csv         # publisher_label → theme mapping (health, economy, sport…)
├── plots/                          # Generated figures (created automatically)
└── requirement.txt
```

## Files

### `causalityTable.py`
`CausalityTable` class — AVRO data loading utility. Supports full load or month-by-month loading.

### `complexity_clusters.py`
Main pipeline. Steps:
1. Load and filter markers (excluding country markers, keeping known publishers)
2. Compute the co-citation matrix then the lift matrix
3. Compute complexity and velocity scores per marker
4. UMAP dimensionality reduction on lift dissimilarity, DBSCAN clustering
5. Visualisations: log-log scatter, boxplots by complexity category, annotated 2D projection

### `complexity_clusters_publisher.py`
For a given DBSCAN cluster, recomputes the lift matrix separately for each publisher and plots the complexity/velocity scatter with a power-law regression (log-log fit) per publisher.

### `marker_clustering.py`
Benchmark on synthetic data. Generates a dependency matrix with cluster structure and compares three methods:
- **K-Means** (k selection by ARI stability)
- **UMAP + HDBSCAN** with Euclidean distance on [C, Cᵀ]
- **UMAP + HDBSCAN** with lift dissimilarity (precomputed matrix)

### `peter_clark_scm.py`
Causal structure discovery via the **PC** algorithm (`causal-learn`, χ² test). For each selected cluster:
1. Computes local complexities within the cluster
2. If the cluster exceeds 25 markers, selects the extremes (least and most complex)
3. Runs the PC algorithm on the binary marker presence matrix per article
4. Saves a causal graph (NetworkX) and an adjacency matrix sorted by complexity

### `kb_visualisation.py`
Interactive command-line tool for exploring a marker's hierarchical tree in the knowledge base. Commands:
- Enter a marker name to visualise its tree
- Append `#N` to set the depth (e.g. `sport#3`)
- Enter `__exit__` to quit

### `basevcx.py`
Synthetic simulation to explore the complexity/velocity relationship on controlled data. Provides:
- **Dependency matrix generators**: chain, tree, cliques, hierarchical, random, random DAG, fractal, funnel, skip-hierarchical, dense progressive, mostly-full, increasing-rank
- **`simulate_markers`**: generates binary documents from a dependency matrix C and unary probabilities u
- **`compute_depth_in_dag`**: depth of each node in the DAG
- Lift computation, complexity scores (sum of lifts per marker), and associated visualisations (heatmaps, scatterplots, barplots)

Usable as a standalone script (`python basevcx.py`) or as an importable module.

## Expected data

All data must be placed in the `data/` folder:
```
data/
├── causalitylink_sample/
│   ├── Markers/      # AVRO files partitioned by year=/month=
│   ├── Tree/         # AVRO files for the marker hierarchy
│   └── KB/           # knowledge base (for kb_visualisation)
├── CausalityLinkPublishers.csv   # columns: publisher, label
└── journaux_themes.csv           # publisher_label → theme mapping
```

Figures are automatically saved to `plots/` (created on first run).

## Installation

```bash
pip install -r requirement.txt
```

Main dependencies: `polars`, `pandas`, `numpy`, `scikit-learn`, `umap-learn`, `hdbscan`, `matplotlib`, `seaborn`, `networkx`, `causal-learn`, `anytree`, `tqdm`.

## Quick start

```python
from pathlib import Path
from complexity_clusters import run_all

filtered_df, markers, conv, journals, lift_matrix, complexities, labels = run_all(
    root=Path("/Data/rc/causalitylink_sample")
)
```

```python
from pathlib import Path
from complexity_clusters_publisher import run_publisher_analysis

run_publisher_analysis(
    root=Path("/Data/rc/causalitylink_sample"),
    cluster_id=12,
    top_n_publishers=10,
    out_prefix="complexity_vs_velocity_publishers",
)
```
