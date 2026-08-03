# complexity-velocity

Reference implementation for **"A Causal Model to Explain Complexity of Topics
and Its Empirical Link with Corpus Velocity"** (Arnaudo, Attarian, Chikhi,
Lehalle).

The code measures the *complexity* of conceptual markers in a text corpus, and
relates it to the *velocity* at which sources publish about them.

---

## The two quantities

A **marker** is a concept (a KPI, an entity, a theme) extracted from news
articles. Both quantities below are computed from marker co-occurrence across
articles.

**Lift** between two markers — the pairwise building block, equal to
`exp(PMI(i, j))`:

```
lift(i, j) = P(i, j) / (P(i) · P(j))
```

**Complexity** `C(M_i)` — the *mean* lift of marker `i` against the other
markers of its own semantic cluster `S(M_i)`:

```
C(M_i) = mean_{j ∈ S(M_i), j ≠ i}  lift(i, j)
```

A marker is complex when its presence reliably drags in a rich constellation of
related concepts; simple when it can stand alone. Complexity is *corpus-relative*
by construction.

**Velocity** `v(M_i)` — how often the corpus publishes about the marker,
measured as its marginal probability `P(M_i = 1)`, i.e. the fraction of
articles mentioning it over the observation window (one month in the paper).
In code this is read off the diagonal of the lift matrix, since
`lift(i, i) = 1 / P(i)`.

**Dissimilarity** used for clustering, derived from the lift:

```
D(i, j) = log(1 + 1 / lift(i, j))
```

---

## Quick start

```bash
pip install -r requirements.txt jupyterlab
jupyter lab demo.ipynb
```

`demo.ipynb` is the entry point. **It runs end to end without the CausalityLink
corpus**: it builds the metric on synthetic data, reproduces the clustering and
PC-recovery benchmarks at reduced scale, and then loads the *precomputed*
corpus results shipped in `clusters/` to reproduce the paper's stylized fact
and the source-selection rule. Runtime is about two minutes on a laptop, and it
is committed with its outputs so it can also just be read on GitHub.

---

## Repository layout

```
complexity-velocity/
├── demo.ipynb                       # ← start here: runnable walkthrough, no proprietary data needed
│
├── causalityTable.py                # AVRO loading utility for the CausalityLink dump
├── complexity_clusters.py           # main pipeline: lift → complexity/velocity → UMAP + DBSCAN
├── complexity_clusters_publisher.py # same, broken down per publisher
├── analyze_selected_clusters.py     # combined figure for clusters 5, 9, 11, 13
├── peter_clark_scm.py               # within-cluster causal graph recovery (PC algorithm)
├── llm_judge.py                     # LLM-as-a-judge cross-validation of the metric
├── kb_visualisation.py              # interactive knowledge-base tree explorer
│
├── basevcx.py                       # synthetic SCM: C generators, document simulation, lifts
├── marker_clustering.py             # clustering benchmark helpers (k-means / UMAP+HDBSCAN)
├── cluster_recovery_fair.py         # C-blind cluster-recovery benchmark  ← the one in the paper
├── cluster_recovery_experiment.py   # earlier single-regime variant (also holds the stat helpers)
├── cluster_recovery_sweep.py        # earlier difficulty sweep, methods allowed to read C
├── empirical_pc_tests.py            # PC skeleton-recovery F1 across nine DAG topologies
│
├── clusters/                        # precomputed per-cluster results on the CausalityLink corpus
├── plots/                           # generated figures (created on first run)
├── results/                         # generated CSVs (created on first run)
├── requirements.txt
└── requirements-llm.txt             # optional, for re-running llm_judge.py
```

### Which script produces which paper result

| Paper element | Script | Command |
|---|---|---|
| Cluster-recovery table & figure (`ARI` vs `p_inter`) | `cluster_recovery_fair.py` | `python cluster_recovery_fair.py` |
| PC skeleton `F1` confidence intervals | `empirical_pc_tests.py` | `python empirical_pc_tests.py --p-min 12 --p-max 12 --n-obs 30000 --reps 20` |
| 2D projection, per-cluster stats, complexity–velocity grid | `complexity_clusters.py` | `python complexity_clusters.py --root <data> --all-clusters` |
| Four-cluster complexity–velocity figure | `analyze_selected_clusters.py` | `python analyze_selected_clusters.py --root <data>` |
| Per-publisher complexity–velocity figure | `complexity_clusters_publisher.py` | `python complexity_clusters_publisher.py --root <data> --cluster-id 12` |
| Intra-cluster PC dependency graphs | `peter_clark_scm.py` | `python peter_clark_scm.py --root <data> --clusters 5 11` |
| LLM-judge CDF figure | `llm_judge.py` | `python llm_judge.py` |
| Synthetic complexity/velocity counterfactuals | `basevcx.py` | `python basevcx.py` |

Every script exposes `--help`. Scripts marked `<data>` need the CausalityLink
corpus (see below); the others run on synthetic data only.

---

## Data

The **CausalityLink** corpus is proprietary and is *not* distributed with this
repository. `data/` is git-ignored. Scripts that consume it expect:

```
data/
├── causalitylink_sample/
│   ├── Markers/                  # AVRO, partitioned as year=YYYY/month=MM/
│   ├── Tree/                     # AVRO, marker ontology (used to drop orphan and country markers)
│   └── KB/                       # knowledge base (kb_visualisation.py only)
├── CausalityLinkPublishers.csv   # columns: publisher, label
└── journaux_themes.csv           # publisher_label → theme (sante, economie, sport, …)
```

Point the scripts elsewhere with `--root` (AVRO folders) and `--data-dir`
(the two CSVs).

**What is shipped instead:** `clusters/` contains the precomputed outputs of the
full pipeline on the January 2025 snapshot — per-cluster marker complexity and
velocity, LLM-judge scores, the aggregate statistics table, and the published
figures. `demo.ipynb` reads these directly, so the empirical claims of the paper
can be checked without access to the corpus.

| File | Contents |
|---|---|
| `clusters/all_clusters_stats.csv` | one row per cluster: `n_kpi`, `n_articles`, mean intra/external lift, `beta0`, `beta1` + 95% CI, `r2`, Kendall `tau`, Pearson `rho`, complexity range |
| `clusters/cluster_<id>_all_markers.csv` | every marker of the cluster with its `complexity` and `velocity` |
| `clusters/cluster_<id>_top_bottom.csv` | the 10 least and 10 most complex markers |
| `clusters/cluster_<id>_llm_classification.csv` | adds `llm_score` (1–10) and `llm_category` (per-cluster tercile) |
| `clusters/llm_classification_summary.csv` | per-cluster tercile counts and mean LLM score |

---

## Pipeline

`complexity_clusters.py --all-clusters` runs, in order:

1. **Load and filter.** Read the monthly `Markers` AVRO snapshot; drop markers
   absent from the `Tree` ontology and markers carrying a country; join
   publisher labels and journal themes.
2. **Select markers.** Keep the most frequent fraction (default 1/3) of markers
   appearing in the retained journal themes.
3. **Co-citation matrix.** Count article-level co-occurrences for every marker
   pair, normalise by the article count. *This is the slow step.*
4. **Lift, complexity, velocity.** Lift from the co-citation probabilities;
   complexity as the mean off-diagonal lift; velocity from the diagonal.
5. **Cluster.** UMAP on the lift-derived dissimilarity (precomputed metric),
   then DBSCAN on the 2D embedding. Each cluster is then re-analysed on its own
   sub-lift matrix, which is what makes complexity a *within-cluster* quantity.

Downstream, `peter_clark_scm.py` recovers the causal skeleton inside a cluster
with the PC algorithm, and `llm_judge.py` cross-checks the resulting complexity
ordering against an independent LLM rater.

### Note on the clustering dissimilarity

The synthetic benchmarks (`marker_clustering.py`, `cluster_recovery_fair.py`)
use the dissimilarity exactly as defined in the paper, `D = log(1 + 1/lift)`.
The corpus clustering in `compute_latent_and_cluster` uses a variant with a
row-wise offset, `D = log(1 + 1/(lift + ε) − P(i))`, which damps very frequent
markers; this is the form that produced the published 21-cluster partition and
is documented in the function's docstring.

---

## Reproducibility

- Every script takes `--seed` (default 42) and passes it to UMAP, DBSCAN
  sampling and the synthetic generators.
- `empirical_pc_tests.py` defaults to the exploratory configuration
  (`p ∈ [12, 20]`, `N_obs = 9000`, 15 repetitions). The paper's table uses
  `p = 12`, `N_obs = 30000`, 20 replications — pass the flags shown in the
  table above.
- `cluster_recovery_fair.py` is the benchmark reported in the paper: no method
  is allowed to read the generating matrix `C`, all three see only the lift
  estimated from simulated documents. `cluster_recovery_sweep.py` and
  `cluster_recovery_experiment.py` are the earlier variants where the baselines
  read `C` directly; they are kept for provenance, and
  `cluster_recovery_fair.py` imports the bootstrap/t-interval helpers from the
  latter.

---

## Citation

```
Baptiste Arnaudo, Keyvan Attarian, Salah Chikhi, Charles-Albert Lehalle.
A Causal Model to Explain Complexity of Topics and Its Empirical Link with
Corpus Velocity.
```

The authors thank Olav Laudy and Pierre Haren for their help with the data.
