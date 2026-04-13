# complexity-velocity

Analyse de la complexité et de la vélocité des markers issus de la base CausalityLink.

## Contexte

Un **marker** est un concept extrait d'articles de presse. Ce projet mesure deux propriétés de chaque marker à partir de leurs co-occurrences dans les articles :

- **Complexité** : degré d'interconnexion d'un marker avec les autres — calculée comme le lift moyen vers tous les autres markers.
- **Vélocité** : fréquence d'apparition isolée d'un marker — calculée comme l'inverse du lift diagonal (1/P(marker)).

Le lift entre deux markers i et j est défini par :

```
lift(i,j) = P(i,j) / (P(i) * P(j))
```

## Structure du projet

```
complexity-velocity/
├── causalityTable.py               # Chargement des fichiers AVRO (markers, tree)
├── complexity_clusters.py          # Pipeline principal : lift, complexité, UMAP, DBSCAN
├── complexity_clusters_publisher.py # Analyse complexité/vélocité par éditeur
├── marker_clustering.py            # Simulation et évaluation de méthodes de clustering
├── peter_clark_scm.py              # Découverte de structure causale (algorithme PC)
├── kb_visualisation.py             # Visualisation interactive de l'arbre KB
├── data/
│   ├── causalitylink_sample/       # Données CausalityLink (AVRO)
│   ├── CausalityLinkPublishers.csv # Mapping publisher_id → label
│   └── journaux_themes.csv         # Mapping éditeur → thème (santé, économie, sport…)
├── plots/                          # Figures générées (créé automatiquement)
└── requirement.txt
```

## Fichiers

### `causalityTable.py`
Classe `CausalityTable` — utilitaire de chargement de données AVRO. Supporte le chargement complet ou mois par mois.

### `complexity_clusters.py`
Pipeline principal. Étapes :
1. Chargement et filtrage des markers (sans pays, avec éditeur connu)
2. Calcul de la matrice de co-citation puis de la matrice de lift
3. Calcul des scores de complexité et vélocité par marker
4. Réduction dimensionnelle UMAP sur la dissimilarité lift, clustering DBSCAN
5. Visualisations : scatter log-log, boxplots par catégorie de complexité, projection 2D annotée

### `complexity_clusters_publisher.py`
Pour un cluster DBSCAN donné, recalcule la matrice de lift séparément pour chaque éditeur et trace le scatter complexité/vélocité avec une régression loi de puissance (fit log-log) par éditeur.

### `marker_clustering.py`
Benchmark sur données synthétiques. Génère une matrice de dépendance avec structure en clusters et compare trois méthodes :
- **K-Means** (sélection de k par stabilité ARI)
- **UMAP + HDBSCAN** en distance euclidéenne sur [C, Cᵀ]
- **UMAP + HDBSCAN** en dissimilarité lift (matrice précomputée)

### `peter_clark_scm.py`
Découverte de structure causale via l'algorithme **PC** (`causal-learn`, test χ²). Pour chaque cluster sélectionné :
1. Calcule les complexités locales au cluster
2. Si le cluster dépasse 25 markers, sélectionne les extrêmes (les moins et les plus complexes)
3. Lance l'algorithme PC sur la matrice binaire de présence des markers par article
4. Sauvegarde un graphe causal (NetworkX) et une matrice d'adjacence triée par complexité

### `kb_visualisation.py`
Outil interactif en ligne de commande pour explorer l'arbre hiérarchique d'un marker dans la base de connaissance. Commandes :
- Entrer un marker pour visualiser son arbre
- Ajouter `#N` pour définir la profondeur (ex : `sport#3`)
- Entrer `__exit__` pour quitter

## Données attendues

Toutes les données doivent être placées dans le dossier `data/` :
```
data/
├── causalitylink_sample/
│   ├── Markers/      # fichiers AVRO partitionnés par year=/month=
│   ├── Tree/         # fichiers AVRO de la hiérarchie des markers
│   └── KB/           # base de connaissance (pour kb_visualisation)
├── CausalityLinkPublishers.csv   # colonnes : publisher, label
└── journaux_themes.csv           # mapping publisher_label → thème
```

Les figures sont sauvegardées automatiquement dans `plots/` (créé à la première exécution).

## Installation

```bash
pip install -r requirement.txt
```

Dépendances principales : `polars`, `pandas`, `numpy`, `scikit-learn`, `umap-learn`, `hdbscan`, `matplotlib`, `seaborn`, `networkx`, `causal-learn`, `anytree`, `tqdm`.

## Utilisation rapide

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
