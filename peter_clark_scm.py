import logging
import sys
import random
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import networkx as nx
from tqdm import tqdm
from umap import UMAP
from sklearn.cluster import DBSCAN

# --- IMPORT CAUSAL-LEARN ---
try:
    from causallearn.search.ConstraintBased.PC import pc
except ImportError:
    print("ERREUR : pip install causal-learn")
    sys.exit(1)

try:
    from causalityTable import CausalityTable
except ImportError:
    logging.warning("Module 'causalityTable' introuvable.")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="levelname)s: %(message)s")

# =============================================================================
# PARTIE 1 : PRE-TRAITEMENT ET CLUSTERING GLOBAL
# =============================================================================

def get_clusters(root_path):
    markerTable = CausalityTable(root_path / "Markers")
    markerTable.load_one_mounth(year=2025, month=1)
    treeTable = CausalityTable(root_path / "Tree")
    treeTable.load_data(date_parsing=False)
    
    markers_filter = treeTable.df.filter(pl.col("country").is_null())["marker"].to_list()
    df_filtered = markerTable.df.filter(pl.col("marker").is_in(markers_filter))
    
    counts = df_filtered.group_by("marker").len().sort("len", descending=True)
    sel_markers = counts.head(2000)["marker"].to_numpy()
    conv = {m: i for i, m in enumerate(sel_markers)}
    
    n = len(sel_markers)
    cm_counts = np.zeros((n, n), dtype=np.int64)
    df_grouped = df_filtered.filter(pl.col("marker").is_in(sel_markers)).group_by("id").agg(pl.col("marker"))
    
    for row in tqdm(df_grouped.iter_rows(), desc="Clustering Global", total=len(df_grouped)):
        idxs = [conv[m] for m in row[1] if m in conv]
        if len(idxs) > 1:
            ix = np.array(idxs)
            cm_counts[np.ix_(ix, ix)] += 1
            
    cm_prob = cm_counts.astype(float) / float(max(1, df_filtered["id"].n_unique()))
    diag_p = np.diag(cm_prob).astype(float)
    denom = diag_p[:, None] * diag_p[None, :]
    lift = np.divide(cm_prob, denom, out=np.zeros_like(cm_prob), where=denom!=0)
    lift = 0.5 * (lift + lift.T)
    
    dist = np.log1p((lift + 1e-4) ** -1)
    X_latent = UMAP(metric="precomputed", n_neighbors=15, random_state=42).fit_transform(dist)
    labels = DBSCAN(eps=0.2, min_samples=10).fit(X_latent).labels_
    
    return df_filtered, sel_markers, labels

# =============================================================================
# PARTIE 2 : CALCUL COMPLEXITÉ LOCALE (AU CLUSTER)
# =============================================================================

def compute_local_complexities(df_cluster, cluster_markers):
    n = len(cluster_markers)
    conv = {m: i for i, m in enumerate(cluster_markers)}
    cm_counts = np.zeros((n, n), dtype=np.int64)
    df_grouped = df_cluster.group_by("id").agg(pl.col("marker"))
    for row in df_grouped.iter_rows():
        idxs = [conv[m] for m in row[1] if m in conv]
        if len(idxs) > 1:
            ix = np.array(idxs)
            cm_counts[np.ix_(ix, ix)] += 1
    cm_prob = cm_counts.astype(float) / float(max(1, df_cluster["id"].n_unique()))
    diag_p = np.diag(cm_prob).astype(float)
    denom = diag_p[:, None] * diag_p[None, :]
    lift_local = np.divide(cm_prob, denom, out=np.zeros_like(cm_prob), where=denom!=0)
    return {m: (np.sum(lift_local[conv[m], :]) - lift_local[conv[m], conv[m]]) / (n - 1) for m in cluster_markers}

# =============================================================================
# PARTIE 3 : VISUALISATION
# =============================================================================

def build_and_save_final_results(cg, markers, complexities, filename, cluster_id):
    G = nx.DiGraph() 
    G.add_nodes_from(markers)
    adj = cg.G.graph
    
    undirected_edges = []
    directed_edges = []

    for i in range(len(markers)):
        for j in range(i + 1, len(markers)):
            m_i, m_j = markers[i], markers[j]
            if adj[i, j] == -1 and adj[j, i] == 1:
                G.add_edge(m_i, m_j)
                directed_edges.append((m_i, m_j))
            elif adj[i, j] == 1 and adj[j, i] == -1:
                G.add_edge(m_j, m_i)
                directed_edges.append((m_j, m_i))
            elif adj[i, j] == -1 and adj[j, i] == -1:
                undirected_edges.append((m_i, m_j))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 12))
    
    # --- DESSIN DU GRAPHE ---
    pos = nx.kamada_kawai_layout(G)
    node_size = 2200 
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=node_size, node_color='#FFDAB9', edgecolors='black')
    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=8, font_weight='bold')
    nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=directed_edges, edge_color='black', 
                               style='solid', width=2, arrowsize=25, arrowstyle='-|>',
                               node_size=node_size, connectionstyle="arc3,rad=0.1")
    nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=undirected_edges, edge_color='orange', 
                               style='solid', width=2, arrows=False, alpha=0.6)

    ax1.set_title(f"Graphe Causal PC - Cluster {cluster_id}", fontsize=14)
    ax1.axis('off')

    # --- MATRICE ADJACENCE TRIÉE PAR COMPLEXITÉ ---
    # Tri des nœuds par ordre décroissant de complexité
    ordered_nodes = sorted(markers, key=lambda n: complexities.get(n, 0), reverse=False)

    labels_with_comp = [f"{n} (LC:{complexities.get(n, 0):.2f})" for n in ordered_nodes]
    
    size = len(ordered_nodes)
    mat = np.zeros((size, size))
    node_to_idx = {n: i for i, n in enumerate(ordered_nodes)}
    
    # Remplissage : 1 pour dirigé, 0.5 pour non-dirigé (symétrique)
    for u, v in directed_edges:
        if u in node_to_idx and v in node_to_idx:
            mat[node_to_idx[u], node_to_idx[v]] = 1
            
    for u, v in undirected_edges:
        if u in node_to_idx and v in node_to_idx:
            mat[node_to_idx[u], node_to_idx[v]] = 0.5
            mat[node_to_idx[v], node_to_idx[u]] = 0.5

    im = ax2.imshow(mat, cmap='Greys', interpolation='nearest', vmin=0, vmax=1)
    ax2.set_xticks(range(size))
    ax2.set_yticks(range(size))
    ax2.set_xticklabels(labels_with_comp, rotation=90, fontsize=9)
    ax2.set_yticklabels(labels_with_comp, fontsize=9)
    ax2.set_title("Matrice d'Adjacence (Triée par Complexité)", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)

# =============================================================================
# BLOC PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    data_path = Path("data/causalitylink_sample")
    Path("plots").mkdir(exist_ok=True)
    
    # 1. Clustering Global
    df_filtered, all_markers, cluster_labels = get_clusters(data_path)
    
    # 2. Identifier les clusters éligibles (entre 16 et 18 markers)
    unique_l, counts_l = np.unique(cluster_labels, return_counts=True)
    eligible_clusters = [lab for lab, count in zip(unique_l, counts_l) 
                         if lab != -1]
    
    if not eligible_clusters:
        logger.info("Aucun cluster correspondant aux critères n'a été trouvé.")
        sys.exit(0)

    # --- AFFICHAGE POUR SÉLECTION ---
    print("\n" + "="*50)
    print(f"{'ID':<6} | {'Taille':<8} | {'Exemples de Markers'}")
    print("-" * 50)
    
    for lab in eligible_clusters:
        c_markers = [all_markers[i] for i, l in enumerate(cluster_labels) if l == lab]
        # On affiche les 5 premiers markers pour donner une idée du contenu
        preview = ", ".join(c_markers[:5]) + ("..." if len(c_markers) > 5 else "")
        print(f"{lab:<6} | {len(c_markers):<8} | {preview}")
    print("="*50 + "\n")

    user_input = input("Entrez les IDs des clusters à traiter (séparés par des virgules) ou 'all' pour tous : ")
    
    if user_input.lower() == 'all':
        clusters_to_process = eligible_clusters
    else:
        try:
            # Nettoyage de l'input pour transformer "1, 2, 3" en [1, 2, 3]
            selected_ids = [int(x.strip()) for x in user_input.split(",") if x.strip()]
            clusters_to_process = [c for c in selected_ids if c in eligible_clusters]
        except ValueError:
            print("Erreur de saisie. Fin du programme.")
            sys.exit(1)

    if not clusters_to_process:
        print("Aucun cluster valide sélectionné. Fin.")
        sys.exit(0)

    # 3. Boucle sur la sélection
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for target_cluster in tqdm(clusters_to_process, desc="Exécution PC"):
        # Récupération initiale de tous les markers du cluster
        cluster_markers = [all_markers[i] for i, l in enumerate(cluster_labels) if l == target_cluster]
        df_cluster_full = df_filtered.filter(pl.col("marker").is_in(cluster_markers))
        
        # Calcul de la complexité sur l'ensemble du cluster pour identifier les extrêmes
        local_complexities = compute_local_complexities(df_cluster_full, cluster_markers)

        # --- LOGIQUE DE SÉLECTION (N > 25) ---
        if len(cluster_markers) > 25:
            logger.info(f"Cluster {target_cluster} trop grand ({len(cluster_markers)}). Sélection des 25 extrêmes.")
            # Tri des markers par complexité
            sorted_by_comp = sorted(cluster_markers, key=lambda m: local_complexities[m])
            
            # On prend les 12 moins complexes (début de liste) et les 13 plus complexes (fin de liste)
            selected_markers = sorted_by_comp[:10] + sorted_by_comp[-8:]
            
            # On restreint les données et les complexités aux 25 sélectionnés
            cluster_markers = selected_markers
            df_cluster = df_filtered.filter(pl.col("marker").is_in(cluster_markers))
            local_complexities = {m: local_complexities[m] for m in cluster_markers}
        else:
            df_cluster = df_cluster_full

        # --- PRÉPARATION DES DONNÉES POUR PC ---
        X_pivot = df_cluster.pivot(index="id", on="marker", values="marker", aggregate_function="len").fill_null(0)
        X_bin = (X_pivot.to_pandas().set_index("id") > 0).astype(int)
        
        if len(X_bin) > 120000:
            X_bin = X_bin.sample(n=120000, random_state=30)
        
        # Lancer PC sur les markers sélectionnés (max 25)
        output_filename = f"plots/causal_cluster{target_cluster}_{timestamp}.png"
        cg = pc(X_bin.to_numpy(), alpha=0.0005, indep_test='chisq', show_progress=True)
        
        # Sauvegarder les résultats
        build_and_save_final_results(cg, X_bin.columns.tolist(), local_complexities, output_filename, target_cluster)