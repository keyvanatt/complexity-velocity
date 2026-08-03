#!/usr/bin/env python3
"""PC skeleton-recovery validation on synthetic DAGs.

For each of nine DAG topologies, generates a graph, simulates binary documents
from the generative model of the paper, runs the PC algorithm and scores the
recovered *undirected skeleton* against the truth with the F1 measure.

Defaults reproduce the paper's table "95% confidence intervals for the F1 score
of the PC algorithm": p = 12 nodes, N_obs = 30 000 observations, 20 replications
per configuration, alpha in {0.01, 0.1, 0.3}, unlimited conditioning-set size.

    python empirical_pc_tests.py

Graph size can also be randomised over a range, e.g. --p-min 12 --p-max 20.
Outputs a CSV under results/ (see --out).
"""

import argparse
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

from causallearn.search.ConstraintBased.PC import pc

# ---------- Defaults (the paper's configuration) ----------
N_REPETITIONS = 20
P_MIN, P_MAX = 12, 12            # graph size; set p_max > p_min to randomise it
N_OBS = 30_000
ALPHA_VALUES = [0.01, 0.1, 0.3]
U_RANGE = (0.05, 0.15)
MAX_PROB = 0.9
RANDOM_SEED = 42
INDEP_TEST = 'chisq'
RESULTS_DIR = Path("results")
DEFAULT_OUT = RESULTS_DIR / "pc_validation_f1.csv"

# ---------- DAG generators (unchanged) ----------
def gen_C_chain(p, strength=0.2):
    C = np.zeros((p, p))
    for i in range(1, p):
        C[i, i-1] = strength
    return C

def gen_C_tree(p, branching=2, strength=0.15):
    C = np.zeros((p, p))
    for i in range(p):
        for b in range(1, branching+1):
            child = i*branching + b
            if child < p:
                C[child, i] = strength
    return C

def random_dag(p, edge_prob=0.2, strength_range=(0.1,0.3)):
    C = np.zeros((p, p))
    for i in range(p):
        for j in range(i+1, p):
            if np.random.rand() < edge_prob:
                C[j, i] = np.random.uniform(*strength_range)
    return C

def gen_star_dag(p, strength=0.25):
    C = np.zeros((p, p))
    C[1:, 0] = strength
    return C

def gen_scale_free_dag(p, strength_range=(0.15,0.35), m=1):
    G = nx.barabasi_albert_graph(p, m, seed=None)
    C = np.zeros((p, p))
    for u, v in G.edges():
        if u < v:
            C[v, u] = np.random.uniform(*strength_range)
        else:
            C[u, v] = np.random.uniform(*strength_range)
    return C

def gen_C_dense_progressive(p, max_lookback=5, base_strength=0.05):
    C = np.zeros((p, p))
    for i in range(1, p):
        C[i, i-1] = base_strength * (1 + i/p)
        lookback = min(i, int(max_lookback * (i/p)) + 1)
        if lookback > 1:
            parents = np.arange(max(0, i-lookback), i-1)
            C[i, parents] = base_strength * (1.5 * i/p)
    return C

def gen_C_mostly_full(p, density=0.8, strength_range=(0.02,0.08)):
    C = np.zeros((p, p))
    strengths = np.linspace(strength_range[0], strength_range[1], p)
    for i in range(1, p):
        for j in range(i):
            if np.random.rand() < density:
                C[i, j] = strengths[i] * np.random.uniform(0.9, 1.1)
    return C

def gen_layered_dag(p, n_layers=4, edge_prob=0.4, strength=0.25):
    layers = np.array_split(np.arange(p), n_layers)
    layers = [arr for arr in layers if len(arr) > 0]
    C = np.zeros((p, p))
    for k in range(1, len(layers)):
        prev_layer = layers[k-1]
        cur_layer = layers[k]
        for target in cur_layer:
            for source in prev_layer:
                if np.random.rand() < edge_prob:
                    C[target, source] = strength * np.random.uniform(0.8, 1.2)
    return C

# ---------- Data generation ----------
def simulate_markers(C, u, n_docs):
    N = len(u)
    ranks = np.sum(C != 0, axis=1).astype(int)
    X = np.zeros((n_docs, N), dtype=int)
    for doc in range(n_docs):
        present = np.zeros(N, dtype=bool)
        for r in range(ranks.max()+1):
            idx = np.where(ranks == r)[0]
            for i in idx:
                p = u[i] + np.dot(C[i, :], present)
                present[i] = np.random.rand() < min(p, 1.0)
        X[doc] = present.astype(int)
    return X

def get_estimated_skeleton(data, alpha):
    try:
        # default max_cond_set_size (None) – no limit
        cg = pc(data, alpha=alpha, indep_test=INDEP_TEST, stable=True)
        est = np.abs(cg.G.graph) > 0
        est = np.maximum(est, est.T)
        return est.astype(int)
    except Exception:
        return None

def compute_f1(true_adj, est_adj):
    triu = np.triu_indices(true_adj.shape[0], k=1)
    true_flat = true_adj[triu]
    est_flat = est_adj[triu]
    tp = np.sum(true_flat & est_flat)
    fp = np.sum(est_flat & ~true_flat)
    fn = np.sum(true_flat & ~est_flat)
    prec = tp / (tp+fp) if (tp+fp)>0 else 0.0
    rec = tp / (tp+fn) if (tp+fn)>0 else 0.0
    return 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0

TOPOLOGIES = {
    'Chain':              gen_C_chain,
    'Tree (b=2)':         gen_C_tree,
    'Star':               gen_star_dag,
    'Random (d=1)':       lambda p: random_dag(p, edge_prob=1/(p-1)),
    'Random (d=2)':       lambda p: random_dag(p, edge_prob=2/(p-1)),
    'Scale-free':         gen_scale_free_dag,
    'Dense progressive':  gen_C_dense_progressive,
    'Mostly full':        gen_C_mostly_full,
    'Layered hierarchy':  gen_layered_dag,
}

# ---------- Worker ----------
def process_config(args):
    topo_name, gen_func, alpha, p_min, p_max, n_obs, reps, seed = args
    # Each worker seeds its own RNG so parallel runs stay reproducible.
    np.random.seed(seed)
    f1_vals = []
    for _ in range(reps):
        p = np.random.randint(p_min, p_max+1)
        C = gen_func(p)
        u = np.random.uniform(*U_RANGE, size=p)
        for i in range(p):
            s = np.sum(C[i, :]) + u[i]
            if s > MAX_PROB:
                C[i, :] *= MAX_PROB / s
        X = simulate_markers(C, u, n_docs=n_obs)
        skel_est = get_estimated_skeleton(X, alpha)
        if skel_est is None:
            continue
        true_skel = np.maximum((C>0).astype(int), (C>0).T.astype(int))
        f1_vals.append(compute_f1(true_skel, skel_est))
    if f1_vals:
        return topo_name, alpha, np.mean(f1_vals), np.std(f1_vals), len(f1_vals)
    else:
        return topo_name, alpha, np.nan, np.nan, 0

# ---------- Run ----------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--p-min", type=int, default=P_MIN, help="Minimum graph size (default: %(default)s)")
    parser.add_argument("--p-max", type=int, default=P_MAX, help="Maximum graph size (default: %(default)s)")
    parser.add_argument("--n-obs", type=int, default=N_OBS,
                        help="Simulated documents per graph (default: %(default)s)")
    parser.add_argument("--reps", type=int, default=N_REPETITIONS,
                        help="Repetitions per configuration (default: %(default)s)")
    parser.add_argument("--alphas", nargs="+", type=float, default=ALPHA_VALUES,
                        help="Significance levels of the independence test (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed (default: %(default)s)")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Parallel workers, -1 for all cores (default: %(default)s)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="Output CSV path (default: %(default)s)")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    tasks = [
        (name, gen, alpha, args.p_min, args.p_max, args.n_obs, args.reps, args.seed + i)
        for i, (name, gen, alpha) in enumerate(
            (name, gen, alpha) for name, gen in TOPOLOGIES.items() for alpha in args.alphas
        )
    ]

    print(f"Running with p in [{args.p_min},{args.p_max}], N_obs={args.n_obs}, "
          f"default max_cond_set_size, {args.reps} reps")
    results_raw = Parallel(n_jobs=args.n_jobs, verbose=10)(delayed(process_config)(t) for t in tasks)

    results = []
    for topo, alpha, f1m, f1s, nsucc in results_raw:
        if nsucc > 0:
            results.append({
                'Topology': topo, 'alpha': alpha,
                'F1_mean': f1m, 'F1_std': f1s, 'N_success': nsucc
            })

    df = pd.DataFrame(results)

    z = 1.96
    df['CI_low'] = df['F1_mean'] - z * df['F1_std'] / np.sqrt(df['N_success'])
    df['CI_high'] = df['F1_mean'] + z * df['F1_std'] / np.sqrt(df['N_success'])

    # Print summary
    print("\n" + "="*80)
    print("95% confidence intervals (default max_cond_set_size)")
    print("="*80)
    for alpha in args.alphas:
        sub = df[df['alpha'] == alpha].set_index('Topology')
        print(f"\n--- alpha = {alpha} ---")
        print(sub[['CI_low', 'F1_mean', 'CI_high']].to_string())
    print("="*80)

    df.to_csv(args.out, index=False)
    print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
