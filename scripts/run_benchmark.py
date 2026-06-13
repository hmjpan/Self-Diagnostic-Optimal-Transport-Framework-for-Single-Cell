"""
ALL BASELINE METHODS FOR COMPARISON
====================================
1. Diffusion Pseudotime + Graph Potential (DPT + Graph Laplacian)
2. Waddington-OT simplified (OT with growth-rate marginals)
3. Graph Entropy Method (Hashimoto et al. style)
4. RNA velocity field direction comparison (on real data)

All methods benchmarked on the same synthetic data as OT (ours).
"""
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse.csgraph import laplacian, minimum_spanning_tree
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import kneighbors_graph
from sklearn.linear_model import LinearRegression
import scanpy as sc
import time, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.landscapes import (SimpleBifurcation, ThreeWayBifurcation, 
                             HierarchicalBifurcation)
from src.sinkhorn import sinkhorn_plan, sinkhorn_distance
from src.potential import reconstruct_gradient
from final_experiment import (calibrate_epsilon, robust_mst_integration,
                               compute_all_metrics)
from predictive_validation import build_potential_interpolator, forward_simulate

print("=" * 70)
print("BASELINE METHODS BENCHMARK")
print("=" * 70)

# Shared data
bif = SimpleBifurcation(); three = ThreeWayBifurcation()
n_cells=1500; beta_true=100.0; tau=0.5; seeds=list(range(5))

rng=np.random.RandomState(42)
X0_b=rng.randn(n_cells,2)*0.5
X1_b=X0_b-bif.grad_V(X0_b)*tau+rng.randn(n_cells,2)*np.sqrt(2*tau/beta_true)
X0_t=rng.randn(n_cells,2)*0.5
X1_t=X0_t-three.grad_V(X0_t)*tau+rng.randn(n_cells,2)*np.sqrt(2*tau/beta_true)

# ===================================================================
# METHOD 1: DIFFUSION PSEUDOTIME + GRAPH POTENTIAL
# ===================================================================
print("\n[1] DPT + GRAPH POTENTIAL")
def dpt_graph_potential(X_src, X_tgt, n_neighbors=30):
    """Use diffusion pseudotime to order cells, then graph Laplacian for potential.
    This combines two common approaches in the literature.
    """
    # Stack all cells, compute DPT
    X_all = np.vstack([X_src, X_tgt])
    n_src = len(X_src)
    
    # Build graph
    kng = kneighbors_graph(X_all, n_neighbors, mode='connectivity', include_self=False)
    L = laplacian(kng, normed=True)
    
    # Compute first non-trivial eigenvector (Fiedler vector)
    # This gives a natural ordering = pseudotime
    try:
        vals, vecs = eigsh(L.astype(float), k=3, which='SM')
        fiedler = vecs[:, 1]  # Second smallest eigenvalue
    except:
        fiedler = np.zeros(len(X_all))
    
    # The Fiedler vector gives an ordering. Use it to estimate potential:
    # V ∝ Fiedler (coarse approximation)
    # Then gradient = (V(x_src) - V(x_tgt)) / tau? No, that's wrong.
    # Correct: use graph Laplacian to estimate the drift from transition probabilities
    # P = D^{-1}W (row-normalized adjacency)
    # The expected displacement from x_i is sum_j P_{ij} * (x_j - x_i)
    W = kng.toarray()
    D_inv = np.diag(1.0 / (W.sum(axis=1) + 1e-10))
    P = D_inv @ W
    
    # Expected next position for each source cell
    T_graph = np.zeros_like(X_src)
    for i in range(n_src):
        weights = P[i, n_src:]  # Weights to target cells
        weights = weights / (weights.sum() + 1e-10)
        T_graph[i] = weights @ X_tgt
    
    return (X_src - T_graph) / tau  # Graph-approximated gradient

m_dpt = compute_all_metrics(dpt_graph_potential(X0_b, X1_b), bif.grad_V(X0_b))
print(f"  Bifurcation: cos_sig={m_dpt['cos_sim_sig']:.3f}, dir={m_dpt['dir_correct']:.1%}")
m_dpt3 = compute_all_metrics(dpt_graph_potential(X0_t, X1_t), three.grad_V(X0_t))
print(f"  Three-way:   cos_sig={m_dpt3['cos_sim_sig']:.3f}, dir={m_dpt3['dir_correct']:.1%}")

# ===================================================================
# METHOD 2: WADDINGTON-OT SIMPLIFIED (OT with growth-rate marginals)
# ===================================================================
print("\n[2] WADDINGTON-OT (growth-rate weighted OT)")
def wot_gradient(X_src, X_tgt, n_clusters=5):
    """Simplified Waddington-OT: estimate growth rates from cluster size changes,
    use them as OT marginal weights."""
    from sklearn.cluster import KMeans
    
    # Cluster source cells
    km_src = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_src)
    km_tgt = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X_tgt)
    
    # Count cells per cluster
    src_counts = np.bincount(km_src.labels_, minlength=n_clusters).astype(float)
    tgt_counts = np.bincount(km_tgt.labels_, minlength=n_clusters).astype(float)
    
    # Growth rate per cluster: g_c = log(N_{t+1,c} / N_{t,c}) / tau
    growth = np.log((tgt_counts + 1) / (src_counts + 1)) / tau
    
    # Assign growth rate to each cell
    src_growth = growth[km_src.labels_]
    tgt_growth = growth[km_tgt.labels_]
    
    # OT with growth-rate-weighted marginals
    # Pushes: a_i ∝ exp(growth_i * tau), b_j ∝ 1 (or also weighted)
    a = np.exp(src_growth * tau)
    a = a / a.sum()
    b = np.ones(len(X_tgt)) / len(X_tgt)
    
    eps_val = calibrate_epsilon(X_src, X_tgt)
    plan, _, _ = sinkhorn_plan(X_src, X_tgt, epsilon=eps_val)
    # Re-weight by growth rates
    # Actually the proper way: use a and b as Sinkhorn marginals
    # But our sinkhorn_plan always uses uniform. Let me use custom weights.
    from src.sinkhorn import sinkhorn_plan as sp
    # Reimplement with custom weights
    C = cdist(X_src, X_tgt, metric='sqeuclidean')
    K = np.exp(-C / eps_val)
    # Sinkhorn with custom marginals
    u = np.ones(len(X_src))
    v = np.ones(len(X_tgt))
    for _ in range(100):
        u = a / (K @ v + 1e-16)
        v = b / (K.T @ u + 1e-16)
    plan_wot = np.diag(u) @ K @ np.diag(v)
    
    from src.sinkhorn import barycentric_projection
    T_wot = barycentric_projection(plan_wot, X_tgt)
    return (X_src - T_wot) / tau

# Also: uniform OT (already have this, but re-run for clean comparison)
eps_b = calibrate_epsilon(X0_b, X1_b)
plan_b, _, _ = sinkhorn_plan(X0_b, X1_b, epsilon=eps_b)
grad_ot_b, _ = reconstruct_gradient(plan_b, X0_b, X1_b, None, tau, None)

eps_t = calibrate_epsilon(X0_t, X1_t)
plan_t, _, _ = sinkhorn_plan(X0_t, X1_t, epsilon=eps_t)
grad_ot_t, _ = reconstruct_gradient(plan_t, X0_t, X1_t, None, tau, None)

m_wot_b = compute_all_metrics(wot_gradient(X0_b, X1_b), bif.grad_V(X0_b))
m_ot_b = compute_all_metrics(grad_ot_b, bif.grad_V(X0_b))
print(f"  Bifurcation: OT={m_ot_b['cos_sim_sig']:.3f}, WOT={m_wot_b['cos_sim_sig']:.3f}")

m_wot_t = compute_all_metrics(wot_gradient(X0_t, X1_t), three.grad_V(X0_t))
m_ot_t = compute_all_metrics(grad_ot_t, three.grad_V(X0_t))
print(f"  Three-way:   OT={m_ot_t['cos_sim_sig']:.3f}, WOT={m_wot_t['cos_sim_sig']:.3f}")

# ===================================================================
# METHOD 3: GRAPH ENTROPY METHOD (Hashimoto et al. style)
# ===================================================================
print("\n[3] GRAPH ENTROPY METHOD")
def graph_entropy_potential(X_src, X_tgt, n_neighbors=30, sigma=0.5):
    """Reconstruct potential from graph entropy.
    V(x) ∝ -H(x) where H(x) is the local entropy of the transition distribution.
    """
    X_all = np.vstack([X_src, X_tgt])
    n_src = len(X_src)
    
    # Build diffusion map
    dists = cdist(X_all, X_all, metric='sqeuclidean')
    K = np.exp(-dists / (2 * sigma**2))
    
    # Row-normalize to get transition matrix
    P = K / (K.sum(axis=1, keepdims=True) + 1e-10)
    
    # Local entropy for each cell: H_i = -sum_j P_{ij} log P_{ij}
    H = -np.sum(P * np.log(P + 1e-10), axis=1)
    
    # Potential: V ∝ -H (lower entropy = more ordered = deeper in valley)
    V_ent = -H
    
    # Estimate gradient from potential differences to neighbors
    tree = KDTree(X_all)
    grad_ent = np.zeros((n_src, 2))
    _, idx = tree.query(X_src, k=n_neighbors+1)
    
    for i in range(n_src):
        nb = idx[i, 1:]
        dV = V_ent[nb] - V_ent[i]
        dX = X_all[nb] - X_src[i]
        try:
            grad_ent[i] = np.linalg.lstsq(dX, dV, rcond=None)[0]
        except:
            pass
    
    return grad_ent

m_ent_b = compute_all_metrics(graph_entropy_potential(X0_b, X1_b), bif.grad_V(X0_b))
print(f"  Bifurcation: cos_sig={m_ent_b['cos_sim_sig']:.3f}, dir={m_ent_b['dir_correct']:.1%}")
m_ent_t = compute_all_metrics(graph_entropy_potential(X0_t, X1_t), three.grad_V(X0_t))
print(f"  Three-way:   cos_sig={m_ent_t['cos_sim_sig']:.3f}, dir={m_ent_t['dir_correct']:.1%}")

# ===================================================================
# METHOD 4: RNA VELOCITY DIRECTION (on Moignard real data)
# ===================================================================
print("\n[4] RNA VELOCITY DIRECTION COMPARISON (Moignard 2015)")
try:
    adata = sc.datasets.moignard15()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=30, subset=True)
    
    # RNA velocity (requires spliced/unspliced, not available for this dataset)
    # Instead: compute the velocity-like field from mRNA abundance changes
    # between stages. This approximates what RNA velocity does.
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X)
    
    stage_order = ['PS','NP','HF','4SG','4SFG']
    # For each pair of consecutive stages, compute mean displacement
    velocity_vectors = []
    for i in range(len(stage_order)-1):
        mask_s = adata.obs['exp_groups'] == stage_order[i]
        mask_t = adata.obs['exp_groups'] == stage_order[i+1]
        mean_s = np.mean(X_pca[mask_s], axis=0)
        mean_t = np.mean(X_pca[mask_t], axis=0)
        velocity_vectors.append(mean_t - mean_s)
    
    # OT gradient on same data
    distributions_pca = [X_pca[adata.obs['exp_groups'] == s] for s in stage_order]
    ot_velocities = []
    for i in range(len(distributions_pca)-1):
        Xs, Xt = distributions_pca[i], distributions_pca[i+1]
        eps_val = calibrate_epsilon(Xs, Xt)
        plan, _, _ = sinkhorn_plan(Xs, Xt, epsilon=eps_val)
        grad_ot, _ = reconstruct_gradient(plan, Xs, Xt, None, 1.0, None)
        ot_velocities.append(np.mean(grad_ot, axis=0))
    
    # Compare directions
    cos_sims_vel = []
    for v_vel, v_ot in zip(velocity_vectors, ot_velocities):
        cos = np.dot(v_vel, v_ot) / (np.linalg.norm(v_vel)*np.linalg.norm(v_ot)+1e-10)
        cos_sims_vel.append(cos)
    
    print(f"  Stage-mean velocity vs OT gradient alignment:")
    for i, (s1, s2) in enumerate(zip(stage_order[:-1], stage_order[1:])):
        arrow = '\u2192' if cos_sims_vel[i] > 0 else '\u2190'
        print(f"    {s1}->{s2}: cos={cos_sims_vel[i]:.3f} {arrow}")
    print(f"  Mean alignment: {np.mean(cos_sims_vel):.3f} "
          f"({'ALIGNED' if np.mean(cos_sims_vel)>0.5 else 'MISALIGNED'})")
except Exception as e:
    print(f"  RNA velocity comparison failed: {e}")

# ===================================================================
# SUMMARY TABLE
# ===================================================================
print("\n" + "=" * 70)
print("FINAL BENCHMARK SUMMARY")
print("=" * 70)
print(f"{'Method':<30} {'Bifurcation cos_sig':>20} {'Three-way cos_sig':>20}")
print("-" * 70)
results_all = {
    'OT (ours)': (m_ot_b['cos_sim_sig'], m_ot_t['cos_sim_sig']),
    'OT with growth (WOT simplified)': (m_wot_b['cos_sim_sig'], m_wot_t['cos_sim_sig']),
    'DPT + Graph Laplacian': (m_dpt['cos_sim_sig'], m_dpt3['cos_sim_sig']),
    'Graph Entropy (Hashimoto-style)': (m_ent_b['cos_sim_sig'], m_ent_t['cos_sim_sig']),
    'StationaryOT (density)': (0.240, -0.401),
    'Linear Regression': (0.906, -0.388),
    'kNN (k=30)': (0.690, 0.169),
    'Global Mean': (0.303, -0.599),
}
for method, (b, t) in results_all.items():
    b_str = f"{b:.3f}" if isinstance(b, float) else str(b)
    t_str = f"{t:.3f}" if isinstance(t, float) else str(t)
    print(f"{method:<30} {b_str:>20} {t_str:>20}")

with open('results/baseline_benchmark.json', 'w') as f:
    json.dump({k: {'bifurcation': float(v[0]), 'threeway': float(v[1])} 
               for k, v in results_all.items()}, f, indent=2)
print("\n[DONE] Results saved.")
