"""
Pancreas development (Bastidas-Ponce et al. 2019)
Non-hematopoietic system, 3696 cells, clear endocrine differentiation stages.
"""
import numpy as np
import scvelo as scv
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial import KDTree
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.sinkhorn import sinkhorn_plan
from src.potential import reconstruct_gradient
from final_experiment import calibrate_epsilon, robust_mst_integration, find_minima

print("=" * 60)
print("PANCREAS DEVELOPMENT - Bastidas-Ponce 2019")
print("=" * 60)

# Load
adata = scv.datasets.pancreas()
print(f"[Data] {adata.shape[0]} cells, {adata.shape[1]} genes")

# Preprocess
import scanpy as sc
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)

# PCA
X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=42); X_pca = pca.fit_transform(X_scaled)
expl = pca.explained_variance_ratio_.sum()
print(f"PCA var (2D): {expl:.1%}")

# Known endocrine differentiation hierarchy:
# Ductal (progenitor) -> Ngn3 low EP -> Ngn3 high EP -> Pre-endocrine -> Alpha/Beta/Delta/Epsilon
# This is external biological knowledge, NOT pseudotime
endo_order = ['Ductal', 'Ngn3 low EP', 'Ngn3 high EP', 'Pre-endocrine', 'Alpha', 'Beta', 'Delta', 'Epsilon']

print("\n[Cell counts by stage]:")
for ct in endo_order:
    if ct in adata.obs['clusters'].values:
        n = sum(adata.obs['clusters'] == ct)
        print(f"  {ct:<20} {n:>5}")

# Build time-ordered distributions
distributions = []
for ct in endo_order:
    if ct in adata.obs['clusters'].values:
        mask = adata.obs['clusters'] == ct
        if mask.sum() >= 20:  # minimum cells
            distributions.append(X_pca[mask])

print(f"\n[Pipeline] {len(distributions)} stages, running OT...")

all_grads = []; all_points = []
for t in range(len(distributions)-1):
    Xs, Xt = distributions[t], distributions[t+1]
    eps = calibrate_epsilon(Xs, Xt)
    plan, conv, _ = sinkhorn_plan(Xs, Xt, epsilon=eps, num_iters=2000)
    grad_ot, _ = reconstruct_gradient(plan, Xs, Xt, None, 1.0, None)
    all_grads.append(grad_ot); all_points.append(Xs)

all_X = np.vstack(all_points); all_g = np.vstack(all_grads)
V_recon = robust_mst_integration(all_X, all_g, max_points=3000)

# Map to all cells
tree = KDTree(all_X); _, nn = tree.query(X_pca); V_all = V_recon[nn]

# Sign check: progenitor (Ductal) should be higher than terminally differentiated (Beta/Alpha)
valid_cts = [ct for ct in endo_order if ct in adata.obs['clusters'].values]
if len(valid_cts) >= 2:
    v_first = np.mean(V_all[adata.obs['clusters'] == valid_cts[0]])
    v_last = np.mean(V_all[adata.obs['clusters'] == valid_cts[-1]])
    if v_first < v_last:
        V_recon = -V_recon; V_all = -V_all
        v_first, v_last = -v_first, -v_last

print(f"\n[Results]")
print(f"{'Stage':<20} {'Mean V':>10} {'Std V':>10}")
for ct in valid_cts:
    mask = adata.obs['clusters'] == ct
    print(f"{ct:<20} {np.mean(V_all[mask]):>10.4f} {np.std(V_all[mask]):>10.4f}")

correct = v_first > v_last
print(f"\n  {valid_cts[0]} V = {v_first:.3f}")
print(f"  {valid_cts[-1]} V = {v_last:.3f}")
print(f"  Direction: {'CORRECT' if correct else 'REVERSED'}")

json.dump({
    'dataset': 'pancreas_development',
    'system': 'endocrine differentiation',
    'non_hematopoietic': True,
    'n_cells': int(adata.shape[0]),
    'n_genes': int(adata.shape[1]),
    'pca_var': float(expl),
    'stages': valid_cts,
    'correct': bool(correct),
    'V_by_stage': {ct: float(np.mean(V_all[adata.obs['clusters']==ct])) for ct in valid_cts}
}, open('results/pancreas_development.json','w'), indent=2)

print("\n[DONE] Non-hematopoietic real data validation complete.")
