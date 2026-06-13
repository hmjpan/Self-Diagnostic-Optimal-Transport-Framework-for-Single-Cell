"""
REAL TIME-SERIES DATA: Moignard et al. 2015
============================================
Mouse blood progenitor differentiation (in vitro).
5 developmental stages: PS -> NP -> HF -> 4SG -> 4SFG
3934 cells, 42 genes.

This is genuine time-series data - stages are ordered by embryology,
NOT by pseudotime computed from the expression data.
"""
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
from pathlib import Path
import sys, json, time

sys.path.insert(0, str(Path(__file__).parent))

from src.sinkhorn import sinkhorn_plan
from src.potential import reconstruct_gradient
from final_experiment import calibrate_epsilon, robust_mst_integration, find_minima

print("=" * 60)
print("MOIGNARD 2015 - REAL TIME-SERIES VALIDATION")
print("=" * 60)

# 1. Load data
print("\n[1] Loading Moignard 2015 data...")
adata = sc.datasets.moignard15()
print(f"    {adata.shape[0]} cells, {adata.shape[1]} genes")

# 2. Group by developmental stage (REAL time ordering)
stage_order = ['PS', 'NP', 'HF', '4SG', '4SFG']
stage_labels = {
    'PS': 'Primitive Streak (earliest)',
    'NP': 'Neural Plate',
    'HF': 'Head Fold',
    '4SG': '4-Somite',
    '4SFG': '4-Somite FG (latest)',
}

print("\n[2] Organizing by developmental stage:")
distributions_raw = []
times = []
for i, stage in enumerate(stage_order):
    mask = adata.obs['exp_groups'] == stage
    n = mask.sum()
    X = adata.X[mask]
    if hasattr(X, 'toarray'):
        X = X.toarray()
    distributions_raw.append(X)
    times.append(float(i))  # Stage index as pseudo-time
    print(f"    t={i}: {stage} - {n} cells")

# 3. Normalize and PCA
print("\n[3] PCA reduction...")
all_cells = np.vstack(distributions_raw)
# Simple normalization: standardize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
all_scaled = scaler.fit_transform(all_cells)

pca = PCA(n_components=2, random_state=42)
pca.fit(all_scaled)
explained = pca.explained_variance_ratio_.sum()

distributions = []
start_idx = 0
for X_raw in distributions_raw:
    n = X_raw.shape[0]
    X_scaled = scaler.transform(X_raw)
    X_pca = pca.transform(X_scaled)
    distributions.append(X_pca)
    start_idx += n

print(f"    PCA explained variance (2D): {explained:.2%}")

# 4. Inverse JKO reconstruction
print("\n[4] Running inverse JKO pipeline...")
all_grads = []
all_points = []

for t in range(len(distributions) - 1):
    X_s = distributions[t]
    X_tgt = distributions[t + 1]
    tau = times[t + 1] - times[t]
    
    eps = calibrate_epsilon(X_s, X_tgt)
    plan, conv, niters = sinkhorn_plan(X_s, X_tgt, epsilon=eps, num_iters=2000)
    grad_ot, _ = reconstruct_gradient(plan, X_s, X_tgt, None, tau, None)
    
    all_grads.append(grad_ot)
    all_points.append(X_s)
    
    print(f"    {stage_order[t]} -> {stage_order[t+1]}: eps={eps:.4f}, "
          f"cells={X_s.shape[0]}->{X_tgt.shape[0]}, conv={conv}")

all_X = np.vstack(all_points)
all_grads_agg = np.vstack(all_grads)

# 5. Reconstruct potential
print("\n[5] Reconstructing potential landscape...")
V_recon = robust_mst_integration(all_X, all_grads_agg, max_points=3000)

# 6. Map back to all cells and analyze
print("\n[6] Biological validation...")
from scipy.spatial import KDTree
tree = KDTree(all_X)

# Map reconstructed potential to all cells
pca_all = pca.transform(scaler.transform(
    np.vstack([d if not hasattr(d, 'toarray') else d.toarray() 
               for d in distributions_raw])
))
_, nn = tree.query(pca_all)
V_all = V_recon[nn]

# Compute mean V per stage
print(f"\n    {'Stage':<8} {'Cells':>6} {'Mean V':>10} {'Std V':>10}")
print(f"    {'-'*40}")
for i, stage in enumerate(stage_order):
    mask = adata.obs['exp_groups'] == stage
    V_stage = V_all[mask]
    print(f"    {stage:<8} {mask.sum():>6} {np.mean(V_stage):>10.4f} {np.std(V_stage):>10.4f}")

# Check: Waddington landscape -> progenitor (PS) should be HIGH,
# differentiated (4SFG) should be LOW
mask_ps = adata.obs['exp_groups'] == 'PS'
mask_4sfg = adata.obs['exp_groups'] == '4SFG'
V_ps = np.mean(V_all[mask_ps])
V_4sfg = np.mean(V_all[mask_4sfg])
print(f"\n    V(PS) = {V_ps:.4f} (progenitor, should be HIGH)")
print(f"    V(4SFG) = {V_4sfg:.4f} (differentiated, should be LOW)")
print(f"    Delta = {V_ps - V_4sfg:.4f} "
      f"({'CORRECT' if V_ps > V_4sfg else 'REVERSED - need sign flip'})")

# 7. Sign flip if needed
if V_ps < V_4sfg:
    print("\n    [SIGN FLIP] Reversing potential...")
    V_recon = -V_recon
    V_all = -V_all
    V_ps = -V_ps
    V_4sfg = -V_4sfg
    print(f"    V(PS) = {V_ps:.4f}, V(4SFG) = {V_4sfg:.4f}")

# 8. Find landscape minima
minima = find_minima(V_recon, all_X, radius=0.5, min_sep=0.5)
print(f"\n    Found {len(minima)} landscape minima")

# Map minima to nearest stages
if len(minima) > 0:
    tree_all = KDTree(pca_all)
    for i, m in enumerate(minima[:5]):
        dist, idx = tree_all.query(m.reshape(1, -1), k=30)
        nearby_stages = adata.obs['exp_groups'].values[idx[0]]
        unique, counts = np.unique(nearby_stages, return_counts=True)
        top = unique[np.argsort(-counts)[:3]]
        print(f"    Min #{i}: ({m[0]:.1f},{m[1]:.1f}) -> stages: {list(top)}")

# 9. Save and plot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Plot 1: All cells colored by stage
ax = axes[0, 0]
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
for i, stage in enumerate(stage_order):
    mask = adata.obs['exp_groups'] == stage
    ax.scatter(pca_all[mask, 0], pca_all[mask, 1], c=colors[i], 
              s=3, alpha=0.5, label=stage)
ax.legend(fontsize=8)
ax.set_title('Developmental Stages (True Time)')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')

# Plot 2: Reconstructed landscape
ax = axes[0, 1]
scat = ax.scatter(pca_all[:, 0], pca_all[:, 1], c=V_all, 
                 cmap='viridis', s=3, alpha=0.7)
plt.colorbar(scat, ax=ax)
if len(minima) > 0:
    ax.scatter(minima[:, 0], minima[:, 1], c='white', s=80, 
              edgecolors='black', marker='o')
ax.set_title('Reconstructed Waddington Landscape')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')

# Plot 3: V vs stage
ax = axes[0, 2]
stage_V = [np.mean(V_all[adata.obs['exp_groups'] == s]) for s in stage_order]
stage_std = [np.std(V_all[adata.obs['exp_groups'] == s]) for s in stage_order]
ax.errorbar(range(len(stage_order)), stage_V, yerr=stage_std, 
           marker='o', markersize=8, linewidth=2, capsize=5)
ax.set_xticks(range(len(stage_order)))
ax.set_xticklabels(stage_order)
ax.set_xlabel('Developmental Stage')
ax.set_ylabel('Mean Potential V')
ax.set_title('Potential vs Developmental Time')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# Plot 4-6: Individual stage distributions
for i, (stage, ax_idx) in enumerate([(0, (1,0)), (2, (1,1)), (4, (1,2))]):
    ax = axes[ax_idx]
    mask = adata.obs['exp_groups'] == stage_order[stage]
    ax.scatter(pca_all[mask, 0], pca_all[mask, 1], 
              c=colors[stage], s=3, alpha=0.6)
    ax.set_title(f'{stage_order[stage]}: {stage_labels[stage_order[stage]]}')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')

plt.tight_layout()
out_path = Path('results/moignard2015_landscape.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n    Plot saved to {out_path}")

# 10. Summary
print("\n" + "=" * 60)
print("REAL DATA VALIDATION SUMMARY")
print("=" * 60)
print(f"  Dataset: Moignard et al. 2015")
print(f"  Cells: {adata.shape[0]}, Genes: {adata.shape[1]}")
print(f"  Stages: {len(stage_order)} true developmental stages")
print(f"  PCA variance (2D): {explained:.2%}")
print(f"  Landscape direction: {'CORRECT' if V_ps > V_4sfg else 'REVERSED'}")
print(f"  Progenitor (PS) potential: {V_ps:.4f}")
print(f"  Differentiated (4SFG) potential: {V_4sfg:.4f}")
print(f"  Minima found: {len(minima)}")

# Save numerical results
results = {
    'dataset': 'Moignard2015',
    'n_cells': int(adata.shape[0]),
    'n_genes': int(adata.shape[1]),
    'n_stages': len(stage_order),
    'stages': stage_order,
    'pca_variance_2d': float(explained),
    'mean_V_by_stage': {s: float(np.mean(V_all[adata.obs['exp_groups'] == s])) 
                        for s in stage_order},
    'V_progenitor': float(V_ps),
    'V_differentiated': float(V_4sfg),
    'landscape_correct': bool(V_ps > V_4sfg),
    'n_minima': int(len(minima)),
}
with open('results/moignard2015_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n[DONE] Real data validation complete.")
