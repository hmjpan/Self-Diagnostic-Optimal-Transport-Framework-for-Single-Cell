"""
FAST Gastrulation Pipeline — subsampled, TruncatedSVD
"""
import numpy as np, gzip, json, sys, time
from scipy.io import mmread
from scipy.sparse import issparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.sinkhorn import sinkhorn_plan
from src.potential import reconstruct_gradient
from final_experiment import calibrate_epsilon, robust_mst_integration

print("=" * 60)
print("GASTRULATION — Fast Pipeline")
print("=" * 60)

# Metadata
with gzip.open('data/gastrulation_meta.tab.gz','rt') as f:
    meta_lines = f.read().strip().split('\n')
header = meta_lines[0].split('\t')
stages_all = [l.split('\t')[header.index('stage')].strip() for l in meta_lines[1:]]
stages = sorted(set(stages_all) - {'mixed_gastrulation'})
print(f"[1] {len(stages_all)} cells, {len(stages)} stages")

# Load sparse, transpose to cells x genes
print("[2] Loading sparse matrix...")
counts = mmread('data/raw_counts.mtx.gz')
if counts.shape[1] == len(stages_all): counts = counts.T.tocsr()
else: counts = counts.tocsr()
print(f"    {counts.shape[0]} cells x {counts.shape[1]} genes, {counts.nnz/1e6:.0f}M nnz")

# Subsample 3000 cells per stage
print("[3] Subsample & HVG selection...")
rng = np.random.RandomState(42)
sub_data = []; sub_labels = []
n_per_stage = 3000
for stage in stages:
    mask = np.array([s == stage for s in stages_all])
    idx = np.where(mask)[0]
    n = min(n_per_stage, len(idx))
    chosen = rng.choice(idx, n, replace=False)
    sub_data.append(counts[chosen, :])
    sub_labels.extend([stage] * n)
    print(f"    {stage}: {n} cells")

# Stack and select HVGs
all_sub = np.vstack([d.toarray() for d in sub_data])
all_log = np.log1p(all_sub)
gene_vars = np.var(all_log, axis=0)
top_genes = np.argsort(-gene_vars)[:2000]
print(f"    {len(top_genes)} HVGs, total cells: {all_sub.shape[0]}")

# Subset to HVGs and normalize
all_sub_hvg = all_sub[:, top_genes]
all_log_hvg = np.log1p(all_sub_hvg)
scaler = StandardScaler()
all_scaled = scaler.fit_transform(all_log_hvg)

# TruncatedSVD (faster than PCA for this)
print("[4] TruncatedSVD...")
svd = TruncatedSVD(n_components=5, random_state=42)
all_pca = svd.fit_transform(all_scaled)
print(f"    Var: 2D={svd.explained_variance_ratio_[:2].sum():.1%}, "
      f"5D={svd.explained_variance_ratio_[:5].sum():.1%}")

# Organize by stage
print("[5] Running OT pipeline (5D)...")
sub_labels = np.array(sub_labels)
dists = [all_pca[sub_labels == s, :5] for s in stages]

all_g = []; all_p = []
for t in range(len(dists)-1):
    Xs, Xt = dists[t], dists[t+1]
    eps = calibrate_epsilon(Xs, Xt)
    plan, conv, _ = sinkhorn_plan(Xs, Xt, epsilon=eps, num_iters=2000)
    grad_ot, _ = reconstruct_gradient(plan, Xs, Xt, None, 1.0, None)
    all_g.append(grad_ot); all_p.append(Xs)
    print(f"    {stages[t]}->{stages[t+1]}: eps={eps:.4f}, n={Xs.shape[0]}->{Xt.shape[0]}, conv={conv}")

all_X = np.vstack(all_p); all_grads = np.vstack(all_g)
V_recon = robust_mst_integration(all_X, all_grads, max_points=5000)

# Map to all cells
from scipy.spatial import KDTree
tree = KDTree(all_X[:, :2])
_, nn = tree.query(all_pca[:, :2])
V_all = V_recon[nn]

# Results
stage_V = {}
for s in stages:
    stage_V[s] = float(np.mean(V_all[sub_labels == s]))

first, last = stages[0], stages[-1]
if stage_V[first] < stage_V[last]:
    for s in stage_V: stage_V[s] = -stage_V[s]

from scipy.stats import spearmanr
stage_nums = [float(s.replace('E','')) for s in stages]
rho, p = spearmanr(stage_nums, [stage_V[s] for s in stages])

print(f"\n[6] Results:")
print(f"    {first} V = {stage_V[first]:.3f}, {last} V = {stage_V[last]:.3f}")
print(f"    Direction: {'CORRECT' if stage_V[first] > stage_V[last] else 'REVERSED'}")
print(f"    Spearman rho = {rho:.3f} (p={p:.4f}) -> {'CORRECT' if rho<-0.3 else 'WEAK'}")
for s in stages:
    print(f"    {s}: V = {stage_V[s]:.3f}")

json.dump({
    'dataset': 'Pijuan-Sala 2019 gastrulation', 'system': 'embryonic development',
    'non_hematopoietic': True, 'n_cells_subsampled': all_sub.shape[0],
    'n_full': len(stages_all), 'direction_correct': bool(stage_V[first] > stage_V[last]),
    'spearman_rho': float(rho), 'spearman_p': float(p),
    'V_by_stage': stage_V, 'pca_var_2d': float(svd.explained_variance_ratio_[:2].sum()),
}, open('results/gastrulation_results.json','w'), indent=2)
print("\n[DONE]")
