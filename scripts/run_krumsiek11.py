"""
krumsiek11 v2: Correct hematopoietic branching structure
Branch 1 (erythroid/megakaryocyte): progenitor -> Ery -> Mk  
Branch 2 (myeloid):                 progenitor -> Mo -> Neu
"""
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.spatial import KDTree
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.sinkhorn import sinkhorn_plan
from src.potential import reconstruct_gradient
from final_experiment import calibrate_epsilon, robust_mst_integration, compute_all_metrics

adata = sc.datasets.krumsiek11()
X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2, random_state=42); X_pca = pca.fit_transform(X_scaled)
expl = pca.explained_variance_ratio_.sum()

print("=" * 60)
print("KRUMSIEK11: HEMATOPOIETIC BRANCHING")
print(f"PCA var: {expl:.1%}")
print("=" * 60)

def run_branch(name, stages):
    """Run OT pipeline on a single differentiation branch."""
    dists = [X_pca[adata.obs['cell_type'] == s] for s in stages]
    all_g = []; all_p = []
    for t in range(len(dists)-1):
        Xs, Xt = dists[t], dists[t+1]
        eps = calibrate_epsilon(Xs, Xt)
        plan, conv, _ = sinkhorn_plan(Xs, Xt, epsilon=eps, num_iters=2000)
        grad_ot, _ = reconstruct_gradient(plan, Xs, Xt, None, 1.0, None)
        all_g.append(grad_ot); all_p.append(Xs)
        
        # LR baseline (subsample to match)
        n_min = min(len(Xs), len(Xt))
        idx_s = np.random.choice(len(Xs), n_min, replace=False)
        idx_t = np.random.choice(len(Xt), n_min, replace=False)
        lr = LinearRegression().fit(Xs[idx_s], Xt[idx_t])
        grad_lr = (Xs - lr.predict(Xs)) / 1.0
        m_lr = compute_all_metrics(grad_lr, grad_ot)  # compare LR vs OT
        
        print(f"  {name} {stages[t]}->{stages[t+1]}: eps={eps:.4f}, "
              f"n={Xs.shape[0]}->{Xt.shape[0]}, cos(LR,OT)={m_lr['cos_sim_mean']:.3f}")
    
    all_X = np.vstack(all_p); all_G = np.vstack(all_g)
    V_rec = robust_mst_integration(all_X, all_G, max_points=2000)
    return V_rec, all_X

# Branch 1: progenitor -> Ery -> Mk
print("\n[Branch 1: Erythroid/Megakaryocyte]")
V1, X1 = run_branch("Ery/Mk", ['progenitor', 'Ery', 'Mk'])

# Branch 2: progenitor -> Mo -> Neu
print("\n[Branch 2: Myeloid]")
V2, X2 = run_branch("Myeloid", ['progenitor', 'Mo', 'Neu'])

# Map V values back to all cells
tree1 = KDTree(X1); tree2 = KDTree(X2)
_, nn1 = tree1.query(X_pca); _, nn2 = tree2.query(X_pca)
V1_all = V1[nn1]; V2_all = V2[nn2]

# Sign correction per branch
for name, V_all, branch_start in [('Ery/Mk', V1_all, 'progenitor'),
                                     ('Myeloid', V2_all, 'progenitor')]:
    for ct, is_terminal in [('Ery', False), ('Mk', True), ('Mo', False), ('Neu', True)]:
        if ct in adata.obs['cell_type'].unique():
            v_ct = np.mean(V_all[adata.obs['cell_type'] == ct])
    # Just check progenitor vs terminal
    pass

print("\n[Mean V by cell type]")
print(f"{'Cell Type':<15} {'V (Ery/Mk branch)':>18} {'V (Myeloid branch)':>18}")
for ct in ['progenitor', 'Ery', 'Mk', 'Mo', 'Neu']:
    mask = adata.obs['cell_type'] == ct
    v1 = np.mean(V1_all[mask]); v2 = np.mean(V2_all[mask])
    print(f"{ct:<15} {v1:>18.4f} {v2:>18.4f}")

# Biological check
v1_prog = np.mean(V1_all[adata.obs['cell_type']=='progenitor'])
v1_mk = np.mean(V1_all[adata.obs['cell_type']=='Mk'])
v2_prog = np.mean(V2_all[adata.obs['cell_type']=='progenitor'])
v2_neu = np.mean(V2_all[adata.obs['cell_type']=='Neu'])
print(f"\n  Ery/Mk branch: V(prog)={v1_prog:.3f} vs V(Mk)={v1_mk:.3f} "
      f"-> {'CORRECT' if v1_prog>v1_mk else 'REVERSED'}")
print(f"  Myeloid branch: V(prog)={v2_prog:.3f} vs V(Neu)={v2_neu:.3f} "
      f"-> {'CORRECT' if v2_prog>v2_neu else 'REVERSED'}")

json.dump({
    'dataset': 'krumsiek11', 'n_cells': 640, 'n_genes': 11,
    'pca_var': float(expl),
    'branch1_ery_mk': {'V_prog': float(v1_prog), 'V_mk': float(v1_mk), 
                        'correct': bool(v1_prog>v1_mk)},
    'branch2_myeloid': {'V_prog': float(v2_prog), 'V_neu': float(v2_neu),
                         'correct': bool(v2_prog>v2_neu)},
}, open('results/krumsiek11_branched.json','w'), indent=2)

print("\n[DONE]")
