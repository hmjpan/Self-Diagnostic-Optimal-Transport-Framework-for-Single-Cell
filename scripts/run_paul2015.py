"""
Real scRNA-seq Data Pipeline
=============================
Apply the inverse JKO Waddington landscape reconstruction to the
Paul et al. 2015 hematopoietic differentiation dataset.

Pipeline:
1. Preprocess: normalize, log-transform, select HVGs, PCA
2. Compute diffusion pseudotime (DPT) from MEP root
3. Bin cells into pseudotime windows
4. Apply inverse JKO to reconstruct the potential landscape in PCA space
5. Validate against known hematopoietic hierarchy
"""
import numpy as np
import scanpy as sc
from scipy.sparse import issparse
from pathlib import Path
import json, time, sys

sys.path.insert(0, str(Path(__file__).parent))
from src.sinkhorn import sinkhorn_plan, barycentric_projection
from src.potential import reconstruct_gradient
from final_experiment import (calibrate_epsilon, robust_mst_integration,
                               find_minima, compute_all_metrics)


def load_and_preprocess(n_top_genes=2000, n_pcs=20):
    """Load Paul 2015 data and preprocess."""
    print("[1/5] Loading Paul et al. 2015 hematopoietic data...")
    adata = sc.datasets.paul15()
    
    print(f"      Raw: {adata.shape[0]} cells, {adata.shape[1]} genes")
    
    # Basic preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Highly variable genes - try seurat_v3 first, fall back to seurat
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, 
                                    subset=True, flavor='seurat_v3')
    except (ModuleNotFoundError, ImportError):
        print("      seurat_v3 not available, using seurat flavor")
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, 
                                    subset=True, flavor='seurat')
    
    print(f"      After HVG: {adata.shape[0]} cells, {adata.shape[1]} genes")
    
    # PCA
    sc.tl.pca(adata, n_comps=n_pcs, svd_solver='arpack')
    
    return adata


def compute_pseudotime(adata, root_cluster='7MEP'):
    """Compute diffusion pseudotime using MEP as root."""
    print("[2/5] Computing diffusion pseudotime...")
    
    # Neighborhood graph
    sc.pp.neighbors(adata, n_neighbors=30, n_pcs=20)
    
    # Diffusion map
    sc.tl.diffmap(adata)
    
    # Root cell: pick one from MEP cluster
    root_idx = np.where(adata.obs['paul15_clusters'] == root_cluster)[0]
    if len(root_idx) == 0:
        # Fallback: use first cell
        root_idx = [0]
        print(f"      Warning: cluster '{root_cluster}' not found, using cell 0")
    
    root_cell = root_idx[0]
    adata.uns['iroot'] = root_cell
    
    # Diffusion pseudotime
    sc.tl.dpt(adata)
    
    print(f"      Root: {root_cluster} (cell {root_cell})")
    print(f"      DPT range: [{adata.obs['dpt_pseudotime'].min():.3f}, "
          f"{adata.obs['dpt_pseudotime'].max():.3f}]")
    
    return adata


def bin_by_pseudotime(adata, n_bins=12, pca_components=2):
    """Bin cells into pseudotime windows and extract PCA coordinates."""
    print(f"[3/5] Binning into {n_bins} pseudotime windows...")
    
    dpt = adata.obs['dpt_pseudotime'].values
    pca_coords = adata.obsm['X_pca'][:, :pca_components].copy()
    
    # Normalize PCA coordinates to [0, 1] per dimension for better OT behavior
    pca_mins = pca_coords.min(axis=0)
    pca_maxs = pca_coords.max(axis=0)
    pca_ranges = pca_maxs - pca_mins
    pca_coords = (pca_coords - pca_mins) / (pca_ranges + 1e-10)
    
    print(f"      PCA range: [{pca_mins[0]:.1f}, {pca_maxs[0]:.1f}] x "
          f"[{pca_mins[1]:.1f}, {pca_maxs[1]:.1f}]")
    print(f"      Normalized PCA to [0, 1] x [0, 1]")
    
    # Create equal-count bins
    bin_edges = np.percentile(dpt, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = dpt.min() - 1e-6
    bin_edges[-1] = dpt.max() + 1e-6
    
    distributions = []
    bin_times = []
    cell_counts = []
    
    for i in range(n_bins):
        mask = (dpt >= bin_edges[i]) & (dpt < bin_edges[i + 1])
        if mask.sum() > 10:
            distributions.append(pca_coords[mask])
            bin_times.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            cell_counts.append(mask.sum())
        else:
            print(f"      Bin {i}: only {mask.sum()} cells, skipping")
    
    print(f"      Created {len(distributions)} bins with "
          f"{np.mean(cell_counts):.0f} ± {np.std(cell_counts):.0f} cells/bin")
    print(f"      Pseudotime range: [{bin_times[0]:.3f}, {bin_times[-1]:.3f}]")
    
    return distributions, np.array(bin_times), cell_counts, pca_mins, pca_ranges


def reconstruct_landscape(distributions, times, output_dir='results'):
    """Run the inverse JKO pipeline on real data."""
    print("[4/5] Reconstructing Waddington landscape...")
    
    n_snapshots = len(distributions)
    all_grads = []
    all_points = []
    
    for t in range(n_snapshots - 1):
        X_s = distributions[t]
        X_tgt = distributions[t + 1]
        tau = times[t + 1] - times[t]
        
        eps = calibrate_epsilon(X_s, X_tgt)
        
        # Subsample large bins for efficiency
        if X_s.shape[0] > 500:
            idx_s = np.random.choice(X_s.shape[0], 500, replace=False)
        else:
            idx_s = np.arange(X_s.shape[0])
        if X_tgt.shape[0] > 500:
            idx_t = np.random.choice(X_tgt.shape[0], 500, replace=False)
        else:
            idx_t = np.arange(X_tgt.shape[0])
        
        plan, conv, niters = sinkhorn_plan(
            X_s[idx_s], X_tgt[idx_t], epsilon=eps, num_iters=2000, tol=1e-8
        )
        grad_ot, _ = reconstruct_gradient(plan, X_s[idx_s], X_tgt[idx_t], None, tau, None)
        
        all_grads.append(grad_ot)
        all_points.append(X_s[idx_s])
        
        if t % 5 == 0:
            print(f"      Interval {t}: eps={eps:.4f}, cells={len(idx_s)}, conv={conv}")
    
    # Aggregate
    all_X = np.vstack(all_points)
    all_grads_agg = np.vstack(all_grads)
    
    print(f"      Aggregated {all_X.shape[0]} gradient estimates")
    
    # Reconstruct potential
    V_recon = robust_mst_integration(all_X, all_grads_agg, max_points=3000)
    
    return V_recon, all_X, all_grads_agg, times


def analyze_landscape(V_recon, all_X, adata, output_dir='results'):
    """Analyze the reconstructed landscape against known biology."""
    print("[5/5] Analyzing reconstructed landscape...")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    from scipy.spatial import KDTree
    
    # Get cluster labels for all cells
    pca_all = adata.obsm['X_pca'][:, :2]
    clusters_all = adata.obs['paul15_clusters'].values
    
    # Build KDTree for reconstructed points
    tree_recon = KDTree(all_X)
    
    # Find minima with appropriate bandwidth for real data scale
    # PCA coordinates span ~50 units, need larger bandwidth
    data_scale = np.sqrt(np.var(all_X[:, 0]) + np.var(all_X[:, 1]))
    minima = find_minima(V_recon, all_X, radius=data_scale*0.05, min_sep=data_scale*0.03)
    
    if len(minima) > 20:
        # Too many minima, use more aggressive filtering
        minima = find_minima(V_recon, all_X, radius=data_scale*0.1, min_sep=data_scale*0.08)
    
    if len(minima) > 10:
        # Take top 10 by potential value
        _, nn = tree_recon.query(minima)
        best_idx = np.argsort(V_recon[nn])[:10]
        minima = minima[best_idx]
    
    print(f"      Found {len(minima)} significant minima (scale={data_scale:.1f})")
    
    # Map minima to cell types
    tree_all = KDTree(pca_all)
    
    if len(minima) > 0:
        print("\n      Minima -> Cell type mapping:")
        for i, m in enumerate(minima):
            dist, idx = tree_all.query(m.reshape(1, -1), k=50)
            nearby_clusters = clusters_all[idx]
            unique, counts = np.unique(nearby_clusters, return_counts=True)
            top = unique[np.argsort(-counts)[:3]]
            print(f"        Min #{i}: ({m[0]:.1f}, {m[1]:.1f}) -> {list(top)}")
    
    # Compute average potential for each cluster
    erythroid = [f'{i}Ery' for i in range(1, 9)]
    myeloid = ['11Mo', '12Mo', '13Mo', '14Mo', '15Mo', 
               '9DC', '10DC', '16Neu', '17Neu', '18Eos']
    
    print("\n      Average potential by cluster:")
    cluster_V = {}
    for cluster in sorted(clusters_all.unique()):
        mask = clusters_all == cluster
        cluster_pca = pca_all[mask]
        _, nn = tree_recon.query(cluster_pca)
        cluster_V[cluster] = float(np.mean(V_recon[nn]))
    
    sorted_clusters = sorted(cluster_V.items(), key=lambda x: x[1])
    for cluster, v in sorted_clusters:
        lineage = 'ERY' if cluster in erythroid else ('MYE' if cluster in myeloid else 'OTHER')
        print(f"        {cluster:12s} V={v:7.3f}  [{lineage}]")
    
    # Key biological check: MEP (root) should be at high V (progenitor),
    # differentiated cells at low V
    mep_V = cluster_V.get('7MEP', float('nan'))
    ery_V = np.mean([cluster_V.get(f'{i}Ery', float('nan')) 
                     for i in range(1, 9) if f'{i}Ery' in cluster_V])
    mye_V = np.mean([cluster_V.get(c, float('nan')) 
                     for c in myeloid if c in cluster_V])
    
    print(f"\n      Biological validation:")
    print(f"        MEP (root):   V = {mep_V:.3f}")
    print(f"        Erythroid:    V = {ery_V:.3f}")
    print(f"        Myeloid:      V = {mye_V:.3f}")
    print(f"        Gradient MEP->Ery: {mep_V - ery_V:.3f} "
          f"({'CORRECT' if mep_V > ery_V else 'REVERSED'})")
    
    results = {
        'n_minima': len(minima),
        'minima': minima.tolist() if len(minima) > 0 else [],
        'cluster_potentials': {k: float(v) for k, v in cluster_V.items()},
        'mep_v': float(mep_V),
        'ery_v': float(ery_V),
        'mye_v': float(mye_V),
    }
    
    with open(out / 'paul15_landscape.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def plot_real_data(adata, V_recon, all_X, minima, distributions, 
                   times, output_path):
    """Visualize the reconstructed landscape on real data."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    pca_all = adata.obsm['X_pca'][:, :2]
    clusters = adata.obs['paul15_clusters'].values
    
    # 1. PCA colored by cluster
    ax = axes[0, 0]
    sc.pl.pca(adata, color='paul15_clusters', ax=ax, show=False, 
              legend_loc='right margin', title='Paul et al. 2015 - Clusters')
    
    # 2. Reconstructed landscape
    ax = axes[0, 1]
    from scipy.spatial import KDTree
    tree_recon = KDTree(all_X)
    _, nn = tree_recon.query(pca_all)
    V_on_all = V_recon[nn]
    
    scat = ax.scatter(pca_all[:, 0], pca_all[:, 1], c=V_on_all, 
                     cmap='viridis', s=3, alpha=0.7)
    plt.colorbar(scat, ax=ax, label='V(x)')
    if len(minima) > 0:
        ax.scatter(minima[:, 0], minima[:, 1], c='white', s=100, 
                  edgecolors='black', marker='o', label='Minima')
        ax.legend(fontsize=8)
    ax.set_title('Reconstructed Waddington Landscape')
    ax.set_xlabel(f'PC1'); ax.set_ylabel('PC2')
    
    # 3. Pseudotime distribution
    ax = axes[0, 2]
    dpt = adata.obs['dpt_pseudotime'].values
    cm = plt.cm.viridis((dpt - dpt.min()) / (dpt.max() - dpt.min()))
    ax.scatter(pca_all[:, 0], pca_all[:, 1], c=cm, s=3, alpha=0.7)
    ax.set_title('Diffusion Pseudotime')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    
    # 4. Distributions over pseudotime
    ax = axes[1, 0]
    colors = plt.cm.plasma(np.linspace(0, 1, len(distributions)))
    for i, (X, c) in enumerate(zip(distributions, colors)):
        ax.scatter(X[:, 0], X[:, 1], c=[c], s=2, alpha=0.4)
    ax.set_title(f'Pseudotime Bins ({len(distributions)} windows)')
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    
    # 5. Gradient field
    ax = axes[1, 1]
    n_q = min(100, len(all_X))
    iq = np.random.choice(len(all_X), n_q, replace=False)
    from final_experiment import compute_all_metrics
    # Get gradients for those points
    # (we stored all_grads_agg for the subsampled points, need to re-interpolate)
    ax.set_title('Reconstructed ∇V Field')
    
    # 6. Potential vs pseudotime
    ax = axes[1, 2]
    ax.scatter(dpt, V_on_all, s=2, alpha=0.3, c=cm)
    ax.set_xlabel('Pseudotime'); ax.set_ylabel('V(x)')
    ax.set_title('Potential along Pseudotime')
    
    # Highlight known lineages
    erythroid_mask = np.array([c.startswith(('1E', '2E', '3E', '4E', '5E', '6E', '7E', '8E')) 
                               for c in clusters])
    myeloid_mask = np.array([c in ['11Mo', '12Mo', '13Mo', '14Mo', '15Mo', '16Neu', '17Neu']
                             for c in clusters])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      Plot saved to {output_path}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--nbins', type=int, default=10)
    p.add_argument('--npcs', type=int, default=2)
    p.add_argument('--output', default='results')
    args = p.parse_args()
    
    # 1. Load and preprocess
    adata = load_and_preprocess()
    
    # 2. Compute pseudotime
    adata = compute_pseudotime(adata)
    
    # 3. Bin by pseudotime
    distributions, times, cell_counts, pca_mins, pca_ranges = bin_by_pseudotime(
        adata, n_bins=args.nbins, pca_components=args.npcs
    )
    
    # 4. Reconstruct landscape
    V_recon, all_X, all_grads_agg, times = reconstruct_landscape(
        distributions, times, args.output
    )
    
    # 5. Analyze
    results = analyze_landscape(V_recon, all_X, adata, args.output)
    
    # 6. Plot
    minima = np.array(results['minima'])
    plot_real_data(adata, V_recon, all_X, minima, distributions, 
                   times, Path(args.output) / 'paul15_landscape.png')
    
    print("\n[Done] Real data analysis complete.")
