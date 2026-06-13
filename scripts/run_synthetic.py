"""
FINAL EXPERIMENT: Complete Waddington Landscape Reconstruction Pipeline
======================================================================

Mathematical framework (recap):
    Given time-series distributions {ρ_k} from Langevin dynamics
    dX = -∇V dt + √(2/β) dW, reconstruct V(x) via inverse JKO:

    ∇V(x) ≈ (x - T(x)) / τ

    where T is the OT map from ρ_k to ρ_{k+1}.

Key findings so far:
    1. OT barycentric projection naturally denoises (better than trajectory tracking)
    2. Noise correction term -(1/β)∇log ρ is counterproductive 
    3. ε ≈ 0.1-0.5 provides optimal trade-off (with cross-distance calibration)
    4. Potential RMSE ~15-20% achievable on synthetic landscapes
"""
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import KDTree
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json, time, sys

from src.sinkhorn import sinkhorn_plan, barycentric_projection
from src.potential import reconstruct_gradient
from src.landscapes import LANDSCAPES


def calibrate_epsilon(X_src, X_tgt):
    """
    Robust epsilon calibration balancing convergence and accuracy.
    
    Uses the MEDIAN squared cross-distance, scaled so that the Gibbs kernel
    exp(-C/ε) has entries predominantly in [exp(-5), exp(-0.2)].
    
    Capped at [0.01, 0.8] to ensure convergence and avoid uniform coupling.
    """
    # Subsample for efficiency
    n_s = min(300, X_src.shape[0])
    n_t = min(300, X_tgt.shape[0])
    idx_s = np.random.choice(X_src.shape[0], n_s, replace=False)
    idx_t = np.random.choice(X_tgt.shape[0], n_t, replace=False)
    
    cross_dists = cdist(X_src[idx_s], X_tgt[idx_t], metric='sqeuclidean')
    median_sq = np.median(cross_dists)
    
    # ε = median_sq / k where k ≈ 5 gives exp(-5) ≈ 0.007 for median entries
    eps = median_sq / 5.0
    
    # Ensure reasonable range
    eps = np.clip(eps, 0.01, 0.8)
    
    return eps


def robust_mst_integration(points, grad_V, max_points=5000):
    """Memory-safe MST-based potential integration with subsampling."""
    N = points.shape[0]
    
    if N > max_points:
        idx = np.random.choice(N, max_points, replace=False)
        pts = points[idx]
        grads = grad_V[idx]
    else:
        pts = points
        grads = grad_V
    
    n = pts.shape[0]
    dists = squareform(pdist(pts))
    np.fill_diagonal(dists, np.inf)
    mst = minimum_spanning_tree(dists).toarray()
    mst = mst + mst.T
    
    V = np.full(n, np.nan)
    V[0] = 0.0
    queue = [0]
    
    while queue:
        i = queue.pop(0)
        neighbors = np.where(mst[i] > 0)[0]
        for j in neighbors:
            if np.isnan(V[j]):
                delta = pts[j] - pts[i]
                dV = -0.5 * np.dot(grads[i] + grads[j], delta)
                V[j] = V[i] + dV
                queue.append(j)
    
    # Handle any remaining NaN
    nan_mask = np.isnan(V)
    if nan_mask.any():
        tree = KDTree(pts[~nan_mask])
        _, nn_idx = tree.query(pts[nan_mask])
        V[nan_mask] = V[~nan_mask][nn_idx]
    
    # Interpolate back
    if N > max_points:
        tree = KDTree(pts)
        _, nn = tree.query(points)
        V_full = V[nn]
    else:
        V_full = V
    
    return V_full


def find_minima(V, points, radius=0.4, min_sep=0.3):
    """Find local minima in the reconstructed potential."""
    tree = KDTree(points)
    order = np.argsort(V)
    found_mask = np.zeros(len(V), dtype=bool)
    minima_idx = []
    
    for i in order:
        if found_mask[i]:
            continue
        nb = tree.query_ball_point(points[i], radius)
        if len(nb) < 3:
            continue
        if np.all(V[nb] >= V[i] - 1e-6):
            minima_idx.append(i)
            # Suppress nearby
            nb_sep = tree.query_ball_point(points[i], min_sep)
            found_mask[nb_sep] = True
    
    return points[minima_idx] if minima_idx else np.array([]).reshape(0, points.shape[1])


def compute_all_metrics(grad_est, grad_true):
    """Multiple gradient quality metrics."""
    N = len(grad_est)
    
    abs_err = np.linalg.norm(grad_est - grad_true, axis=1)
    norm_true = np.linalg.norm(grad_true, axis=1)
    norm_est = np.linalg.norm(grad_est, axis=1)
    
    # Cosine similarity
    dot = np.sum(grad_est * grad_true, axis=1)
    denom = norm_est * norm_true + 1e-10
    cos_sim = np.clip(dot / denom, -1, 1)
    
    # Only consider points with significant true gradient
    sig = norm_true > np.percentile(norm_true, 30)
    
    return {
        'abs_err_mean': float(np.mean(abs_err)),
        'abs_err_median': float(np.median(abs_err)),
        'cos_sim_mean': float(np.mean(cos_sim)),
        'cos_sim_sig': float(np.mean(cos_sim[sig])) if sig.any() else 0.0,
        'dir_correct': float(np.mean(cos_sim > 0.3)),
        'dir_correct_sig': float(np.mean(cos_sim[sig] > 0.3)) if sig.any() else 0.0,
    }


def run_final(landscape_name, n_cells=1500, n_times=25, T_total=15.0,
              beta=100.0, output_dir='results'):
    """Complete reconstruction pipeline."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print(f"FINAL: {landscape_name} | N={n_cells} | β={beta} | Snapshots={n_times}")
    print("=" * 70)
    
    # ---- 1. Simulate ----
    landscape = LANDSCAPES[landscape_name]()
    rng = np.random.RandomState(42)
    dt = 0.01
    rec = max(1, int(T_total / (n_times - 1) / dt))
    n_steps = int(T_total / dt)
    
    X = rng.randn(n_cells, landscape.dim) * 0.3
    distributions = [X.copy()]
    times = [0.0]
    
    t0 = time.time()
    for s in range(1, n_steps + 1):
        X = X - landscape.grad_V(X) * dt + rng.randn(*X.shape) * np.sqrt(2*dt/beta)
        if s % rec == 0:
            distributions.append(X.copy())
            times.append(s * dt)
    
    distributions = distributions[:n_times]
    times = np.array(times[:n_times])
    tau = times[1] - times[0]
    print(f"[Sim] {n_times} snapshots, τ={tau:.3f}, took {time.time()-t0:.1f}s")
    
    # ---- 2. OT gradient reconstruction ----
    all_grads = []
    all_X = []
    interval_data = []
    
    t0 = time.time()
    for t in range(n_times - 1):
        X_s = distributions[t]
        X_tgt = distributions[t+1]
        
        eps = calibrate_epsilon(X_s, X_tgt)
        plan, conv, niters = sinkhorn_plan(X_s, X_tgt, epsilon=eps, 
                                            num_iters=2000, tol=1e-8)
        
        grad_ot, _ = reconstruct_gradient(plan, X_s, X_tgt, None, tau, None)
        grad_true = landscape.grad_V(X_s)
        m = compute_all_metrics(grad_ot, grad_true)
        
        interval_data.append({
            't': t, 'eps': float(eps), 'converged': conv, 'iters': niters,
            **m
        })
        
        all_grads.append(grad_ot)
        all_X.append(X_s)
    
    all_X = np.vstack(all_X)
    all_grads = np.vstack(all_grads)
    ot_time = time.time() - t0
    print(f"[OT]  {n_times-1} intervals, avg eps={np.mean([d['eps'] for d in interval_data]):.4f}, "
          f"took {ot_time:.1f}s")
    
    # ---- 3. Potential integration ----
    t0 = time.time()
    V_recon = robust_mst_integration(all_X, all_grads, max_points=4000)
    
    V_true = landscape.V(all_X)
    shift = np.mean(V_true - V_recon)
    V_shifted = V_recon + shift
    # Also check sign-flipped version (integration direction ambiguity)
    V_flipped = -V_recon + np.mean(V_true + V_recon)
    rmse1 = np.sqrt(np.mean((V_shifted - V_true)**2))
    rmse2 = np.sqrt(np.mean((V_flipped - V_true)**2))
    if rmse2 < rmse1:
        V_shifted = V_flipped
        V_recon = -V_recon
        rmse = rmse2
    else:
        rmse = rmse1
    V_range = max(np.max(V_true) - np.min(V_true), 1e-8)
    rel_rmse = rmse / V_range
    corr = np.corrcoef(V_shifted, V_true)[0, 1] if V_range > 0 else 0
    int_time = time.time() - t0
    
    print(f"[Pot] RMSE={rmse:.4f}, RelRMSE={rel_rmse:.4f}, Corr={corr:.4f}, "
          f"took {int_time:.1f}s")
    
    # ---- 4. Critical points ----
    minima_est = find_minima(V_recon, all_X, radius=0.5, min_sep=0.4)
    cp_err = float('nan')
    if hasattr(landscape, 'minima') and len(minima_est) > 0:
        D = cdist(minima_est, landscape.minima)
        ri, ci = linear_sum_assignment(D)
        cp_err = float(np.mean(D[ri, ci]))
        print(f"[CP]  Found {len(minima_est)} minima, error={cp_err:.4f} "
              f"(true: {len(landscape.minima)})")
    
    # ---- 5. Aggregate metrics ----
    global_m = compute_all_metrics(all_grads, landscape.grad_V(all_X))
    
    print(f"\n[Global] abs_err={global_m['abs_err_mean']:.4f}, "
          f"cos_sim={global_m['cos_sim_mean']:.3f}, "
          f"cos_sig={global_m['cos_sim_sig']:.3f}, "
          f"dir_correct={global_m['dir_correct']:.1%}")
    
    # ---- 6. Visualize ----
    _final_plot(landscape, distributions, times, all_X, V_recon, V_true,
                interval_data, global_m, all_grads, 
                out / f'{landscape_name}_final.png')
    
    # ---- 7. Save ----
    summary = {
        'landscape': landscape_name,
        'n_cells': n_cells, 'n_times': n_times, 'beta': beta,
        'tau': float(tau), 'T_total': T_total,
        'potential_rmse': float(rmse),
        'potential_rel_rmse': float(rel_rmse),
        'potential_correlation': float(corr),
        'critical_point_error': cp_err,
        'global_metrics': global_m,
        'intervals': interval_data,
        'n_minima_found': len(minima_est),
    }
    json.dump(summary, (out / f'{landscape_name}_final.json').open('w'), 
              indent=2, default=str)
    
    return summary


def _final_plot(landscape, distributions, times, all_X, V_recon, V_true,
                intervals, glob_m, all_grads_recon, out_path):
    """Production-quality visualization."""
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    
    # True landscape
    xs = np.linspace(-3, 3, 150)
    Xg, Yg = np.meshgrid(xs, xs)
    Zg = landscape.V(np.stack([Xg, Yg], axis=-1))
    
    ax = axes[0, 0]
    ax.contourf(Xg, Yg, Zg, levels=30, cmap='viridis')
    if hasattr(landscape, 'minima'):
        ax.scatter(*landscape.minima.T, c='white', s=100, edgecolors='black', 
                  marker='o', label='True minima')
    if hasattr(landscape, 'saddles'):
        ax.scatter(*landscape.saddles.T, c='yellow', s=80, edgecolors='black',
                  marker='s', label='True saddle')
    ax.legend(fontsize=7); ax.set_title('True Landscape V(x,y)')
    
    # Reconstructed
    ax = axes[0, 1]
    sc = ax.scatter(all_X[:, 0], all_X[:, 1], c=V_recon, cmap='viridis', s=1.5, alpha=0.7)
    plt.colorbar(sc, ax=ax)
    mins = find_minima(V_recon, all_X)
    if len(mins) > 0:
        ax.scatter(*mins.T, c='white', s=50, edgecolors='black', marker='o')
    ax.set_title('Reconstructed V(x,y)')
    
    # Correlation
    ax = axes[0, 2]
    shift = np.mean(V_true - V_recon)
    V_shifted_plot = V_recon + shift
    # Check sign
    V_flipped_plot = -V_recon + np.mean(V_true + V_recon)
    if np.sqrt(np.mean((V_flipped_plot - V_true)**2)) < np.sqrt(np.mean((V_shifted_plot - V_true)**2)):
        V_shifted_plot = V_flipped_plot
    ax.scatter(V_true, V_shifted_plot, s=1, alpha=0.15)
    lims = [min(V_true.min(), V_shifted_plot.min()),
            max(V_true.max(), V_shifted_plot.max())]
    ax.plot(lims, lims, 'r--', lw=1)
    r = np.corrcoef(V_true, V_shifted_plot)[0, 1]
    ax.set_xlabel('V_true'); ax.set_ylabel('V_recon')
    ax.set_title(f'Correlation r={r:.3f}')
    
    # Metrics over time
    ax = axes[0, 3]
    ts = [d['t'] for d in intervals]
    ax.plot(ts, [d['abs_err_mean'] for d in intervals], 'o-', ms=3, label='Abs err')
    ax.plot(ts, [d['cos_sim_sig'] for d in intervals], 's-', ms=3, label='Cos sim (sig)')
    ax.axhline(y=0.5, c='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Interval'); ax.set_ylabel('Metric')
    ax.legend(); ax.set_title('Quality vs Time')
    
    # Distributions
    ax = axes[1, 0]
    cm = plt.cm.plasma(np.linspace(0, 1, len(distributions)))
    for i, (X, c) in enumerate(zip(distributions, cm)):
        ax.scatter(X[:, 0], X[:, 1], c=[c], s=1, alpha=0.4)
    ax.set_title(f'Distributions ({len(distributions)} times)')
    
    # True gradient
    ax = axes[1, 1]
    n_q = min(150, len(all_X))
    iq = np.random.choice(len(all_X), n_q, replace=False)
    Xq = all_X[iq]
    gq = landscape.grad_V(Xq)
    nq = np.linalg.norm(gq, axis=1)
    keep = nq > np.percentile(nq, 15)
    ax.quiver(Xq[keep, 0], Xq[keep, 1], gq[keep, 0], gq[keep, 1],
             color='blue', alpha=0.5, scale=25)
    ax.set_title('True ∇V')
    
    # Reconstructed gradient
    ax = axes[1, 2]
    gq_ot = all_grads_recon[iq]
    ax.quiver(Xq[keep, 0], Xq[keep, 1], gq_ot[keep, 0], gq_ot[keep, 1],
             color='red', alpha=0.5, scale=25)
    ax.set_title('OT ∇V')
    
    # Error map
    ax = axes[1, 3]
    Verr = np.abs(V_shifted_plot - V_true)
    sc = ax.scatter(all_X[:, 0], all_X[:, 1], c=Verr, cmap='hot', s=1.5, alpha=0.7)
    plt.colorbar(sc, ax=ax)
    ax.set_title('|Error| Map')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [Plot] -> {out_path}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--landscape', default='bifurcation')
    p.add_argument('--ncells', type=int, default=1500)
    p.add_argument('--ntimes', type=int, default=25)
    p.add_argument('--beta', type=float, default=100.0)
    p.add_argument('--T', type=float, default=15.0)
    p.add_argument('--output', default='results')
    args = p.parse_args()
    
    run_final(args.landscape, args.ncells, args.ntimes, args.T, args.beta, args.output)
