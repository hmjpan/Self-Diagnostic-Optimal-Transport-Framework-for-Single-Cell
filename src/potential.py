"""
Waddington Potential Reconstruction via Inverse JKO Scheme

Core mathematical innovation: Given a time series of cell distributions,
recover the driving potential function V(x) by inverting the
Jordan-Kinderlehrer-Otto variational scheme using optimal transport maps.

Key formula:
    ∇V(x) ≈ (x - T(x)) / τ - (1/β) ∇log ρ(x)

where T is the optimal transport map from time t_k to t_{k+1}.
"""
import numpy as np
from scipy.spatial import KDTree
from scipy.sparse.linalg import cg, spsolve
from scipy.sparse import diags, eye, kron, csr_matrix
from scipy.linalg import lstsq
from .sinkhorn import sinkhorn_plan, barycentric_projection


def estimate_log_density_gradient(points, k=30):
    """
    Estimate ∇log ρ(x) at each point using k-NN density estimation.
    
    For a k-NN density estimator ρ(x) ∝ k / (N * V_d * r_k(x)^d),
    we have log ρ(x) = const - d * log r_k(x).
    Hence ∇log ρ(x) ≈ -d * ∇log r_k(x).
    
    Parameters
    ----------
    points : ndarray (N, d)
        Point cloud
    k : int
        Number of neighbors for density estimation
    
    Returns
    -------
    grad_log_rho : ndarray (N, d)
        Gradient of log density at each point
    """
    N, d = points.shape
    tree = KDTree(points)
    distances, indices = tree.query(points, k=k + 1)
    
    # distances to k-th neighbor (excluding self)
    r_k = distances[:, -1]  # (N,)
    
    # ∇log r_k via finite differences with nearest neighbors
    grad_log_rho = np.zeros((N, d))
    
    for i in range(N):
        nn_idx = indices[i, 1:k + 1]  # k neighbors
        nn_vecs = points[nn_idx] - points[i]  # (k, d)
        nn_dists = distances[i, 1:k + 1][:, np.newaxis]  # (k, 1)
        
        # Weighted least squares gradient estimation
        # Minimize sum_j w_j ||g^T (x_j - x_i) - (log r(x_j) - log r(x_i))||^2
        with np.errstate(divide='ignore'):
            weights = 1.0 / (nn_dists[:, 0] + 1e-10)
        weights /= weights.sum()
        
        dlogr = np.log(r_k[nn_idx] + 1e-20) - np.log(r_k[i] + 1e-20)
        
        # Solve weighted least squares: A^T W A g = A^T W b
        A = nn_vecs  # (k, d)
        W = np.diag(weights)
        b = dlogr  # (k,)
        
        try:
            AtWA = A.T @ W @ A
            AtWb = A.T @ (weights * dlogr)
            if np.linalg.cond(AtWA) < 1e8:
                g = np.linalg.solve(AtWA, AtWb)
            else:
                g = np.linalg.lstsq(A, dlogr, rcond=None)[0]
        except np.linalg.LinAlgError:
            g = np.zeros(d)
        
        grad_log_rho[i] = -d * g
    
    return grad_log_rho


def reconstruct_gradient(plan, X_source, Y_target, X_density_grad=None,
                         tau=1.0, beta=None):
    """
    Reconstruct ∇V at each source point from the OT plan.
    
    Core formula:
        ∇V(x_i) = (x_i - T(x_i)) / τ - (1/β) ∇log ρ(x_i)
    
    Parameters
    ----------
    plan : ndarray (N, M)
        OT plan P from source to target
    X_source : ndarray (N, d)
        Source point cloud
    Y_target : ndarray (M, d)
        Target point cloud
    X_density_grad : ndarray (N, d), optional
        Precomputed ∇log ρ at source points
    tau : float
        Time step between distributions
    beta : float, optional
        Inverse noise level. If None, only deterministic part is used.
    
    Returns
    -------
    grad_V : ndarray (N, d)
        Reconstructed potential gradient at each source point
    T : ndarray (N, d)
        Barycentric projection (transport map)
    """
    T = barycentric_projection(plan, Y_target)
    
    # Deterministic drift component (from OT displacement)
    grad_V = (X_source - T) / tau
    
    # Add noise correction if beta is provided
    if beta is not None and X_density_grad is not None:
        grad_V -= (1.0 / beta) * X_density_grad
    
    return grad_V, T


def estimate_noise_level(grad_V, displacement, d, tau):
    """
    Estimate inverse temperature β from the fluctuation-dissipation relation.
    
    For the Langevin dynamics dX = -∇V dt + sqrt(2/β) dW:
    
    E[||ΔX_drift||²] = τ² E[||∇V||²]    (deterministic part)
    E[||ΔX_noise||²] = (2dτ)/β          (stochastic part)
    
    The OT map recovers the DETERMINISTIC drift, not the full trajectory.
    The displacement we measure is: ΔX_OT ≈ τ∇V + (τ/β)∇log ρ_k
    
    At stationarity ∇log ρ ≈ -β∇V, so ΔX_OT ≈ 0, but away from equilibrium
    the density gradient term contributes.
    
    We estimate β by considering the SCATTER of individual particle 
    displacements around the deterministic flow. Specifically:
    
    For particles near point x, the variance of their one-step displacement
    (in the direction orthogonal to drift) should be ~2/β.
    
    Alternative: use the entropy production rate.
    
    For now, use a robust estimator based on the scatter:
    σ²_noise = Var[ ||ΔX - E[ΔX|X]||² ] / d
    1/β ≈ σ²_noise / (2τ)
    
    Parameters
    ----------
    grad_V, displacement, d, tau : same as before
    
    Returns
    -------
    beta : float
        Estimated inverse temperature (capped at [1, 10^4])
    """
    # Method 1: Use the total mean squared displacement
    # E[||ΔX||²] = τ²E[||∇V||²] + (2dτ)/β + cross_term
    # The cross_term = -(2τ/β)E[∇V · ∇log ρ] averages to 2d/β at stationarity.
    # So total E[||ΔX||²] = τ²E[||∇V||²] + (4dτ)/β at stationarity.
    # Away from equilibrium it varies.
    
    E_disp_sq = np.mean(np.sum(displacement ** 2, axis=1))
    E_grad_sq = np.mean(np.sum(grad_V ** 2, axis=1))
    
    # The ∇V we estimated already includes noise correction if beta was known.
    # If grad_V was computed without noise correction, then:
    # displacement = τ∇V_true + (τ/β)∇log ρ + noise residual
    # The naive estimate overestimates the deterministic term.
    
    # Use a self-consistent approach:
    # 1. Estimate using the simpler formula (valid near stationarity):
    #    E[||ΔX||²] ≈ 4dτ/β + τ²E[||∇V||²]
    # 2. Solve for β
    
    drift_variance = tau ** 2 * E_grad_sq
    excess_variance = max(E_disp_sq - drift_variance, 1e-10)
    
    # Conservative estimation: attribute half the excess to noise
    diffusion_est = excess_variance / (4 * d * tau)
    diffusion_est = max(diffusion_est, 1e-6)
    
    beta_est = 1.0 / diffusion_est
    
    # Sanity bounds
    return np.clip(beta_est, 0.1, 1e6)


def solve_poisson_on_grid(grad_V, points, bounds, grid_resolution=64):
    """
    Reconstruct V on a regular grid from estimated gradient field ∇V
    by solving the Poisson equation ΔV = ∇·(∇V).
    
    This projects the per-point gradient estimates onto a consistent
    scalar potential field.
    
    Parameters
    ----------
    grad_V : ndarray (N, d)
        Gradient at each data point
    points : ndarray (N, d)
        Data point coordinates
    bounds : ndarray (d, 2)
        [[min_1, max_1], ..., [min_d, max_d]]
    grid_resolution : int
        Number of grid points per dimension
    
    Returns
    -------
    V_grid : ndarray (grid_resolution, ..., grid_resolution)
        Reconstructed potential on the grid
    grid_axes : list of ndarray
        Axis coordinates for each dimension
    """
    from scipy.interpolate import LinearNDInterpolator
    
    d = points.shape[1]
    
    # Create grid
    axes = [np.linspace(bounds[i, 0], bounds[i, 1], grid_resolution) 
            for i in range(d)]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), axis=-1)
    grid_shape = (grid_resolution,) * d
    N_grid = grid_resolution ** d
    
    # Interpolate gradient components onto grid
    div_term = np.zeros(N_grid)
    grid_flat = grid.reshape(N_grid, d)
    
    for dim in range(d):
        interp = LinearNDInterpolator(points, grad_V[:, dim], fill_value=0.0)
        grad_component = interp(grid_flat).reshape(grid_shape)
        
        # Finite difference divergence contribution
        grad_component_pad = np.pad(grad_component, 1, mode='edge')
        dx = (bounds[dim, 1] - bounds[dim, 0]) / (grid_resolution - 1)
        dg_dx = (grad_component_pad[2:, 1:-1] if dim == 0 else 
                 grad_component_pad[1:-1, 2:] if dim == 1 else
                 grad_component_pad[1:-1, 1:-1]) 
        # Simplified: use central difference
        slices_plus = [slice(1, -1)] * d
        slices_minus = [slice(1, -1)] * d
        slices_plus[dim] = slice(2, None)
        slices_minus[dim] = slice(0, -2)
        
        dg = (grad_component_pad[tuple(slices_plus)] - 
              grad_component_pad[tuple(slices_minus)]) / (2 * dx)
        div_term += dg.ravel()
    
    # Solve Poisson: -ΔV = f, where f = ∇·(∇V)
    # Laplacian on grid using finite differences
    f = -div_term  # right-hand side
    
    # Construct sparse Laplacian matrix (2D case, extend for d > 2)
    if d == 2:
        nx, ny = grid_resolution, grid_resolution
        n = nx * ny
        dx2 = ((bounds[0, 1] - bounds[0, 0]) / (nx - 1)) ** 2
        dy2 = ((bounds[1, 1] - bounds[1, 0]) / (ny - 1)) ** 2
        
        main_diag = np.ones(n) * (2.0 / dx2 + 2.0 / dy2)
        x_diag = np.ones(n - 1) * (-1.0 / dx2)
        y_diag = np.ones(n - nx) * (-1.0 / dy2)
        
        # Remove connections across x-boundaries
        for i in range(1, ny):
            x_diag[i * nx - 1] = 0.0
        
        A = diags([main_diag, x_diag, x_diag, y_diag, y_diag],
                  [0, 1, -1, nx, -nx], format='csr')
        
        # Fix gauge: set V[0,0] = 0
        A = A[1:, 1:]
        f = f[1:]
        
        V_flat = spsolve(A, f)
        V_flat = np.concatenate([[0.0], V_flat])
        V_grid = V_flat.reshape(grid_shape)
    else:
        # For d > 2, use a simpler approach: path integration
        V_flat = np.zeros(N_grid)
        for i in range(1, N_grid):
            # Integrate along coordinate axes from origin
            idx = np.unravel_index(i, grid_shape)
            prev_idx = list(idx)
            for j in range(d):
                if idx[j] > 0:
                    prev_idx[j] = idx[j] - 1
            prev_i = np.ravel_multi_index(tuple(prev_idx), grid_shape)
            delta = grid_flat[i] - grid_flat[prev_i]
            V_flat[i] = V_flat[prev_i] - np.dot(
                (grad_V_pts_or_zero(grid_flat[i], grad_V, points) + 
                 grad_V_pts_or_zero(grid_flat[prev_i], grad_V, points)) / 2,
                delta
            )
        V_grid = V_flat.reshape(grid_shape)
    
    return V_grid, axes


def grad_V_pts_or_zero(x, grad_V, points, threshold=1.0):
    """Nearest-neighbor lookup of gradient field."""
    tree = KDTree(points)
    dist, idx = tree.query(x.reshape(1, -1))
    if dist[0] < threshold:
        return grad_V[idx[0]]
    return np.zeros_like(x)


def reconstruct_from_time_series(distributions, times, epsilon=0.01,
                                  beta=None, estimate_beta=False,
                                  k_density=30, epsilon_mode='auto'):
    """
    Full pipeline: reconstruct Waddington potential from time-series
    distributions.
    
    Parameters
    ----------
    distributions : list of ndarray
        [X_0, X_1, ..., X_T] where each X_t is (N_t, d)
    times : ndarray (T+1,)
        Time points corresponding to each distribution
    epsilon : float
        Sinkhorn regularization (used only if epsilon_mode='fixed')
    beta : float or None
        Inverse noise level. If None and estimate_beta=True, estimated from data.
    estimate_beta : bool
        Whether to estimate β from data
    k_density : int
        Number of neighbors for density gradient estimation
    epsilon_mode : str
        'auto': ε = 0.01 * median squared distance (recommended)
        'fixed': use the provided epsilon value

    Returns
    -------
    grad_V_all : list of ndarray
        Estimated ∇V at each time point (except last)
    T_maps : list of ndarray
        Transport maps for each time interval
    betas : list of float
        Estimated β values for each time interval
    """
    T = len(distributions) - 1
    grad_V_all = []
    T_maps = []
    betas = []
    
    for t in range(T):
        X_src = distributions[t]
        X_tgt = distributions[t + 1]
        tau = times[t + 1] - times[t]
        
        # Adaptive epsilon: scale to typical squared distance in the data
        if epsilon_mode == 'auto':
            from scipy.spatial.distance import cdist
            # Sample distances to estimate scale
            n_sample = min(200, X_src.shape[0], X_tgt.shape[0])
            idx_src = np.random.choice(X_src.shape[0], n_sample, replace=False)
            idx_tgt = np.random.choice(X_tgt.shape[0], n_sample, replace=False)
            sample_dists = cdist(X_src[idx_src], X_tgt[idx_tgt], metric='sqeuclidean')
            median_sq_dist = np.median(sample_dists)
            # ε should be ~0.5-5% of median squared distance for clear cost contrast
            eps_auto = max(median_sq_dist * 0.02, 0.001)
        else:
            eps_auto = epsilon
        
        # Compute OT plan
        plan, converged, n_iters = sinkhorn_plan(
            X_src, X_tgt, epsilon=eps_auto
        )
        
        if not converged:
            print(f"  [Warning] Sinkhorn did not converge for interval {t} "
                  f"(iters={n_iters}, eps={eps_auto:.4f})")
        
        # Density gradient at source
        if beta is not None or estimate_beta:
            grad_log = estimate_log_density_gradient(X_src, k=k_density)
        else:
            grad_log = None
        
        # Initial deterministic gradient estimate (without noise correction)
        grad_V_det, T_map = reconstruct_gradient(plan, X_src, X_tgt, None, tau, None)
        
        # Apply noise correction if beta is available
        if estimate_beta and grad_log is not None:
            # First pass: estimate beta from deterministic gradient
            displacement = X_src - T_map
            beta_est = estimate_noise_level(grad_V_det, displacement, 
                                            X_src.shape[1], tau)
            betas.append(beta_est)
            
            # Second pass: refine with noise correction
            grad_V, T_map = reconstruct_gradient(plan, X_src, X_tgt, grad_log, 
                                                  tau, beta_est)
        elif beta is not None and grad_log is not None:
            grad_V, T_map = reconstruct_gradient(plan, X_src, X_tgt, grad_log, 
                                                  tau, beta)
            betas.append(beta)
        else:
            grad_V = grad_V_det
            if estimate_beta:
                betas.append(np.nan)
        
        grad_V_all.append(grad_V)
        T_maps.append(T_map)
    
    return grad_V_all, T_maps, betas


def integrate_potential(points, grad_V):
    """
    Reconstruct scalar potential from gradient field by path integration
    using minimum spanning tree.
    
    Parameters
    ----------
    points : ndarray (N, d)
        Point coordinates
    grad_V : ndarray (N, d)
        Gradient values at each point
    
    Returns
    -------
    V : ndarray (N,)
        Potential values (up to additive constant)
    """
    from scipy.sparse.csgraph import minimum_spanning_tree
    from scipy.spatial.distance import pdist, squareform
    
    N = points.shape[0]
    
    # Build Euclidean distance graph
    dists = squareform(pdist(points))
    
    # Build minimum spanning tree
    mst = minimum_spanning_tree(dists)
    mst = mst.toarray()
    mst = mst + mst.T  # Make symmetric
    
    # BFS from node 0 to compute potentials
    V = np.full(N, np.nan)
    V[0] = 0.0
    queue = [0]
    
    while queue:
        i = queue.pop(0)
        for j in range(N):
            if mst[i, j] > 0 and np.isnan(V[j]):
                delta = points[j] - points[i]
                # Trapezoidal rule for line integral
                dV = -0.5 * np.dot(grad_V[i] + grad_V[j], delta)
                V[j] = V[i] + dV
                queue.append(j)
    
    return V
