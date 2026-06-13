"""
Sinkhorn Algorithm for Entropy-Regularized Optimal Transport

Computes the optimal coupling between two empirical measures under the
entropy-regularized Kantorovich formulation:

    minimize  <P, C> - epsilon * H(P)
    s.t.      P 1 = a, P^T 1 = b

via iterative Bregman projections (Sinkhorn-Knopp algorithm).

Reference: Cuturi (2013), "Sinkhorn Distances: Lightspeed Computation of Optimal Transport"
"""
import numpy as np
from scipy.spatial.distance import cdist
from numba import njit
import warnings


@njit(cache=True, parallel=False)
def _sinkhorn_kernel(K, a, b, num_iters=1000, tol=1e-9):
    """
    Core Sinkhorn iteration (Numba-accelerated).
    
    Parameters
    ----------
    K : ndarray (N, M)
        Gibbs kernel exp(-C / epsilon)
    a : ndarray (N,)
        Source marginal (sum to 1)
    b : ndarray (M,)
        Target marginal (sum to 1)
    num_iters : int
        Maximum iterations
    tol : float
        Convergence tolerance in L1 norm of marginals
    
    Returns
    -------
    u : ndarray (N,)
    v : ndarray (M,)
    converged : bool
    n_iters : int
    """
    N, M = K.shape
    u = np.ones(N)
    v = np.ones(M)
    
    for iteration in range(num_iters):
        # Scale row-wise: u = a / (K v)
        Kv = np.zeros(N)
        for i in range(N):
            s = 0.0
            for j in range(M):
                s += K[i, j] * v[j]
            Kv[i] = s
        
        u_old = u.copy()
        for i in range(N):
            if Kv[i] > 0:
                u[i] = a[i] / Kv[i]
            else:
                u[i] = 0.0
        
        # Scale column-wise: v = b / (K^T u)
        KTu = np.zeros(M)
        for j in range(M):
            s = 0.0
            for i in range(N):
                s += K[i, j] * u[i]
            KTu[j] = s
        
        for j in range(M):
            if KTu[j] > 0:
                v[j] = b[j] / KTu[j]
            else:
                v[j] = 0.0
        
        # Check marginal error
        err = 0.0
        for i in range(N):
            err += abs(Kv[i] * u_old[i] - a[i])
        if err < tol:
            return u, v, True, iteration + 1
    
    return u, v, False, num_iters


def sinkhorn_plan(X, Y, a=None, b=None, epsilon=0.01, num_iters=1000, tol=1e-9):
    """
    Compute entropy-regularized optimal transport plan between two point clouds.

    Parameters
    ----------
    X : ndarray (N, d)
        Source point cloud
    Y : ndarray (M, d)
        Target point cloud
    a : ndarray (N,), optional
        Source weights. Uniform if None.
    b : ndarray (M,), optional
        Target weights. Uniform if None.
    epsilon : float
        Entropy regularization strength
    num_iters : int
        Maximum Sinkhorn iterations
    tol : float
        Convergence tolerance

    Returns
    -------
    plan : ndarray (N, M)
        Optimal transport plan P
    converged : bool
    n_iters : int
    """
    N = X.shape[0]
    M = Y.shape[0]
    
    if a is None:
        a = np.ones(N) / N
    if b is None:
        b = np.ones(M) / M
    
    # Cost matrix: squared Euclidean distance
    C = cdist(X, Y, metric='sqeuclidean')
    
    # Gibbs kernel
    K = np.exp(-C / epsilon)
    
    u, v, converged, n_iters = _sinkhorn_kernel(K, a, b, num_iters, tol)
    
    # Build transport plan
    plan = np.diag(u) @ K @ np.diag(v)
    
    return plan, converged, n_iters


def sinkhorn_distance(X, Y, a=None, b=None, epsilon=0.01, num_iters=1000, tol=1e-9):
    """
    Compute entropy-regularized Wasserstein-2 distance between two point clouds.
    
    W_{2,epsilon}(X, Y)^2 = <C, P>
    """
    plan, converged, n_iters = sinkhorn_plan(X, Y, a, b, epsilon, num_iters, tol)
    C = cdist(X, Y, metric='sqeuclidean')
    distance = np.sqrt(np.sum(plan * C))
    return distance, plan, converged, n_iters


def barycentric_projection(plan, Y):
    """
    Compute the barycentric projection (conditional expectation) of the
    transport plan: T(x_i) = E_{Y|X=x_i}[Y]
    
    Parameters
    ----------
    plan : ndarray (N, M)
        Optimal transport plan P
    Y : ndarray (M, d)
        Target point cloud

    Returns
    -------
    T : ndarray (N, d)
        Barycentric projection (transport map)
    """
    N = plan.shape[0]
    d = Y.shape[1]
    T = np.zeros((N, d))
    
    for i in range(N):
        row_sum = plan[i, :].sum()
        if row_sum > 1e-12:
            T[i] = plan[i, :] @ Y / row_sum
        else:
            T[i] = Y[np.argmax(plan[i, :])]
    
    return T


def displacement_interpolation(X, T, t):
    """
    McCann's displacement interpolation (Wasserstein geodesic).
    
    X_t = (1-t) * id + t * T
    
    Parameters
    ----------
    X : ndarray (N, d)
        Source points
    T : ndarray (N, d)
        Transport map
    t : float in [0, 1]
        Interpolation parameter

    Returns
    -------
    X_t : ndarray (N, d)
        Interpolated points
    """
    return (1 - t) * X + t * T
