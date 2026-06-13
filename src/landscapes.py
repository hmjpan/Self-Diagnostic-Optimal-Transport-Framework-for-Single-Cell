"""
Synthetic Waddington Landscape Generator and Simulator

Generates known potential functions for validation and simulates
Langevin dynamics on them.
"""
import numpy as np
from scipy.stats import multivariate_normal


class SyntheticLandscape:
    """
    Base class for synthetic Waddington landscapes with known ground truth.
    """
    
    def V(self, X):
        """Potential function. X: (..., d) -> (...)"""
        raise NotImplementedError
    
    def grad_V(self, X):
        """Gradient of potential."""
        raise NotImplementedError
    
    def simulate(self, n_cells, X0, T_total, dt=0.01, beta=10.0, 
                 seed=None, record_every=100):
        """
        Simulate Langevin dynamics on the landscape.
        
        dX_t = -∇V(X_t) dt + sqrt(2/β) dW_t
        
        Parameters
        ----------
        n_cells : int
            Number of cells (particles)
        X0 : ndarray (n_cells, d)
            Initial positions
        T_total : float
            Total simulation time
        dt : float
            Integration time step
        beta : float
            Inverse temperature
        seed : int, optional
            Random seed
        record_every : int
            Record state every N steps
        
        Returns
        -------
        trajectory : list of ndarray
            Recorded distributions at each snapshot
        times : ndarray
            Time points corresponding to each snapshot
        """
        rng = np.random.RandomState(seed)
        n_steps = int(T_total / dt)
        n_snapshots = n_steps // record_every + 1
        
        trajectory = []
        times = np.zeros(n_snapshots)
        
        X = X0.copy()
        trajectory.append(X.copy())
        times[0] = 0.0
        
        for step in range(1, n_steps + 1):
            # Euler-Maruyama
            dW = rng.randn(*X.shape) * np.sqrt(dt)
            X = X - self.grad_V(X) * dt + np.sqrt(2.0 / beta) * dW
            
            if step % record_every == 0:
                idx = step // record_every
                trajectory.append(X.copy())
                times[idx] = step * dt
        
        return trajectory, times
    
    def steady_state_sample(self, n_cells, beta=10.0, n_burnin=5000, 
                            n_thin=10, seed=None):
        """
        Sample from the stationary distribution π(x) ∝ exp(-β V(x))
        using Langevin MCMC.
        """
        rng = np.random.RandomState(seed)
        d = self.dim
        
        # Initialize from Gaussian
        X = rng.randn(n_cells, d)
        
        dt = 0.01
        
        # Burn-in
        for _ in range(n_burnin):
            dW = rng.randn(n_cells, d) * np.sqrt(dt)
            X = X - self.grad_V(X) * dt + np.sqrt(2.0 / beta) * dW
        
        # Sampling with thinning
        samples = []
        for i in range(n_cells * n_thin):
            dW = rng.randn(n_cells, d) * np.sqrt(dt)
            X = X - self.grad_V(X) * dt + np.sqrt(2.0 / beta) * dW
            if i % n_thin == 0:
                samples.append(X.copy())
        
        return np.vstack(samples)


class SimpleBifurcation(SyntheticLandscape):
    """
    Simple bifurcation landscape.
    V(x, y) = (1/4)(x^2 - 1)^2 + (1/2) y^2
    
    Has two minima at (±1, 0) and a saddle at (0, 0).
    """
    def __init__(self):
        self.dim = 2
        self.minima = np.array([[-1.0, 0.0], [1.0, 0.0]])
        self.saddles = np.array([[0.0, 0.0]])
    
    def V(self, X):
        x, y = X[..., 0], X[..., 1]
        return 0.25 * (x**2 - 1)**2 + 0.5 * y**2
    
    def grad_V(self, X):
        x, y = X[..., 0], X[..., 1]
        dVdx = x * (x**2 - 1)
        dVdy = y
        return np.stack([dVdx, dVdy], axis=-1)


class ThreeWayBifurcation(SyntheticLandscape):
    """
    Three-way bifurcation with cubic symmetry.
    V(x, y) = (x^2 + y^2 - 1)^2 + (1/2)(x^3 - 3xy^2)
    
    Has three minima arranged at 120° intervals.
    """
    def __init__(self):
        self.dim = 2
        theta = np.linspace(0, 2*np.pi, 4)[:3]
        self.minima = np.column_stack([np.cos(theta), np.sin(theta)])
    
    def V(self, X):
        x, y = X[..., 0], X[..., 1]
        r2 = x**2 + y**2
        return (r2 - 1)**2 + 0.5 * (x**3 - 3*x*y**2)
    
    def grad_V(self, X):
        x, y = X[..., 0], X[..., 1]
        r2 = x**2 + y**2
        dVdx = 4 * x * (r2 - 1) + 0.5 * (3*x**2 - 3*y**2)
        dVdy = 4 * y * (r2 - 1) + 0.5 * (-6*x*y)
        return np.stack([dVdx, dVdy], axis=-1)


class HierarchicalBifurcation(SyntheticLandscape):
    """
    Hierarchical bifurcation: first splits left/right, then each branch 
    bifurcates again.
    
    V(x, y) = (1/4)(x^2 - 1)^4 + (1/2)(y - tanh(4x))^2 (y + tanh(4x))^2
    """
    def __init__(self):
        self.dim = 2
        self.minima = np.array([[-1.0, -0.7616], [-1.0, 0.7616],
                                [1.0, -0.7616], [1.0, 0.7616]])
    
    def V(self, X):
        x, y = X[..., 0], X[..., 1]
        th = np.tanh(4 * x)
        return 0.25 * (x**2 - 1)**4 + 0.5 * (y**2 - th**2)**2
    
    def grad_V(self, X):
        x, y = X[..., 0], X[..., 1]
        th = np.tanh(4 * x)
        dth = 4 * (1 - th**2)
        
        dVdx = 2 * (x**2 - 1)**3 * x + 2 * (y**2 - th**2) * (-2 * th * dth)
        dVdy = 2 * (y**2 - th**2) * y
        return np.stack([dVdx, dVdy], axis=-1)


class HighDimensionalLandscape(SyntheticLandscape):
    """
    High-dimensional generalization: d-dimensional landscape with
    m attractors embedded in a d-dimensional space.
    
    V(x) = -log(∑_{i=1}^m exp(-||x - μ_i||_{Σ_i}^2 / 2))
    
    where ||x||_{Σ_i} = x^T Σ_i^{-1} x
    """
    def __init__(self, means, covs=None, n_dims=None, n_attractors=None, seed=42):
        """
        Parameters
        ----------
        means : ndarray (m, d) or None
            Attractor locations
        covs : list of ndarray or None
            Covariance matrices for each attractor
        n_dims : int
            Dimensionality (if means not provided)
        n_attractors : int
            Number of attractors (if means not provided)
        seed : int
            Random seed for generating random landscapes
        """
        rng = np.random.RandomState(seed)
        
        if means is not None:
            self.means = means
            self.dim = means.shape[1]
            self.n_attractors = means.shape[0]
        else:
            self.dim = n_dims
            self.n_attractors = n_attractors
            self.means = rng.randn(n_attractors, n_dims) * 2.0
        
        if covs is not None:
            self.inv_covs = [np.linalg.inv(c) for c in covs]
        else:
            # Random positive definite matrices
            self.inv_covs = []
            for i in range(self.n_attractors):
                A = rng.randn(self.dim, self.dim) * 0.5
                cov = A.T @ A + np.eye(self.dim) * 0.5
                self.inv_covs.append(np.linalg.inv(cov))
    
    def V(self, X):
        """Soft-min (log-sum-exp) potential."""
        logits = np.zeros((*X.shape[:-1], self.n_attractors))
        for i in range(self.n_attractors):
            delta = X - self.means[i]
            mahal = np.sum(delta @ self.inv_covs[i] * delta, axis=-1)
            logits[..., i] = -0.5 * mahal
        max_logit = np.max(logits, axis=-1, keepdims=True)
        return -np.log(np.sum(np.exp(logits - max_logit), axis=-1)) - max_logit[..., 0]
    
    def grad_V(self, X):
        """Gradient of soft-min potential."""
        logits = np.zeros((*X.shape[:-1], self.n_attractors))
        grads = np.zeros((*X.shape[:-1], self.dim, self.n_attractors))
        
        for i in range(self.n_attractors):
            delta = X - self.means[i]
            mahal = np.sum(delta @ self.inv_covs[i] * delta, axis=-1)
            logits[..., i] = -0.5 * mahal
            grads[..., :, i] = delta @ self.inv_covs[i]
        
        max_logit = np.max(logits, axis=-1, keepdims=True)
        weights = np.exp(logits - max_logit)
        weights /= np.sum(weights, axis=-1, keepdims=True)
        
        return np.sum(weights[..., np.newaxis, :] * grads, axis=-1)


# Registry of landscapes for testing
LANDSCAPES = {
    'bifurcation': SimpleBifurcation,
    'threeway': ThreeWayBifurcation,
    'hierarchical': HierarchicalBifurcation,
}
