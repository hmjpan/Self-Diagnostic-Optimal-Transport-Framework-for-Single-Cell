"""
Waddington Landscape Reconstruction via Inverse Optimal Transport
=================================================================

Mathematical framework: ∇V(x) ≈ (x - T(x)) / τ
where T is the OT barycentric projection between consecutive time points.
"""
from .sinkhorn import sinkhorn_plan, sinkhorn_distance, barycentric_projection
from .potential import (reconstruct_from_time_series, reconstruct_gradient,
                         integrate_potential, solve_poisson_on_grid,
                         estimate_noise_level)
from .landscapes import (SimpleBifurcation, ThreeWayBifurcation,
                          HierarchicalBifurcation, HighDimensionalLandscape,
                          LANDSCAPES)

__version__ = "2.0.0"
