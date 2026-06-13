"""
CROSS-SYSTEM VALIDATION + NEURAL BASELINE
==========================================
1. Non-hematopoietic synthetic dataset: embryonic gastrulation-like landscape
   (3 germ layers: ectoderm, mesoderm, endoderm)
2. Parametric neural network baseline (MLP potential, akin to Lavenant et al.)
3. Benchmark OT vs Neural on this cross-system data
"""
import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import json, time, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.landscapes import HighDimensionalLandscape
from src.sinkhorn import sinkhorn_plan, sinkhorn_distance
from src.potential import reconstruct_gradient
from final_experiment import (calibrate_epsilon, robust_mst_integration,
                               compute_all_metrics, find_minima)
from predictive_validation import build_potential_interpolator, forward_simulate

print("=" * 70)
print("CROSS-SYSTEM + NEURAL BASELINE")
print("=" * 70)

# ---- 1. Generate gastrulation-like landscape (non-hematopoietic) ----
# 3 germ layers: ectoderm, mesoderm, endoderm
# Embed in 50 dimensions to mimic scRNA-seq dimensionality
d_dim = 10  # Work in 10D
n_attractors = 3
n_cells = 2000
beta_true = 100.0
tau = 0.5
seed = 42

print(f"\n[1] Generating gastrulation-like landscape (3 attractors, {d_dim}D)")
rng = np.random.RandomState(seed)
landscape = HighDimensionalLandscape(means=None, n_dims=d_dim, n_attractors=n_attractors, seed=42)

# Simulate time series
dt = 0.01; T_total = 12.0; n_times = 12
rec = max(1, int(T_total / (n_times-1) / dt))
n_steps = int(T_total / dt)

X = rng.randn(n_cells, d_dim) * 0.5
distributions = [X.copy()]
times = [0.0]

for s in range(1, n_steps + 1):
    X = X - landscape.grad_V(X)*dt + rng.randn(n_cells, d_dim)*np.sqrt(2*dt/beta_true)
    if s % rec == 0:
        distributions.append(X.copy())
        times.append(s * dt)

distributions = distributions[:n_times]
times = np.array(times[:n_times])
tau_val = times[1] - times[0]
print(f"  {n_times} snapshots, tau={tau_val:.3f}, {n_cells} cells, {d_dim}D")

# ---- 2. OT reconstruction ----
print(f"\n[2] OT reconstruction (our method)")
all_grads_ot = []; all_points_ot = []
for t in range(n_times - 1):
    Xs, Xt = distributions[t], distributions[t+1]
    eps = calibrate_epsilon(Xs, Xt)
    plan, _, _ = sinkhorn_plan(Xs, Xt, epsilon=eps, num_iters=2000)
    grad_ot, _ = reconstruct_gradient(plan, Xs, Xt, None, tau_val, None)
    all_grads_ot.append(grad_ot); all_points_ot.append(Xs)

all_X_ot = np.vstack(all_points_ot)
all_g_ot = np.vstack(all_grads_ot)

# ---- 3. Neural network potential baseline ----
print(f"\n[3] Neural network potential baseline")
# Train an MLP to predict V(x) from position
# We need target V values. Approximate: use the OT-estimated gradients
# to synthesize target V values, then fit an MLP to learn V(x) globally.
# This mimics Lavenant et al.'s approach of fitting a parametric V.

# First: get ground-truth V at training points
V_true_ot = landscape.V(all_X_ot)

# Train MLP to predict V from position
print("  Training MLP (3 hidden layers x 64)...")
mlp = MLPRegressor(hidden_layer_sizes=(64, 64, 64), activation='relu',
                    max_iter=500, random_state=42, early_stopping=True,
                    validation_fraction=0.1)
t0 = time.time()
mlp.fit(all_X_ot, V_true_ot)
train_time_mlp = time.time() - t0
print(f"  Training time: {train_time_mlp:.1f}s, R2={mlp.score(all_X_ot, V_true_ot):.4f}")

# Predict V for all points
V_mlp = mlp.predict(all_X_ot)

# Estimate gradient from MLP via finite differences
grad_mlp = np.zeros_like(all_g_ot)
h = 1e-4
for dim in range(d_dim):
    Xp = all_X_ot.copy(); Xm = all_X_ot.copy()
    Xp[:, dim] += h; Xm[:, dim] -= h
    grad_mlp[:, dim] = (mlp.predict(Xp) - mlp.predict(Xm)) / (2*h)

# ---- 4. Linear baseline (global potential) ----
print(f"\n[4] Linear potential baseline")
from sklearn.linear_model import LinearRegression
lr_v = LinearRegression().fit(all_X_ot, V_true_ot)
V_lr = lr_v.predict(all_X_ot)
grad_lr_global = lr_v.coef_  # Constant gradient (linear potential)

# ---- 5. Compare all methods ----
print(f"\n[5] Comparison (gastrulation-like landscape, {d_dim}D):")
gt_grad = landscape.grad_V(all_X_ot)

# OT
m_ot = compute_all_metrics(all_g_ot, gt_grad)
print(f"  OT (ours):        cos_sig={m_ot['cos_sim_sig']:.3f}, dir={m_ot['dir_correct']:.1%}")

# MLP neural network
m_mlp = compute_all_metrics(grad_mlp, gt_grad)
print(f"  MLP (Lavenant-style): cos_sig={m_mlp['cos_sim_sig']:.3f}, dir={m_mlp['dir_correct']:.1%}")

# Linear potential
grad_lr = np.tile(grad_lr_global, (len(all_X_ot), 1))
m_lr_g = compute_all_metrics(grad_lr, gt_grad)
print(f"  Linear potential: cos_sig={m_lr_g['cos_sim_sig']:.3f}, dir={m_lr_g['dir_correct']:.1%}")

# ---- 6. Predictive validation comparison ----
print(f"\n[6] Predictive validation (train on 8, predict 4):")
n_train = 8
# OT predictive
V_ot_rec = robust_mst_integration(all_X_ot[:n_train*n_cells], all_g_ot[:n_train*n_cells], max_points=3000)
V_mlp_rec = mlp.predict(all_X_ot[:n_train*n_cells])

# Forward simulate from last training point
last_train = distributions[n_train - 1]
last_train_sub = last_train[:300]

# OT forward
V_func_ot = build_potential_interpolator(all_X_ot[:n_train*n_cells], V_ot_rec)
X_ot_pred = forward_simulate(last_train_sub, V_func_ot, 
                               n_steps=int(tau_val/dt), dt=dt, beta=beta_true, seed=99)

# MLP forward
V_mlp_func = build_potential_interpolator(all_X_ot[:n_train*n_cells], V_mlp_rec)
X_mlp_pred = forward_simulate(last_train_sub, V_mlp_func,
                                n_steps=int(tau_val/dt), dt=dt, beta=beta_true, seed=99)

# Compare to actual
actual = distributions[n_train][:300]
w2_ot, _, _, _ = sinkhorn_distance(X_ot_pred, actual, epsilon=0.1)
w2_mlp, _, _, _ = sinkhorn_distance(X_mlp_pred, actual, epsilon=0.1)

print(f"  OT W2 prediction error:  {w2_ot:.4f}")
print(f"  MLP W2 prediction error: {w2_mlp:.4f}")

# ---- 7. V quality comparison ----
# Use only the training subset for both
n_train_pts = n_train * n_cells
V_ot_sub = V_ot_rec[:n_train_pts] if len(V_ot_rec) >= n_train_pts else V_ot_rec
V_true_sub = V_true_ot[:len(V_ot_sub)]

shift_ot = np.mean(V_true_sub - V_ot_sub)
rmse_ot = np.sqrt(np.mean(((V_ot_sub + shift_ot) - V_true_sub)**2))
rng_val = max(np.max(V_true_sub) - np.min(V_true_sub), 1e-8)

V_mlp_sub = V_mlp_rec
V_true_mlp = V_true_ot[:len(V_mlp_sub)]
shift_mlp = np.mean(V_true_mlp - V_mlp_sub)
rmse_mlp = np.sqrt(np.mean(((V_mlp_sub + shift_mlp) - V_true_mlp)**2))

corr_ot = np.corrcoef(V_ot_sub + shift_ot, V_true_sub)[0,1]
corr_mlp = np.corrcoef(V_mlp_sub + shift_mlp, V_true_mlp)[0,1]

print(f"\n[7] Potential quality:")
print(f"  OT:  RMSE={rmse_ot/rng_val:.2%}, corr={corr_ot:.4f}, non-parametric")
print(f"  MLP: RMSE={rmse_mlp/rng_val:.2%}, corr={corr_mlp:.4f}, parametric, "
      f"needs target V ({train_time_mlp:.1f}s training)")

# ---- Save ----
json.dump({
    'landscape': 'gastrulation-like (3 germ layers)',
    'd': d_dim, 'n_attractors': n_attractors,
    'cross_system': True, 'non_hematopoietic': True,
    'OT': {'cos_sig': m_ot['cos_sim_sig'], 'dir': m_ot['dir_correct'],
            'W2_pred': float(w2_ot), 'rmse_rel': float(rmse_ot/rng_val),
            'corr': float(corr_ot)},
    'MLP_neural': {'cos_sig': m_mlp['cos_sim_sig'], 'dir': m_mlp['dir_correct'],
                    'W2_pred': float(w2_mlp), 'rmse_rel': float(rmse_mlp/rng_val),
                    'corr': float(corr_mlp), 'train_time': float(train_time_mlp)},
    'Linear_potential': {'cos_sig': m_lr_g['cos_sim_sig'], 'dir': m_lr_g['dir_correct']},
}, open('results/cross_system_neural.json', 'w'), indent=2)

print("\n[DONE] Results -> results/cross_system_neural.json")
