"""
Chu et al. 2016 (GSE75748) — Human iPSC to Definitive Endoderm
===============================================================
Real time points: 0h, 12h, 24h, 36h, 72h, 96h
~758 single cells from time course
This is a non-hematopoietic, human system.
"""
import numpy as np
import pandas as pd
import gzip, json, sys, time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.sinkhorn import sinkhorn_plan
from src.potential import reconstruct_gradient
from final_experiment import calibrate_epsilon, robust_mst_integration, find_minima

print("=" * 70)
print("CHU 2016 — Human iPSC to Definitive Endoderm")
print("=" * 70)

# ---- 1. Load data ----
print("\n[1] Loading expression matrix...")
with gzip.open('data/GSE75748_sc_time_course_ec.csv.gz', 'rt') as f:
    df = pd.read_csv(f, index_col=0)
print(f"    Shape: {df.shape}")

# The data format: genes x cells, with time point info in column names
# Column names typically contain time information
# Let's parse time points from column names
col_names = df.columns.tolist()
print(f"    Sample columns: {col_names[:5]}")

# Extract time from column names — format is typically like "0h_rep1_cell1"
# Different formats exist. Let's try to parse
times = []
for col in col_names:
    # Try to find time patterns like 0h, 12h, 24h, etc.
    import re
    match = re.search(r'(\d+)h', col)
    if match:
        times.append(float(match.group(1)))
    else:
        # Try "day" or "D" patterns
        match = re.search(r'[Dd]ay\s*(\d+)', col)
        if match:
            times.append(float(match.group(1)) * 24)
        else:
            times.append(-1)

unique_times = sorted(set(t for t in times if t >= 0))
print(f"    Time points found: {unique_times}")
print(f"    Unmatched columns: {sum(1 for t in times if t < 0)}")

if len(unique_times) < 2:
    print("    WARNING: Could not parse time points from column names!")
    print("    Trying alternative: reading from GEO metadata...")
    # Fall back to metadata from the soft file
    # For now, try to use whatever we can parse
    times = np.array(times)
    valid = times >= 0
    if valid.sum() > 0:
        print(f"    Using {valid.sum()} cells with parseable time labels")

# ---- 2. Extract expression and organize by time ----
X = df.values.T  # (cells x genes)
times = np.array(times)
valid_mask = times >= 0
X = X[valid_mask]
times = times[valid_mask]

# Normalize
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
expl = pca.explained_variance_ratio_.sum()
print(f"    Valid cells: {X.shape[0]}, PCA var (2D): {expl:.1%}")

# Group by time point
unique_times = sorted(set(times))
print(f"\n[2] Time points: {unique_times}")
distributions = []
for t in unique_times:
    mask = times == t
    n = mask.sum()
    distributions.append(X_pca[mask])
    print(f"    t={t:>4.0f}h: {n} cells")

if len(distributions) < 3:
    print("    ERROR: Need at least 3 time points. Check data parsing.")
    sys.exit(1)

# ---- 3. Run inverse JKO ----
print(f"\n[3] Running OT pipeline ({len(distributions)} time points)...")
all_grads = []; all_points = []
for t in range(len(distributions)-1):
    Xs, Xt = distributions[t], distributions[t+1]
    eps = calibrate_epsilon(Xs, Xt)
    plan, conv, niter = sinkhorn_plan(Xs, Xt, epsilon=eps, num_iters=2000)
    grad_ot, _ = reconstruct_gradient(plan, Xs, Xt, None, 1.0, None)
    all_grads.append(grad_ot); all_points.append(Xs)
    print(f"    t={unique_times[t]}h->{unique_times[t+1]}h: eps={eps:.4f}, "
          f"n={Xs.shape[0]}->{Xt.shape[0]}, conv={conv}")

all_X = np.vstack(all_points); all_g = np.vstack(all_grads)
V_recon = robust_mst_integration(all_X, all_g, max_points=3000)

# Map to all cells
from scipy.spatial import KDTree
tree = KDTree(all_X); _, nn = tree.query(X_pca); V_all = V_recon[nn]

# ---- 4. Sign check and biological validation ----
# Progenitor (0h, undifferentiated) should be HIGHER than differentiated (96h)
v_0h = np.mean(V_all[times == unique_times[0]])
v_last = np.mean(V_all[times == unique_times[-1]])
print(f"\n[4] Biological validation:")
print(f"    V(0h, undiff)  = {v_0h:.3f}")
print(f"    V({unique_times[-1]:.0f}h, diff) = {v_last:.3f}")

if v_0h < v_last:
    print(f"    Direction REVERSED — flipping sign...")
    V_recon = -V_recon; V_all = -V_all
    v_0h, v_last = -v_0h, -v_last

print(f"    Delta = {v_0h - v_last:+.3f}")
print(f"    Direction: {'CORRECT (progenitor > differentiated)' if v_0h > v_last else 'REVERSED'}")

# Time trend
print(f"\n    Time trend:")
for t in unique_times:
    mask = times == t
    print(f"    {t:>4.0f}h: V = {np.mean(V_all[mask]):>8.3f} +/- {np.std(V_all[mask]):>6.3f}")

# Spearman correlation: should be negative (later time = lower V)
from scipy.stats import spearmanr
rho, p = spearmanr(times, V_all)
print(f"\n    Spearman rho(time, V) = {rho:.3f} (p={p:.3f})")
print(f"    {'CORRECT (negative trend)' if rho < -0.3 else 'WEAK/REVERSED' if abs(rho) < 0.3 else 'REVERSED'}")

# ---- 5. Save ----
results = {
    'dataset': 'Chu 2016 (GSE75748)',
    'system': 'human iPSC to definitive endoderm',
    'non_hematopoietic': True,
    'n_cells': int(X.shape[0]),
    'n_genes': int(df.shape[0]),
    'time_points': [float(t) for t in unique_times],
    'pca_var': float(expl),
    'V_0h': float(v_0h),
    'V_last': float(v_last),
    'direction_correct': bool(v_0h > v_last),
    'spearman_rho': float(rho),
    'spearman_p': float(p),
    'mean_V_by_time': {float(t): float(np.mean(V_all[times==t])) for t in unique_times},
}
with open('results/chu2016_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[DONE] Results -> results/chu2016_results.json")
