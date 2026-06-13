# Self-Diagnostic Optimal Transport for Waddington Landscape Reconstruction

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A non-parametric framework for reconstructing Waddington epigenetic landscapes
from time-series single-cell data. The core formula `∇V(x) ≈ (x - T(x))/τ`
follows from the conditional expectation of Langevin dynamics, where `T(x)` is
the optimal transport barycentric projection between consecutive time points.
A self-diagnostic module evaluates reconstruction quality through systematic
checks, classifying results as Pass, Warning, or Fail.

## Repository Structure

```
├── src/                            Core library
│   ├── sinkhorn.py                 Sinkhorn OT algorithm (Numba)
│   ├── potential.py                Gradient reconstruction + MST integration
│   └── landscapes.py               Synthetic landscape generators
├── scripts/                        Reproducible pipeline scripts
│   ├── run_synthetic.py            Synthetic landscape validation
│   ├── run_predictive.py           Non-circular predictive validation
│   ├── run_benchmark.py            8-method systematic benchmark
│   ├── run_moignard.py             Real data: Moignard 2015
│   ├── run_krumsiek11.py           Real data: Krumsiek11
│   ├── run_pancreas.py             Real data: Pancreas development
│   ├── run_pancreas_deep.py        Pancreas deep analysis (PCA stability, markers)
│   ├── run_chu2016.py              Real data: Chu 2016 iPSC
│   ├── run_gastrulation.py         Real data: Gastrulation atlas
│   ├── run_paul2015.py             Real data: Paul 2015 pseudotime trap
│   ├── run_reviewer_fixes.py       Ablation and robustness experiments
│   ├── run_cross_system.py         Cross-system + neural baseline
│   ├── run_highdim.py              High-dimensional validation
│   ├── run_wot_comparison.py       Waddington-OT comparison
│   ├── generate_figures.py         Publication figure generation
│   └── generate_marker_figure.py   Pancreas marker figure
├── tools/                          Utilities
│   ├── download_chu.py             Download Chu 2016 data
│   └── download_gastrulation.py    Download gastrulation data
├── data/                           Input data (gitignored, see data/README.md)
├── requirements.txt                Python dependencies
└── .gitignore
```

## Quick Start

### Installation
```bash
git clone <repo-url>
cd math
pip install -r requirements.txt
```

### Reproducing Results

**Synthetic validation:**
```bash
python scripts/run_synthetic.py --landscape bifurcation --ncells 1500 --beta 100
python scripts/run_synthetic.py --landscape threeway --ncells 1500 --beta 100
python scripts/run_synthetic.py --landscape hierarchical --ncells 1500 --beta 100
```

**Systematic benchmark (8 methods):**
```bash
python scripts/run_benchmark.py
```

**Real data analysis:**
```bash
python scripts/run_moignard.py
python scripts/run_krumsiek11.py
python scripts/run_pancreas.py
python scripts/run_chu2016.py         # Requires data download (see data/README.md)
python scripts/run_gastrulation.py    # Requires data download (see data/README.md)
python scripts/run_paul2015.py
```

**Robustness experiments:**
```bash
python scripts/run_reviewer_fixes.py
python scripts/run_cross_system.py
python scripts/run_highdim.py
python scripts/run_wot_comparison.py
```

**Figures:**
```bash
python scripts/generate_figures.py
python scripts/generate_marker_figure.py
```

**Verification:**
```bash
python scripts/check_results.py
```

## Key Results

| Benchmark | OT (ours) | Best Competitor |
|-----------|-----------|-----------------|
| Bifurcation gradient cos_sim | **0.972** | 0.927 (WOT) |
| Three-way gradient cos_sim | **0.863** | 0.879 (WOT) |

OT is the **only method** that correctly recovers gradient direction on the
three-way branching landscape.

**Real data**: Validated on 6 datasets spanning 4 biological systems.
Correct developmental ordering in 5/6 cases; diagnostic system correctly
flags the marginal case (gastrulation atlas).

**Biological insight**: In pancreas development, marker regulatory dynamics
(Pax4 19-fold downregulation, Arx activation, Neurog3 silencing) reveal a
distinct Alpha lineage specification program independent of the landscape model.

## Citation

If you use this code, please cite the accompanying paper.

## License

MIT License.
