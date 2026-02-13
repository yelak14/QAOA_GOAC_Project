# QAOA-GOAC Project

Quantum Approximate Optimization Algorithm for crystal structure configuration optimization using the Generalized Occupation-based Atomic Cluster (GOAC) framework.

## Overview

This project implements QAOA for finding optimal atomic configurations in crystal materials. The code is general-purpose and can be applied to any system described by GOAC coefficients (on-site energies, pairwise interactions, and a constant offset).

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

# Run with your own data
python -m runners.simulator.dicke_xy --data-dir data/input --n-particles 2 --p-value 20

# Run the Li2Co8O16 example
python -m examples.Li2Co8O16.runners.simulator.dicke_xy
```

## Methods

| Method | Initial State | Mixer | Valid % | Script |
|--------|--------------|-------|---------|--------|
| Standard + Penalty | Hadamard | X | ~40-60% | `runners.simulator.standard_penalty` |
| **Dicke + XY** | Dicke | XY | **100%** | `runners.simulator.dicke_xy` |
| Dicke + XY Multi-start | Dicke | XY | **100%** | `runners.simulator.dicke_xy_multistart` |

### Method Comparison

- **Standard + Penalty**: Uses penalty term lambda(N-k)^2 to discourage invalid configurations. Simple but may sample invalid states.
- **Dicke + XY (Recommended)**: Initializes in Dicke state and uses XY mixer that preserves particle number. Always produces valid configurations.
- **Multi-start**: Runs multiple short optimizations then fine-tunes the best. More robust convergence.

## Command-Line Arguments

All runners accept command-line arguments for configuration:

```bash
# Simulator runners
python -m runners.simulator.dicke_xy \
    --data-dir path/to/data \
    --n-particles 2 \
    --p-value 20 \
    --shots 8192 \
    --maxiter 10000 \
    --num-runs 4 \
    --output-dir results/my_run

# Hardware runners (additional options)
python -m runners.hardware.dicke_xy \
    --data-dir path/to/data \
    --n-particles 2 \
    --p-value 5 \
    --shots 4096 \
    --backend ibm_brisbane \
    --optimization-level 3 \
    --resilience-level 1
```

## Running on IBM Quantum Hardware

1. Create an IBM Quantum account at https://quantum.ibm.com
2. Save your credentials:
```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")
```
3. Run hardware scripts:
```bash
python -m runners.hardware.dicke_xy --data-dir data/input --n-particles 2
```

## Postprocessing

```bash
# Compare Standard vs Dicke methods
python -m postprocessing.compare_methods

# Compare hardware vs simulator results
python -m postprocessing.compare_hardware_sim

# Generate publication-ready figures
python -m postprocessing.publication_figures

# Statistical analysis across all results
python -m postprocessing.analyze_results
```

## Examples

### Li2Co8O16

The `examples/Li2Co8O16/` directory contains a complete example for the Li2Co8O16 cathode material system (8 qubits, 2 particles, 28 valid configurations). This example includes brute force comparison, which is feasible for this small system.

```bash
python -m examples.Li2Co8O16.runners.simulator.dicke_xy
```

See `examples/Li2Co8O16/README.md` for details.

## Using Your Own System

1. Prepare your GOAC coefficient files:
   - `alpha`: On-site energies (one per line)
   - `beta`: Pairwise interactions (upper triangle or full matrix)
   - `const`: Constant energy offset
2. Place them in a directory (e.g., `data/input/`)
3. Run with appropriate parameters:
```bash
python -m runners.simulator.dicke_xy \
    --data-dir data/input \
    --n-particles <your_particle_count> \
    --p-value 20
```

## Project Structure

```
QAOA_GOAC_Project/
├── data/                          # User's input data directory
│   └── README.md                  # Instructions for input format
├── qaoa/                          # Core QAOA module
│   ├── __init__.py
│   ├── hamiltonian.py
│   ├── circuits.py
│   ├── mixers.py
│   ├── utils.py
│   └── brute_force.py             # Optional brute force utility
├── runners/                       # Generalized runner scripts
│   ├── simulator/
│   │   ├── dicke_xy.py
│   │   ├── dicke_xy_multistart.py
│   │   └── standard_penalty.py
│   └── hardware/
│       ├── dicke_xy.py
│       ├── dicke_xy_multistart.py
│       └── standard_penalty.py
├── postprocessing/                # Analysis and visualization
│   ├── __init__.py
│   ├── plotting.py                # Reusable plots (bitstring vs probability)
│   ├── plots.py                   # Additional plot functions
│   ├── analysis.py
│   ├── analyze_results.py
│   ├── compare_methods.py
│   ├── compare_hardware_sim.py
│   ├── publication_figures.py
│   └── visualization_3d.py
├── examples/
│   └── Li2Co8O16/
│       ├── README.md
│       ├── data/                   # alpha, beta, const, CIF
│       ├── runners/                # Hardcoded for this system, includes brute force
│       └── results/                # Pre-computed results
├── results/                       # Output directory for user runs
├── requirements.txt
├── PROJECT_SPECIFICATION.md
└── README.md
```

## Output Files

Each runner generates:
- `summary.json` - Run configuration and summary statistics
- `run_X_convergence.csv` - Energy vs iteration data
- `run_X_energy_distribution.csv` - Final energy distribution
- `run_X_final_distribution.csv` - Final state probabilities
- `bitstring_probability.csv` - Bitstring vs probability data
- `bitstring_probability.png` - Bitstring probability bar chart
- `*.png` - Visualization plots

## References

- [GOAC Paper](https://www.nature.com/articles/s41524-025-01690-7)
- [IBM QAOA Tutorial](https://quantum.cloud.ibm.com/docs/en/tutorials/advanced-techniques-for-qaoa)
- [XY-Mixer / Particle-Number Conserving QAOA](https://doi.org/10.3390/a12020034)
- [Dicke State Preparation](https://arxiv.org/abs/1904.07358)
