# QAOA-GOAC Project

Quantum Approximate Optimization Algorithm for Li₂Co₈O₁₆ cathode material configuration optimization.

## System
- **Material**: Li₂Co₈O₁₆ (75% delithiated LiCoO₂)
- **Problem**: Find optimal placement of 2 Li atoms in 8 possible sites
- **Qubits**: 8
- **Valid Configurations**: 28

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

# Run simulator (recommended: Dicke + XY)
python -m runners.simulator.dicke_xy

# Run hardware (requires IBM Quantum account)
python -m runners.hardware.dicke_xy
```

## Methods

| Method | Initial State | Mixer | Valid % | Script |
|--------|--------------|-------|---------|--------|
| Standard + Penalty | Hadamard | X | ~40-60% | `runners.simulator.standard_penalty` |
| **Dicke + XY** | Dicke | XY | **100%** | `runners.simulator.dicke_xy` |
| Dicke + XY Multi-start | Dicke | XY | **100%** | `runners.simulator.dicke_xy_multistart` |

### Method Comparison

- **Standard + Penalty**: Uses penalty term λ(N-2)² to discourage invalid configurations. Simple but may sample invalid states.
- **Dicke + XY (Recommended)**: Initializes in Dicke state |D₂⁸⟩ and uses XY mixer that preserves particle number. Always produces valid configurations with exactly 2 particles.
- **Multi-start**: Runs multiple short optimizations then fine-tunes the best. More robust convergence.

## Running on IBM Quantum Hardware

1. Create an IBM Quantum account at https://quantum.ibm.com
2. Save your credentials:
```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")
```
3. Run hardware scripts:
```bash
python -m runners.hardware.dicke_xy
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

## Project Structure

```
QAOA_GOAC_Project/
├── data/input/                    # GOAC coefficients
│   ├── alpha                      # On-site energies
│   ├── beta                       # Interaction energies
│   └── const                      # Constant energy term
├── qaoa/                          # Core QAOA modules
│   ├── __init__.py
│   ├── hamiltonian.py             # Hamiltonian construction
│   ├── circuits.py                # Shared circuit functions
│   ├── utils.py                   # Utility functions
│   └── ...
├── runners/
│   ├── simulator/                 # Aer simulator runners
│   │   ├── standard_penalty.py    # Standard QAOA + Penalty
│   │   ├── dicke_xy.py            # Dicke + XY QAOA
│   │   └── dicke_xy_multistart.py # Multi-start optimization
│   └── hardware/                  # IBM Quantum runners
│       ├── standard_penalty.py
│       ├── dicke_xy.py
│       └── dicke_xy_multistart.py
├── postprocessing/                # Analysis tools
│   ├── compare_methods.py         # Compare Standard vs Dicke
│   ├── compare_hardware_sim.py    # Compare hardware vs simulator
│   ├── publication_figures.py     # Paper-ready figures
│   └── analyze_results.py         # Statistical analysis
├── results/                       # Auto-generated outputs
│   ├── simulator/
│   │   ├── standard_penalty/
│   │   ├── dicke_xy/
│   │   └── dicke_xy_multistart/
│   └── hardware/
├── PROJECT_SPECIFICATION.md
├── README.md
└── requirements.txt
```

## Output Files

Each runner generates:
- `summary.json` - Run configuration and summary statistics
- `run_X_convergence.csv` - Energy vs iteration data
- `run_X_energy_distribution.csv` - Final energy distribution
- `run_X_final_distribution.csv` - Final state probabilities
- `brute_force_*.csv` - Reference classical solutions
- `*.png` - Visualization plots

## For Different Systems

1. Replace files in `data/input/` (alpha, beta, const)
2. Update `N_PARTICLES` in runner script to match your system
3. Run!

## References

- [GOAC Paper](https://www.nature.com/articles/s41524-025-01690-7)
- [IBM QAOA Tutorial](https://quantum.cloud.ibm.com/docs/en/tutorials/advanced-techniques-for-qaoa)
- [XY-Mixer / Particle-Number Conserving QAOA](https://doi.org/10.3390/a12020034)
- [Dicke State Preparation](https://arxiv.org/abs/1904.07358)
