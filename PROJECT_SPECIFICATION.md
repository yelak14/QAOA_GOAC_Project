# QAOA-GOAC Project Specification

## Project Overview

This project applies the **Quantum Approximate Optimization Algorithm (QAOA)** to solve atomistic configuration optimization problems for **Li₂Co₈O₁₆** (75% delithiated LiCoO₂ cathode material) using binary optimization coefficients from the **GOAC** package.

### System Details
| Property | Value |
|----------|-------|
| Material | Li₂Co₈O₁₆ (LCO cathode, 75% delithiated) |
| Qubits Required | **8** (one per Li site) |
| Occupied Sites | 2 Li atoms |
| Vacant Sites | 6 empty Li sites |
| Total Configurations | C(8,2) = 28 |
| E_const | -933.755 eV |

### Objective
Find the ground state configuration (which 2 of 8 Li sites are occupied) that minimizes the Coulomb energy.

---

## Project Structure

```
QAOA_GOAC_Project/
├── PROJECT_SPECIFICATION.md
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── data/
│   ├── input/                    # GOAC coefficients and CIF
│   │   ├── alpha
│   │   ├── beta
│   │   ├── const
│   │   └── POSCAR-sc.cif
│   └── output/                   # Results will be saved here
│
├── qaoa/
│   ├── __init__.py
│   ├── hamiltonian.py            # Cost Hamiltonian construction
│   ├── mixers.py                 # Standard and constraint-preserving mixers
│   ├── circuit.py                # QAOA circuit construction
│   ├── optimizer.py              # CVaR cost function and optimization
│   └── utils.py                  # Helper functions
│
├── runners/
│   ├── __init__.py
│   ├── simulator/
│   │   ├── __init__.py
│   │   ├── standard_mixer.py     # Simulator + Standard QAOA
│   │   └── constrained_mixer.py  # Simulator + Constraint-preserving QAOA
│   └── hardware/
│       ├── __init__.py
│       ├── standard_mixer.py     # Real hardware + Standard QAOA
│       └── constrained_mixer.py  # Real hardware + Constraint-preserving QAOA
│
├── postprocessing/
│   ├── __init__.py
│   ├── plots.py                  # All visualization functions
│   ├── analysis.py               # Results analysis
│   └── visualization_3d.py       # Crystal structure visualization
│
├── notebooks/
│   └── demo.ipynb
│
└── results/
    ├── simulator/
    │   ├── standard/
    │   └── constrained/
    └── hardware/
        ├── standard/
        └── constrained/
```

---

## Execution Modes

### 1. IBM Simulator
- **Backend**: `AerSimulator` or `ibmq_qasm_simulator`
- **Advantages**: Fast, no queue, unlimited shots, no noise
- **Use for**: Development, testing, benchmarking

### 2. Real Quantum Hardware
- **Backend**: IBM Eagle/Heron processors (e.g., `ibm_brisbane`)
- **Advantages**: True quantum computation
- **Requires**: Error mitigation (dynamical decoupling, twirling, CVaR)

---

## Mixer Types

### 1. Standard QAOA Mixer
- **Mixer Hamiltonian**: H_M = Σ X_i (transverse field)
- **Pros**: Simple, well-studied
- **Cons**: Does NOT preserve particle number constraint
- **Result**: May sample invalid configurations (not exactly 2 Li atoms)
- **Mitigation**: Post-select valid configurations, use penalty terms

### 2. Constraint-Preserving Mixer (XY-Mixer)
- **Mixer Hamiltonian**: H_M = Σ (X_i X_j + Y_i Y_j) for connected pairs
- **Pros**: PRESERVES particle number (always exactly 2 Li atoms)
- **Cons**: More complex circuit, requires careful initialization
- **Initial State**: Must start in valid subspace (e.g., |11000000⟩)
- **Reference**: Hadfield et al., "From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz"

---

## Post-Processing Analysis

### Basic Analysis
1. Optimization convergence plot (cost vs. iteration)
2. Energy probability distribution (p(X) vs E(X))
3. Ground state identification and probability

### Quantum-Specific Analysis
4. Approximation ratio vs. QAOA depth (p)
5. Parameter landscape heatmap (cost vs. γ, β)
6. Circuit depth and gate count analysis

### Configuration Analysis
7. Bitstring → atomic configuration visualization
8. Degeneracy analysis (states with similar energies)
9. Comparison with classical GOAC results (MC/GA)

### Error & Noise Analysis (Hardware)
10. Shot noise convergence
11. CVaR α sensitivity analysis
12. Error mitigation effectiveness comparison

### Physical Interpretation
13. Site occupation statistics
14. Li-Li distance distribution in top configurations
15. Energy decomposition (first-order vs. second-order)

---

## Workflow

### Phase 1: Setup
1. Load GOAC coefficients (α, β, E_const)
2. Build cost Hamiltonian
3. Choose execution mode (simulator/hardware) and mixer type

### Phase 2: QAOA Execution
4. Initialize circuit with appropriate initial state
5. Apply QAOA layers with chosen mixer
6. Run optimization (COBYLA with CVaR cost function)

### Phase 3: Post-Processing
7. Generate all analysis plots
8. Identify ground state configuration
9. Map bitstring to physical Li arrangement
10. Compare with classical enumeration (validation)

---

## Expected Results

For Li₂Co₈O₁₆ with 8 sites and 2 occupied:
- Ground state should maximize Li-Li separation
- Expect Li atoms in **alternating layers** (z ≈ 0.26 and z ≈ 0.76)
- Can verify against brute-force enumeration of all 28 configurations

---

## References

1. GOAC: Köster et al., npj Computational Materials (2025)
2. IBM QAOA Tutorial: Advanced techniques for QAOA
3. CVaR: Barkoutsos et al., Quantum 4, 256 (2020)
4. XY-Mixer: Hadfield et al., Algorithms 12, 34 (2019)
5. SWAP Strategy: Weidenfeller et al., Quantum 6, 870 (2022)

---

*Last updated: January 2025*
