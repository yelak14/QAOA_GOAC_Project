# Li2Co8O16 Example

Example QAOA configuration for the Li2Co8O16 (75% delithiated LiCoO2) cathode material system.

## System Details

- **Material**: Li2Co8O16
- **Problem**: Optimal placement of 2 Li atoms in 8 possible sites
- **Qubits**: 8
- **Particles**: 2 (Li atoms)
- **Valid Configurations**: C(8,2) = 28
- **Ground State**: `00100100` (sites 2 and 5)

## Data Files

- `data/alpha` - On-site energies (8 values)
- `data/beta` - Pairwise interaction energies (28 values, upper triangle)
- `data/const` - Constant energy offset
- `data/POSCAR-sc.cif` - Crystal structure file

## Running

```bash
# From the project root directory:

# Simulator - Dicke + XY (recommended)
python -m examples.Li2Co8O16.runners.simulator.dicke_xy

# Simulator - Multi-start
python -m examples.Li2Co8O16.runners.simulator.dicke_xy_multistart

# Simulator - Standard + Penalty
python -m examples.Li2Co8O16.runners.simulator.standard_penalty

# Hardware (requires IBM Quantum account)
python -m examples.Li2Co8O16.runners.hardware.dicke_xy
python -m examples.Li2Co8O16.runners.hardware.dicke_xy_multistart
python -m examples.Li2Co8O16.runners.hardware.standard_penalty
```

## Notes

- These example runners include brute force comparison, which is feasible for this small 8-qubit system (2^8 = 256 configurations)
- For larger systems, use the generalized runners in `runners/` which omit brute force
- Default settings: p=20, shots=8192, matching the published results
