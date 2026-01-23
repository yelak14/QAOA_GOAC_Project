# QAOA-GOAC Project

Quantum Approximate Optimization Algorithm for Li₂Co₈O₁₆ cathode material configuration optimization.

## System
- **Material**: Li₂Co₈O₁₆ (75% delithiated LiCoO₂)
- **Problem**: Find optimal placement of 2 Li atoms in 8 possible sites
- **Qubits**: 8
- **Configurations**: 28 possible

## Quick Start

```bash
# 1. Create environment
python -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run simulator with standard mixer
python -m runners.simulator.standard_mixer

# 4. Run simulator with constraint-preserving mixer
python -m runners.simulator.constrained_mixer
```

## Execution Modes

| Mode | Mixer | Script |
|------|-------|--------|
| Simulator | Standard | `runners/simulator/standard_mixer.py` |
| Simulator | Constrained | `runners/simulator/constrained_mixer.py` |
| Hardware | Standard | `runners/hardware/standard_mixer.py` |
| Hardware | Constrained | `runners/hardware/constrained_mixer.py` |

## Mixer Types

### Standard Mixer (X-Mixer)
- Simple transverse field: H_M = Σ X_i
- Does NOT preserve particle number
- May sample invalid configurations

### Constraint-Preserving Mixer (XY-Mixer)
- Preserves particle number: H_M = Σ (X_i X_j + Y_i Y_j)
- Always samples valid configurations (exactly 2 Li atoms)
- Requires initialization in valid subspace

## Project Structure

```
├── data/input/          # GOAC coefficients (alpha, beta, const, CIF)
├── qaoa/                # Core QAOA implementation
├── runners/             # Execution scripts (simulator/hardware × mixer type)
├── postprocessing/      # Analysis and visualization
└── results/             # Output files organized by mode
```

## References

- [GOAC Paper](https://www.nature.com/articles/s41524-025-01690-7)
- [IBM QAOA Tutorial](https://quantum.cloud.ibm.com/docs/en/tutorials/advanced-techniques-for-qaoa)
- [XY-Mixer Paper](https://doi.org/10.3390/a12020034)
