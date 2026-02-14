# Input Data Directory

Place your GOAC coefficient files in a subdirectory here (e.g., `data/input/`).

## Required Files

| File | Description | Format |
|------|-------------|--------|
| `alpha` | On-site energies | One value per line, n values total |
| `beta` | Pairwise interaction energies | Either n*(n-1)/2 values (upper triangle) or full n x n matrix |
| `const` | Constant energy offset | Single value |

## Optional Files

| File | Description |
|------|-------------|
| `*.cif` | Crystal structure file (for reference) |

## Example

For a system with n=8 sites and 2 particles:
- `alpha`: 8 values (on-site energies for each site)
- `beta`: 28 values (C(8,2) pairwise interactions) or 8x8 matrix
- `const`: 1 value (constant energy term)

See `examples/Li2Co8O16/data/` for a complete example.

## Usage

```bash
# Run with custom data directory
python -m runners.simulator.dicke_xy --data-dir data/input --n-particles 2

# Run with example data
python -m examples.Li2Co8O16.runners.simulator.dicke_xy
```
