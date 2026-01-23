import numpy as np
from itertools import combinations
from pathlib import Path


def load_coefficients(data_dir="data/input"):
    """Load GOAC coefficients (alpha, beta, E_const) from files.

    The beta file can be either:
    - A flat list of n*(n-1)/2 values (upper triangle: pairs (0,1),(0,2),...,(n-2,n-1))
    - A full n×n matrix
    """
    data_path = Path(data_dir)

    alpha = np.loadtxt(data_path / "alpha")
    n = len(alpha)
    E_const = float(np.loadtxt(data_path / "const"))

    beta_raw = np.loadtxt(data_path / "beta")

    if beta_raw.ndim == 1:
        # Flat upper triangle: reconstruct symmetric matrix
        n_pairs = n * (n - 1) // 2
        assert len(beta_raw) == n_pairs, \
            f"Expected {n_pairs} beta values for {n} qubits, got {len(beta_raw)}"
        beta = np.zeros((n, n))
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                beta[i, j] = beta_raw[idx]
                beta[j, i] = beta_raw[idx]
                idx += 1
    else:
        beta = beta_raw

    return alpha, beta, E_const


def bitstring_to_config(bitstring):
    """Convert a bitstring to a list of occupied site indices.

    Args:
        bitstring: str like '11000000' or list/array of 0s and 1s

    Returns:
        List of indices where sites are occupied (1).
    """
    if isinstance(bitstring, str):
        return [i for i, b in enumerate(bitstring) if b == '1']
    return [i for i, b in enumerate(bitstring) if b == 1]


def is_valid_config(bitstring, n_particles=2):
    """Check if a configuration has exactly n_particles occupied sites."""
    if isinstance(bitstring, str):
        return bitstring.count('1') == n_particles
    return sum(bitstring) == n_particles


def enumerate_valid_configs(n_qubits=8, n_particles=2):
    """Generate all valid bitstrings with exactly n_particles ones.

    Returns:
        List of bitstrings (str) with exactly n_particles '1's.
    """
    configs = []
    for occupied in combinations(range(n_qubits), n_particles):
        bits = ['0'] * n_qubits
        for idx in occupied:
            bits[idx] = '1'
        configs.append(''.join(bits))
    return configs
