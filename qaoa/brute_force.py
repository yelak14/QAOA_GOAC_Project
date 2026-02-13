"""Brute force energy computation utility.

WARNING: This scales as O(2^n) and is only feasible for small systems (~20 qubits or fewer).
For larger systems, use QAOA directly without brute force comparison.
"""

import numpy as np


def compute_brute_force_energies(alpha, beta_coeff, E_const, n_qubits):
    """Compute energies for ALL 2^n configurations.

    Args:
        alpha: On-site energies (array of length n_qubits)
        beta_coeff: Interaction coefficients (flat array or n x n matrix)
        E_const: Constant energy offset
        n_qubits: Number of qubits/sites

    Returns:
        dict of {bitstring: (energy, n_particles)}
    """
    alpha = np.array(alpha).flatten()

    beta_coeff_array = np.array(beta_coeff)
    beta_matrix = np.zeros((n_qubits, n_qubits))

    if beta_coeff_array.ndim == 2:
        beta_matrix = beta_coeff_array.copy()
    elif beta_coeff_array.ndim == 1:
        idx = 0
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                beta_matrix[i, j] = beta_coeff_array[idx]
                beta_matrix[j, i] = beta_coeff_array[idx]
                idx += 1

    all_configs = {}

    for idx in range(2**n_qubits):
        bitstring = format(idx, f'0{n_qubits}b')
        n_particles = bitstring.count('1')

        energy = E_const
        for i, bit in enumerate(bitstring):
            if bit == '1':
                energy += alpha[i]

        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if bitstring[i] == '1' and bitstring[j] == '1':
                    energy += beta_matrix[i, j]

        all_configs[bitstring] = (energy, n_particles)

    return all_configs
