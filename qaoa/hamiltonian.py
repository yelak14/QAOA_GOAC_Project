import numpy as np
from qiskit.quantum_info import SparsePauliOp
from .utils import enumerate_valid_configs


def build_cost_hamiltonian(alpha, beta, E_const):
    """Build the cost Hamiltonian as a SparsePauliOp.

    The QUBO energy is: E(x) = E_const + sum_i alpha_i * x_i + sum_{i<j} beta_ij * x_i * x_j

    Using the mapping x_i = (1 - Z_i) / 2, we convert to an Ising Hamiltonian:
    H = offset + sum_i h_i * Z_i + sum_{i<j} J_ij * Z_i * Z_j

    Args:
        alpha: array of shape (n,) - linear coefficients
        beta: array of shape (n, n) - quadratic coefficients (symmetric)
        E_const: float - constant energy offset

    Returns:
        SparsePauliOp representing the cost Hamiltonian
    """
    n = len(alpha)

    # Convert QUBO to Ising
    # x_i = (1 - Z_i) / 2
    # x_i * x_j = (1 - Z_i)(1 - Z_j) / 4 = (1 - Z_i - Z_j + Z_i*Z_j) / 4

    # Constant term
    offset = E_const
    for i in range(n):
        offset += alpha[i] / 2.0
    for i in range(n):
        for j in range(i + 1, n):
            offset += beta[i, j] / 4.0

    # Linear terms (Z_i coefficients)
    h = np.zeros(n)
    for i in range(n):
        h[i] -= alpha[i] / 2.0
        for j in range(n):
            if i != j:
                h[i] -= beta[min(i, j), max(i, j)] / 4.0 if i < j else beta[j, i] / 4.0 if j < i else 0.0

    # Recalculate h properly
    h = np.zeros(n)
    for i in range(n):
        h[i] = -alpha[i] / 2.0
    for i in range(n):
        for j in range(i + 1, n):
            h[i] -= beta[i, j] / 4.0
            h[j] -= beta[i, j] / 4.0

    # Quadratic terms (Z_i Z_j coefficients)
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = beta[i, j] / 4.0

    # Build SparsePauliOp
    pauli_list = []

    # Identity (constant offset)
    pauli_list.append(("I" * n, offset))

    # Single Z terms
    for i in range(n):
        if abs(h[i]) > 1e-10:
            label = ['I'] * n
            # Qiskit uses little-endian: qubit 0 is rightmost
            label[n - 1 - i] = 'Z'
            pauli_list.append((''.join(label), h[i]))

    # ZZ terms
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-10:
                label = ['I'] * n
                label[n - 1 - i] = 'Z'
                label[n - 1 - j] = 'Z'
                pauli_list.append((''.join(label), J[i, j]))

    return SparsePauliOp.from_list(pauli_list)


def evaluate_energy(bitstring, alpha, beta, E_const):
    """Evaluate the QUBO energy for a given configuration.

    Args:
        bitstring: str of '0's and '1's
        alpha: linear coefficients
        beta: quadratic coefficients
        E_const: constant offset

    Returns:
        Energy value (float)
    """
    x = np.array([int(b) for b in bitstring])
    energy = E_const
    energy += np.dot(alpha, x)
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            energy += beta[i, j] * x[i] * x[j]
    return energy


def get_exact_solution(alpha, beta, E_const, n_particles=2, n_qubits=8):
    """Find the ground state by brute-force enumeration.

    Args:
        alpha: linear coefficients
        beta: quadratic coefficients
        E_const: constant offset
        n_particles: number of occupied sites
        n_qubits: number of qubits/sites

    Returns:
        Tuple of (ground_state_bitstring, ground_state_energy, all_energies_dict)
    """
    configs = enumerate_valid_configs(n_qubits, n_particles)
    energies = {}

    for config in configs:
        energies[config] = evaluate_energy(config, alpha, beta, E_const)

    ground_state = min(energies, key=energies.get)
    return ground_state, energies[ground_state], energies
