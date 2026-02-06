"""Shared quantum circuit functions for QAOA.

This module contains circuit-building functions used by both
simulator and hardware runners.
"""

import numpy as np
from math import comb
from itertools import combinations
from qiskit import QuantumCircuit


def create_hadamard_initial_state(n_qubits):
    """Create initial state |+>^n (all Hadamard).

    Used for Standard QAOA.

    Args:
        n_qubits: Number of qubits

    Returns:
        QuantumCircuit with Hadamard gates
    """
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    return qc


def create_dicke_initial_state(n_qubits, k):
    """Create Dicke state |D_k^n> using statevector initialization.

    Dicke state is equal superposition of all states with exactly k ones:
    |D_k^n> = (n choose k)^(-1/2) * sum_perm |1^k 0^(n-k)>

    Used for Particle-Number-Conserving QAOA.

    Args:
        n_qubits: Number of qubits (n)
        k: Number of particles (number of 1s)

    Returns:
        QuantumCircuit initialized to Dicke state
    """
    n_states = comb(n_qubits, k)
    amplitude = 1.0 / np.sqrt(n_states)

    sv_array = np.zeros(2**n_qubits, dtype=complex)

    for positions in combinations(range(n_qubits), k):
        idx = sum(2**p for p in positions)
        sv_array[idx] = amplitude

    qc = QuantumCircuit(n_qubits)
    qc.initialize(sv_array, range(n_qubits))

    return qc


def apply_cost_layer(qc, gamma, alpha, beta_matrix, n_qubits):
    """Apply cost unitary: exp(-i * gamma * H_C).

    H_C = sum_i alpha_i n_i + sum_{i<j} beta_ij n_i n_j

    Args:
        qc: QuantumCircuit to modify
        gamma: Cost layer parameter
        alpha: On-site energies (array)
        beta_matrix: Interaction matrix (2D array)
        n_qubits: Number of qubits
    """
    # Single qubit Z rotations from alpha_i
    for i in range(n_qubits):
        angle = -gamma * alpha[i]
        qc.rz(angle, i)

    # Two qubit ZZ rotations from beta_ij
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            bij = beta_matrix[i, j]
            if abs(bij) < 1e-10:
                continue
            angle = gamma * bij / 2
            qc.cx(i, j)
            qc.rz(angle, j)
            qc.cx(i, j)


def apply_x_mixer_layer(qc, beta, n_qubits):
    """Apply X mixer: exp(-i * beta * H_X).

    H_X = sum_i X_i (standard transverse field mixer)

    Does NOT preserve particle number.

    Args:
        qc: QuantumCircuit to modify
        beta: Mixer parameter
        n_qubits: Number of qubits
    """
    for i in range(n_qubits):
        qc.rx(2 * beta, i)


def apply_xy_mixer_layer(qc, beta, n_qubits, connectivity='full'):
    """Apply XY mixer: exp(-i * beta * H_XY).

    H_XY = sum_{i<j} (X_i X_j + Y_i Y_j)

    Preserves particle number (Hamming weight).

    Args:
        qc: QuantumCircuit to modify
        beta: Mixer parameter
        n_qubits: Number of qubits
        connectivity: 'full' for all-to-all, 'nearest' for nearest-neighbor
    """
    if connectivity == 'full':
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qc.rxx(2 * beta, i, j)
                qc.ryy(2 * beta, i, j)
    elif connectivity == 'nearest':
        for i in range(n_qubits - 1):
            qc.rxx(2 * beta, i, i + 1)
            qc.ryy(2 * beta, i, i + 1)


def build_qaoa_circuit(n_qubits, p, gammas, betas, alpha, beta_matrix,
                       method='dicke_xy', n_particles=None, add_measurements=True):
    """Build complete QAOA circuit.

    Args:
        n_qubits: Number of qubits
        p: QAOA depth (number of layers)
        gammas: Cost layer parameters (length p)
        betas: Mixer layer parameters (length p)
        alpha: On-site energies
        beta_matrix: Interaction matrix
        method: 'standard' for X mixer, 'dicke_xy' for Dicke + XY
        n_particles: Number of particles (required for dicke_xy)
        add_measurements: Whether to add measurement gates

    Returns:
        QuantumCircuit
    """
    # Initial state
    if method == 'standard':
        qc = create_hadamard_initial_state(n_qubits)
    elif method == 'dicke_xy':
        if n_particles is None:
            raise ValueError("n_particles required for dicke_xy method")
        qc = create_dicke_initial_state(n_qubits, n_particles)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Apply p layers
    for layer in range(p):
        apply_cost_layer(qc, gammas[layer], alpha, beta_matrix, n_qubits)

        if method == 'standard':
            apply_x_mixer_layer(qc, betas[layer], n_qubits)
        elif method == 'dicke_xy':
            apply_xy_mixer_layer(qc, betas[layer], n_qubits)

    # Add measurements
    if add_measurements:
        qc.measure_all()

    return qc


def save_circuit_diagram(qc, filepath, fold=80):
    """Save circuit diagram as PNG.

    Args:
        qc: QuantumCircuit
        filepath: Output file path (should end in .png)
        fold: Number of gates before folding to new line
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig = qc.draw(output='mpl', fold=fold)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)


def get_circuit_stats(qc):
    """Get circuit statistics.

    Args:
        qc: QuantumCircuit

    Returns:
        dict with depth, gate counts, etc.
    """
    return {
        'depth': qc.depth(),
        'num_qubits': qc.num_qubits,
        'total_gates': qc.size(),
        'gate_counts': dict(qc.count_ops())
    }
