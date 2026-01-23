from qiskit.circuit import QuantumCircuit, Parameter
import numpy as np
from itertools import combinations
from math import comb, sqrt


def standard_mixer(n_qubits, beta_param):
    """Create the standard transverse-field mixer: H_M = sum_i X_i.

    Applies Rx(2*beta) to each qubit.

    Args:
        n_qubits: number of qubits
        beta_param: Parameter or float for the mixer angle

    Returns:
        QuantumCircuit implementing the mixer unitary
    """
    qc = QuantumCircuit(n_qubits)
    for i in range(n_qubits):
        qc.rx(2 * beta_param, i)
    return qc


def xy_mixer(n_qubits, beta_param, connectivity='full'):
    """Create the XY (particle-preserving) mixer.

    H_M = sum_{i<j} (X_i X_j + Y_i Y_j) for connected pairs.
    This preserves the total number of 1s (particle number).

    The XY interaction between qubits i and j is implemented as:
    exp(-i * beta * (XX + YY)) using a CNOT-Ry-CNOT decomposition.

    Args:
        n_qubits: number of qubits
        beta_param: Parameter or float for the mixer angle
        connectivity: 'full' for all-to-all, 'ring' for ring topology

    Returns:
        QuantumCircuit implementing the XY mixer unitary
    """
    qc = QuantumCircuit(n_qubits)

    if connectivity == 'full':
        pairs = [(i, j) for i in range(n_qubits) for j in range(i + 1, n_qubits)]
    elif connectivity == 'ring':
        pairs = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
    else:
        raise ValueError(f"Unknown connectivity: {connectivity}")

    for i, j in pairs:
        # XY interaction: exp(-i * beta * (XX + YY) / 2)
        # Decomposition using partial SWAP
        qc.cx(i, j)
        qc.ry(2 * beta_param, i)
        qc.cx(j, i)
        qc.ry(-2 * beta_param, i)
        qc.cx(i, j)

    return qc


def get_initial_state(n_qubits, n_particles=2, mixer_type='standard'):
    """Create the initial state circuit.

    Args:
        n_qubits: number of qubits
        n_particles: number of particles (for constrained mixer)
        mixer_type: 'standard' or 'xy'

    Returns:
        QuantumCircuit preparing the initial state
    """
    qc = QuantumCircuit(n_qubits)

    if mixer_type == 'standard':
        # Uniform superposition
        qc.h(range(n_qubits))
    elif mixer_type == 'xy':
        # Dicke state |D_k^n>: equal superposition over all C(n,k) states
        # with exactly k ones.
        statevector = _dicke_statevector(n_qubits, n_particles)
        qc.initialize(statevector, range(n_qubits))
    else:
        raise ValueError(f"Unknown mixer type: {mixer_type}")

    return qc


def _dicke_statevector(n, k):
    """Compute the Dicke state |D_k^n> as a statevector.

    The Dicke state is an equal superposition over all computational basis
    states with exactly k ones:
    |D_k^n> = (1/sqrt(C(n,k))) * sum_{|x|=k} |x>

    Args:
        n: number of qubits
        k: number of ones (particles)

    Returns:
        numpy array of length 2^n representing the statevector
    """
    dim = 2 ** n
    n_states = comb(n, k)
    amplitude = 1.0 / sqrt(n_states)

    statevector = np.zeros(dim)
    for occupied in combinations(range(n), k):
        # Build the basis state index
        # Qiskit convention: qubit 0 is least significant bit
        index = 0
        for qubit in occupied:
            index |= (1 << qubit)
        statevector[index] = amplitude

    return statevector
