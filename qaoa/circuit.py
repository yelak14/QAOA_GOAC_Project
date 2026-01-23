from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.quantum_info import SparsePauliOp
import numpy as np

from .mixers import standard_mixer, xy_mixer, get_initial_state


def cost_layer(hamiltonian, gamma, n_qubits):
    """Create a single cost layer: exp(-i * gamma * H_C).

    Implements the diagonal cost unitary using Rz and Rzz gates.

    Args:
        hamiltonian: SparsePauliOp representing the cost Hamiltonian
        gamma: Parameter or float for the cost angle
        n_qubits: number of qubits

    Returns:
        QuantumCircuit implementing the cost layer
    """
    qc = QuantumCircuit(n_qubits)

    for pauli_label, coeff in zip(hamiltonian.paulis.to_labels(), hamiltonian.coeffs):
        coeff = np.real(coeff)
        if abs(coeff) < 1e-10:
            continue

        # Find which qubits have Z operators
        z_qubits = []
        for pos, char in enumerate(pauli_label):
            if char == 'Z':
                # Qiskit labels: leftmost char = highest qubit index
                qubit_idx = n_qubits - 1 - pos
                z_qubits.append(qubit_idx)

        if len(z_qubits) == 0:
            # Identity term - contributes global phase, skip
            continue
        elif len(z_qubits) == 1:
            # Single Z: Rz gate
            qc.rz(2 * gamma * coeff, z_qubits[0])
        elif len(z_qubits) == 2:
            # ZZ term: CNOT-Rz-CNOT decomposition
            i, j = z_qubits
            qc.cx(i, j)
            qc.rz(2 * gamma * coeff, j)
            qc.cx(i, j)

    return qc


def build_qaoa_circuit(hamiltonian, mixer_type='standard', p=1,
                       n_qubits=8, n_particles=2):
    """Build a parameterized QAOA circuit.

    Args:
        hamiltonian: SparsePauliOp cost Hamiltonian
        mixer_type: 'standard' or 'xy'
        p: number of QAOA layers
        n_qubits: number of qubits
        n_particles: number of particles (for XY mixer)

    Returns:
        Tuple of (QuantumCircuit, list of gamma Parameters, list of beta Parameters)
    """
    # Create parameters
    gammas = [Parameter(f'γ_{i}') for i in range(p)]
    betas = [Parameter(f'β_{i}') for i in range(p)]

    # Initialize circuit with initial state
    qc = get_initial_state(n_qubits, n_particles, mixer_type)

    # Apply p layers of cost + mixer
    for layer in range(p):
        # Cost layer
        qc.compose(cost_layer(hamiltonian, gammas[layer], n_qubits), inplace=True)

        # Mixer layer
        if mixer_type == 'standard':
            qc.compose(standard_mixer(n_qubits, betas[layer]), inplace=True)
        elif mixer_type == 'xy':
            qc.compose(xy_mixer(n_qubits, betas[layer]), inplace=True)

    # Measurement
    qc.measure_all()

    return qc, gammas, betas
