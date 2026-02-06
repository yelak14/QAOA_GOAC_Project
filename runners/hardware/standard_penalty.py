"""Standard QAOA with Penalty on IBM Quantum Hardware.

Features:
- IBM Quantum credential handling (qiskit-ibm-runtime)
- Error mitigation (resilience level)
- Transpilation for hardware topology
- Job queuing and monitoring
- Circuit visualization
- CSV export for OriginLab
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from math import comb
import time

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Session, SamplerV2 as Sampler
from qiskit_ibm_runtime.options import SamplerOptions
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qaoa.hamiltonian import evaluate_energy, get_exact_solution
from qaoa.utils import load_coefficients, is_valid_config
from qaoa.circuits import (
    create_hadamard_initial_state, apply_cost_layer, apply_x_mixer_layer,
    build_qaoa_circuit, save_circuit_diagram, get_circuit_stats
)


# ===========================================
# CONFIGURATION
# ===========================================
P_VALUE = 5               # Lower depth for hardware (noise)
PENALTY_LAMBDA = 500.0    # Penalty strength
N_PARTICLES = 2           # Target particle number
SHOTS = 4096              # Shots per circuit
MAXITER = 100             # Fewer iterations for hardware

# IBM Quantum settings
BACKEND_NAME = "ibm_brisbane"  # Or: ibm_kyoto, ibm_osaka
OPTIMIZATION_LEVEL = 3         # Transpiler optimization (0-3)
RESILIENCE_LEVEL = 1           # Error mitigation (0-2)
# ===========================================


def setup_ibm_quantum():
    """Setup IBM Quantum service.

    First time setup (run once):
    >>> from qiskit_ibm_runtime import QiskitRuntimeService
    >>> QiskitRuntimeService.save_account(channel="ibm_quantum", token="YOUR_TOKEN")
    """
    service = QiskitRuntimeService(channel="ibm_quantum")
    backend = service.backend(BACKEND_NAME)

    print(f"Connected to IBM Quantum")
    print(f"  Backend: {backend.name}")
    print(f"  Qubits: {backend.num_qubits}")
    print(f"  Status: {backend.status().status_msg}")

    return service, backend


def transpile_for_hardware(qc, backend):
    """Transpile circuit for hardware topology."""
    pm = generate_preset_pass_manager(
        backend=backend,
        optimization_level=OPTIMIZATION_LEVEL
    )
    return pm.run(qc)


def run_circuit_on_hardware(qc, backend, shots):
    """Run circuit on hardware with error mitigation."""
    options = SamplerOptions()
    options.resilience_level = RESILIENCE_LEVEL
    options.default_shots = shots

    with Session(backend=backend) as session:
        sampler = Sampler(mode=session, options=options)
        job = sampler.run([qc])

        print(f"  Job ID: {job.job_id()}")
        print(f"  Waiting for results...")

        result = job.result()

    # Extract counts
    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()

    return counts, job.job_id()


def compute_expectation_and_n_from_counts(counts, alpha_transformed, beta_matrix_transformed,
                                           E_const_transformed, n_qubits):
    """Compute <H'> and <N> from measurement counts."""
    total_shots = sum(counts.values())
    expected_energy = 0.0
    expected_n = 0.0

    for bitstring, count in counts.items():
        prob = count / total_shots
        bs = bitstring[::-1]

        n_particles = bs.count('1')
        expected_n += prob * n_particles

        energy = E_const_transformed
        for i, bit in enumerate(bs):
            if bit == '1':
                energy += alpha_transformed[i]

        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if bs[i] == '1' and bs[j] == '1':
                    energy += beta_matrix_transformed[i, j]

        expected_energy += prob * energy

    return expected_energy, expected_n


def main():
    print("=" * 70)
    print("Standard QAOA + Penalty on IBM Quantum Hardware")
    print("=" * 70)

    # Load coefficients
    data_dir = project_root / "data" / "input"
    alpha, beta_coeff, E_const = load_coefficients(str(data_dir))

    alpha = np.array(alpha).flatten()
    n_qubits = len(alpha)

    # Build beta matrix
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

    # Apply penalty transformation
    alpha_transformed = alpha + PENALTY_LAMBDA * (1 - 2 * N_PARTICLES)
    beta_matrix_transformed = beta_matrix + 2 * PENALTY_LAMBDA
    E_const_transformed = E_const + PENALTY_LAMBDA * N_PARTICLES**2

    print(f"\nSystem: {n_qubits} qubits, {N_PARTICLES} particles (target)")
    print(f"Penalty lambda = {PENALTY_LAMBDA}")
    print(f"QAOA depth p = {P_VALUE} ({2*P_VALUE} parameters)")
    print(f"Shots: {SHOTS}")

    # Get exact ground state
    ground_state, ground_energy, all_energies = get_exact_solution(
        alpha, beta_coeff, E_const, N_PARTICLES, n_qubits
    )
    print(f"\nGround state: {ground_state}")
    print(f"Ground energy (original): {ground_energy:.4f} eV")

    # Setup IBM Quantum
    print("\n" + "=" * 70)
    print("Setting up IBM Quantum connection...")
    print("=" * 70)

    try:
        service, backend = setup_ibm_quantum()
    except Exception as e:
        print(f"Error connecting to IBM Quantum: {e}")
        print("Please run the following to save your credentials:")
        print("  from qiskit_ibm_runtime import QiskitRuntimeService")
        print("  QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN')")
        return

    # Output directory
    output_dir = project_root / "results" / "hardware" / "standard_penalty"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use random parameters for demonstration
    np.random.seed(42)
    gammas = np.random.uniform(0, np.pi, P_VALUE)
    betas = np.random.uniform(0, np.pi, P_VALUE)

    # Build circuit
    print("\n" + "=" * 70)
    print("Building and transpiling circuit...")
    print("=" * 70)

    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))

    for layer in range(P_VALUE):
        apply_cost_layer(qc, gammas[layer], alpha_transformed,
                        beta_matrix_transformed, n_qubits)
        apply_x_mixer_layer(qc, betas[layer], n_qubits)

    qc.measure_all()

    # Get circuit stats before transpilation
    stats_before = get_circuit_stats(qc)
    print(f"  Before transpilation: depth={stats_before['depth']}, gates={stats_before['total_gates']}")

    # Transpile for hardware
    qc_transpiled = transpile_for_hardware(qc, backend)
    stats_after = get_circuit_stats(qc_transpiled)
    print(f"  After transpilation: depth={stats_after['depth']}, gates={stats_after['total_gates']}")

    # Save circuit diagrams
    try:
        save_circuit_diagram(qc, output_dir / "circuit_original.png")
        save_circuit_diagram(qc_transpiled, output_dir / "circuit_transpiled.png", fold=100)
        print(f"  Saved circuit diagrams")
    except Exception as e:
        print(f"  Warning: Could not save circuit diagrams: {e}")

    # Run on hardware
    print("\n" + "=" * 70)
    print("Running on IBM Quantum hardware...")
    print("=" * 70)

    start_time = time.time()
    counts, job_id = run_circuit_on_hardware(qc_transpiled, backend, SHOTS)
    elapsed_time = time.time() - start_time

    print(f"  Completed in {elapsed_time:.1f}s")

    # Process results
    total_shots_final = sum(counts.values())
    final_distribution = {bs[::-1]: count/total_shots_final
                         for bs, count in counts.items()}

    valid_prob = sum(prob for bs, prob in final_distribution.items()
                    if is_valid_config(bs, N_PARTICLES))

    # Energy distribution
    energy_dist = {}
    for bs, prob in final_distribution.items():
        energy = E_const_transformed
        for i, bit in enumerate(bs):
            if bit == '1':
                energy += alpha_transformed[i]
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if bs[i] == '1' and bs[j] == '1':
                    energy += beta_matrix_transformed[i, j]

        energy_rounded = round(energy, 0)
        if energy_rounded not in energy_dist:
            energy_dist[energy_rounded] = 0.0
        energy_dist[energy_rounded] += prob

    min_energy = min(energy_dist.keys())
    gs_prob_summed = energy_dist[min_energy]

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Valid fraction: {valid_prob:.2%}")
    print(f"  GS prob (summed): {gs_prob_summed:.4f}")

    # Top 10 states
    top_states = sorted(final_distribution.items(), key=lambda x: -x[1])[:10]
    print("\n  Top 10 measured states:")
    for bs, prob in top_states:
        n_p = bs.count('1')
        valid_str = "OK" if n_p == N_PARTICLES else "X"
        print(f"    {bs}: prob={prob:.4f}, N={n_p} {valid_str}")

    # Save results
    print("\n" + "=" * 70)
    print("Saving results...")
    print("=" * 70)

    # Final distribution
    final_dist_df = pd.DataFrame({
        'bitstring': list(final_distribution.keys()),
        'probability': list(final_distribution.values()),
        'n_particles': [bs.count('1') for bs in final_distribution.keys()]
    })
    final_dist_df.to_csv(output_dir / "final_distribution.csv", index=False)

    # Energy distribution
    sorted_energies = sorted(energy_dist.keys())
    energy_dist_df = pd.DataFrame({
        'energy': sorted_energies,
        'probability': [energy_dist[e] for e in sorted_energies]
    })
    energy_dist_df.to_csv(output_dir / "energy_distribution.csv", index=False)

    # Summary
    summary = {
        'settings': {
            'method': 'standard_penalty_hardware',
            'backend': BACKEND_NAME,
            'p': P_VALUE,
            'penalty_lambda': PENALTY_LAMBDA,
            'n_particles': N_PARTICLES,
            'shots': SHOTS,
            'optimization_level': OPTIMIZATION_LEVEL,
            'resilience_level': RESILIENCE_LEVEL
        },
        'job_id': job_id,
        'ground_state': ground_state,
        'ground_energy': float(ground_energy),
        'results': {
            'valid_fraction': valid_prob,
            'gs_prob_summed': gs_prob_summed,
            'gs_energy': min_energy,
            'elapsed_time': elapsed_time
        },
        'circuit_stats': {
            'original_depth': stats_before['depth'],
            'original_gates': stats_before['total_gates'],
            'transpiled_depth': stats_after['depth'],
            'transpiled_gates': stats_after['total_gates']
        }
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved to: {output_dir}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
