"""QAOA with constraint-preserving XY mixer on IBM Quantum hardware."""

import sys
import json
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from qaoa.utils import load_coefficients, is_valid_config
from qaoa.hamiltonian import build_cost_hamiltonian, evaluate_energy, get_exact_solution
from qaoa.circuit import build_qaoa_circuit

from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Session
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def main():
    print("=" * 60)
    print("QAOA with XY Mixer (IBM Hardware)")
    print("=" * 60)

    # Load GOAC coefficients
    data_dir = project_root / "data" / "input"
    alpha, beta_coeff, E_const = load_coefficients(str(data_dir))
    n_qubits = len(alpha)
    n_particles = 2

    print(f"\nSystem: {n_qubits} qubits, {n_particles} particles")
    print(f"E_const = {E_const:.3f} eV")
    print(f"Initial state: |{'1' * n_particles + '0' * (n_qubits - n_particles)}>")

    # Get exact solution for reference
    ground_state, ground_energy, all_energies = get_exact_solution(
        alpha, beta_coeff, E_const, n_particles, n_qubits
    )
    print(f"Exact ground state: {ground_state}, E = {ground_energy:.6f} eV")

    # Connect to IBM Quantum
    print("\nConnecting to IBM Quantum...")
    try:
        service = QiskitRuntimeService()
    except Exception:
        print("ERROR: IBM Quantum credentials not configured.")
        print("Run: from qiskit_ibm_runtime import QiskitRuntimeService")
        print("     QiskitRuntimeService.save_account(channel='ibm_quantum', token='YOUR_TOKEN')")
        return

    # Select backend
    backend = service.least_busy(simulator=False, min_num_qubits=n_qubits)
    print(f"Backend: {backend.name}")

    # Build QAOA circuit with XY mixer
    hamiltonian = build_cost_hamiltonian(alpha, beta_coeff, E_const)
    p = 1
    circuit, gammas, betas = build_qaoa_circuit(
        hamiltonian, mixer_type='xy', p=p,
        n_qubits=n_qubits, n_particles=n_particles
    )

    # Transpile for hardware
    pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
    transpiled = pm.run(circuit)
    print(f"Transpiled circuit depth: {transpiled.depth()}")
    print(f"Gate counts: {dict(transpiled.count_ops())}")

    # Optimization loop
    print("\nRunning optimization...")
    shots = 4096
    history = {'costs': [], 'params': []}

    def objective(params):
        param_dict = {}
        for i in range(p):
            param_dict[gammas[i]] = params[i]
            param_dict[betas[i]] = params[p + i]

        bound = circuit.assign_parameters(param_dict)
        transpiled_bound = pm.run(bound)

        with Session(service=service, backend=backend) as session:
            sampler = SamplerV2(session=session)
            job = sampler.run([transpiled_bound], shots=shots)
            result = job.result()

        counts = result[0].data.meas.get_counts()

        from qaoa.optimizer import cvar_cost_function
        cost = cvar_cost_function(counts, alpha, beta_coeff, E_const, cvar_alpha=0.2)
        history['costs'].append(cost)
        history['params'].append(params.tolist())
        print(f"  Eval {len(history['costs'])}: cost = {cost:.6f}")
        return cost

    from scipy.optimize import minimize
    x0 = np.random.uniform(0, np.pi, 2 * p)
    result = minimize(objective, x0, method='COBYLA',
                      options={'maxiter': 50, 'rhobeg': 0.5})

    # Final measurement
    param_dict = {}
    for i in range(p):
        param_dict[gammas[i]] = result.x[i]
        param_dict[betas[i]] = result.x[p + i]

    bound = circuit.assign_parameters(param_dict)
    transpiled_bound = pm.run(bound)

    with Session(service=service, backend=backend) as session:
        sampler = SamplerV2(session=session)
        job = sampler.run([transpiled_bound], shots=shots * 4)
        final_result = job.result()

    counts = final_result[0].data.meas.get_counts()

    # Analyze
    valid_counts = {k: v for k, v in counts.items() if is_valid_config(k[::-1], n_particles)}
    total = sum(counts.values())
    valid_fraction = sum(valid_counts.values()) / total

    if valid_counts:
        best_bs = max(valid_counts, key=valid_counts.get)[::-1]
        best_energy = evaluate_energy(best_bs, alpha, beta_coeff, E_const)
    else:
        best_bs, best_energy = "N/A", float('inf')

    print(f"\n--- Results ---")
    print(f"Optimal cost: {result.fun:.6f} eV")
    print(f"Valid fraction: {valid_fraction:.1%}")
    print(f"Best config: {best_bs}, E = {best_energy:.6f} eV")
    print(f"Exact ground: {ground_state}, E = {ground_energy:.6f} eV")

    if valid_fraction > 0.99:
        print("[OK] XY mixer successfully preserved particle number on hardware")
    else:
        print(f"[WARN] Hardware noise caused {1-valid_fraction:.1%} constraint violations")

    # Save results
    output_dir = project_root / "results" / "hardware" / "constrained"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        'backend': backend.name,
        'p': p,
        'optimal_cost': float(result.fun),
        'optimal_params': result.x.tolist(),
        'valid_fraction': valid_fraction,
        'best_config': best_bs,
        'best_energy': float(best_energy),
        'ground_state': ground_state,
        'ground_energy': ground_energy,
        'convergence_history': history['costs'],
        'counts': counts
    }
    with open(output_dir / "results.json", 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
