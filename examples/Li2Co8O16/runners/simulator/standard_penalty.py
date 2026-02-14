"""Standard QAOA with Penalty - Aer Simulator + COBYLA (Enhanced Version).

ENHANCED FEATURES:
1) Save all plot data as CSV for OriginLab
2) Plot and save expected_N vs iterations
3) Brute force comparison
4) Parameter landscape visualization
5) Circuit visualization

Original settings:
- Aer simulator (shot-based, realistic)
- COBYLA optimizer (gradient-free, robust to noise)
- Configurable p and penalty_lambda

Example runner for Li2Co8O16.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from math import comb
import time

# Add project root to path (4 levels up from examples/Li2Co8O16/runners/simulator/)
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Example root (2 levels up from runners/simulator/)
example_root = Path(__file__).resolve().parent.parent

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

from qaoa.hamiltonian import evaluate_energy, get_exact_solution
from qaoa.utils import load_coefficients, is_valid_config
from qaoa.circuits import (
    create_hadamard_initial_state, apply_cost_layer, apply_x_mixer_layer,
    build_qaoa_circuit, save_circuit_diagram, get_circuit_stats
)
from postprocessing.plotting import plot_bitstring_probability


def compute_expectation_and_n_from_counts(counts, alpha_transformed, beta_matrix_transformed,
                                           E_const_transformed, n_qubits):
    """Compute <H'> and <N> from measurement counts.

    Returns:
        Tuple of (expected_energy, expected_N)
    """
    total_shots = sum(counts.values())
    expected_energy = 0.0
    expected_n = 0.0

    for bitstring, count in counts.items():
        prob = count / total_shots
        bs = bitstring[::-1]  # Reverse for qubit ordering

        # Count particles
        n_particles = bs.count('1')
        expected_n += prob * n_particles

        # Calculate energy
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


def run_qaoa_aer_cobyla_enhanced(alpha, beta_coeff, E_const, penalty_lambda, n_target,
                                  n_qubits, p, shots=8192, maxiter=2500, num_runs=4, verbose=True):
    """Run QAOA with Aer simulator + COBYLA (enhanced with N tracking)."""
    alpha = np.array(alpha).flatten()

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
    alpha_transformed = alpha + penalty_lambda * (1 - 2 * n_target)
    beta_matrix_transformed = beta_matrix + 2 * penalty_lambda
    E_const_transformed = E_const + penalty_lambda * n_target**2

    if verbose:
        print(f"Penalty transformation applied:")
        print(f"  alpha'[0] = {alpha_transformed[0]:.4f}")
        print(f"  beta'[0,1] = {beta_matrix_transformed[0,1]:.4f}")

    backend = AerSimulator()
    all_results = []

    for run_idx in range(num_runs):
        if verbose:
            print(f"\n{'='*60}")
            print(f"RUN {run_idx + 1}/{num_runs}")
            print(f"{'='*60}")

        # Enhanced history tracking (includes expected_N)
        history = {
            'energies': [],
            'expected_n': [],
            'iterations': [],
            'params': []
        }
        n_evals = [0]

        def objective(params):
            """Compute <H'> using Aer simulator."""
            gammas = params[:p]
            betas = params[p:]

            qc = QuantumCircuit(n_qubits)
            qc.h(range(n_qubits))

            for layer in range(p):
                apply_cost_layer(qc, gammas[layer], alpha_transformed,
                                beta_matrix_transformed, n_qubits)
                apply_x_mixer_layer(qc, betas[layer], n_qubits)

            qc.measure_all()

            transpiled = transpile(qc, backend)
            job = backend.run(transpiled, shots=shots)
            counts = job.result().get_counts()

            # Compute both energy and expected N
            energy, exp_n = compute_expectation_and_n_from_counts(
                counts, alpha_transformed, beta_matrix_transformed,
                E_const_transformed, n_qubits
            )

            # Record history
            history['energies'].append(energy)
            history['expected_n'].append(exp_n)
            history['iterations'].append(n_evals[0])
            history['params'].append(params.tolist())
            n_evals[0] += 1

            if verbose and n_evals[0] % 100 == 0:
                print(f"  Eval {n_evals[0]}: Re(H) = {energy:.4f}, <N> = {exp_n:.4f}")

            return energy

        np.random.seed(42 + run_idx * 100)
        x0 = np.random.uniform(0, 1.0, 2 * p)

        if verbose:
            print(f"Starting COBYLA optimization (gradient-free)...")
            print(f"  Parameters: {2*p} (p={p})")
            print(f"  Shots per evaluation: {shots}")
            print(f"  Max iterations: {maxiter}")

        start_time = time.time()

        result = minimize(
            objective,
            x0,
            method='COBYLA',
            options={
                'maxiter': maxiter,
                'rhobeg': 0.5,
                'disp': False
            }
        )

        elapsed_time = time.time() - start_time

        if verbose:
            print(f"\nOptimization finished in {elapsed_time:.1f}s")
            print(f"  Evaluations: {n_evals[0]}")
            print(f"  Final Re(H): {result.fun:.4f}")
            print(f"  Success: {result.success}")

        # Final measurement
        final_params = result.x
        gammas = final_params[:p]
        betas = final_params[p:]

        qc = QuantumCircuit(n_qubits)
        qc.h(range(n_qubits))
        for layer in range(p):
            apply_cost_layer(qc, gammas[layer], alpha_transformed,
                            beta_matrix_transformed, n_qubits)
            apply_x_mixer_layer(qc, betas[layer], n_qubits)
        qc.measure_all()

        transpiled = transpile(qc, backend)
        job = backend.run(transpiled, shots=shots * 4)
        final_counts = job.result().get_counts()

        total_shots_final = sum(final_counts.values())
        final_distribution = {bs[::-1]: count/total_shots_final
                             for bs, count in final_counts.items()}

        valid_prob = sum(prob for bs, prob in final_distribution.items()
                        if is_valid_config(bs, n_target))

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

        if verbose:
            print(f"\nResults:")
            print(f"  Valid fraction: {valid_prob:.2%}")
            print(f"  GS prob (summed at E={min_energy:.0f}): {gs_prob_summed:.4f}")

        all_results.append({
            'run_idx': run_idx,
            'history': history,
            'final_distribution': final_distribution,
            'energy_distribution': energy_dist,
            'valid_fraction': valid_prob,
            'gs_prob_summed': gs_prob_summed,
            'gs_energy': min_energy,
            'final_cost': result.fun,
            'final_params': result.x.tolist(),
            'n_evals': n_evals[0],
            'elapsed_time': elapsed_time,
            'success': result.success
        })

    return all_results, alpha_transformed, beta_matrix_transformed, E_const_transformed


def compute_brute_force_energies(alpha, beta_coeff, E_const, n_qubits):
    """Compute energies for ALL 2^n configurations (brute force).

    Returns:
        dict of {bitstring: (energy, n_particles)}
    """
    alpha = np.array(alpha).flatten()

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


def scan_parameter_landscape(alpha_transformed, beta_matrix_transformed, E_const_transformed,
                              n_qubits, p, optimal_params, backend, shots=4096,
                              gamma_range=(-np.pi, np.pi), beta_range=(-np.pi, np.pi),
                              n_points=21):
    """Scan 2D parameter landscape around optimal parameters.

    For simplicity, scans gamma_0 and beta_0 while keeping others fixed.

    Returns:
        dict with gamma_values, beta_values, energy_landscape (2D array)
    """
    gamma_values = np.linspace(gamma_range[0], gamma_range[1], n_points)
    beta_values = np.linspace(beta_range[0], beta_range[1], n_points)

    energy_landscape = np.zeros((n_points, n_points))

    base_params = optimal_params.copy()

    for i, gamma_0 in enumerate(gamma_values):
        for j, beta_0 in enumerate(beta_values):
            # Modify only gamma_0 and beta_0
            params = base_params.copy()
            params[0] = gamma_0      # gamma_0
            params[p] = beta_0       # beta_0

            gammas = params[:p]
            betas = params[p:]

            qc = QuantumCircuit(n_qubits)
            qc.h(range(n_qubits))

            for layer in range(p):
                apply_cost_layer(qc, gammas[layer], alpha_transformed,
                                beta_matrix_transformed, n_qubits)
                apply_x_mixer_layer(qc, betas[layer], n_qubits)

            qc.measure_all()

            transpiled = transpile(qc, backend)
            job = backend.run(transpiled, shots=shots)
            counts = job.result().get_counts()

            energy, _ = compute_expectation_and_n_from_counts(
                counts, alpha_transformed, beta_matrix_transformed,
                E_const_transformed, n_qubits
            )

            energy_landscape[i, j] = energy

    return {
        'gamma_values': gamma_values,
        'beta_values': beta_values,
        'energy_landscape': energy_landscape
    }


def main():
    print("=" * 70)
    print("Standard QAOA + Penalty (Aer Simulator + COBYLA) - ENHANCED")
    print("With: CSV export, <N> tracking, brute force, parameter landscape")
    print("=" * 70)

    # ===========================================
    # CONFIGURATION
    # ===========================================
    P_VALUE = 20              # QAOA depth (40 parameters)
    PENALTY_LAMBDA = 500.0    # Penalty strength
    N_PARTICLES = 2           # Target particle number
    SHOTS = 8192              # Shots per evaluation
    MAXITER = 10000           # COBYLA iterations
    NUM_RUNS = 4              # Number of different initializations

    # Landscape scan settings
    SCAN_LANDSCAPE = True
    LANDSCAPE_POINTS = 21     # Grid resolution for landscape
    # ===========================================

    # Load coefficients
    data_dir = example_root / "data" / "input"
    alpha, beta_coeff, E_const = load_coefficients(str(data_dir))

    alpha = np.array(alpha).flatten()
    n_qubits = len(alpha)

    print(f"\nSystem: {n_qubits} qubits, {N_PARTICLES} particles (target)")
    print(f"Penalty lambda = {PENALTY_LAMBDA}")
    print(f"QAOA depth p = {P_VALUE} ({2*P_VALUE} parameters)")
    print(f"Optimizer: COBYLA (gradient-free)")
    print(f"Shots per evaluation: {SHOTS}")
    print(f"Max iterations: {MAXITER}")
    print(f"Number of runs: {NUM_RUNS}")

    # Get exact ground state
    ground_state, ground_energy, all_energies = get_exact_solution(
        alpha, beta_coeff, E_const, N_PARTICLES, n_qubits
    )
    print(f"\nGround state: {ground_state}")
    print(f"Ground energy (original): {ground_energy:.4f} eV")
    print(f"Number of valid configs: {len(all_energies)}")

    # Run QAOA
    all_results, alpha_transformed, beta_matrix_transformed, E_const_transformed = \
        run_qaoa_aer_cobyla_enhanced(
            alpha, beta_coeff, E_const, PENALTY_LAMBDA, N_PARTICLES,
            n_qubits, P_VALUE, shots=SHOTS, maxiter=MAXITER,
            num_runs=NUM_RUNS, verbose=True
        )

    # Output directory
    output_dir = example_root / "results" / "simulator" / "standard_penalty"
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # SAVE CIRCUIT DIAGRAMS
    # =========================================================================
    print("\n" + "=" * 70)
    print("Saving circuit diagrams...")
    print("=" * 70)

    try:
        # 1. Initial state circuit
        initial_qc = QuantumCircuit(n_qubits)
        initial_qc.h(range(n_qubits))
        save_circuit_diagram(initial_qc, output_dir / "circuit_01_initial_state.png")
        print(f"  Saved: circuit_01_initial_state.png")

        # 2. Single QAOA layer
        single_layer_qc = QuantumCircuit(n_qubits)
        single_layer_qc.h(range(n_qubits))
        apply_cost_layer(single_layer_qc, 0.5, alpha_transformed, beta_matrix_transformed, n_qubits)
        apply_x_mixer_layer(single_layer_qc, 0.5, n_qubits)
        save_circuit_diagram(single_layer_qc, output_dir / "circuit_02_single_layer.png")
        print(f"  Saved: circuit_02_single_layer.png")

        # 3. Full ansatz (p layers, with example parameters)
        example_gammas = [0.5] * P_VALUE
        example_betas = [0.5] * P_VALUE
        full_qc = build_qaoa_circuit(n_qubits, P_VALUE, example_gammas, example_betas,
                                      alpha_transformed, beta_matrix_transformed,
                                      method='standard', add_measurements=True)
        save_circuit_diagram(full_qc, output_dir / "circuit_03_full_ansatz.png", fold=40)
        print(f"  Saved: circuit_03_full_ansatz.png")

        # Get circuit stats
        stats = get_circuit_stats(full_qc)
        print(f"  Circuit stats: depth={stats['depth']}, gates={stats['total_gates']}")
    except Exception as e:
        print(f"  Warning: Could not save circuit diagrams: {e}")

    # =========================================================================
    # 1) SAVE ALL PLOT DATA AS CSV FOR ORIGINLAB
    # =========================================================================
    print("\n" + "=" * 70)
    print("1) Saving plot data as CSV for OriginLab...")
    print("=" * 70)

    for run_idx, res in enumerate(all_results):
        # Convergence data: iteration, Re(H), expected_N
        convergence_df = pd.DataFrame({
            'iteration': res['history']['iterations'],
            'Re_H': res['history']['energies'],
            'expected_N': res['history']['expected_n']
        })
        convergence_df.to_csv(output_dir / f"run_{run_idx+1}_convergence.csv", index=False)

        # Energy distribution data
        energy_dist = res['energy_distribution']
        sorted_energies = sorted(energy_dist.keys())
        energy_dist_df = pd.DataFrame({
            'energy': sorted_energies,
            'probability': [energy_dist[e] for e in sorted_energies]
        })
        energy_dist_df.to_csv(output_dir / f"run_{run_idx+1}_energy_distribution.csv", index=False)

        # Final state distribution
        final_dist = res['final_distribution']
        final_dist_df = pd.DataFrame({
            'bitstring': list(final_dist.keys()),
            'probability': list(final_dist.values()),
            'n_particles': [bs.count('1') for bs in final_dist.keys()]
        })
        final_dist_df.to_csv(output_dir / f"run_{run_idx+1}_final_distribution.csv", index=False)

        # Parameters
        params_df = pd.DataFrame({
            'param_index': list(range(len(res['final_params']))),
            'param_value': res['final_params']
        })
        params_df.to_csv(output_dir / f"run_{run_idx+1}_final_params.csv", index=False)

    print(f"  Saved convergence, energy distribution, final distribution, and params for each run")

    # =========================================================================
    # 2) PLOT AND SAVE EXPECTED_N VS ITERATIONS
    # =========================================================================
    print("\n" + "=" * 70)
    print("2) Plotting expected_N vs iterations...")
    print("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Combined plot for all runs
    fig, axes = plt.subplots(NUM_RUNS, 2, figsize=(14, 4 * NUM_RUNS))

    if NUM_RUNS == 1:
        axes = [axes]

    for row_idx, res in enumerate(all_results):
        # Left: Re(H) vs Iteration
        ax_left = axes[row_idx][0]
        ax_left.plot(res['history']['iterations'], res['history']['energies'], 'b-', linewidth=0.5)
        ax_left.set_xlabel('Iteration', fontsize=11)
        ax_left.set_ylabel('Re(H)', fontsize=11)
        ax_left.set_title(f'Run {row_idx + 1} - Energy Convergence', fontsize=12)
        ax_left.grid(True, alpha=0.3)

        # Right: Expected N vs Iteration
        ax_right = axes[row_idx][1]
        ax_right.plot(res['history']['iterations'], res['history']['expected_n'], 'r-', linewidth=0.5)
        ax_right.axhline(y=N_PARTICLES, color='g', linestyle='--', label=f'Target N={N_PARTICLES}')
        ax_right.set_xlabel('Iteration', fontsize=11)
        ax_right.set_ylabel('<N>', fontsize=11)
        ax_right.set_title(f'Run {row_idx + 1} - Expected Particle Number', fontsize=12)
        ax_right.legend()
        ax_right.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "expected_n_vs_iterations.png", dpi=150)
    plt.close()
    print(f"  Saved: expected_n_vs_iterations.png")

    # =========================================================================
    # 3) BRUTE FORCE COMPARISON
    # =========================================================================
    print("\n" + "=" * 70)
    print("3) Computing brute force comparison...")
    print("=" * 70)

    brute_force_configs = compute_brute_force_energies(alpha, beta_coeff, E_const, n_qubits)

    # Sort by energy
    sorted_configs = sorted(brute_force_configs.items(), key=lambda x: x[1][0])

    # Save brute force data
    brute_force_df = pd.DataFrame({
        'bitstring': [cfg[0] for cfg in sorted_configs],
        'energy_eV': [cfg[1][0] for cfg in sorted_configs],
        'n_particles': [cfg[1][1] for cfg in sorted_configs],
        'is_valid': [cfg[1][1] == N_PARTICLES for cfg in sorted_configs]
    })
    brute_force_df.to_csv(output_dir / "brute_force_all_configs.csv", index=False)

    # Filter valid configs
    valid_configs = [(bs, e, n) for bs, (e, n) in sorted_configs if n == N_PARTICLES]
    valid_configs_df = pd.DataFrame({
        'bitstring': [cfg[0] for cfg in valid_configs],
        'energy_eV': [cfg[1] for cfg in valid_configs],
        'n_particles': [cfg[2] for cfg in valid_configs]
    })
    valid_configs_df.to_csv(output_dir / "brute_force_valid_configs.csv", index=False)

    print(f"  Total configurations: {len(sorted_configs)}")
    print(f"  Valid configurations (N={N_PARTICLES}): {len(valid_configs)}")
    print(f"  Ground state: {valid_configs[0][0]} with E={valid_configs[0][1]:.4f} eV")

    # Plot: Brute force energy spectrum
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # All configurations
    all_energies_list = [cfg[1][0] for cfg in sorted_configs]
    all_n_particles = [cfg[1][1] for cfg in sorted_configs]
    colors = ['red' if n == N_PARTICLES else 'blue' for n in all_n_particles]

    ax1.scatter(range(len(sorted_configs)), all_energies_list, c=colors, s=10, alpha=0.6)
    ax1.set_xlabel('Configuration Index (sorted by energy)', fontsize=11)
    ax1.set_ylabel('Energy (eV)', fontsize=11)
    ax1.set_title('Brute Force: All Configurations', fontsize=12)
    ax1.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label=f'N={N_PARTICLES} (valid)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Other N')
    ])
    ax1.grid(True, alpha=0.3)

    # Energy vs N
    energy_by_n = {}
    for bs, (e, n) in sorted_configs:
        if n not in energy_by_n:
            energy_by_n[n] = []
        energy_by_n[n].append(e)

    n_values = sorted(energy_by_n.keys())
    min_energies = [min(energy_by_n[n]) for n in n_values]

    ax2.bar(n_values, min_energies, color='steelblue', edgecolor='black')
    ax2.axvline(x=N_PARTICLES, color='red', linestyle='--', label=f'Target N={N_PARTICLES}')
    ax2.set_xlabel('Number of Particles (N)', fontsize=11)
    ax2.set_ylabel('Minimum Energy (eV)', fontsize=11)
    ax2.set_title('Minimum Energy vs Particle Number', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "brute_force_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: brute_force_comparison.png")

    # Compare QAOA results with brute force
    print("\n  Comparison with QAOA results:")
    for run_idx, res in enumerate(all_results):
        # Get top states from QAOA
        top_states = sorted(res['final_distribution'].items(), key=lambda x: -x[1])[:5]
        print(f"\n  Run {run_idx + 1} - Top 5 measured states:")
        for bs, prob in top_states:
            bf_energy, bf_n = brute_force_configs[bs]
            valid_str = "OK" if bf_n == N_PARTICLES else "X"
            print(f"    {bs}: prob={prob:.4f}, E={bf_energy:.4f} eV, N={bf_n} {valid_str}")

    # Save comparison data
    comparison_data = []
    for run_idx, res in enumerate(all_results):
        for bs, prob in res['final_distribution'].items():
            bf_energy, bf_n = brute_force_configs[bs]
            comparison_data.append({
                'run': run_idx + 1,
                'bitstring': bs,
                'qaoa_probability': prob,
                'brute_force_energy': bf_energy,
                'n_particles': bf_n,
                'is_valid': bf_n == N_PARTICLES
            })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / "qaoa_vs_brute_force_comparison.csv", index=False)

    # =========================================================================
    # 4) PARAMETER LANDSCAPE
    # =========================================================================
    if SCAN_LANDSCAPE:
        print("\n" + "=" * 70)
        print("4) Scanning parameter landscape...")
        print("=" * 70)

        backend = AerSimulator()

        # Use the best run's parameters
        best_run_idx = np.argmax([res['valid_fraction'] for res in all_results])
        best_params = np.array(all_results[best_run_idx]['final_params'])

        print(f"  Using parameters from Run {best_run_idx + 1} (best valid fraction)")
        print(f"  Scanning {LANDSCAPE_POINTS}x{LANDSCAPE_POINTS} grid...")

        landscape_data = scan_parameter_landscape(
            alpha_transformed, beta_matrix_transformed, E_const_transformed,
            n_qubits, P_VALUE, best_params, backend,
            shots=2048, n_points=LANDSCAPE_POINTS
        )

        # Save landscape data
        landscape_df = pd.DataFrame(landscape_data['energy_landscape'],
                                     index=landscape_data['gamma_values'],
                                     columns=landscape_data['beta_values'])
        landscape_df.to_csv(output_dir / "parameter_landscape.csv")

        # Also save as flat format for easier plotting
        flat_landscape = []
        for i, gamma in enumerate(landscape_data['gamma_values']):
            for j, beta in enumerate(landscape_data['beta_values']):
                flat_landscape.append({
                    'gamma_0': gamma,
                    'beta_0': beta,
                    'energy': landscape_data['energy_landscape'][i, j]
                })
        flat_landscape_df = pd.DataFrame(flat_landscape)
        flat_landscape_df.to_csv(output_dir / "parameter_landscape_flat.csv", index=False)

        # Plot landscape
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(landscape_data['energy_landscape'],
                       extent=[landscape_data['beta_values'][0], landscape_data['beta_values'][-1],
                               landscape_data['gamma_values'][-1], landscape_data['gamma_values'][0]],
                       aspect='auto', cmap='viridis')

        # Mark optimal point
        ax.scatter([best_params[P_VALUE]], [best_params[0]],
                   c='red', s=100, marker='*', label='Optimal')

        ax.set_xlabel('beta_0', fontsize=12)
        ax.set_ylabel('gamma_0', fontsize=12)
        ax.set_title('Parameter Landscape (gamma_0 vs beta_0)', fontsize=14)
        plt.colorbar(im, ax=ax, label='Re(H)')
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "parameter_landscape.png", dpi=150)
        plt.close()
        print(f"  Saved: parameter_landscape.png")

        # 3D surface plot
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(landscape_data['beta_values'], landscape_data['gamma_values'])
        ax.plot_surface(X, Y, landscape_data['energy_landscape'], cmap='viridis', alpha=0.8)

        ax.set_xlabel('beta_0', fontsize=11)
        ax.set_ylabel('gamma_0', fontsize=11)
        ax.set_zlabel('Re(H)', fontsize=11)
        ax.set_title('Parameter Landscape (3D)', fontsize=14)

        plt.tight_layout()
        plt.savefig(output_dir / "parameter_landscape_3d.png", dpi=150)
        plt.close()
        print(f"  Saved: parameter_landscape_3d.png")

    # =========================================================================
    # GENERATE PROFESSOR-STYLE PLOTS (original)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Generating Professor-style plots...")
    print("=" * 70)

    fig, axes = plt.subplots(NUM_RUNS, 2, figsize=(14, 4 * NUM_RUNS))

    if NUM_RUNS == 1:
        axes = [axes]

    for row_idx, res in enumerate(all_results):
        # Left: Re(H) vs Iteration
        ax_left = axes[row_idx][0]
        ax_left.plot(res['history']['iterations'], res['history']['energies'], 'b-', linewidth=0.5)
        ax_left.set_xlabel('Iteration', fontsize=11)
        ax_left.set_ylabel('Re(H)', fontsize=11)
        ax_left.set_title(f'Run {row_idx + 1}', fontsize=12)
        ax_left.grid(True, alpha=0.3)

        # Right: Energy Distribution
        ax_right = axes[row_idx][1]
        energy_dist = res['energy_distribution']
        sorted_energies = sorted(energy_dist.keys())
        sorted_probs = [energy_dist[e] for e in sorted_energies]

        x_pos = np.arange(len(sorted_energies))
        ax_right.bar(x_pos, sorted_probs, color='steelblue', edgecolor='black', linewidth=0.3)

        n_labels = min(25, len(sorted_energies))
        step = max(1, len(sorted_energies) // n_labels)
        tick_positions = x_pos[::step]
        tick_labels = [f'{sorted_energies[i]:.0f}' for i in range(0, len(sorted_energies), step)]
        ax_right.set_xticks(tick_positions)
        ax_right.set_xticklabels(tick_labels, rotation=90, fontsize=8)

        ax_right.set_xlabel('Energy Value', fontsize=11)
        ax_right.set_ylabel('Total Probability', fontsize=11)
        ax_right.set_title(f'Valid: {res["valid_fraction"]:.1%}, GS: {res["gs_prob_summed"]:.3f}', fontsize=11)
        ax_right.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "professor_style_plots.png", dpi=150)
    plt.close()
    print(f"  Saved: professor_style_plots.png")

    # =========================================================================
    # BITSTRING PROBABILITY PLOT (using best run)
    # =========================================================================
    best_run_idx = np.argmax([res['gs_prob_summed'] for res in all_results])
    best_final_distribution = all_results[best_run_idx]['final_distribution']

    # Bitstring probability plot
    plot_bitstring_probability(best_final_distribution, N_PARTICLES, ground_state, output_dir)
    print(f"  Saved: bitstring_probability.png, bitstring_probability.csv")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Run':<6} {'Valid %':<12} {'GS Prob (summed)':<18} {'Final Re(H)':<15} {'Time (s)':<10}")
    print("-" * 70)
    for res in all_results:
        print(f"{res['run_idx']+1:<6} {res['valid_fraction']*100:<12.2f} {res['gs_prob_summed']:<18.4f} {res['final_cost']:<15.2f} {res['elapsed_time']:<10.1f}")

    avg_valid = np.mean([res['valid_fraction'] for res in all_results])
    avg_gs_prob = np.mean([res['gs_prob_summed'] for res in all_results])

    print("-" * 70)
    print(f"{'AVG':<6} {avg_valid*100:<12.2f} {avg_gs_prob:<18.4f}")

    print(f"\nGround state energy (original): {ground_energy:.4f} eV")
    print(f"Number of valid configurations: {comb(n_qubits, N_PARTICLES)}")

    # Save summary JSON
    summary = {
        'settings': {
            'method': 'standard_penalty',
            'p': P_VALUE,
            'n_parameters': 2 * P_VALUE,
            'penalty_lambda': PENALTY_LAMBDA,
            'n_particles': N_PARTICLES,
            'shots': SHOTS,
            'maxiter': MAXITER,
            'num_runs': NUM_RUNS
        },
        'ground_state': ground_state,
        'ground_energy_original': float(ground_energy),
        'results': [
            {
                'run': res['run_idx'] + 1,
                'valid_fraction': res['valid_fraction'],
                'gs_prob_summed': res['gs_prob_summed'],
                'gs_energy': res['gs_energy'],
                'final_cost': res['final_cost'],
                'n_evals': res['n_evals'],
                'elapsed_time': res['elapsed_time']
            }
            for res in all_results
        ],
        'averages': {
            'valid_fraction': float(avg_valid),
            'gs_prob_summed': float(avg_gs_prob)
        }
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs saved to: {output_dir}")
    print("\nCSV files for OriginLab:")
    print("  - run_X_convergence.csv (iteration, Re_H, expected_N)")
    print("  - run_X_energy_distribution.csv (energy, probability)")
    print("  - run_X_final_distribution.csv (bitstring, probability, n_particles)")
    print("  - run_X_final_params.csv (param_index, param_value)")
    print("  - brute_force_all_configs.csv")
    print("  - brute_force_valid_configs.csv")
    print("  - qaoa_vs_brute_force_comparison.csv")
    print("  - parameter_landscape.csv")
    print("  - parameter_landscape_flat.csv")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
