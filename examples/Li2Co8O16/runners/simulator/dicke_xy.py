"""Particle-Number-Conserving QAOA with Dicke Initial State + XY Mixer (Enhanced).

This implements the IMPROVED QAOA algorithm:
- Initial state: Dicke state |D_k^N> (equal superposition of all N-particle states)
- Mixer: XY mixer (preserves particle number)
- Result: 100% valid configurations, high ground state probability

ENHANCED FEATURES:
1) Save all plot data as CSV for OriginLab
2) Plot and save expected_N vs iterations
3) Brute force comparison
4) Parameter landscape visualization
5) Circuit visualization

Example runner for Li2Co8O16.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from math import comb, sqrt
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
    create_dicke_initial_state, apply_cost_layer, apply_xy_mixer_layer,
    build_qaoa_circuit, save_circuit_diagram, get_circuit_stats
)
from postprocessing.plotting import plot_bitstring_probability


def compute_expectation_and_n_from_counts(counts, alpha, beta_matrix, E_const, n_qubits):
    """Compute <H> and <N> from measurement counts.

    Note: Using ORIGINAL Hamiltonian (no penalty transformation needed
    since particle number is conserved).
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

        # Calculate energy (original Hamiltonian)
        energy = E_const
        for i, bit in enumerate(bs):
            if bit == '1':
                energy += alpha[i]

        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if bs[i] == '1' and bs[j] == '1':
                    energy += beta_matrix[i, j]

        expected_energy += prob * energy

    return expected_energy, expected_n


def run_dicke_xy_qaoa(alpha, beta_coeff, E_const, n_target, n_qubits, p,
                      shots=8192, maxiter=2500, num_runs=4,
                      use_full_connectivity=True, verbose=True):
    """Run particle-conserving QAOA with Dicke state + XY mixer.

    Args:
        alpha: on-site energies
        beta_coeff: interaction coefficients
        E_const: constant energy
        n_target: target particle number
        n_qubits: number of qubits
        p: QAOA depth
        shots: measurement shots
        maxiter: optimizer iterations
        num_runs: number of optimization runs
        use_full_connectivity: if True, use all-to-all XY mixer
        verbose: print progress

    Returns:
        Tuple of (all_results, alpha, beta_matrix)
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

    backend = AerSimulator()
    all_results = []

    for run_idx in range(num_runs):
        if verbose:
            print(f"\n{'='*60}")
            print(f"RUN {run_idx + 1}/{num_runs}")
            print(f"{'='*60}")

        history = {
            'energies': [],
            'expected_n': [],
            'iterations': [],
            'params': []
        }
        n_evals = [0]

        def objective(params):
            gammas = params[:p]
            betas = params[p:]

            # Create circuit with Dicke state initialization
            qc = create_dicke_initial_state(n_qubits, n_target)

            # Apply p layers of cost + XY mixer
            for layer in range(p):
                apply_cost_layer(qc, gammas[layer], alpha, beta_matrix, n_qubits)
                connectivity = 'full' if use_full_connectivity else 'nearest'
                apply_xy_mixer_layer(qc, betas[layer], n_qubits, connectivity=connectivity)

            qc.measure_all()

            # Run
            transpiled = transpile(qc, backend)
            job = backend.run(transpiled, shots=shots)
            counts = job.result().get_counts()

            # Compute expectation (original Hamiltonian - no penalty needed!)
            energy, exp_n = compute_expectation_and_n_from_counts(
                counts, alpha, beta_matrix, E_const, n_qubits
            )

            # Record history
            history['energies'].append(energy)
            history['expected_n'].append(exp_n)
            history['iterations'].append(n_evals[0])
            history['params'].append(params.tolist())
            n_evals[0] += 1

            if verbose and n_evals[0] % 100 == 0:
                print(f"  Eval {n_evals[0]}: <H> = {energy:.4f} eV, <N> = {exp_n:.4f}")

            return energy

        # Random initial parameters
        np.random.seed(42 + run_idx * 100)
        x0 = np.random.uniform(0, 1.0, 2 * p)

        if verbose:
            print(f"Starting COBYLA optimization...")
            print(f"  Parameters: {2*p} (p={p})")
            print(f"  Shots: {shots}")
            print(f"  Max iterations: {maxiter}")
            print(f"  XY mixer: {'Full connectivity' if use_full_connectivity else 'Nearest-neighbor'}")

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
            print(f"  Final <H>: {result.fun:.4f} eV")
            print(f"  Success: {result.success}")

        # Final measurement with more shots
        final_params = result.x
        gammas = final_params[:p]
        betas = final_params[p:]

        qc = create_dicke_initial_state(n_qubits, n_target)
        for layer in range(p):
            apply_cost_layer(qc, gammas[layer], alpha, beta_matrix, n_qubits)
            connectivity = 'full' if use_full_connectivity else 'nearest'
            apply_xy_mixer_layer(qc, betas[layer], n_qubits, connectivity=connectivity)
        qc.measure_all()

        transpiled = transpile(qc, backend)
        job = backend.run(transpiled, shots=shots * 4)
        final_counts = job.result().get_counts()

        total_shots_final = sum(final_counts.values())
        final_distribution = {bs[::-1]: count/total_shots_final
                             for bs, count in final_counts.items()}

        # Check valid fraction (should be ~100% for Dicke + XY!)
        valid_prob = sum(prob for bs, prob in final_distribution.items()
                        if is_valid_config(bs, n_target))

        # Energy distribution (original energies - no penalty)
        energy_dist = {}
        for bs, prob in final_distribution.items():
            energy = E_const
            for i, bit in enumerate(bs):
                if bit == '1':
                    energy += alpha[i]
            for i in range(n_qubits):
                for j in range(i + 1, n_qubits):
                    if bs[i] == '1' and bs[j] == '1':
                        energy += beta_matrix[i, j]

            energy_rounded = round(energy, 2)  # Finer resolution for valid states
            if energy_rounded not in energy_dist:
                energy_dist[energy_rounded] = 0.0
            energy_dist[energy_rounded] += prob

        min_energy = min(energy_dist.keys())
        gs_prob_summed = energy_dist[min_energy]

        if verbose:
            print(f"\nResults:")
            print(f"  Valid fraction: {valid_prob:.2%}")
            print(f"  GS prob (summed at E={min_energy:.2f}): {gs_prob_summed:.4f}")

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

    return all_results, alpha, beta_matrix


def compute_brute_force_energies(alpha, beta_coeff, E_const, n_qubits):
    """Compute energies for ALL 2^n configurations."""
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


def scan_parameter_landscape(alpha, beta_matrix, E_const, n_qubits, n_target, p,
                              optimal_params, backend, shots=4096,
                              gamma_range=(-np.pi, np.pi), beta_range=(-np.pi, np.pi),
                              n_points=21, use_full_connectivity=True):
    """Scan 2D parameter landscape for Dicke + XY QAOA."""
    gamma_values = np.linspace(gamma_range[0], gamma_range[1], n_points)
    beta_values = np.linspace(beta_range[0], beta_range[1], n_points)

    energy_landscape = np.zeros((n_points, n_points))
    base_params = optimal_params.copy()

    for i, gamma_0 in enumerate(gamma_values):
        for j, beta_0 in enumerate(beta_values):
            params = base_params.copy()
            params[0] = gamma_0
            params[p] = beta_0

            gammas = params[:p]
            betas = params[p:]

            qc = create_dicke_initial_state(n_qubits, n_target)

            for layer in range(p):
                apply_cost_layer(qc, gammas[layer], alpha, beta_matrix, n_qubits)
                connectivity = 'full' if use_full_connectivity else 'nearest'
                apply_xy_mixer_layer(qc, betas[layer], n_qubits, connectivity=connectivity)

            qc.measure_all()

            transpiled = transpile(qc, backend)
            job = backend.run(transpiled, shots=shots)
            counts = job.result().get_counts()

            energy, _ = compute_expectation_and_n_from_counts(
                counts, alpha, beta_matrix, E_const, n_qubits
            )

            energy_landscape[i, j] = energy

    return {
        'gamma_values': gamma_values,
        'beta_values': beta_values,
        'energy_landscape': energy_landscape
    }


def main():
    print("=" * 70)
    print("Particle-Number-Conserving QAOA (Dicke + XY Mixer) - ENHANCED")
    print("=" * 70)

    # ===========================================
    # CONFIGURATION
    # ===========================================
    P_VALUE = 20              # QAOA depth
    N_PARTICLES = 2           # Target particle number
    SHOTS = 8192              # Shots per evaluation
    MAXITER = 10000           # COBYLA iterations
    NUM_RUNS = 4              # Number of runs
    USE_FULL_CONNECTIVITY = True  # Full or nearest-neighbor XY mixer

    # Landscape scan settings
    SCAN_LANDSCAPE = True
    LANDSCAPE_POINTS = 21
    # ===========================================

    # Load coefficients
    data_dir = example_root / "data"
    alpha, beta_coeff, E_const = load_coefficients(str(data_dir))

    alpha_arr = np.array(alpha).flatten()
    n_qubits = len(alpha_arr)

    print(f"\nSystem: {n_qubits} qubits, {N_PARTICLES} particles (conserved)")
    print(f"QAOA depth p = {P_VALUE} ({2*P_VALUE} parameters)")
    print(f"Initial state: Dicke state |D_{N_PARTICLES}^{n_qubits}>")
    print(f"Mixer: XY ({'full connectivity' if USE_FULL_CONNECTIVITY else 'nearest-neighbor'})")
    print(f"Optimizer: COBYLA")
    print(f"Shots: {SHOTS}")
    print(f"Max iterations: {MAXITER}")
    print(f"Number of runs: {NUM_RUNS}")

    # Valid configurations
    n_valid = comb(n_qubits, N_PARTICLES)
    print(f"\nNumber of valid configurations: {n_valid}")

    # Get exact ground state
    ground_state, ground_energy, all_valid_energies = get_exact_solution(
        alpha, beta_coeff, E_const, N_PARTICLES, n_qubits
    )
    print(f"Ground state: {ground_state}")
    print(f"Ground energy: {ground_energy:.4f} eV")

    # Run QAOA
    all_results, alpha_arr, beta_matrix = run_dicke_xy_qaoa(
        alpha, beta_coeff, E_const, N_PARTICLES, n_qubits, P_VALUE,
        shots=SHOTS, maxiter=MAXITER, num_runs=NUM_RUNS,
        use_full_connectivity=USE_FULL_CONNECTIVITY, verbose=True
    )

    # Output directory
    output_dir = example_root / "results" / "simulator" / "dicke_xy"
    output_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # SAVE CIRCUIT DIAGRAMS
    # =========================================================================
    print("\n" + "=" * 70)
    print("Saving circuit diagrams...")
    print("=" * 70)

    try:
        # 1. Initial Dicke state circuit
        initial_qc = create_dicke_initial_state(n_qubits, N_PARTICLES)
        save_circuit_diagram(initial_qc, output_dir / "circuit_01_dicke_initial.png")
        print(f"  Saved: circuit_01_dicke_initial.png")

        # 2. Single QAOA layer
        single_layer_qc = create_dicke_initial_state(n_qubits, N_PARTICLES)
        apply_cost_layer(single_layer_qc, 0.5, alpha_arr, beta_matrix, n_qubits)
        apply_xy_mixer_layer(single_layer_qc, 0.5, n_qubits, connectivity='full')
        save_circuit_diagram(single_layer_qc, output_dir / "circuit_02_single_layer.png")
        print(f"  Saved: circuit_02_single_layer.png")

        # 3. Full ansatz
        example_gammas = [0.5] * P_VALUE
        example_betas = [0.5] * P_VALUE
        full_qc = build_qaoa_circuit(n_qubits, P_VALUE, example_gammas, example_betas,
                                      alpha_arr, beta_matrix,
                                      method='dicke_xy', n_particles=N_PARTICLES,
                                      add_measurements=True)
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
        # Convergence data
        convergence_df = pd.DataFrame({
            'iteration': res['history']['iterations'],
            'energy_eV': res['history']['energies'],
            'expected_N': res['history']['expected_n']
        })
        convergence_df.to_csv(output_dir / f"run_{run_idx+1}_convergence.csv", index=False)

        # Energy distribution
        energy_dist = res['energy_distribution']
        sorted_energies = sorted(energy_dist.keys())
        energy_dist_df = pd.DataFrame({
            'energy_eV': sorted_energies,
            'probability': [energy_dist[e] for e in sorted_energies]
        })
        energy_dist_df.to_csv(output_dir / f"run_{run_idx+1}_energy_distribution.csv", index=False)

        # Final distribution
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

    print(f"  Saved CSV files for all {NUM_RUNS} runs")

    # =========================================================================
    # 2) PLOT EXPECTED_N VS ITERATIONS
    # =========================================================================
    print("\n" + "=" * 70)
    print("2) Plotting expected_N vs iterations...")
    print("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(NUM_RUNS, 2, figsize=(14, 4 * NUM_RUNS))

    if NUM_RUNS == 1:
        axes = [axes]

    for row_idx, res in enumerate(all_results):
        # Left: Energy convergence
        ax_left = axes[row_idx][0]
        ax_left.plot(res['history']['iterations'], res['history']['energies'], 'b-', linewidth=0.5)
        ax_left.axhline(y=ground_energy, color='r', linestyle='--', label=f'Ground E={ground_energy:.2f}')
        ax_left.set_xlabel('Iteration', fontsize=11)
        ax_left.set_ylabel('<H> (eV)', fontsize=11)
        ax_left.set_title(f'Run {row_idx + 1} - Energy Convergence', fontsize=12)
        ax_left.legend()
        ax_left.grid(True, alpha=0.3)

        # Right: Expected N (should be exactly N_PARTICLES!)
        ax_right = axes[row_idx][1]
        ax_right.plot(res['history']['iterations'], res['history']['expected_n'], 'g-', linewidth=0.5)
        ax_right.axhline(y=N_PARTICLES, color='r', linestyle='--', label=f'Target N={N_PARTICLES}')
        ax_right.set_xlabel('Iteration', fontsize=11)
        ax_right.set_ylabel('<N>', fontsize=11)
        ax_right.set_title(f'Run {row_idx + 1} - Particle Number (should be constant!)', fontsize=12)
        ax_right.set_ylim(N_PARTICLES - 0.5, N_PARTICLES + 0.5)
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
    sorted_configs = sorted(brute_force_configs.items(), key=lambda x: x[1][0])

    # Save brute force data
    brute_force_df = pd.DataFrame({
        'bitstring': [cfg[0] for cfg in sorted_configs],
        'energy_eV': [cfg[1][0] for cfg in sorted_configs],
        'n_particles': [cfg[1][1] for cfg in sorted_configs],
        'is_valid': [cfg[1][1] == N_PARTICLES for cfg in sorted_configs]
    })
    brute_force_df.to_csv(output_dir / "brute_force_all_configs.csv", index=False)

    valid_configs = [(bs, e, n) for bs, (e, n) in sorted_configs if n == N_PARTICLES]
    valid_configs_df = pd.DataFrame({
        'bitstring': [cfg[0] for cfg in valid_configs],
        'energy_eV': [cfg[1] for cfg in valid_configs]
    })
    valid_configs_df.to_csv(output_dir / "brute_force_valid_configs.csv", index=False)

    print(f"  Total configurations: {len(sorted_configs)}")
    print(f"  Valid configurations (N={N_PARTICLES}): {len(valid_configs)}")
    print(f"  Ground state: {valid_configs[0][0]} with E={valid_configs[0][1]:.4f} eV")

    # Plot brute force
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Valid configs only (sorted by energy)
    valid_energies = [cfg[1] for cfg in valid_configs]
    valid_bitstrings = [cfg[0] for cfg in valid_configs]

    ax1.bar(range(len(valid_energies)), valid_energies, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Configuration Index', fontsize=11)
    ax1.set_ylabel('Energy (eV)', fontsize=11)
    ax1.set_title(f'Valid Configurations (N={N_PARTICLES}) - Sorted by Energy', fontsize=12)
    ax1.axhline(y=ground_energy, color='r', linestyle='--', label=f'Ground E={ground_energy:.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Energy histogram of valid configs
    ax2.hist(valid_energies, bins=20, color='steelblue', edgecolor='black')
    ax2.axvline(x=ground_energy, color='r', linestyle='--', label=f'Ground E={ground_energy:.2f}')
    ax2.set_xlabel('Energy (eV)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Energy Distribution of Valid Configurations', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "brute_force_comparison.png", dpi=150)
    plt.close()
    print(f"  Saved: brute_force_comparison.png")

    # QAOA vs brute force comparison
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
        best_run_idx = np.argmax([res['gs_prob_summed'] for res in all_results])
        best_params = np.array(all_results[best_run_idx]['final_params'])

        print(f"  Using parameters from Run {best_run_idx + 1} (best GS probability)")
        print(f"  Scanning {LANDSCAPE_POINTS}x{LANDSCAPE_POINTS} grid...")

        landscape_data = scan_parameter_landscape(
            alpha_arr, beta_matrix, E_const, n_qubits, N_PARTICLES, P_VALUE,
            best_params, backend, shots=2048, n_points=LANDSCAPE_POINTS,
            use_full_connectivity=USE_FULL_CONNECTIVITY
        )

        # Save landscape data
        landscape_df = pd.DataFrame(landscape_data['energy_landscape'],
                                     index=landscape_data['gamma_values'],
                                     columns=landscape_data['beta_values'])
        landscape_df.to_csv(output_dir / "parameter_landscape.csv")

        flat_landscape = []
        for i, gamma in enumerate(landscape_data['gamma_values']):
            for j, beta in enumerate(landscape_data['beta_values']):
                flat_landscape.append({
                    'gamma_0': gamma,
                    'beta_0': beta,
                    'energy_eV': landscape_data['energy_landscape'][i, j]
                })
        flat_landscape_df = pd.DataFrame(flat_landscape)
        flat_landscape_df.to_csv(output_dir / "parameter_landscape_flat.csv", index=False)

        # Plot 2D heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(landscape_data['energy_landscape'],
                       extent=[landscape_data['beta_values'][0], landscape_data['beta_values'][-1],
                               landscape_data['gamma_values'][-1], landscape_data['gamma_values'][0]],
                       aspect='auto', cmap='viridis')

        ax.scatter([best_params[P_VALUE]], [best_params[0]],
                   c='red', s=100, marker='*', label='Optimal')

        ax.set_xlabel('beta_0', fontsize=12)
        ax.set_ylabel('gamma_0', fontsize=12)
        ax.set_title('Parameter Landscape (Dicke + XY QAOA)', fontsize=14)
        plt.colorbar(im, ax=ax, label='<H> (eV)')
        ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "parameter_landscape.png", dpi=150)
        plt.close()

        # 3D plot
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        X, Y = np.meshgrid(landscape_data['beta_values'], landscape_data['gamma_values'])
        ax.plot_surface(X, Y, landscape_data['energy_landscape'], cmap='viridis', alpha=0.8)

        ax.set_xlabel('beta_0', fontsize=11)
        ax.set_ylabel('gamma_0', fontsize=11)
        ax.set_zlabel('<H> (eV)', fontsize=11)
        ax.set_title('Parameter Landscape 3D (Dicke + XY QAOA)', fontsize=14)

        plt.tight_layout()
        plt.savefig(output_dir / "parameter_landscape_3d.png", dpi=150)
        plt.close()
        print(f"  Saved: parameter_landscape.png, parameter_landscape_3d.png")

    # =========================================================================
    # PROFESSOR-STYLE PLOTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("Generating Professor-style plots...")
    print("=" * 70)

    fig, axes = plt.subplots(NUM_RUNS, 2, figsize=(14, 4 * NUM_RUNS))

    if NUM_RUNS == 1:
        axes = [axes]

    for row_idx, res in enumerate(all_results):
        # Left: Energy convergence
        ax_left = axes[row_idx][0]
        ax_left.plot(res['history']['iterations'], res['history']['energies'], 'b-', linewidth=0.5)
        ax_left.axhline(y=ground_energy, color='r', linestyle='--', alpha=0.7)
        ax_left.set_xlabel('Iteration', fontsize=11)
        ax_left.set_ylabel('<H> (eV)', fontsize=11)
        ax_left.set_title(f'Run {row_idx + 1}', fontsize=12)
        ax_left.grid(True, alpha=0.3)

        # Right: Energy distribution (should show concentrated probability!)
        ax_right = axes[row_idx][1]
        energy_dist = res['energy_distribution']
        sorted_energies = sorted(energy_dist.keys())
        sorted_probs = [energy_dist[e] for e in sorted_energies]

        x_pos = np.arange(len(sorted_energies))
        ax_right.bar(x_pos, sorted_probs, color='black', edgecolor='black', linewidth=0.3)

        n_labels = min(15, len(sorted_energies))
        step = max(1, len(sorted_energies) // n_labels)
        tick_positions = x_pos[::step]
        tick_labels = [f'{sorted_energies[i]:.2f}' for i in range(0, len(sorted_energies), step)]
        ax_right.set_xticks(tick_positions)
        ax_right.set_xticklabels(tick_labels, rotation=90, fontsize=8)

        ax_right.set_xlabel('E(X) (eV)', fontsize=11)
        ax_right.set_ylabel('p(X)', fontsize=11)
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
    print("SUMMARY - Dicke + XY QAOA (Particle-Number Conserving)")
    print("=" * 70)
    print(f"{'Run':<6} {'Valid %':<12} {'GS Prob (summed)':<18} {'Final <H> (eV)':<15} {'Time (s)':<10}")
    print("-" * 70)
    for res in all_results:
        print(f"{res['run_idx']+1:<6} {res['valid_fraction']*100:<12.2f} {res['gs_prob_summed']:<18.4f} {res['final_cost']:<15.4f} {res['elapsed_time']:<10.1f}")

    avg_valid = np.mean([res['valid_fraction'] for res in all_results])
    avg_gs_prob = np.mean([res['gs_prob_summed'] for res in all_results])
    avg_energy = np.mean([res['final_cost'] for res in all_results])

    print("-" * 70)
    print(f"{'AVG':<6} {avg_valid*100:<12.2f} {avg_gs_prob:<18.4f} {avg_energy:<15.4f}")

    print(f"\nGround state: {ground_state}")
    print(f"Ground energy: {ground_energy:.4f} eV")
    print(f"Valid configurations: {n_valid}")

    # Save summary
    summary = {
        'settings': {
            'method': 'dicke_xy',
            'initial_state': 'dicke',
            'mixer': 'xy_full' if USE_FULL_CONNECTIVITY else 'xy_nearest_neighbor',
            'p': P_VALUE,
            'n_parameters': 2 * P_VALUE,
            'n_particles': N_PARTICLES,
            'shots': SHOTS,
            'maxiter': MAXITER,
            'num_runs': NUM_RUNS
        },
        'ground_state': ground_state,
        'ground_energy': float(ground_energy),
        'n_valid_configs': n_valid,
        'results': [
            {
                'run': res['run_idx'] + 1,
                'valid_fraction': res['valid_fraction'],
                'gs_prob_summed': res['gs_prob_summed'],
                'gs_energy': res['gs_energy'],
                'final_energy': res['final_cost'],
                'n_evals': res['n_evals'],
                'elapsed_time': res['elapsed_time']
            }
            for res in all_results
        ],
        'averages': {
            'valid_fraction': float(avg_valid),
            'gs_prob_summed': float(avg_gs_prob),
            'final_energy': float(avg_energy)
        }
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs saved to: {output_dir}")
    print("\nCSV files for OriginLab:")
    print("  - run_X_convergence.csv")
    print("  - run_X_energy_distribution.csv")
    print("  - run_X_final_distribution.csv")
    print("  - run_X_final_params.csv")
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
