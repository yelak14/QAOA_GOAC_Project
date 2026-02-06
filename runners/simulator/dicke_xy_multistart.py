"""Particle-Number-Conserving QAOA with Multi-Start Optimization.

MULTI-START STRATEGY:
1. Stage 1: Run many SHORT optimizations (explore many starting points)
2. Stage 2: Select the BEST result from Stage 1
3. Stage 3: Continue optimizing from that best point (fine-tune)

This approach is more reliable than running a few long independent optimizations.

Features:
- Dicke state initialization (particle number conserving)
- XY mixer (preserves particle number)
- Multi-start optimization for better consistency
- All enhanced features (CSV export, plots, circuit visualization)
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import combinations
from math import comb, sqrt
import time

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

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


def compute_expectation_and_n_from_counts(counts, alpha, beta_matrix, E_const, n_qubits):
    """Compute <H> and <N> from measurement counts."""
    total_shots = sum(counts.values())
    expected_energy = 0.0
    expected_n = 0.0

    for bitstring, count in counts.items():
        prob = count / total_shots
        bs = bitstring[::-1]

        n_particles = bs.count('1')
        expected_n += prob * n_particles

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


def run_single_optimization(alpha, beta_matrix, E_const, n_target, n_qubits, p,
                            initial_params, backend, shots=8192, maxiter=500):
    """Run a single QAOA optimization from given initial parameters.

    Returns:
        dict with final_params, final_energy, n_evals, history
    """
    history = {'energies': [], 'expected_n': []}
    n_evals = [0]

    def objective(params):
        gammas = params[:p]
        betas = params[p:]

        qc = create_dicke_initial_state(n_qubits, n_target)

        for layer in range(p):
            apply_cost_layer(qc, gammas[layer], alpha, beta_matrix, n_qubits)
            apply_xy_mixer_layer(qc, betas[layer], n_qubits, connectivity='full')

        qc.measure_all()

        transpiled = transpile(qc, backend)
        job = backend.run(transpiled, shots=shots)
        counts = job.result().get_counts()

        energy, exp_n = compute_expectation_and_n_from_counts(
            counts, alpha, beta_matrix, E_const, n_qubits
        )

        history['energies'].append(energy)
        history['expected_n'].append(exp_n)
        n_evals[0] += 1

        return energy

    result = minimize(
        objective,
        initial_params,
        method='COBYLA',
        options={
            'maxiter': maxiter,
            'rhobeg': 0.5,
            'disp': False
        }
    )

    return {
        'final_params': result.x,
        'final_energy': result.fun,
        'n_evals': n_evals[0],
        'history': history,
        'success': result.success
    }


def run_multistart_qaoa(alpha, beta_coeff, E_const, n_target, n_qubits, p,
                        shots=8192,
                        stage1_runs=20, stage1_maxiter=500,
                        stage2_maxiter=5000,
                        verbose=True):
    """Run multi-start QAOA optimization.

    Stage 1: Run many short optimizations to explore the landscape
    Stage 2: Continue from the best result for fine-tuning

    Args:
        alpha: on-site energies
        beta_coeff: interaction coefficients
        E_const: constant energy
        n_target: target particle number
        n_qubits: number of qubits
        p: QAOA depth
        shots: measurement shots
        stage1_runs: number of short runs in Stage 1
        stage1_maxiter: max iterations per Stage 1 run
        stage2_maxiter: max iterations for Stage 2 refinement
        verbose: print progress

    Returns:
        dict with all results
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

    # =========================================================================
    # STAGE 1: Many short optimizations
    # =========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("STAGE 1: Exploring landscape with many short runs")
        print(f"  Runs: {stage1_runs}")
        print(f"  Max iterations per run: {stage1_maxiter}")
        print("=" * 60)

    stage1_results = []
    stage1_start_time = time.time()

    for run_idx in range(stage1_runs):
        # Random initial parameters
        np.random.seed(42 + run_idx * 37)  # Different seed for each run
        x0 = np.random.uniform(0, 2 * np.pi, 2 * p)

        result = run_single_optimization(
            alpha, beta_matrix, E_const, n_target, n_qubits, p,
            x0, backend, shots=shots, maxiter=stage1_maxiter
        )

        stage1_results.append({
            'run_idx': run_idx,
            'initial_params': x0.tolist(),
            'final_params': result['final_params'].tolist(),
            'final_energy': result['final_energy'],
            'n_evals': result['n_evals'],
            'history': result['history']
        })

        if verbose:
            print(f"  Run {run_idx + 1:2d}/{stage1_runs}: <H> = {result['final_energy']:.4f} eV")

    stage1_time = time.time() - stage1_start_time

    # Find best result from Stage 1
    best_stage1_idx = np.argmin([r['final_energy'] for r in stage1_results])
    best_stage1 = stage1_results[best_stage1_idx]

    if verbose:
        print(f"\n  Best from Stage 1: Run {best_stage1_idx + 1}")
        print(f"  Best energy: {best_stage1['final_energy']:.4f} eV")
        print(f"  Stage 1 time: {stage1_time:.1f}s")

    # =========================================================================
    # STAGE 2: Fine-tune from best result
    # =========================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("STAGE 2: Fine-tuning from best Stage 1 result")
        print(f"  Starting from Run {best_stage1_idx + 1}")
        print(f"  Max iterations: {stage2_maxiter}")
        print("=" * 60)

    stage2_start_time = time.time()

    # Continue optimization from best Stage 1 parameters
    stage2_history = {'energies': [], 'expected_n': [], 'iterations': []}
    n_evals = [0]

    def objective_stage2(params):
        gammas = params[:p]
        betas = params[p:]

        qc = create_dicke_initial_state(n_qubits, n_target)

        for layer in range(p):
            apply_cost_layer(qc, gammas[layer], alpha, beta_matrix, n_qubits)
            apply_xy_mixer_layer(qc, betas[layer], n_qubits, connectivity='full')

        qc.measure_all()

        transpiled = transpile(qc, backend)
        job = backend.run(transpiled, shots=shots)
        counts = job.result().get_counts()

        energy, exp_n = compute_expectation_and_n_from_counts(
            counts, alpha, beta_matrix, E_const, n_qubits
        )

        stage2_history['energies'].append(energy)
        stage2_history['expected_n'].append(exp_n)
        stage2_history['iterations'].append(n_evals[0])
        n_evals[0] += 1

        if verbose and n_evals[0] % 100 == 0:
            print(f"  Eval {n_evals[0]}: <H> = {energy:.4f} eV")

        return energy

    # Start from best Stage 1 parameters
    x0_stage2 = np.array(best_stage1['final_params'])

    stage2_result = minimize(
        objective_stage2,
        x0_stage2,
        method='COBYLA',
        options={
            'maxiter': stage2_maxiter,
            'rhobeg': 0.1,  # Smaller step size for fine-tuning
            'disp': False
        }
    )

    stage2_time = time.time() - stage2_start_time

    if verbose:
        print(f"\n  Stage 2 finished in {stage2_time:.1f}s")
        print(f"  Final <H>: {stage2_result.fun:.4f} eV")
        print(f"  Evaluations: {n_evals[0]}")

    # =========================================================================
    # Final measurement
    # =========================================================================
    final_params = stage2_result.x
    gammas = final_params[:p]
    betas = final_params[p:]

    qc = create_dicke_initial_state(n_qubits, n_target)
    for layer in range(p):
        apply_cost_layer(qc, gammas[layer], alpha, beta_matrix, n_qubits)
        apply_xy_mixer_layer(qc, betas[layer], n_qubits, connectivity='full')
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
        energy = E_const
        for i, bit in enumerate(bs):
            if bit == '1':
                energy += alpha[i]
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                if bs[i] == '1' and bs[j] == '1':
                    energy += beta_matrix[i, j]

        energy_rounded = round(energy, 2)
        if energy_rounded not in energy_dist:
            energy_dist[energy_rounded] = 0.0
        energy_dist[energy_rounded] += prob

    min_energy = min(energy_dist.keys())
    gs_prob_summed = energy_dist[min_energy]

    if verbose:
        print(f"\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"  Valid fraction: {valid_prob:.2%}")
        print(f"  GS prob (summed at E={min_energy:.2f}): {gs_prob_summed:.4f}")
        print(f"  Total time: {stage1_time + stage2_time:.1f}s")

    return {
        'stage1_results': stage1_results,
        'best_stage1_idx': best_stage1_idx,
        'stage2_history': stage2_history,
        'final_params': final_params.tolist(),
        'final_energy': stage2_result.fun,
        'final_distribution': final_distribution,
        'energy_distribution': energy_dist,
        'valid_fraction': valid_prob,
        'gs_prob_summed': gs_prob_summed,
        'gs_energy': min_energy,
        'stage1_time': stage1_time,
        'stage2_time': stage2_time,
        'total_time': stage1_time + stage2_time,
        'alpha': alpha,
        'beta_matrix': beta_matrix
    }


def compute_brute_force_energies(alpha, beta_matrix, E_const, n_qubits):
    """Compute energies for ALL 2^n configurations."""
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
                              n_points=21):
    """Scan 2D parameter landscape."""
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
                apply_xy_mixer_layer(qc, betas[layer], n_qubits, connectivity='full')

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
    print("Dicke + XY QAOA with MULTI-START OPTIMIZATION")
    print("=" * 70)

    # ===========================================
    # CONFIGURATION
    # ===========================================
    P_VALUE = 20              # QAOA depth
    N_PARTICLES = 2           # Target particle number
    SHOTS = 8192              # Shots per evaluation

    # Multi-start settings
    STAGE1_RUNS = 20          # Number of short runs in Stage 1
    STAGE1_MAXITER = 500      # Max iterations per Stage 1 run
    STAGE2_MAXITER = 5000     # Max iterations for Stage 2 refinement

    # Landscape scan settings
    SCAN_LANDSCAPE = True
    LANDSCAPE_POINTS = 21
    # ===========================================

    # Load coefficients
    data_dir = project_root / "data" / "input"
    alpha, beta_coeff, E_const = load_coefficients(str(data_dir))

    alpha_arr = np.array(alpha).flatten()
    n_qubits = len(alpha_arr)

    print(f"\nSystem: {n_qubits} qubits, {N_PARTICLES} particles (conserved)")
    print(f"QAOA depth p = {P_VALUE} ({2*P_VALUE} parameters)")
    print(f"Initial state: Dicke state |D_{N_PARTICLES}^{n_qubits}>")
    print(f"Mixer: XY (full connectivity)")
    print(f"\nMulti-start settings:")
    print(f"  Stage 1: {STAGE1_RUNS} runs x {STAGE1_MAXITER} iterations")
    print(f"  Stage 2: {STAGE2_MAXITER} iterations (fine-tuning)")

    # Valid configurations
    n_valid = comb(n_qubits, N_PARTICLES)
    print(f"\nNumber of valid configurations: {n_valid}")

    # Get exact ground state
    ground_state, ground_energy, all_valid_energies = get_exact_solution(
        alpha, beta_coeff, E_const, N_PARTICLES, n_qubits
    )
    print(f"Ground state: {ground_state}")
    print(f"Ground energy: {ground_energy:.4f} eV")

    # Run multi-start QAOA
    results = run_multistart_qaoa(
        alpha, beta_coeff, E_const, N_PARTICLES, n_qubits, P_VALUE,
        shots=SHOTS,
        stage1_runs=STAGE1_RUNS, stage1_maxiter=STAGE1_MAXITER,
        stage2_maxiter=STAGE2_MAXITER,
        verbose=True
    )

    # Output directory
    output_dir = project_root / "results" / "simulator" / "dicke_xy_multistart"
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
        apply_cost_layer(single_layer_qc, 0.5, results['alpha'], results['beta_matrix'], n_qubits)
        apply_xy_mixer_layer(single_layer_qc, 0.5, n_qubits, connectivity='full')
        save_circuit_diagram(single_layer_qc, output_dir / "circuit_02_single_layer.png")
        print(f"  Saved: circuit_02_single_layer.png")

        # 3. Full ansatz
        example_gammas = [0.5] * P_VALUE
        example_betas = [0.5] * P_VALUE
        full_qc = build_qaoa_circuit(n_qubits, P_VALUE, example_gammas, example_betas,
                                      results['alpha'], results['beta_matrix'],
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

    # Stage 1 summary
    stage1_summary = pd.DataFrame({
        'run': [r['run_idx'] + 1 for r in results['stage1_results']],
        'final_energy': [r['final_energy'] for r in results['stage1_results']],
        'n_evals': [r['n_evals'] for r in results['stage1_results']]
    })
    stage1_summary.to_csv(output_dir / "stage1_summary.csv", index=False)

    # Stage 2 convergence
    stage2_convergence = pd.DataFrame({
        'iteration': results['stage2_history']['iterations'],
        'energy_eV': results['stage2_history']['energies'],
        'expected_N': results['stage2_history']['expected_n']
    })
    stage2_convergence.to_csv(output_dir / "stage2_convergence.csv", index=False)

    # Combined convergence (Stage 1 best + Stage 2)
    best_stage1 = results['stage1_results'][results['best_stage1_idx']]
    combined_energies = best_stage1['history']['energies'] + results['stage2_history']['energies']
    combined_iterations = list(range(len(combined_energies)))
    stage1_end = len(best_stage1['history']['energies'])

    combined_convergence = pd.DataFrame({
        'iteration': combined_iterations,
        'energy_eV': combined_energies,
        'stage': ['Stage 1'] * stage1_end + ['Stage 2'] * len(results['stage2_history']['energies'])
    })
    combined_convergence.to_csv(output_dir / "combined_convergence.csv", index=False)

    # Energy distribution
    energy_dist = results['energy_distribution']
    sorted_energies = sorted(energy_dist.keys())
    energy_dist_df = pd.DataFrame({
        'energy_eV': sorted_energies,
        'probability': [energy_dist[e] for e in sorted_energies]
    })
    energy_dist_df.to_csv(output_dir / "final_energy_distribution.csv", index=False)

    # Final distribution
    final_dist = results['final_distribution']
    final_dist_df = pd.DataFrame({
        'bitstring': list(final_dist.keys()),
        'probability': list(final_dist.values()),
        'n_particles': [bs.count('1') for bs in final_dist.keys()]
    })
    final_dist_df.to_csv(output_dir / "final_state_distribution.csv", index=False)

    # Final parameters
    params_df = pd.DataFrame({
        'param_index': list(range(len(results['final_params']))),
        'param_value': results['final_params']
    })
    params_df.to_csv(output_dir / "final_params.csv", index=False)

    print(f"  Saved CSV files")

    # =========================================================================
    # 2) PLOT STAGE 1 COMPARISON + STAGE 2 CONVERGENCE
    # =========================================================================
    print("\n" + "=" * 70)
    print("2) Plotting convergence...")
    print("=" * 70)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Stage 1 final energies (bar chart)
    ax1 = axes[0, 0]
    stage1_energies = [r['final_energy'] for r in results['stage1_results']]
    colors = ['red' if i == results['best_stage1_idx'] else 'steelblue'
              for i in range(len(stage1_energies))]
    ax1.bar(range(1, len(stage1_energies) + 1), stage1_energies, color=colors, edgecolor='black')
    ax1.axhline(y=ground_energy, color='green', linestyle='--', label=f'Ground E={ground_energy:.2f}')
    ax1.set_xlabel('Stage 1 Run', fontsize=11)
    ax1.set_ylabel('Final <H> (eV)', fontsize=11)
    ax1.set_title('Stage 1: Exploring Landscape (red = best)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Top-right: Stage 2 convergence
    ax2 = axes[0, 1]
    ax2.plot(results['stage2_history']['iterations'], results['stage2_history']['energies'],
             'b-', linewidth=0.8)
    ax2.axhline(y=ground_energy, color='green', linestyle='--', label=f'Ground E={ground_energy:.2f}')
    ax2.set_xlabel('Iteration', fontsize=11)
    ax2.set_ylabel('<H> (eV)', fontsize=11)
    ax2.set_title('Stage 2: Fine-tuning from Best', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Bottom-left: Combined convergence (Stage 1 + Stage 2)
    ax3 = axes[1, 0]
    ax3.plot(range(stage1_end), best_stage1['history']['energies'], 'b-', linewidth=0.8, label='Stage 1')
    ax3.plot(range(stage1_end, len(combined_energies)), results['stage2_history']['energies'],
             'r-', linewidth=0.8, label='Stage 2')
    ax3.axhline(y=ground_energy, color='green', linestyle='--', label=f'Ground E={ground_energy:.2f}')
    ax3.axvline(x=stage1_end, color='gray', linestyle=':', alpha=0.7)
    ax3.set_xlabel('Iteration', fontsize=11)
    ax3.set_ylabel('<H> (eV)', fontsize=11)
    ax3.set_title('Combined Convergence (Stage 1 -> Stage 2)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Bottom-right: Final energy distribution
    ax4 = axes[1, 1]
    x_pos = np.arange(len(sorted_energies))
    ax4.bar(x_pos, [energy_dist[e] for e in sorted_energies], color='black', edgecolor='black')

    n_labels = min(15, len(sorted_energies))
    step = max(1, len(sorted_energies) // n_labels)
    tick_positions = x_pos[::step]
    tick_labels = [f'{sorted_energies[i]:.2f}' for i in range(0, len(sorted_energies), step)]
    ax4.set_xticks(tick_positions)
    ax4.set_xticklabels(tick_labels, rotation=90, fontsize=8)

    ax4.set_xlabel('E(X) (eV)', fontsize=11)
    ax4.set_ylabel('p(X)', fontsize=11)
    ax4.set_title(f'Final Distribution (Valid: {results["valid_fraction"]:.1%}, GS: {results["gs_prob_summed"]:.3f})', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "multistart_convergence.png", dpi=150)
    plt.close()
    print(f"  Saved: multistart_convergence.png")

    # =========================================================================
    # 3) BRUTE FORCE COMPARISON
    # =========================================================================
    print("\n" + "=" * 70)
    print("3) Computing brute force comparison...")
    print("=" * 70)

    brute_force_configs = compute_brute_force_energies(
        results['alpha'], results['beta_matrix'], E_const, n_qubits
    )
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

    # QAOA vs brute force comparison
    comparison_data = []
    for bs, prob in results['final_distribution'].items():
        bf_energy, bf_n = brute_force_configs[bs]
        comparison_data.append({
            'bitstring': bs,
            'qaoa_probability': prob,
            'brute_force_energy': bf_energy,
            'n_particles': bf_n,
            'is_valid': bf_n == N_PARTICLES
        })

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / "qaoa_vs_brute_force_comparison.csv", index=False)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    valid_energies = [cfg[1] for cfg in valid_configs]
    ax1.bar(range(len(valid_energies)), valid_energies, color='steelblue', edgecolor='black')
    ax1.axhline(y=ground_energy, color='r', linestyle='--', label=f'Ground E={ground_energy:.2f}')
    ax1.set_xlabel('Configuration Index', fontsize=11)
    ax1.set_ylabel('Energy (eV)', fontsize=11)
    ax1.set_title(f'Valid Configurations (N={N_PARTICLES})', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

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

    # =========================================================================
    # 4) PARAMETER LANDSCAPE
    # =========================================================================
    if SCAN_LANDSCAPE:
        print("\n" + "=" * 70)
        print("4) Scanning parameter landscape...")
        print("=" * 70)

        backend = AerSimulator()

        print(f"  Scanning {LANDSCAPE_POINTS}x{LANDSCAPE_POINTS} grid...")

        landscape_data = scan_parameter_landscape(
            results['alpha'], results['beta_matrix'], E_const,
            n_qubits, N_PARTICLES, P_VALUE,
            np.array(results['final_params']), backend,
            shots=2048, n_points=LANDSCAPE_POINTS
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

        final_params = np.array(results['final_params'])
        ax.scatter([final_params[P_VALUE]], [final_params[0]],
                   c='red', s=100, marker='*', label='Optimal')

        ax.set_xlabel('beta_0', fontsize=12)
        ax.set_ylabel('gamma_0', fontsize=12)
        ax.set_title('Parameter Landscape (Multi-start Dicke + XY)', fontsize=14)
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
        ax.set_title('Parameter Landscape 3D', fontsize=14)

        plt.tight_layout()
        plt.savefig(output_dir / "parameter_landscape_3d.png", dpi=150)
        plt.close()
        print(f"  Saved: parameter_landscape.png, parameter_landscape_3d.png")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY - Multi-Start Dicke + XY QAOA")
    print("=" * 70)
    print(f"\nStage 1: {STAGE1_RUNS} runs x {STAGE1_MAXITER} iterations")
    print(f"  Best run: {results['best_stage1_idx'] + 1}")
    print(f"  Best energy: {results['stage1_results'][results['best_stage1_idx']]['final_energy']:.4f} eV")
    print(f"  Time: {results['stage1_time']:.1f}s")

    print(f"\nStage 2: {STAGE2_MAXITER} iterations")
    print(f"  Final energy: {results['final_energy']:.4f} eV")
    print(f"  Time: {results['stage2_time']:.1f}s")

    print(f"\nFinal Results:")
    print(f"  Valid fraction: {results['valid_fraction']:.2%}")
    print(f"  GS prob (summed): {results['gs_prob_summed']:.4f}")
    print(f"  Total time: {results['total_time']:.1f}s")

    print(f"\nGround state: {ground_state}")
    print(f"Ground energy: {ground_energy:.4f} eV")

    # Save summary (convert numpy types to native Python types for JSON)
    summary = {
        'settings': {
            'method': 'dicke_xy_multistart',
            'p': int(P_VALUE),
            'n_particles': int(N_PARTICLES),
            'shots': int(SHOTS),
            'stage1_runs': int(STAGE1_RUNS),
            'stage1_maxiter': int(STAGE1_MAXITER),
            'stage2_maxiter': int(STAGE2_MAXITER)
        },
        'ground_state': ground_state,
        'ground_energy': float(ground_energy),
        'results': {
            'best_stage1_run': int(results['best_stage1_idx'] + 1),
            'best_stage1_energy': float(results['stage1_results'][results['best_stage1_idx']]['final_energy']),
            'final_energy': float(results['final_energy']),
            'valid_fraction': float(results['valid_fraction']),
            'gs_prob_summed': float(results['gs_prob_summed']),
            'stage1_time': float(results['stage1_time']),
            'stage2_time': float(results['stage2_time']),
            'total_time': float(results['total_time'])
        }
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll outputs saved to: {output_dir}")
    print("\nCSV files for OriginLab:")
    print("  - stage1_summary.csv")
    print("  - stage2_convergence.csv")
    print("  - combined_convergence.csv")
    print("  - final_energy_distribution.csv")
    print("  - final_state_distribution.csv")
    print("  - final_params.csv")
    print("  - brute_force_all_configs.csv")
    print("  - brute_force_valid_configs.csv")
    print("  - parameter_landscape.csv")
    print("  - parameter_landscape_flat.csv")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
