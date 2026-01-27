"""QAOA with constraint-preserving XY mixer on AerSimulator."""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from qaoa.utils import load_coefficients, is_valid_config
from qaoa.hamiltonian import build_cost_hamiltonian, evaluate_energy, get_exact_solution
from qaoa.circuit import build_qaoa_circuit
from qaoa.optimizer import run_optimization, cvar_cost_function


def main():
    print("=" * 60)
    print("QAOA with XY (Constraint-Preserving) Mixer (Simulator)")
    print("=" * 60)

    # Load GOAC coefficients
    data_dir = project_root / "data" / "input"
    alpha, beta, E_const = load_coefficients(str(data_dir))
    n_qubits = len(alpha)
    n_particles = 2

    print(f"\nSystem: {n_qubits} qubits, {n_particles} particles")
    print(f"E_const = {E_const:.3f} eV")
    print(f"Initial state: |{'1' * n_particles + '0' * (n_qubits - n_particles)}>")

    # Get exact solution for reference
    print("\n--- Exact Solution (brute-force) ---")
    ground_state, ground_energy, all_energies = get_exact_solution(
        alpha, beta, E_const, n_particles, n_qubits
    )
    max_energy = max(all_energies.values())
    print(f"Ground state: {ground_state} (sites {[i for i, b in enumerate(ground_state) if b == '1']})")
    print(f"Ground energy: {ground_energy:.6f} eV")

    # Build cost Hamiltonian
    hamiltonian = build_cost_hamiltonian(alpha, beta, E_const)

    # Run QAOA for different depths
    results_by_p = {}
    for p in [1, 2, 3]:
        print(f"\n--- QAOA p={p} (XY Mixer) ---")
        circuit, gammas, betas = build_qaoa_circuit(
            hamiltonian, mixer_type='xy', p=p,
            n_qubits=n_qubits, n_particles=n_particles
        )
        print(f"Circuit: {circuit.num_qubits} qubits, depth={circuit.depth()}")

        result = run_optimization(
            circuit, gammas, betas, alpha, beta, E_const,
            shots=4096, maxiter=150, cvar_alpha=0.2, p=p,
            n_particles=n_particles
        )

        # Analyze results
        counts = result['optimal_counts']
        total_shots = sum(counts.values())
        valid_counts = {k: v for k, v in counts.items() if is_valid_config(k[::-1], n_particles)}
        valid_fraction = sum(valid_counts.values()) / total_shots

        # Find most probable valid configuration
        if valid_counts:
            best_bitstring = max(valid_counts, key=valid_counts.get)
            best_bs_corrected = best_bitstring[::-1]
            best_energy = evaluate_energy(best_bs_corrected, alpha, beta, E_const)
            best_prob = valid_counts[best_bitstring] / total_shots
        else:
            best_bs_corrected = "N/A"
            best_energy = float('inf')
            best_prob = 0.0

        print(f"Optimal cost (CVaR): {result['optimal_cost']:.6f} eV")
        print(f"Valid configurations: {valid_fraction:.1%}")
        print(f"Best valid config: {best_bs_corrected} (p={best_prob:.3f})")
        print(f"Best config energy: {best_energy:.6f} eV")
        print(f"Evaluations: {result['n_evals']}")

        # Check constraint preservation
        invalid_fraction = 1.0 - valid_fraction
        if invalid_fraction < 0.01:
            print(f"  [OK] Constraint preserved (invalid: {invalid_fraction:.3%})")
        else:
            print(f"  [WARN] Constraint violation detected (invalid: {invalid_fraction:.1%})")

        results_by_p[p] = {
            'result': result,
            'valid_fraction': valid_fraction,
            'best_config': best_bs_corrected,
            'best_energy': best_energy,
            'best_prob': best_prob
        }

    # --- Parameter landscape scan (p=1) ---
    print("\nComputing parameter landscape (p=1)...")
    from qiskit_aer import AerSimulator
    from qiskit import transpile as qk_transpile

    n_grid = 15
    gamma_range = np.linspace(0, 2 * np.pi, n_grid)
    beta_scan_range = np.linspace(0, np.pi, n_grid)
    cost_landscape = np.zeros((n_grid, n_grid))

    circuit_p1, gammas_p1, betas_p1 = build_qaoa_circuit(
        hamiltonian, mixer_type='xy', p=1,
        n_qubits=n_qubits, n_particles=n_particles
    )
    sim_backend = AerSimulator()

    for gi, g_val in enumerate(gamma_range):
        for bi, b_val in enumerate(beta_scan_range):
            pdict = {gammas_p1[0]: g_val, betas_p1[0]: b_val}
            bound = circuit_p1.assign_parameters(pdict)
            t_circ = qk_transpile(bound, sim_backend)
            job = sim_backend.run(t_circ, shots=1024)
            c = job.result().get_counts()
            cost_landscape[gi, bi] = cvar_cost_function(
                c, alpha, beta, E_const, cvar_alpha=0.2
            )
    print("Parameter landscape computed.")

    # Save results
    output_dir = project_root / "results" / "simulator" / "constrained"
    output_dir.mkdir(parents=True, exist_ok=True)

    save_data = {
        'ground_state': ground_state,
        'ground_energy': ground_energy,
        'all_energies': all_energies,
        'results_by_p': {}
    }
    for p, res in results_by_p.items():
        save_data['results_by_p'][str(p)] = {
            'optimal_cost': float(res['result']['optimal_cost']),
            'optimal_params': res['result']['optimal_params'].tolist(),
            'valid_fraction': res['valid_fraction'],
            'best_config': res['best_config'],
            'best_energy': float(res['best_energy']),
            'best_prob': float(res['best_prob']),
            'convergence_history': [float(c) for c in res['result']['history']['costs']],
            'counts': res['result']['optimal_counts']
        }

    with open(output_dir / "results.json", 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_dir / 'results.json'}")

    # --- Save CSV files ---
    from postprocessing.analysis import analyze_results, compute_approximation_ratio, compare_with_exact

    best_p = max(results_by_p.keys())
    final_counts = results_by_p[best_p]['result']['optimal_counts']
    total_shots = sum(final_counts.values())

    # 1. Brute-force enumeration
    bf_rows = [{'bitstring': bs, 'energy_eV': e,
                'sites': str([i for i, b in enumerate(bs) if b == '1'])}
               for bs, e in sorted(all_energies.items(), key=lambda x: x[1])]
    pd.DataFrame(bf_rows).to_csv(output_dir / "brute_force_enumeration.csv", index=False)

    # 2. Convergence history for each p
    for p, res in results_by_p.items():
        conv_df = pd.DataFrame({
            'iteration': range(1, len(res['result']['history']['costs']) + 1),
            'cvar_cost_eV': res['result']['history']['costs']
        })
        conv_df.to_csv(output_dir / f"convergence_p{p}.csv", index=False)

    # 3. QAOA results analysis for each p (uses analyze_results)
    for p, res in results_by_p.items():
        df = analyze_results(res['result']['optimal_counts'], alpha, beta, E_const)
        df.to_csv(output_dir / f"qaoa_results_p{p}.csv", index=False)

    # 4. Brute-force comparison data (for best p)
    comparison_rows = []
    for bs, e in sorted(all_energies.items(), key=lambda x: x[1]):
        bs_qiskit = bs[::-1]
        qaoa_prob = final_counts.get(bs_qiskit, 0) / total_shots
        comparison_rows.append({
            'bitstring': bs,
            'sites': str([i for i, b in enumerate(bs) if b == '1']),
            'exact_energy_eV': e,
            'qaoa_probability': qaoa_prob
        })
    pd.DataFrame(comparison_rows).to_csv(output_dir / "brute_force_comparison.csv", index=False)

    # 5. Site occupation data
    df_final = analyze_results(final_counts, alpha, beta, E_const)
    valid_df = df_final[df_final['valid']]
    occupation = np.zeros(n_qubits)
    total_prob = valid_df['probability'].sum()
    for _, row in valid_df.iterrows():
        for i, b in enumerate(row['bitstring']):
            if b == '1':
                occupation[i] += row['probability']
    if total_prob > 0:
        occupation /= total_prob
    pd.DataFrame({'site': range(n_qubits), 'occupation_probability': occupation}).to_csv(
        output_dir / "site_occupation.csv", index=False)

    # 6. Approximation ratio for each p (uses compute_approximation_ratio)
    approx_rows = []
    for p, res in results_by_p.items():
        ratio = compute_approximation_ratio(res['best_energy'], ground_energy, max_energy)
        approx_rows.append({
            'p': p, 'best_energy_eV': res['best_energy'],
            'exact_ground_energy_eV': ground_energy,
            'max_energy_eV': max_energy,
            'approximation_ratio': ratio
        })
    pd.DataFrame(approx_rows).to_csv(output_dir / "approximation_ratio.csv", index=False)

    # 7. Compare with exact solution (uses compare_with_exact)
    comparison = compare_with_exact(final_counts, alpha, beta, E_const, n_particles, n_qubits)
    comp_summary = {
        'ground_state': comparison['ground_state'],
        'ground_energy_eV': comparison['ground_energy'],
        'qaoa_found_ground': comparison['qaoa_found_ground'],
        'ground_state_probability': comparison['ground_state_probability'],
        'valid_fraction': comparison.get('valid_fraction', 0.0),
        'expected_energy_eV': comparison.get('expected_energy', float('nan')),
        'approximation_ratio': comparison['approximation_ratio']
    }
    pd.DataFrame([comp_summary]).to_csv(output_dir / "compare_with_exact.csv", index=False)
    if comparison.get('top_5_configs'):
        pd.DataFrame(comparison['top_5_configs']).to_csv(
            output_dir / "top_5_configs.csv", index=False)

    # 8. Energy distribution data
    valid_df_energy = df_final[df_final['valid']].copy()
    if not valid_df_energy.empty:
        energy_dist = valid_df_energy.groupby('energy')['probability'].sum().reset_index()
        energy_dist.columns = ['energy_eV', 'probability']
        energy_dist.to_csv(output_dir / "energy_distribution.csv", index=False)

    # 9. Config vs energy data
    config_energy_rows = []
    for bitstring, count in final_counts.items():
        bs = bitstring[::-1]
        energy = evaluate_energy(bs, alpha, beta, E_const)
        prob = count / total_shots
        valid = is_valid_config(bs, n_particles)
        config_energy_rows.append({
            'bitstring': bs, 'energy_eV': energy,
            'probability': prob, 'valid': valid
        })
    pd.DataFrame(config_energy_rows).to_csv(output_dir / "config_vs_energy.csv", index=False)

    # 10. Parameter landscape data
    landscape_rows = []
    for gi, g_val in enumerate(gamma_range):
        for bi, b_val in enumerate(beta_scan_range):
            landscape_rows.append({
                'gamma': g_val, 'beta': b_val,
                'cost_eV': cost_landscape[gi, bi]
            })
    pd.DataFrame(landscape_rows).to_csv(output_dir / "parameter_landscape.csv", index=False)

    print("CSV files saved.")

    # --- Generate plots ---
    from postprocessing.plots import (plot_convergence, plot_energy_distribution,
                                      plot_site_occupation, plot_brute_force_comparison,
                                      plot_config_vs_energy, plot_approximation_ratio_vs_depth,
                                      plot_parameter_landscape)

    for p, res in results_by_p.items():
        plot_convergence(res['result']['history']['costs'],
                         save_path=str(output_dir / f"convergence_p{p}.png"),
                         title=f"XY Mixer p={p}")

    df_final = analyze_results(final_counts, alpha, beta, E_const)
    plot_energy_distribution(df_final, save_path=str(output_dir / "energy_distribution.png"))
    plot_site_occupation(df_final, n_qubits=n_qubits,
                         save_path=str(output_dir / "site_occupation.png"))

    # Brute-force comparison plot
    plot_brute_force_comparison(all_energies, final_counts, alpha, beta, E_const,
                                save_path=str(output_dir / "brute_force_comparison.png"),
                                title=f"XY Mixer (p={best_p}) vs Brute-Force")

    # Config vs energy plot
    plot_config_vs_energy(final_counts, alpha, beta, E_const,
                          save_path=str(output_dir / "config_vs_energy.png"),
                          title=f"XY Mixer (p={best_p}) - Config vs Energy")

    # Approximation ratio vs depth
    plot_approximation_ratio_vs_depth(
        results_by_p, ground_energy, max_energy,
        save_path=str(output_dir / "approximation_ratio_vs_depth.png"),
        title="XY Mixer - Approximation Ratio vs Depth"
    )

    # Parameter landscape
    plot_parameter_landscape(
        cost_landscape, gamma_range, beta_scan_range,
        save_path=str(output_dir / "parameter_landscape.png"),
        title="XY Mixer (p=1) - Parameter Landscape"
    )

    print("Plots saved to results/simulator/constrained/")
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
