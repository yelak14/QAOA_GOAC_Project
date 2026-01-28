"""QAOA with constraint-preserving XY mixer on IBM Quantum hardware."""

from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Session
from qiskit import transpile
from qaoa.optimizer import cvar_cost_function
from qaoa.circuit import build_qaoa_circuit
from qaoa.hamiltonian import build_cost_hamiltonian, evaluate_energy, get_exact_solution
from qaoa.utils import load_coefficients, is_valid_config
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


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
    print(
        f"Initial state: |{'1' * n_particles + '0' * (n_qubits - n_particles)}>")

    # Get exact solution for reference
    ground_state, ground_energy, all_energies = get_exact_solution(
        alpha, beta_coeff, E_const, n_particles, n_qubits
    )
    max_energy = max(all_energies.values())
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

        cost = cvar_cost_function(
            counts, alpha, beta_coeff, E_const, cvar_alpha=0.2)
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
    valid_counts = {k: v for k, v in counts.items(
    ) if is_valid_config(k[::-1], n_particles)}
    total = sum(counts.values())
    valid_fraction = sum(valid_counts.values()) / total

    if valid_counts:
        best_bs_qiskit = max(valid_counts, key=valid_counts.get)
        best_bs = best_bs_qiskit[::-1]
        best_energy = evaluate_energy(best_bs, alpha, beta_coeff, E_const)
        best_prob = valid_counts[best_bs_qiskit] / total
    else:
        best_bs, best_energy, best_prob = "N/A", float('inf'), 0.0

    print(f"\n--- Results ---")
    print(f"Optimal cost: {result.fun:.6f} eV")
    print(f"Valid fraction: {valid_fraction:.1%}")
    print(f"Best config: {best_bs}, E = {best_energy:.6f} eV")
    print(f"Exact ground: {ground_state}, E = {ground_energy:.6f} eV")

    if valid_fraction > 0.99:
        print("[OK] XY mixer successfully preserved particle number on hardware")
    else:
        print(
            f"[WARN] Hardware noise caused {1-valid_fraction:.1%} constraint violations")

    # Build results_by_p for approximation ratio vs depth plot
    results_by_p = {p: {'best_energy': best_energy}}

    # --- Parameter landscape scan (p=1, 5x5 grid on hardware) ---
    print("\nComputing parameter landscape (p=1, 5x5 grid on hardware)...")
    n_grid = 5
    gamma_range = np.linspace(0, 2 * np.pi, n_grid)
    beta_scan_range = np.linspace(0, np.pi, n_grid)
    cost_landscape = np.zeros((n_grid, n_grid))

    with Session(service=service, backend=backend) as session:
        sampler = SamplerV2(session=session)
        for gi, g_val in enumerate(gamma_range):
            for bi, b_val in enumerate(beta_scan_range):
                pdict = {gammas[0]: g_val, betas[0]: b_val}
                bound_scan = circuit.assign_parameters(pdict)
                transpiled_scan = pm.run(bound_scan)
                job = sampler.run([transpiled_scan], shots=1024)
                scan_result = job.result()
                scan_counts = scan_result[0].data.meas.get_counts()
                cost_landscape[gi, bi] = cvar_cost_function(
                    scan_counts, alpha, beta_coeff, E_const, cvar_alpha=0.2
                )
    print("Parameter landscape computed.")

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
        'best_energy': float(best_energy) if best_energy != float('inf') else None,
        'ground_state': ground_state,
        'ground_energy': ground_energy,
        'all_energies': all_energies,
        'convergence_history': history['costs'],
        'counts': counts
    }
    with open(output_dir / "results.json", 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_dir / 'results.json'}")

    # --- Save CSV files ---
    from postprocessing.analysis import analyze_results, compute_approximation_ratio, compare_with_exact

    total_shots = total

    # 1. Brute-force enumeration
    bf_rows = [{'bitstring': bs, 'energy_eV': e,
                'sites': str([i for i, b in enumerate(bs) if b == '1'])}
               for bs, e in sorted(all_energies.items(), key=lambda x: x[1])]
    pd.DataFrame(bf_rows).to_csv(
        output_dir / "brute_force_enumeration.csv", index=False)

    # 2. Convergence history
    conv_df = pd.DataFrame({
        'iteration': range(1, len(history['costs']) + 1),
        'cvar_cost_eV': history['costs']
    })
    conv_df.to_csv(output_dir / f"convergence_p{p}.csv", index=False)

    # 3. QAOA results analysis (uses analyze_results)
    df = analyze_results(counts, alpha, beta_coeff, E_const, n_particles)
    df.to_csv(output_dir / f"qaoa_results_p{p}.csv", index=False)

    # 4. Brute-force comparison data
    comparison_rows = []
    for bs, e in sorted(all_energies.items(), key=lambda x: x[1]):
        bs_qiskit = bs[::-1]
        qaoa_prob = counts.get(bs_qiskit, 0) / total_shots
        comparison_rows.append({
            'bitstring': bs,
            'sites': str([i for i, b in enumerate(bs) if b == '1']),
            'exact_energy_eV': e,
            'qaoa_probability': qaoa_prob
        })
    pd.DataFrame(comparison_rows).to_csv(
        output_dir / "brute_force_comparison.csv", index=False)

    # 5. Site occupation data
    df_final = analyze_results(counts, alpha, beta_coeff, E_const, n_particles)
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

    # 6. Approximation ratio (uses compute_approximation_ratio)
    ratio = compute_approximation_ratio(best_energy, ground_energy, max_energy)
    pd.DataFrame([{
        'p': p, 'best_energy_eV': best_energy,
        'exact_ground_energy_eV': ground_energy,
        'max_energy_eV': max_energy,
        'approximation_ratio': ratio
    }]).to_csv(output_dir / "approximation_ratio.csv", index=False)

    # 7. Compare with exact solution (uses compare_with_exact)
    comparison = compare_with_exact(
        counts, alpha, beta_coeff, E_const, n_particles, n_qubits)
    comp_summary = {
        'ground_state': comparison['ground_state'],
        'ground_energy_eV': comparison['ground_energy'],
        'qaoa_found_ground': comparison['qaoa_found_ground'],
        'ground_state_probability': comparison['ground_state_probability'],
        'valid_fraction': comparison.get('valid_fraction', 0.0),
        'expected_energy_eV': comparison.get('expected_energy', float('nan')),
        'approximation_ratio': comparison['approximation_ratio']
    }
    pd.DataFrame([comp_summary]).to_csv(
        output_dir / "compare_with_exact.csv", index=False)
    if comparison.get('top_5_configs'):
        pd.DataFrame(comparison['top_5_configs']).to_csv(
            output_dir / "top_5_configs.csv", index=False)

    # 8. Energy distribution data
    valid_df_energy = df_final[df_final['valid']].copy()
    if not valid_df_energy.empty:
        energy_dist = valid_df_energy.groupby(
            'energy')['probability'].sum().reset_index()
        energy_dist.columns = ['energy_eV', 'probability']
        energy_dist.to_csv(output_dir / "energy_distribution.csv", index=False)

    # 9. Config vs energy data
    config_energy_rows = []
    for bitstring, count_val in counts.items():
        bs = bitstring[::-1]
        energy = evaluate_energy(bs, alpha, beta_coeff, E_const)
        prob = count_val / total_shots
        valid = is_valid_config(bs, n_particles)
        config_energy_rows.append({
            'bitstring': bs, 'energy_eV': energy,
            'probability': prob, 'valid': valid
        })
    pd.DataFrame(config_energy_rows).to_csv(
        output_dir / "config_vs_energy.csv", index=False)

    # 10. Parameter landscape data
    landscape_rows = []
    for gi, g_val in enumerate(gamma_range):
        for bi, b_val in enumerate(beta_scan_range):
            landscape_rows.append({
                'gamma': g_val, 'beta': b_val,
                'cost_eV': cost_landscape[gi, bi]
            })
    pd.DataFrame(landscape_rows).to_csv(
        output_dir / "parameter_landscape.csv", index=False)

    print("CSV files saved.")

    # --- Generate plots ---
    from postprocessing.plots import (plot_convergence, plot_energy_distribution,
                                      plot_site_occupation, plot_brute_force_comparison,
                                      plot_config_vs_energy, plot_approximation_ratio_vs_depth,
                                      plot_parameter_landscape, plot_dual_panel_energy_probability)

    # Convergence
    plot_convergence(history['costs'],
                     save_path=str(output_dir / f"convergence_p{p}.png"),
                     title=f"XY Mixer p={p} (Hardware)")

    # Energy distribution
    plot_energy_distribution(df_final, save_path=str(output_dir / "energy_distribution.png"),
                             title="Energy Distribution (Hardware, XY Mixer)")

    # Site occupation
    plot_site_occupation(df_final, n_qubits=n_qubits,
                         save_path=str(output_dir / "site_occupation.png"),
                         title="Site Occupation (Hardware, XY Mixer)")

    # Brute-force comparison
    plot_brute_force_comparison(all_energies, counts, alpha, beta_coeff, E_const,
                                save_path=str(
                                    output_dir / "brute_force_comparison.png"),
                                title=f"XY Mixer (p={p}, Hardware) vs Brute-Force")

    # Config vs energy
    plot_config_vs_energy(counts, alpha, beta_coeff, E_const,
                          save_path=str(output_dir / "config_vs_energy.png"),
                          title=f"XY Mixer (p={p}, Hardware) - Config vs Energy")

    # Dual-panel energy/probability plots for each p
    for p, res in results_by_p.items():
        plot_dual_panel_energy_probability(
            all_energies,
            res['result']['optimal_counts'],
            mixer_type="XY (Constrained)",
            p=p,
            save_path=str(output_dir / f"all_configurations_energy_p{p}.png")
        )

    # Also save combined version (using best p)
    plot_dual_panel_energy_probability(
        all_energies,
        final_counts,
        mixer_type="XY (Constrained)",
        p=best_p,
        save_path=str(output_dir / "all_configurations_energy.png")
    )

    # Approximation ratio vs depth
    plot_approximation_ratio_vs_depth(
        results_by_p, ground_energy, max_energy,
        save_path=str(output_dir / "approximation_ratio_vs_depth.png"),
        title="XY Mixer (Hardware) - Approximation Ratio vs Depth"
    )

    # Parameter landscape
    plot_parameter_landscape(
        cost_landscape, gamma_range, beta_scan_range,
        save_path=str(output_dir / "parameter_landscape.png"),
        title="XY Mixer (p=1, Hardware) - Parameter Landscape"
    )

    print(f"Plots saved to {output_dir}")
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
