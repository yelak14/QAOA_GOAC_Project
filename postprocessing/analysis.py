import numpy as np
import pandas as pd
from qaoa.hamiltonian import evaluate_energy
from qaoa.utils import is_valid_config, bitstring_to_config


def analyze_results(counts, alpha, beta, E_const, n_particles=2):
    """Analyze measurement results into a DataFrame.

    Args:
        counts: dict of {bitstring: count}
        alpha: GOAC linear coefficients
        beta: GOAC quadratic coefficients
        E_const: constant energy offset
        n_particles: required particle number

    Returns:
        DataFrame with columns: bitstring, energy, count, probability, valid, sites
    """
    total_shots = sum(counts.values())
    rows = []

    for bitstring, count in counts.items():
        # Qiskit returns little-endian, reverse for physical interpretation
        bs = bitstring[::-1]
        energy = evaluate_energy(bs, alpha, beta, E_const)
        valid = is_valid_config(bs, n_particles)
        sites = bitstring_to_config(bs) if valid else []

        rows.append({
            'bitstring': bs,
            'bitstring_qiskit': bitstring,
            'energy': energy,
            'count': count,
            'probability': count / total_shots,
            'valid': valid,
            'sites': str(sites)
        })

    df = pd.DataFrame(rows)
    df = df.sort_values('energy').reset_index(drop=True)
    return df


def compute_approximation_ratio(result_energy, exact_ground_energy, max_energy):
    """Compute the approximation ratio.

    r = (max_energy - result_energy) / (max_energy - exact_ground_energy)

    A ratio of 1.0 means the exact ground state was found.

    Args:
        result_energy: energy obtained by QAOA
        exact_ground_energy: exact ground state energy
        max_energy: maximum energy in the problem

    Returns:
        Approximation ratio (float in [0, 1])
    """
    if abs(max_energy - exact_ground_energy) < 1e-10:
        return 1.0
    return (max_energy - result_energy) / (max_energy - exact_ground_energy)


def compare_with_exact(qaoa_counts, alpha, beta, E_const, n_particles=2, n_qubits=8):
    """Compare QAOA results with exact enumeration.

    Args:
        qaoa_counts: measurement counts from QAOA
        alpha, beta, E_const: GOAC coefficients
        n_particles: particle number constraint
        n_qubits: number of qubits

    Returns:
        Dict with comparison metrics
    """
    from qaoa.hamiltonian import get_exact_solution

    # Get exact solution
    ground_state, ground_energy, all_energies = get_exact_solution(
        alpha, beta, E_const, n_particles, n_qubits
    )

    # Analyze QAOA results
    df = analyze_results(qaoa_counts, alpha, beta, E_const, n_particles)
    valid_df = df[df['valid']]

    if valid_df.empty:
        return {
            'ground_state': ground_state,
            'ground_energy': ground_energy,
            'qaoa_found_ground': False,
            'ground_state_probability': 0.0,
            'approximation_ratio': 0.0
        }

    # Check if ground state was found
    total_shots = sum(qaoa_counts.values())
    ground_state_qiskit = ground_state[::-1]  # Convert to Qiskit convention
    ground_prob = qaoa_counts.get(ground_state_qiskit, 0) / total_shots

    # Compute expected energy from valid samples
    valid_expected_energy = (valid_df['energy'] * valid_df['probability']).sum() / valid_df['probability'].sum()

    # Approximation ratio
    max_energy = max(all_energies.values())
    approx_ratio = compute_approximation_ratio(valid_expected_energy, ground_energy, max_energy)

    return {
        'ground_state': ground_state,
        'ground_energy': ground_energy,
        'qaoa_found_ground': ground_prob > 0,
        'ground_state_probability': ground_prob,
        'valid_fraction': valid_df['probability'].sum(),
        'expected_energy': valid_expected_energy,
        'approximation_ratio': approx_ratio,
        'top_5_configs': valid_df.head(5)[['bitstring', 'energy', 'probability']].to_dict('records')
    }
