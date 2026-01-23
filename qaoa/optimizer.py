import numpy as np
from scipy.optimize import minimize
from qiskit_aer import AerSimulator
from qiskit import transpile

from .hamiltonian import evaluate_energy
from .utils import is_valid_config


def cvar_cost_function(counts, alpha, beta, E_const, cvar_alpha=0.2, n_particles=2):
    """Compute the CVaR (Conditional Value at Risk) cost from measurement counts.

    CVaR focuses on the lowest-energy fraction (alpha) of samples,
    providing better optimization signal than the simple expectation value.

    Args:
        counts: dict of {bitstring: count}
        alpha: GOAC linear coefficients
        beta: GOAC quadratic coefficients
        E_const: constant energy offset
        cvar_alpha: fraction of lowest-energy samples to average (0, 1]
        n_particles: required number of particles for validity check

    Returns:
        CVaR cost value (float)
    """
    # Compute energy for each measured bitstring
    energies = []
    for bitstring, count in counts.items():
        # Qiskit returns bitstrings in little-endian, reverse for our convention
        bs = bitstring[::-1]
        energy = evaluate_energy(bs, alpha, beta, E_const)
        energies.extend([energy] * count)

    energies.sort()

    # Take the lowest cvar_alpha fraction
    n_samples = len(energies)
    n_cvar = max(1, int(np.ceil(cvar_alpha * n_samples)))
    cvar_value = np.mean(energies[:n_cvar])

    return cvar_value


def run_optimization(circuit, gammas, betas, alpha, beta, E_const,
                     backend=None, shots=4096, maxiter=200,
                     cvar_alpha=0.2, p=1, n_particles=2):
    """Run QAOA parameter optimization using COBYLA.

    Args:
        circuit: parameterized QuantumCircuit
        gammas: list of gamma Parameter objects
        betas: list of beta Parameter objects
        alpha: GOAC linear coefficients
        beta: GOAC quadratic coefficients
        E_const: constant energy offset
        backend: Qiskit backend (default: AerSimulator)
        shots: number of measurement shots
        maxiter: maximum optimizer iterations
        cvar_alpha: CVaR alpha parameter
        p: number of QAOA layers
        n_particles: required particle number

    Returns:
        Dict with keys: 'optimal_params', 'optimal_cost', 'history',
                        'optimal_counts', 'n_evals'
    """
    if backend is None:
        backend = AerSimulator()

    history = {'costs': [], 'params': []}
    n_evals = [0]

    def objective(params):
        # Bind parameters
        param_dict = {}
        for i in range(p):
            param_dict[gammas[i]] = params[i]
            param_dict[betas[i]] = params[p + i]

        bound_circuit = circuit.assign_parameters(param_dict)
        transpiled = transpile(bound_circuit, backend)
        job = backend.run(transpiled, shots=shots)
        counts = job.result().get_counts()

        cost = cvar_cost_function(counts, alpha, beta, E_const,
                                  cvar_alpha=cvar_alpha, n_particles=n_particles)

        history['costs'].append(cost)
        history['params'].append(params.copy())
        n_evals[0] += 1

        return cost

    # Initial parameters: random in [0, pi]
    x0 = np.random.uniform(0, np.pi, 2 * p)

    result = minimize(objective, x0, method='COBYLA',
                      options={'maxiter': maxiter, 'rhobeg': 0.5})

    # Get final counts with optimal parameters
    param_dict = {}
    for i in range(p):
        param_dict[gammas[i]] = result.x[i]
        param_dict[betas[i]] = result.x[p + i]

    bound_circuit = circuit.assign_parameters(param_dict)
    transpiled = transpile(bound_circuit, backend)
    job = backend.run(transpiled, shots=shots * 4)  # More shots for final result
    optimal_counts = job.result().get_counts()

    return {
        'optimal_params': result.x,
        'optimal_cost': result.fun,
        'history': history,
        'optimal_counts': optimal_counts,
        'n_evals': n_evals[0],
        'scipy_result': result
    }
