"""Postprocessing tools for QAOA results analysis."""

from .compare_methods import compare_standard_vs_dicke
from .compare_hardware_sim import compare_hardware_simulator
from .publication_figures import generate_publication_figures
from .analyze_results import analyze_all_results

# Legacy imports for backwards compatibility
from .analysis import analyze_results, compute_approximation_ratio, compare_with_exact
from .plots import (plot_convergence, plot_energy_distribution,
                    plot_site_occupation, plot_approximation_ratio_vs_depth)
from .plotting import plot_bitstring_probability

__all__ = [
    'compare_standard_vs_dicke',
    'compare_hardware_simulator',
    'generate_publication_figures',
    'analyze_all_results',
    # Legacy
    'analyze_results',
    'compute_approximation_ratio',
    'compare_with_exact',
    'plot_convergence',
    'plot_energy_distribution',
    'plot_site_occupation',
    'plot_approximation_ratio_vs_depth',
    'plot_bitstring_probability'
]
