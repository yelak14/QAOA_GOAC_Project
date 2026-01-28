import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def plot_convergence(cost_history, save_path=None, title="QAOA Convergence"):
    """Plot optimization convergence (cost vs. iteration).

    Args:
        cost_history: list of cost values per iteration
        save_path: if provided, save figure to this path
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cost_history) + 1), cost_history, 'b-o', markersize=3)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('CVaR Cost (eV)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_energy_distribution(df, save_path=None, title="Energy Distribution"):
    """Plot energy probability distribution from results DataFrame.

    Args:
        df: DataFrame from analyze_results with 'energy' and 'probability' columns
        save_path: if provided, save figure to this path
        title: plot title
    """
    valid_df = df[df['valid']].copy()
    if valid_df.empty:
        print("No valid configurations to plot.")
        return

    # Aggregate by energy (some configs may have same energy)
    energy_probs = valid_df.groupby(
        'energy')['probability'].sum().reset_index()
    energy_probs = energy_probs.sort_values('energy')

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(energy_probs)), energy_probs['probability'],
           color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(energy_probs)))
    ax.set_xticklabels([f"{e:.4f}" for e in energy_probs['energy']],
                       rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Probability')
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_site_occupation(df, n_qubits=8, save_path=None, title="Site Occupation Frequency"):
    """Plot how frequently each site is occupied across measured configurations.

    Args:
        df: DataFrame from analyze_results
        n_qubits: number of sites
        save_path: if provided, save figure to this path
        title: plot title
    """
    valid_df = df[df['valid']].copy()
    if valid_df.empty:
        print("No valid configurations to plot.")
        return

    # Compute weighted occupation frequency
    occupation = np.zeros(n_qubits)
    total_prob = 0
    for _, row in valid_df.iterrows():
        for i, b in enumerate(row['bitstring']):
            if b == '1':
                occupation[i] += row['probability']
        total_prob += row['probability']

    if total_prob > 0:
        occupation /= total_prob

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(n_qubits), occupation, color='coral', alpha=0.8)
    ax.set_xlabel('Site Index')
    ax.set_ylabel('Occupation Probability')
    ax.set_title(title)
    ax.set_xticks(range(n_qubits))
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_approximation_ratio_vs_depth(results_by_p, exact_ground_energy, max_energy,
                                      save_path=None, title="Approximation Ratio vs QAOA Depth"):
    """Plot approximation ratio as a function of QAOA depth p.

    Args:
        results_by_p: dict mapping p -> {'best_energy': float, ...}
        exact_ground_energy: exact ground state energy
        max_energy: maximum energy among valid configs
        save_path: if provided, save figure
        title: plot title
    """
    from .analysis import compute_approximation_ratio

    ps = sorted(results_by_p.keys())
    ratios = []
    for p in ps:
        r = compute_approximation_ratio(
            results_by_p[p]['best_energy'],
            exact_ground_energy,
            max_energy
        )
        ratios.append(r)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ps, ratios, 'ro-', markersize=8, linewidth=2)
    ax.set_xlabel('QAOA Depth (p)')
    ax.set_ylabel('Approximation Ratio')
    ax.set_title(title)
    ax.set_ylim(0, 1.1)
    ax.axhline(y=1.0, color='g', linestyle='--', alpha=0.5, label='Optimal')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_brute_force_comparison(all_energies, qaoa_counts, alpha, beta, E_const,
                                n_particles=2, save_path=None,
                                title="QAOA vs Brute-Force Comparison"):
    """Plot exact energies of all valid configurations with QAOA probabilities overlaid.

    Args:
        all_energies: dict {bitstring: energy} from brute-force enumeration
        qaoa_counts: dict {bitstring: count} from QAOA measurement
        alpha, beta, E_const: GOAC coefficients
        n_particles: particle number constraint
        save_path: if provided, save figure
        title: plot title
    """
    # Sort configurations by energy
    sorted_configs = sorted(all_energies.items(), key=lambda x: x[1])
    config_labels = [bs for bs, _ in sorted_configs]
    ground_energy = sorted_configs[0][1]
    # Plot relative energies (E - E_ground) so differences are visible
    exact_energies = [e - ground_energy for _, e in sorted_configs]

    # Compute QAOA probabilities for each valid configuration
    total_shots = sum(qaoa_counts.values())
    qaoa_probs = []
    for bs in config_labels:
        # Convert to Qiskit convention (reversed)
        bs_qiskit = bs[::-1]
        count = qaoa_counts.get(bs_qiskit, 0)
        qaoa_probs.append(count / total_shots)

    # Create figure with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    x = np.arange(len(config_labels))
    width = 0.4

    # Bar chart: exact energies
    bars1 = ax1.bar(x - width/2, exact_energies, width, color='steelblue',
                    alpha=0.7, label='Exact Energy')

    # Bar chart: QAOA probabilities
    bars2 = ax2.bar(x + width/2, qaoa_probs, width, color='coral',
                    alpha=0.7, label='QAOA Probability')

    # Highlight ground state(s)
    min_energy = min(exact_energies)
    for i, e in enumerate(exact_energies):
        if abs(e - min_energy) < 1e-8:
            ax1.bar(x[i] - width/2, e, width, color='darkblue',
                    alpha=0.9, edgecolor='gold', linewidth=2)

    ax1.set_xlabel('Configuration (sorted by energy)')
    ax1.set_ylabel('Energy - E_ground (eV)', color='steelblue')
    ax2.set_ylabel('QAOA Probability', color='coral')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='coral')

    # X-axis labels: show site indices
    site_labels = [
        f"{bs}\n{[i for i, b in enumerate(bs) if b == '1']}" for bs in config_labels]
    ax1.set_xticks(x)
    ax1.set_xticklabels(site_labels, rotation=90,
                        fontsize=7, family='monospace')

    ax1.set_title(title)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax1.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_config_vs_energy(counts, alpha, beta, E_const, n_particles=2,
                          save_path=None, title="Configuration vs Energy"):
    """Plot measured configurations (bitstrings) vs their energy.

    Shows each sampled bitstring on x-axis and its energy on y-axis,
    with marker size proportional to measurement probability.

    Args:
        counts: dict {bitstring: count} from QAOA measurement
        alpha, beta, E_const: GOAC coefficients
        n_particles: particle number constraint
        save_path: if provided, save figure
        title: plot title
    """
    from qaoa.hamiltonian import evaluate_energy
    from qaoa.utils import is_valid_config

    total_shots = sum(counts.values())

    valid_bs = []
    valid_energies = []
    valid_probs = []
    invalid_bs = []
    invalid_energies = []
    invalid_probs = []

    for bitstring, count in counts.items():
        bs = bitstring[::-1]  # Reverse from Qiskit convention
        energy = evaluate_energy(bs, alpha, beta, E_const)
        prob = count / total_shots

        if is_valid_config(bs, n_particles):
            valid_bs.append(bs)
            valid_energies.append(energy)
            valid_probs.append(prob)
        else:
            invalid_bs.append(bs)
            invalid_energies.append(energy)
            invalid_probs.append(prob)

    # Sort valid by energy
    if valid_bs:
        sorted_valid = sorted(
            zip(valid_bs, valid_energies, valid_probs), key=lambda x: x[1])
        valid_bs, valid_energies, valid_probs = zip(*sorted_valid)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot invalid configs (gray, smaller)
    if invalid_bs:
        invalid_sizes = [max(10, p * 5000) for p in invalid_probs]
        ax.scatter(range(len(invalid_bs)), invalid_energies,
                   s=invalid_sizes, c='lightgray', alpha=0.4,
                   edgecolors='gray', linewidth=0.5, label=f'Invalid ({len(invalid_bs)})')

    # Plot valid configs (colored by energy, larger)
    if valid_bs:
        valid_sizes = [max(30, p * 5000) for p in valid_probs]
        x_valid = range(len(invalid_bs), len(invalid_bs) + len(valid_bs))
        scatter = ax.scatter(x_valid, valid_energies,
                             s=valid_sizes, c=valid_energies, cmap='coolwarm_r',
                             edgecolors='black', linewidth=0.8,
                             label=f'Valid ({len(valid_bs)})')
        plt.colorbar(scatter, ax=ax, label='Energy (eV)', shrink=0.8)

        # Annotate top-3 lowest energy valid configs
        for i in range(min(3, len(valid_bs))):
            ax.annotate(valid_bs[i],
                        (x_valid[i], valid_energies[i]),
                        textcoords="offset points", xytext=(5, 10),
                        fontsize=7, family='monospace',
                        arrowprops=dict(arrowstyle='->', color='black', lw=0.8))

    ax.set_xlabel('Configuration index')
    ax.set_ylabel('Energy (eV)')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.2)

    # Add ground state energy line
    if valid_energies:
        ax.axhline(y=min(valid_energies), color='green', linestyle='--',
                   alpha=0.6, label='Ground state')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_parameter_landscape(cost_values, gamma_range, beta_range,
                             save_path=None, title="Parameter Landscape"):
    """Plot 2D heatmap of cost function over gamma-beta space.

    Args:
        cost_values: 2D array of cost values [n_gamma, n_beta]
        gamma_range: array of gamma values
        beta_range: array of beta values
        save_path: if provided, save figure
        title: plot title
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cost_values, aspect='auto', origin='lower',
                   extent=[beta_range[0], beta_range[-1],
                           gamma_range[0], gamma_range[-1]],
                   cmap='viridis')
    ax.set_xlabel('β')
    ax.set_ylabel('γ')
    ax.set_title(title)
    plt.colorbar(im, label='CVaR Cost (eV)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_dual_panel_energy_probability(all_energies, qaoa_counts, n_particles=2,
                                       save_path=None, title=None, mixer_type="XY (Constrained)", p=1):
    """Plot dual-panel figure: energy above ground (top) and QAOA probability (bottom).

    This creates the exact visualization style with:
    - Top panel: Bar chart of energy above ground state for all valid configurations
    - Bottom panel: Bar chart of QAOA sampling probability with uniform baseline
    - Color gradient from dark red (high prob) to light coral (low prob)
    - Configurations sorted by energy and labeled with bitstring + site indices

    Args:
        all_energies: dict {bitstring: energy} from brute-force enumeration
        qaoa_counts: dict {bitstring: count} from QAOA measurement
        n_particles: particle number constraint (default 2 for Li2Co8O16)
        save_path: if provided, save figure to this path
        title: plot title (auto-generated if None)
        mixer_type: string describing mixer ("XY (Constrained)" or "X (Standard)")
        p: QAOA depth
    """
    # Sort configurations by energy
    sorted_configs = sorted(all_energies.items(), key=lambda x: x[1])
    config_labels = [bs for bs, _ in sorted_configs]
    ground_energy = sorted_configs[0][1]

    # Compute relative energies (E - E_ground)
    relative_energies = [e - ground_energy for _, e in sorted_configs]

    # Compute QAOA probabilities for each valid configuration
    total_shots = sum(qaoa_counts.values())
    qaoa_probs = []
    for bs in config_labels:
        # Convert to Qiskit convention (reversed bitstring)
        bs_qiskit = bs[::-1]
        count = qaoa_counts.get(bs_qiskit, 0)
        qaoa_probs.append(count / total_shots)

    n_configs = len(config_labels)
    uniform_prob = 1.0 / n_configs  # Baseline for random sampling

    # Create figure with two subplots (shared x-axis)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={'height_ratios': [1, 1.2], 'hspace': 0.05})

    x = np.arange(n_configs)

    # ============ TOP PANEL: Energy above ground ============
    # Color bars by energy (darker blue for lower energy)
    energy_colors = plt.cm.Blues(np.linspace(0.3, 0.7, n_configs))
    ax1.bar(x, relative_energies, color=energy_colors,
            edgecolor='none', alpha=0.8)

    ax1.set_ylabel('Energy above ground (eV)', fontsize=11)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.3, axis='y')

    # Title
    if title is None:
        title = f"{mixer_type} Mixer (p={p}) - All Valid Configurations"
    ax1.set_title(title, fontsize=13, fontweight='bold')

    # ============ BOTTOM PANEL: QAOA Probability ============
    # Create color gradient based on probability (dark red = high, light coral = low)
    max_prob = max(qaoa_probs) if max(qaoa_probs) > 0 else 1
    prob_normalized = [prob / max_prob for prob in qaoa_probs]

    # Custom colormap: light coral to dark red
    colors = []
    for pn in prob_normalized:
        # Interpolate between light coral (1.0, 0.6, 0.6) and dark red (0.5, 0.0, 0.0)
        r = 1.0 - 0.5 * pn
        g = 0.6 - 0.6 * pn
        b = 0.5 - 0.5 * pn
        colors.append((r, g, b))

    ax2.bar(x, qaoa_probs, color=colors, edgecolor='none', alpha=0.9)

    # Add uniform probability baseline
    ax2.axhline(y=uniform_prob, color='coral',
                linestyle=':', linewidth=1.5, alpha=0.8)
    ax2.text(n_configs - 1, uniform_prob + 0.005, f'Uniform (1/{n_configs}={uniform_prob:.4f})',
             ha='right', va='bottom', fontsize=9, color='coral')

    ax2.set_ylabel('QAOA Probability', fontsize=11)
    ax2.set_xlabel('Configuration', fontsize=11)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3, axis='y')

    # X-axis labels: bitstring + site indices
    site_labels = [
        f"{bs}\n{[i for i, b in enumerate(bs) if b == '1']}" for bs in config_labels]
    ax2.set_xticks(x)
    ax2.set_xticklabels(site_labels, rotation=90,
                        fontsize=7, family='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Figure saved to: {save_path}")
    else:
        plt.show()

    return fig
