"""Generate publication-quality figures for QAOA results.

This module creates paper-ready figures with proper formatting,
font sizes, and export options.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 10
    rcParams['axes.labelsize'] = 12
    rcParams['axes.titlesize'] = 14
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 10
    rcParams['figure.dpi'] = 150
    rcParams['savefig.dpi'] = 300
    rcParams['savefig.bbox'] = 'tight'
    rcParams['axes.linewidth'] = 1.0
    rcParams['xtick.major.width'] = 1.0
    rcParams['ytick.major.width'] = 1.0


def load_summary(results_dir):
    """Load summary.json from results directory."""
    summary_path = results_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            return json.load(f)
    return None


def load_convergence_data(results_dir, run=1):
    """Load convergence CSV data."""
    csv_path = results_dir / f"run_{run}_convergence.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def load_energy_distribution(results_dir, run=1):
    """Load energy distribution CSV data."""
    csv_path = results_dir / f"run_{run}_energy_distribution.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def generate_convergence_figure(output_dir):
    """Generate convergence comparison figure."""
    print("Generating convergence comparison figure...")

    results_root = project_root / "results" / "simulator"

    # Load data for different methods
    standard_data = load_convergence_data(results_root / "standard_penalty")
    dicke_data = load_convergence_data(results_root / "dicke_xy")

    if standard_data is None and dicke_data is None:
        print("  No convergence data found")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    if standard_data is not None:
        energy_col = 'Re_H' if 'Re_H' in standard_data.columns else 'energy_eV'
        ax.plot(standard_data['iteration'], standard_data[energy_col],
                'b-', linewidth=1, alpha=0.8, label='Standard + Penalty')

    if dicke_data is not None:
        energy_col = 'energy_eV' if 'energy_eV' in dicke_data.columns else 'Re_H'
        ax.plot(dicke_data['iteration'], dicke_data[energy_col],
                'g-', linewidth=1, alpha=0.8, label='Dicke + XY')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Energy (eV)')
    ax.set_title('QAOA Convergence Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(output_dir / "convergence_comparison.png", dpi=300)
    plt.savefig(output_dir / "convergence_comparison.pdf")
    plt.close()
    print(f"  Saved: convergence_comparison.png/pdf")


def generate_energy_distribution_figure(output_dir):
    """Generate energy distribution comparison figure."""
    print("Generating energy distribution figure...")

    results_root = project_root / "results" / "simulator"

    # Load data
    standard_data = load_energy_distribution(results_root / "standard_penalty")
    dicke_data = load_energy_distribution(results_root / "dicke_xy")

    if standard_data is None and dicke_data is None:
        print("  No energy distribution data found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if standard_data is not None:
        ax = axes[0]
        energy_col = 'energy' if 'energy' in standard_data.columns else 'energy_eV'
        x = np.arange(len(standard_data))
        ax.bar(x, standard_data['probability'], color='steelblue', edgecolor='black', linewidth=0.5)

        # Set tick labels
        n_labels = min(10, len(standard_data))
        step = max(1, len(standard_data) // n_labels)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([f'{e:.0f}' for e in standard_data[energy_col].values[::step]],
                          rotation=45, ha='right')

        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Probability')
        ax.set_title('Standard + Penalty')
        ax.grid(True, alpha=0.3, axis='y')

    if dicke_data is not None:
        ax = axes[1]
        energy_col = 'energy_eV' if 'energy_eV' in dicke_data.columns else 'energy'
        x = np.arange(len(dicke_data))
        ax.bar(x, dicke_data['probability'], color='green', edgecolor='black', linewidth=0.5)

        n_labels = min(10, len(dicke_data))
        step = max(1, len(dicke_data) // n_labels)
        ax.set_xticks(x[::step])
        ax.set_xticklabels([f'{e:.1f}' for e in dicke_data[energy_col].values[::step]],
                          rotation=45, ha='right')

        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Probability')
        ax.set_title('Dicke + XY')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "energy_distribution_comparison.png", dpi=300)
    plt.savefig(output_dir / "energy_distribution_comparison.pdf")
    plt.close()
    print(f"  Saved: energy_distribution_comparison.png/pdf")


def generate_method_comparison_bars(output_dir):
    """Generate method comparison bar chart."""
    print("Generating method comparison bar chart...")

    results_root = project_root / "results" / "simulator"

    # Load summaries
    methods = {
        'Standard\n+ Penalty': load_summary(results_root / "standard_penalty"),
        'Dicke\n+ XY': load_summary(results_root / "dicke_xy"),
        'Dicke + XY\n(Multi-start)': load_summary(results_root / "dicke_xy_multistart")
    }

    # Extract data
    valid_fracs = []
    gs_probs = []
    labels = []

    for name, summary in methods.items():
        if summary is None:
            continue
        labels.append(name)

        if 'averages' in summary:
            valid_fracs.append(summary['averages']['valid_fraction'] * 100)
            gs_probs.append(summary['averages']['gs_prob_summed'])
        elif 'results' in summary and isinstance(summary['results'], dict):
            valid_fracs.append(summary['results']['valid_fraction'] * 100)
            gs_probs.append(summary['results']['gs_prob_summed'])
        else:
            valid_fracs.append(0)
            gs_probs.append(0)

    if not labels:
        print("  No data found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    x = np.arange(len(labels))
    colors = ['steelblue', 'green', 'darkgreen'][:len(labels)]

    # Valid fraction
    ax1 = axes[0]
    bars1 = ax1.bar(x, valid_fracs, color=colors, edgecolor='black')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel('Valid Configurations (%)')
    ax1.set_title('(a) Constraint Satisfaction')
    ax1.set_ylim(0, 110)
    ax1.axhline(y=100, color='red', linestyle='--', alpha=0.5, linewidth=0.8)

    for bar, val in zip(bars1, valid_fracs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    # GS probability
    ax2 = axes[1]
    bars2 = ax2.bar(x, gs_probs, color=colors, edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Ground State Probability')
    ax2.set_title('(b) Solution Quality')
    ax2.set_ylim(0, max(gs_probs) * 1.3 if gs_probs else 1)

    for bar, val in zip(bars2, gs_probs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "method_comparison_bars.png", dpi=300)
    plt.savefig(output_dir / "method_comparison_bars.pdf")
    plt.close()
    print(f"  Saved: method_comparison_bars.png/pdf")


def generate_publication_figures(output_dir=None):
    """Generate all publication-quality figures.

    Args:
        output_dir: Output directory. If None, uses results/figures/
    """
    print("=" * 70)
    print("Generating Publication Figures")
    print("=" * 70)

    # Setup matplotlib style
    setup_publication_style()

    # Output directory
    if output_dir is None:
        output_dir = project_root / "results" / "figures"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate figures
    generate_convergence_figure(output_dir)
    generate_energy_distribution_figure(output_dir)
    generate_method_comparison_bars(output_dir)

    print("\n" + "=" * 70)
    print(f"All figures saved to: {output_dir}")
    print("=" * 70)


def main():
    """Generate publication figures."""
    generate_publication_figures()


if __name__ == "__main__":
    main()
