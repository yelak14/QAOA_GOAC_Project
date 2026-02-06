"""Compare hardware vs simulator results.

This module loads results from both simulator and hardware runs and
analyzes the differences due to hardware noise and error mitigation.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def load_summary(results_dir):
    """Load summary.json from results directory."""
    summary_path = results_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            return json.load(f)
    return None


def compare_hardware_simulator(method='dicke_xy', output_dir=None):
    """Compare hardware vs simulator results for a given method.

    Args:
        method: 'standard_penalty', 'dicke_xy', or 'dicke_xy_multistart'
        output_dir: Output directory. If None, uses results/comparison/

    Returns:
        DataFrame with comparison data
    """
    print("=" * 70)
    print(f"Comparing Hardware vs Simulator ({method})")
    print("=" * 70)

    results_root = project_root / "results"

    # Load summaries
    sim_summary = load_summary(results_root / "simulator" / method)
    hw_summary = load_summary(results_root / "hardware" / method)

    if sim_summary is None:
        print(f"Warning: Simulator results for {method} not found")
    if hw_summary is None:
        print(f"Warning: Hardware results for {method} not found")

    if sim_summary is None and hw_summary is None:
        print("No results found to compare!")
        return None

    # Output directory
    if output_dir is None:
        output_dir = project_root / "results" / "comparison"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build comparison data
    comparison_data = []

    if sim_summary:
        # Handle different summary structures
        if 'averages' in sim_summary:
            valid_frac = sim_summary['averages']['valid_fraction']
            gs_prob = sim_summary['averages']['gs_prob_summed']
        elif 'results' in sim_summary and isinstance(sim_summary['results'], dict):
            valid_frac = sim_summary['results']['valid_fraction']
            gs_prob = sim_summary['results']['gs_prob_summed']
        else:
            valid_frac = np.mean([r['valid_fraction'] for r in sim_summary['results']])
            gs_prob = np.mean([r['gs_prob_summed'] for r in sim_summary['results']])

        comparison_data.append({
            'Platform': 'Simulator',
            'Method': method,
            'Valid %': valid_frac * 100,
            'GS Probability': gs_prob,
            'p': sim_summary['settings']['p']
        })

    if hw_summary:
        if 'results' in hw_summary and isinstance(hw_summary['results'], dict):
            valid_frac = hw_summary['results']['valid_fraction']
            gs_prob = hw_summary['results'].get('gs_prob_summed', 0)
        elif 'best_result' in hw_summary:
            valid_frac = hw_summary['best_result']['valid_fraction']
            gs_prob = hw_summary['best_result'].get('gs_prob_summed', 0)
        else:
            valid_frac = 0
            gs_prob = 0

        comparison_data.append({
            'Platform': 'Hardware',
            'Method': method,
            'Valid %': valid_frac * 100,
            'GS Probability': gs_prob,
            'p': hw_summary['settings']['p']
        })

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / f"hardware_sim_comparison_{method}.csv", index=False)

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(comparison_df.to_string(index=False))

    # Calculate differences
    if len(comparison_data) == 2:
        sim_data = comparison_data[0]
        hw_data = comparison_data[1]

        print("\n" + "=" * 70)
        print("DIFFERENCES (Hardware - Simulator)")
        print("=" * 70)
        print(f"  Valid %: {hw_data['Valid %'] - sim_data['Valid %']:+.2f}%")
        print(f"  GS Probability: {hw_data['GS Probability'] - sim_data['GS Probability']:+.4f}")

        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        platforms = ['Simulator', 'Hardware']
        valid_pcts = [sim_data['Valid %'], hw_data['Valid %']]
        gs_probs = [sim_data['GS Probability'], hw_data['GS Probability']]
        colors = ['steelblue', 'coral']

        # Valid percentage
        ax1 = axes[0]
        bars1 = ax1.bar(platforms, valid_pcts, color=colors, edgecolor='black')
        ax1.set_ylabel('Valid Configurations (%)', fontsize=12)
        ax1.set_title(f'Valid Configuration Fraction\n({method})', fontsize=14)
        ax1.set_ylim(0, 110)

        for bar, val in zip(bars1, valid_pcts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=11)

        # GS probability
        ax2 = axes[1]
        bars2 = ax2.bar(platforms, gs_probs, color=colors, edgecolor='black')
        ax2.set_ylabel('Ground State Probability', fontsize=12)
        ax2.set_title(f'Ground State Probability\n({method})', fontsize=14)
        ax2.set_ylim(0, max(gs_probs) * 1.3 if max(gs_probs) > 0 else 1)

        for bar, val in zip(bars2, gs_probs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=11)

        plt.tight_layout()
        plt.savefig(output_dir / f"hardware_sim_comparison_{method}.png", dpi=150)
        plt.close()
        print(f"\nSaved: hardware_sim_comparison_{method}.png")

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

    return comparison_df


def compare_all_methods():
    """Compare hardware vs simulator for all available methods."""
    methods = ['standard_penalty', 'dicke_xy', 'dicke_xy_multistart']

    all_comparisons = []
    for method in methods:
        df = compare_hardware_simulator(method)
        if df is not None:
            all_comparisons.append(df)

    if all_comparisons:
        combined_df = pd.concat(all_comparisons, ignore_index=True)
        output_dir = project_root / "results" / "comparison"
        combined_df.to_csv(output_dir / "hardware_sim_comparison_all.csv", index=False)
        return combined_df

    return None


def main():
    """Run hardware vs simulator comparison for all methods."""
    compare_all_methods()


if __name__ == "__main__":
    main()
