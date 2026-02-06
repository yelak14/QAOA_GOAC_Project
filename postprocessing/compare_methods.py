"""Compare Standard QAOA vs Dicke + XY QAOA results.

This module loads results from both methods and generates comparison plots
and tables for analysis.
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


def compare_standard_vs_dicke(output_dir=None):
    """Compare Standard QAOA + Penalty vs Dicke + XY QAOA.

    Loads results from:
    - results/simulator/standard_penalty/
    - results/simulator/dicke_xy/

    Generates:
    - Comparison bar charts
    - Summary table (CSV)
    - Comparison plot (PNG)

    Args:
        output_dir: Output directory for results. If None, uses results/comparison/
    """
    print("=" * 70)
    print("Comparing Standard QAOA vs Dicke + XY QAOA")
    print("=" * 70)

    results_root = project_root / "results" / "simulator"

    # Load summaries
    standard_summary = load_summary(results_root / "standard_penalty")
    dicke_summary = load_summary(results_root / "dicke_xy")

    if standard_summary is None:
        print("Warning: Standard penalty results not found")
    if dicke_summary is None:
        print("Warning: Dicke + XY results not found")

    if standard_summary is None and dicke_summary is None:
        print("No results found to compare!")
        return

    # Output directory
    if output_dir is None:
        output_dir = project_root / "results" / "comparison"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build comparison data
    comparison_data = []

    if standard_summary:
        comparison_data.append({
            'Method': 'Standard + Penalty',
            'Valid %': standard_summary['averages']['valid_fraction'] * 100,
            'GS Probability': standard_summary['averages']['gs_prob_summed'],
            'p': standard_summary['settings']['p'],
            'Parameters': standard_summary['settings']['n_parameters'],
            'Num Runs': standard_summary['settings']['num_runs']
        })

    if dicke_summary:
        comparison_data.append({
            'Method': 'Dicke + XY',
            'Valid %': dicke_summary['averages']['valid_fraction'] * 100,
            'GS Probability': dicke_summary['averages']['gs_prob_summed'],
            'p': dicke_summary['settings']['p'],
            'Parameters': dicke_summary['settings']['n_parameters'],
            'Num Runs': dicke_summary['settings']['num_runs']
        })

    # Check for multistart results
    multistart_summary = load_summary(results_root / "dicke_xy_multistart")
    if multistart_summary:
        comparison_data.append({
            'Method': 'Dicke + XY (Multi-start)',
            'Valid %': multistart_summary['results']['valid_fraction'] * 100,
            'GS Probability': multistart_summary['results']['gs_prob_summed'],
            'p': multistart_summary['settings']['p'],
            'Parameters': multistart_summary['settings']['p'] * 2,
            'Num Runs': multistart_summary['settings']['stage1_runs']
        })

    # Create comparison DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(output_dir / "method_comparison.csv", index=False)
    print(f"\nSaved: method_comparison.csv")

    # Print comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(comparison_df.to_string(index=False))

    # Create comparison plots
    if len(comparison_data) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        methods = [d['Method'] for d in comparison_data]
        valid_pcts = [d['Valid %'] for d in comparison_data]
        gs_probs = [d['GS Probability'] for d in comparison_data]

        # Valid percentage comparison
        ax1 = axes[0]
        colors = ['steelblue' if 'Standard' in m else 'green' for m in methods]
        bars1 = ax1.bar(methods, valid_pcts, color=colors, edgecolor='black')
        ax1.set_ylabel('Valid Configurations (%)', fontsize=12)
        ax1.set_title('Valid Configuration Fraction', fontsize=14)
        ax1.set_ylim(0, 110)
        ax1.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Ideal (100%)')
        ax1.legend()
        ax1.tick_params(axis='x', rotation=15)

        # Add value labels
        for bar, val in zip(bars1, valid_pcts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

        # Ground state probability comparison
        ax2 = axes[1]
        bars2 = ax2.bar(methods, gs_probs, color=colors, edgecolor='black')
        ax2.set_ylabel('Ground State Probability', fontsize=12)
        ax2.set_title('Ground State Probability', fontsize=14)
        ax2.set_ylim(0, max(gs_probs) * 1.3)
        ax2.tick_params(axis='x', rotation=15)

        # Add value labels
        for bar, val in zip(bars2, gs_probs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_dir / "method_comparison.png", dpi=150)
        plt.close()
        print(f"Saved: method_comparison.png")

    # Detailed per-run comparison if available
    if standard_summary and dicke_summary:
        print("\n" + "=" * 70)
        print("PER-RUN COMPARISON")
        print("=" * 70)

        print("\nStandard + Penalty runs:")
        for r in standard_summary['results']:
            print(f"  Run {r['run']}: Valid={r['valid_fraction']*100:.1f}%, GS={r['gs_prob_summed']:.4f}")

        print("\nDicke + XY runs:")
        for r in dicke_summary['results']:
            print(f"  Run {r['run']}: Valid={r['valid_fraction']*100:.1f}%, GS={r['gs_prob_summed']:.4f}")

    print("\n" + "=" * 70)
    print(f"Results saved to: {output_dir}")
    print("=" * 70)

    return comparison_df


def main():
    """Run method comparison."""
    compare_standard_vs_dicke()


if __name__ == "__main__":
    main()
