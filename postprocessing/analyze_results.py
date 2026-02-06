"""Statistical analysis of QAOA results.

This module provides functions for analyzing QAOA results across
multiple runs, calculating statistics, and generating summary reports.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

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


def compute_statistics(values):
    """Compute statistics for a list of values.

    Returns:
        dict with mean, std, min, max, median
    """
    if not values:
        return {'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'median': 0}

    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'min': np.min(values),
        'max': np.max(values),
        'median': np.median(values)
    }


def analyze_method_results(method_name, results_dir):
    """Analyze results for a single method.

    Args:
        method_name: Name of the method
        results_dir: Directory containing results

    Returns:
        dict with analysis results
    """
    summary = load_summary(results_dir)

    if summary is None:
        return None

    analysis = {
        'method': method_name,
        'settings': summary.get('settings', {}),
        'ground_state': summary.get('ground_state', ''),
        'ground_energy': summary.get('ground_energy', 0)
    }

    # Handle different summary structures
    if 'results' in summary:
        if isinstance(summary['results'], list):
            # Multiple runs
            valid_fracs = [r['valid_fraction'] for r in summary['results']]
            gs_probs = [r['gs_prob_summed'] for r in summary['results']]
            energies = [r.get('final_cost', r.get('final_energy', 0)) for r in summary['results']]
            times = [r.get('elapsed_time', 0) for r in summary['results']]

            analysis['num_runs'] = len(summary['results'])
            analysis['valid_fraction_stats'] = compute_statistics(valid_fracs)
            analysis['gs_prob_stats'] = compute_statistics(gs_probs)
            analysis['energy_stats'] = compute_statistics(energies)
            analysis['time_stats'] = compute_statistics(times)

        elif isinstance(summary['results'], dict):
            # Single result (e.g., multistart)
            analysis['num_runs'] = 1
            analysis['valid_fraction'] = summary['results'].get('valid_fraction', 0)
            analysis['gs_prob'] = summary['results'].get('gs_prob_summed', 0)
            analysis['final_energy'] = summary['results'].get('final_energy', 0)
            analysis['total_time'] = summary['results'].get('total_time', 0)

    if 'averages' in summary:
        analysis['averages'] = summary['averages']

    return analysis


def analyze_all_results(output_dir=None):
    """Analyze results from all available methods.

    Args:
        output_dir: Output directory. If None, uses results/analysis/

    Returns:
        DataFrame with summary statistics
    """
    print("=" * 70)
    print("Statistical Analysis of QAOA Results")
    print("=" * 70)

    results_root = project_root / "results"

    # Define methods to analyze
    methods = {
        'standard_penalty': results_root / "simulator" / "standard_penalty",
        'dicke_xy': results_root / "simulator" / "dicke_xy",
        'dicke_xy_multistart': results_root / "simulator" / "dicke_xy_multistart",
        'standard_penalty_hw': results_root / "hardware" / "standard_penalty",
        'dicke_xy_hw': results_root / "hardware" / "dicke_xy",
        'dicke_xy_multistart_hw': results_root / "hardware" / "dicke_xy_multistart"
    }

    # Analyze each method
    analyses = {}
    for name, path in methods.items():
        if path.exists():
            analysis = analyze_method_results(name, path)
            if analysis is not None:
                analyses[name] = analysis
                print(f"\nFound results for: {name}")

    if not analyses:
        print("No results found to analyze!")
        return None

    # Output directory
    if output_dir is None:
        output_dir = project_root / "results" / "analysis"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build summary table
    summary_rows = []

    for name, analysis in analyses.items():
        row = {
            'Method': name,
            'Platform': 'Hardware' if '_hw' in name else 'Simulator',
            'p': analysis['settings'].get('p', ''),
            'Ground Energy': analysis.get('ground_energy', '')
        }

        if 'valid_fraction_stats' in analysis:
            row['Valid % (mean)'] = analysis['valid_fraction_stats']['mean'] * 100
            row['Valid % (std)'] = analysis['valid_fraction_stats']['std'] * 100
            row['GS Prob (mean)'] = analysis['gs_prob_stats']['mean']
            row['GS Prob (std)'] = analysis['gs_prob_stats']['std']
            row['Energy (mean)'] = analysis['energy_stats']['mean']
            row['Num Runs'] = analysis['num_runs']
        elif 'valid_fraction' in analysis:
            row['Valid % (mean)'] = analysis['valid_fraction'] * 100
            row['Valid % (std)'] = 0
            row['GS Prob (mean)'] = analysis['gs_prob']
            row['GS Prob (std)'] = 0
            row['Energy (mean)'] = analysis.get('final_energy', 0)
            row['Num Runs'] = 1
        elif 'averages' in analysis:
            row['Valid % (mean)'] = analysis['averages'].get('valid_fraction', 0) * 100
            row['GS Prob (mean)'] = analysis['averages'].get('gs_prob_summed', 0)
            row['Num Runs'] = analysis.get('num_runs', 1)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Save summary
    summary_df.to_csv(output_dir / "statistical_summary.csv", index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    # Detailed analysis for each method
    detailed_output = []
    for name, analysis in analyses.items():
        detailed_output.append(f"\n{'='*50}")
        detailed_output.append(f"Method: {name}")
        detailed_output.append(f"{'='*50}")
        detailed_output.append(f"Ground state: {analysis.get('ground_state', 'N/A')}")
        detailed_output.append(f"Ground energy: {analysis.get('ground_energy', 'N/A'):.4f} eV")

        if 'valid_fraction_stats' in analysis:
            stats = analysis['valid_fraction_stats']
            detailed_output.append(f"\nValid Fraction:")
            detailed_output.append(f"  Mean: {stats['mean']*100:.2f}%")
            detailed_output.append(f"  Std:  {stats['std']*100:.2f}%")
            detailed_output.append(f"  Min:  {stats['min']*100:.2f}%")
            detailed_output.append(f"  Max:  {stats['max']*100:.2f}%")

            stats = analysis['gs_prob_stats']
            detailed_output.append(f"\nGround State Probability:")
            detailed_output.append(f"  Mean:   {stats['mean']:.4f}")
            detailed_output.append(f"  Std:    {stats['std']:.4f}")
            detailed_output.append(f"  Min:    {stats['min']:.4f}")
            detailed_output.append(f"  Max:    {stats['max']:.4f}")

    # Save detailed analysis
    with open(output_dir / "detailed_analysis.txt", 'w') as f:
        f.write('\n'.join(detailed_output))

    # Save full analysis as JSON
    with open(output_dir / "full_analysis.json", 'w') as f:
        # Convert numpy types to Python types
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(convert_types(analyses), f, indent=2)

    print(f"\nDetailed analysis saved to: {output_dir}")
    print(f"Files: statistical_summary.csv, detailed_analysis.txt, full_analysis.json")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return summary_df


def main():
    """Run statistical analysis."""
    analyze_all_results()


if __name__ == "__main__":
    main()
