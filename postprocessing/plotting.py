"""Reusable plotting functions for QAOA results.

Includes bitstring vs. probability bar chart and related utilities.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


def plot_bitstring_probability(final_distribution, n_target, ground_state=None, output_path=None,
                                max_display=50):
    """Plot bitstring vs. probability bar chart.

    Shows individual bitstring configurations on x-axis and their measured
    probabilities on y-axis, with valid configurations highlighted.

    Args:
        final_distribution: dict {bitstring: probability}
        n_target: target particle number for validity check
        ground_state: ground state bitstring (str), or None to skip GS highlighting
        output_path: directory path to save output files
        max_display: maximum number of bitstrings to display (default 50)

    Returns:
        dict with valid_fraction and (optionally) gs_probability
    """
    output_path = Path(output_path)

    # Sort by probability descending
    sorted_items = sorted(final_distribution.items(), key=lambda x: -x[1])

    # Limit display
    display_items = sorted_items[:max_display]
    bitstrings = [item[0] for item in display_items]
    probabilities = [item[1] for item in display_items]

    # Classify each bitstring
    colors = []
    for bs in bitstrings:
        n_particles = bs.count('1')
        if ground_state is not None and bs == ground_state:
            colors.append('#FFD700')  # Gold for ground state
        elif n_particles == n_target:
            colors.append('#2ecc71')  # Green for valid
        else:
            colors.append('#e74c3c')  # Red for invalid

    # Compute statistics
    valid_fraction = sum(prob for bs, prob in final_distribution.items()
                         if bs.count('1') == n_target)

    result = {'valid_fraction': valid_fraction}

    if ground_state is not None:
        gs_probability = final_distribution.get(ground_state, 0.0)
        result['gs_probability'] = gs_probability

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))

    x_pos = np.arange(len(bitstrings))
    bars = ax.bar(x_pos, probabilities, color=colors, edgecolor='black', linewidth=0.3)

    # Mark ground state with a star
    if ground_state is not None and ground_state in bitstrings:
        gs_idx = bitstrings.index(ground_state)
        ax.plot(gs_idx, probabilities[gs_idx], '*', color='black',
                markersize=15, zorder=5)

    ax.set_xlabel('Bitstring Configuration', fontsize=11)
    ax.set_ylabel('Probability p(x)', fontsize=11)

    if ground_state is not None:
        ax.set_title(f'Bitstring Probability Distribution '
                     f'(Valid: {valid_fraction:.1%}, GS prob: {gs_probability:.4f})',
                     fontsize=12)
    else:
        ax.set_title(f'Bitstring Probability Distribution '
                     f'(Valid: {valid_fraction:.1%})',
                     fontsize=12)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(bitstrings, rotation=90, fontsize=7, family='monospace')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', edgecolor='black', label=f'Valid (N={n_target})'),
        Patch(facecolor='#e74c3c', edgecolor='black', label='Invalid'),
    ]
    if ground_state is not None:
        legend_elements.append(
            Patch(facecolor='#FFD700', edgecolor='black', label=f'Ground state ({ground_state})')
        )
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_path / "bitstring_probability.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Save CSV for OriginLab
    all_sorted = sorted(final_distribution.items(), key=lambda x: -x[1])
    csv_data = {
        'bitstring': [item[0] for item in all_sorted],
        'probability': [item[1] for item in all_sorted],
        'n_particles': [item[0].count('1') for item in all_sorted],
        'is_valid': [item[0].count('1') == n_target for item in all_sorted],
    }
    if ground_state is not None:
        csv_data['is_ground_state'] = [item[0] == ground_state for item in all_sorted]
    csv_df = pd.DataFrame(csv_data)
    csv_df.to_csv(output_path / "bitstring_probability.csv", index=False)

    return result
