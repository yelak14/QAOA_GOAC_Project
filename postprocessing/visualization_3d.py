"""3D crystal structure visualization for Li configurations.

Can be run as a standalone script after computation:
    python postprocessing/visualization_3d.py --results-dir results/simulator/constrained
    python postprocessing/visualization_3d.py --results-dir results/hardware/standard --top-n 5
"""

import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def visualize_configuration(cif_file, occupied_sites, save_path=None,
                             title="Li₂Co₈O₁₆ Configuration"):
    """Visualize the crystal structure with occupied Li sites highlighted.

    Args:
        cif_file: path to the CIF/POSCAR file
        occupied_sites: list of occupied site indices
        save_path: if provided, save figure
        title: plot title
    """
    try:
        from pymatgen.core import Structure
        structure = Structure.from_file(cif_file)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot all atoms
        for i, site in enumerate(structure):
            coords = site.frac_coords
            element = str(site.specie)

            if element == 'Li':
                # Check if this Li site is occupied
                if i in occupied_sites:
                    ax.scatter(*coords, c='green', s=200, marker='o',
                               edgecolors='black', linewidth=2, label='Li (occupied)')
                else:
                    ax.scatter(*coords, c='lightgreen', s=100, marker='o',
                               alpha=0.3, label='Li (vacant)')
            elif element == 'Co':
                ax.scatter(*coords, c='blue', s=80, marker='^', alpha=0.6)
            elif element == 'O':
                ax.scatter(*coords, c='red', s=40, marker='s', alpha=0.4)

        ax.set_xlabel('a')
        ax.set_ylabel('b')
        ax.set_zlabel('c')
        ax.set_title(title)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    except ImportError:
        print("pymatgen not available. Using simple visualization.")
        _simple_visualization(occupied_sites, save_path, title)
    except Exception as e:
        print(f"CIF loading failed ({e}). Using simple visualization.")
        _simple_visualization(occupied_sites, save_path, title)


def _simple_visualization(occupied_sites, save_path=None, title="Li Site Configuration"):
    """Simple 2D visualization of site occupation."""
    n_sites = 8
    fig, ax = plt.subplots(figsize=(8, 3))

    for i in range(n_sites):
        color = 'green' if i in occupied_sites else 'lightgray'
        circle = plt.Circle((i, 0), 0.3, color=color, ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(i, 0, str(i), ha='center', va='center', fontsize=12, fontweight='bold')

    ax.set_xlim(-0.5, n_sites - 0.5)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('Site Index')
    ax.set_yticks([])

    occupied_str = ', '.join(str(s) for s in occupied_sites)
    ax.text(n_sites / 2, -0.7, f'Occupied sites: [{occupied_str}]',
            ha='center', fontsize=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    """Standalone entry point: read saved results and generate visualizations."""
    parser = argparse.ArgumentParser(
        description="Visualize QAOA results as crystal structure diagrams. "
                    "Reads results from saved CSV/JSON files produced by runner scripts."
    )
    parser.add_argument(
        '--results-dir', type=str, required=True,
        help='Path to results directory containing results.json and CSV files'
    )
    parser.add_argument(
        '--cif-file', type=str, default=None,
        help='Path to CIF/POSCAR file (default: data/input/POSCAR-sc.cif)'
    )
    parser.add_argument(
        '--top-n', type=int, default=3,
        help='Number of top configurations to visualize (default: 3)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory for figures (default: same as results-dir)'
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine project root for finding CIF file
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    if args.cif_file:
        cif_file = args.cif_file
    else:
        cif_file = str(project_root / "data" / "input" / "POSCAR-sc.cif")

    # --- Load results from JSON ---
    results_json = results_dir / "results.json"
    results = None
    if results_json.exists():
        with open(results_json) as f:
            results = json.load(f)
        print(f"Loaded results from {results_json}")

        # Find best configuration
        # Simulator format: results_by_p dict
        if 'results_by_p' in results:
            best_p = max(results['results_by_p'].keys())
            best_config = results['results_by_p'][best_p].get('best_config', '')
        else:
            # Hardware format: flat dict
            best_config = results.get('best_config', '')

        if best_config and best_config != 'N/A':
            occupied = [i for i, b in enumerate(best_config) if b == '1']
            print(f"Best configuration: {best_config} -> sites {occupied}")
            save_path = str(output_dir / "best_config_3d.png")
            visualize_configuration(
                cif_file, occupied, save_path=save_path,
                title=f"Best Config: sites {occupied}"
            )
            print(f"Saved: {save_path}")

        # Visualize exact ground state
        ground_state = results.get('ground_state', '')
        if ground_state:
            occupied = [i for i, b in enumerate(ground_state) if b == '1']
            save_path = str(output_dir / "ground_state_3d.png")
            visualize_configuration(
                cif_file, occupied, save_path=save_path,
                title=f"Exact Ground State: sites {occupied}"
            )
            print(f"Saved: {save_path}")
    else:
        print(f"No results.json found in {results_dir}")

    # --- Load top-N configs from CSV ---
    csv_files = sorted(results_dir.glob("qaoa_results_p*.csv"))
    if csv_files:
        # Use the last (highest p) CSV
        csv_path = csv_files[-1]
        print(f"Loading configurations from {csv_path.name}")
        df = pd.read_csv(csv_path)

        valid_df = df[df['valid'] == True].head(args.top_n)
        for rank, (_, row) in enumerate(valid_df.iterrows(), start=1):
            bs = row['bitstring']
            occupied = [i for i, b in enumerate(bs) if b == '1']
            energy = row['energy']
            site_str = '_'.join(map(str, occupied))
            save_name = f"config_rank{rank}_sites{site_str}.png"
            save_path = str(output_dir / save_name)
            visualize_configuration(
                cif_file, occupied, save_path=save_path,
                title=f"Rank {rank}: sites {occupied}, E={energy:.4f} eV"
            )
            print(f"Saved: {save_path}")
    else:
        print("No qaoa_results_p*.csv files found. Skipping top-N visualization.")

    # --- Visualize from compare_with_exact CSV ---
    compare_csv = results_dir / "compare_with_exact.csv"
    if compare_csv.exists():
        comp_df = pd.read_csv(compare_csv)
        if not comp_df.empty:
            gs = comp_df.iloc[0].get('ground_state', '')
            if gs and gs != 'N/A':
                occupied = [i for i, b in enumerate(gs) if b == '1']
                gs_prob = comp_df.iloc[0].get('ground_state_probability', 0)
                approx_ratio = comp_df.iloc[0].get('approximation_ratio', 0)
                save_path = str(output_dir / "exact_comparison_3d.png")
                visualize_configuration(
                    cif_file, occupied, save_path=save_path,
                    title=f"Ground State (prob={gs_prob:.3f}, r={approx_ratio:.3f})"
                )
                print(f"Saved: {save_path}")

    # --- Visualize top-5 from top_5_configs CSV ---
    top5_csv = results_dir / "top_5_configs.csv"
    if top5_csv.exists():
        top5_df = pd.read_csv(top5_csv)
        for rank, (_, row) in enumerate(top5_df.iterrows(), start=1):
            bs = row['bitstring']
            occupied = [i for i, b in enumerate(bs) if b == '1']
            energy = row['energy']
            prob = row['probability']
            save_name = f"top5_rank{rank}.png"
            save_path = str(output_dir / save_name)
            visualize_configuration(
                cif_file, occupied, save_path=save_path,
                title=f"Top-5 #{rank}: sites {occupied}, E={energy:.4f} eV, p={prob:.3f}"
            )
            print(f"Saved: {save_path}")

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()
