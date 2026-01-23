"""3D crystal structure visualization for Li configurations."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
