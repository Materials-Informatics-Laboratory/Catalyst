from pathlib import Path
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from ase.build import bulk

from catalyst.graph.alignnd import alignn_gen


path = Path(__file__).parent


def _get_plot_positions(data):
    """
    Return 2D plotting positions for a realignnd graph.

    If atoms were stored, this handles either:
      - data.atoms as a single ASE Atoms object
      - data.atoms as a list of ASE Atoms objects from realignnd

    Otherwise, it falls back to a circular layout.
    """
    x_atm = data.x_atm.detach().cpu().numpy()
    n_atoms = x_atm.shape[0]

    atoms_obj = getattr(data, "atoms", None)

    if atoms_obj is None:
        theta = np.linspace(0, 2 * np.pi, n_atoms, endpoint=False)
        return np.column_stack([np.cos(theta), np.sin(theta)])

    # realignnd may store a list of ASE Atoms objects.
    if isinstance(atoms_obj, list):
        positions = []
        x_offset = 0.0

        for atoms in atoms_obj:
            pos = atoms.get_positions()[:, :2].copy()

            # Shift each structure to the right so they do not overlap visually.
            pos[:, 0] += x_offset

            positions.append(pos)

            if len(pos) > 0:
                x_span = pos[:, 0].max() - pos[:, 0].min()
                x_offset = pos[:, 0].max() + max(5.0, 0.25 * x_span)

        return np.vstack(positions)

    # Single ASE Atoms object.
    return atoms_obj.get_positions()[:, :2]


def visualize_graph(data):
    colors = np.array([
        "aqua", "mediumslateblue", "peru", "limegreen",
        "darkorange", "salmon", "brown", "gold"
    ])

    edge_index_G = data.edge_index_G.detach().cpu().numpy()
    x_atm = data.x_atm.detach().cpu().numpy()

    pos = _get_plot_positions(data)

    if pos.shape[0] != x_atm.shape[0]:
        raise ValueError(
            f"Number of plotting positions ({pos.shape[0]}) does not match "
            f"number of graph atoms ({x_atm.shape[0]})."
        )

    # Works for one-hot labels. If atom_labels='atomic_number', this still
    # chooses the active feature column.
    color_ids = np.argmax(x_atm != 0.0, axis=1)
    color_map = colors[color_ids % len(colors)]

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # ------------------------
    # Graph G: atomic graph
    # ------------------------
    if edge_index_G.shape[1] > 0:
        segments_G = pos[edge_index_G.T]

        ax[0].add_collection(
            LineCollection(
                segments_G,
                colors="dimgrey",
                linewidths=0.4,
                zorder=1,
            )
        )

    ax[0].scatter(
        pos[:, 0],
        pos[:, 1],
        s=100,
        c=color_map,
        edgecolors="black",
        linewidths=0.4,
        zorder=2,
    )

    ax[0].set_title("Graph G (1,2 body graph)")
    ax[0].axis("equal")
    ax[0].axis("off")

    # ------------------------
    # Graph A: angular graph
    # Nodes are bonds from G.
    # Positions are bond midpoints.
    # ------------------------
    if data.edge_index_A is not None:
        edge_index_A = data.edge_index_A.detach().cpu().numpy()

        bond_pos = 0.5 * (
            pos[edge_index_G[0]] + pos[edge_index_G[1]]
        )

        if edge_index_A.shape[1] > 0:
            segments_A = bond_pos[edge_index_A.T]

            ax[1].add_collection(
                LineCollection(
                    segments_A,
                    colors="dimgrey",
                    linewidths=0.4,
                    zorder=1,
                )
            )

        ax[1].scatter(
            bond_pos[:, 0],
            bond_pos[:, 1],
            s=50,
            c="lightgrey",
            edgecolors="black",
            linewidths=0.3,
            zorder=2,
        )

        ax[1].set_title("Graph A (2,3 body graph)")
        ax[1].axis("equal")
        ax[1].axis("off")

    else:
        ax[1].set_title("Graph A not included")
        ax[1].axis("off")

    plt.tight_layout()
    plt.show()


def build_realignnd_structures(lattice_constant=4.05, repeat=(2, 2, 2)):
    """Build three related periodic Al FCC structures entirely with ASE.

    The three states are intentionally different so the realignnd example still
    demonstrates combining multiple structures:

      1. pristine FCC Al
      2. 3% isotropically expanded FCC Al
      3. sheared FCC Al

    No external VASP files are needed.
    """
    pristine = bulk("Al", "fcc", a=lattice_constant, cubic=True).repeat(repeat)
    pristine.pbc = True

    expanded = pristine.copy()
    expanded.set_cell(expanded.cell * 1.03, scale_atoms=True)

    sheared = pristine.copy()
    sheared_cell = sheared.cell.array.copy()
    sheared_cell[0] += 0.08 * sheared_cell[1]
    sheared.set_cell(sheared_cell, scale_atoms=True)

    return [pristine, expanded, sheared]


print("Building ASE structures and generating realignnd graph...")
start = time.time()

structures = build_realignnd_structures()
for idx, structure in enumerate(structures):
    print(f"  structure {idx}: {len(structure)} Al atoms, cell={structure.cell.lengths()}")

data = {
    "type": "realignnd",
    "raw_data": structures,

    # [cutoff, k]
    # k = -1 means cutoff-based graph generation in your atoms2graph setup.
    "neighbor_params": [3.0, 4],

    "is_dihedral": False,

    # Set this to True if you want visualize_graph() to use real positions.
    # If False, visualization falls back to a circular layout.
    "store_raw_data": True,

    "use_pt": False,
    "include_angs": True,
    "node_labels": None,
    "element_list": ["Al"],
    "store_atoms_type": "ase-atoms",

    # New retry / robustness controls.
    "auto_retry_graph": True,
    "max_graph_attempts": 8,
    "k_step": 4,
    "cutoff_scale": 1.25,
    "max_k": None,
    "max_cutoff": 6.0,
    "require_bonds": True,
    "require_angles": True,
    "require_dihedrals": False,
    "retry_verbose": True,

    # If your updated realignnd parallelizes over structures, this controls it.
    # Use 1 for serial, -1 for all available cores, or a fixed number like 8.
    "cpu_cores": -1,
}

graph_data = alignn_gen(data=data)

print(graph_data)

if hasattr(graph_data, "graph_cutoff_used"):
    print("Cutoff used:", graph_data.graph_cutoff_used)

if hasattr(graph_data, "graph_k_used"):
    print("k used:", graph_data.graph_k_used)

if hasattr(graph_data, "graph_build_attempts"):
    print("Build attempts:", graph_data.graph_build_attempts)

out_file = path / "graph_data.pt"
torch.save(graph_data, out_file)

print(f"Saved: {out_file}")
print(f"Time elapsed: {time.time() - start:.4f} seconds")

visualize_graph(graph_data)