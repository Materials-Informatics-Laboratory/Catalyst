"""Build a standard ALIGNN graph from a small ASE-generated Al FCC crystal.

This example is intentionally self-contained: no VASP OUTCAR or other external
structure file is required.  The structure is generated at runtime with ASE.
"""

from pathlib import Path

import torch
from ase.build import bulk

from catalyst.graph.alignnd import alignn_gen


path = Path(__file__).parent


def build_al_fcc_structure(lattice_constant: float = 4.05, repeat=(3, 3, 3)):
    """Return a periodic conventional-cell Al FCC supercell."""
    structure = bulk("Al", "fcc", a=lattice_constant, cubic=True)
    structure = structure.repeat(repeat)
    structure.pbc = True
    return structure


# Build the example directly with ASE instead of loading an OUTCAR.
structures = [build_al_fcc_structure()]

dataset = []

for i, structure in enumerate(structures):
    print(
        f"Generating graph for structure {i + 1} of {len(structures)} "
        f"({len(structure)} atoms)"
    )

    data = {
        "type": "alignnd",
        "raw_data": structure,

        # [cutoff, k]
        # k = -1 means cutoff-based graph generation in the atoms2graph setup.
        # Al FCC nearest-neighbor distance is ~2.86 A for a = 4.05 A, so 3.0 A
        # captures the first coordination shell.
        "neighbor_params": [3.0, -1],

        "is_dihedral": False,
        "store_raw_data": False,
        "use_pt": False,
        "include_angs": True,
        "node_labels": None,
        "element_list": ["Al"],
        "store_atoms_type": "ase-atoms",

        # Retry / robustness controls.
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

        # Safe to leave at 1 when not using internal angle parallelism.
        "cpu_cores": 1,
    }

    graph_data = alignn_gen(data=data)
    dataset.append(graph_data)

torch.save(dataset, path / "graph_data.pt")
print(f"Saved {len(dataset)} graph(s) to {path / 'graph_data.pt'}")
