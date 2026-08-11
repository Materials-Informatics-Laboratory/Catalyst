"""Build atom-centered ALIGNN graphs from an ASE-generated Al FCC crystal.

This example is intentionally self-contained: no VASP OUTCAR or other external
structure file is required.  The structure is generated at runtime with ASE.
"""

from pathlib import Path
import time

import torch
from ase.build import bulk

from catalyst.graph.alignnd import alignn_gen


path = Path(__file__).parent


def build_al_fcc_structure(lattice_constant: float = 4.05, repeat=(2, 2, 2)):
    """Return a small periodic conventional-cell Al FCC supercell."""
    structure = bulk("Al", "fcc", a=lattice_constant, cubic=True)
    structure = structure.repeat(repeat)
    structure.pbc = True
    return structure


start = time.time()

# atomic_alignnd creates one local graph per atom, so a 2x2x2 conventional FCC
# supercell (32 atoms) is large enough to exercise periodic neighborhoods while
# keeping this source-code example quick to run.
structures = [build_al_fcc_structure()]

for i, structure in enumerate(structures):
    print(
        f"Generating atom-centered graphs for structure {i + 1} of "
        f"{len(structures)} ({len(structure)} atoms)"
    )

    data = {
        "type": "atomic_alignnd",
        "raw_data": structure,

        # [cutoff, k]
        # k = -1 means cutoff-based graph generation in the atoms2graph setup.
        "neighbor_params": [3.0, -1],

        "is_dihedral": False,
        "store_raw_data": False,
        "use_pt": False,
        "include_angs": True,
        "node_labels": None,
        "element_list": ["Al"],
        "store_atoms_type": "ase-atoms",

        # atomic_alignnd parallelizes over atoms.
        # Use -1 for all available cores, or set a fixed number such as 8.
        "cpu_cores": -1,

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
    }

    graph_data = alignn_gen(data=data)

    print(f"Generated {len(graph_data)} atom-centered graphs")

    out_file = path / f"graph_data-{i}.pt"
    torch.save(graph_data, out_file)

    print(f"Saved: {out_file}")

print(f"Time elapsed: {time.time() - start:.4f} seconds")
