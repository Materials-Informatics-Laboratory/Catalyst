from pathlib import Path
import os
import time

import torch
from ase.io import read

from catalyst.graph.alignnd import alignn_gen


path = Path(__file__).parent

start = time.time()

structures = read(
    path / "OUTCAR-0",
    index=":",
    format="vasp-out",
)

for i, structure in enumerate(structures):
    print(f"Generating atomic graph for structure {i + 1} of {len(structures)}")

    data = {
        "type": "atomic_alignnd",
        "raw_data": structure,

        # [cutoff, k]
        # k = -1 means cutoff-based graph generation in your atoms2graph setup.
        "neighbor_params": [3.0, -1],

        "is_dihedral": False,
        "store_raw_data": False,
        "use_pt": False,
        "include_angs": True,
        "node_labels": None,
        "element_list": ["Al"],
        "store_atoms_type": "ase-atoms",

        # Parallelism for atomic_alignnd.
        # atomic_alignnd parallelizes over atoms, so this is useful.
        # Use -1 for all available cores, or set a fixed number like 8 or 16.
        "cpu_cores": -1,

        # New retry / robustness controls
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